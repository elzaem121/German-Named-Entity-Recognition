import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# -------------------------
# CRF (decode only) + load transitions from state_dict
# -------------------------
class CRFDecode(nn.Module):
    """
    Minimal CRF layer that supports Viterbi decoding.
    Parameters are loaded from head.pt:
      - start_transitions: (num_labels,)
      - end_transitions:   (num_labels,)
      - transitions:       (num_labels, num_labels)  [from -> to]
    """
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor):
        """
        emissions: (B, T, C)
        mask:      (B, T) bool
        returns: list[list[int]] of length B
        """
        B, T, C = emissions.shape
        mask = mask.bool()

        # score: (B, C) for t=0
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B,C)
        history = []

        for t in range(1, T):
            emit_t = emissions[:, t].unsqueeze(1)  # (B,1,C)
            # broadcast:
            # score.unsqueeze(2): (B,C,1)
            # transitions:        (C,C)  -> (1,C,C)
            # sum -> (B,C,C) where dim1 is prev_tag, dim2 is next_tag
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t  # (B,C,C)
            # best prev for each next_tag
            best_score, best_path = next_score.max(dim=1)  # (B,C), (B,C)
            history.append(best_path)

            # if mask is False at t, keep previous score (no update)
            m = mask[:, t].unsqueeze(1)  # (B,1)
            score = torch.where(m, best_score, score)

        score = score + self.end_transitions.unsqueeze(0)  # (B,C)

        # backtrack
        seq_ends = mask.long().sum(dim=1) - 1  # (B,)
        best_last_score, best_last_tag = score.max(dim=1)  # (B,)

        best_paths = []
        for i in range(B):
            end_t = int(seq_ends[i].item())
            tag = int(best_last_tag[i].item())
            path = [tag]

            # history length = T-1, but effective length may be shorter due to mask
            for t in range(end_t - 1, -1, -1):
                tag = int(history[t][i, tag].item())
                path.append(tag)

            path.reverse()
            best_paths.append(path)

        return best_paths


# -------------------------
# Head (same as training, but no torchcrf dependency)
# -------------------------
class NERHeadWithCRF(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256, num_labels=7, dropout=0.5):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            512,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRFDecode(num_labels)

    def compress_to_word_level(self, features, word_indices, max_words: int):
        """
        Take first subword embedding per word_id (like your training code).
        features: (B, S, D)
        word_indices: (B, S) with -1 for special tokens
        """
        B, S, D = features.shape
        device = features.device
        max_words = max(int(max_words), 1)

        word_features = torch.zeros(B, max_words, D, device=device)
        word_mask = torch.zeros(B, max_words, dtype=torch.bool, device=device)

        for i in range(B):
            seen = set()
            for j in range(S):
                w = int(word_indices[i, j].item())
                if w >= 0 and w < max_words and w not in seen:
                    word_features[i, w] = features[i, j]
                    word_mask[i, w] = True
                    seen.add(w)

        return word_features, word_mask

    @torch.no_grad()
    def decode(self, features, word_indices):
        valid = word_indices[word_indices >= 0]
        W = int(valid.max().item()) + 1 if valid.numel() else 1

        word_features, word_mask = self.compress_to_word_level(features, word_indices, max_words=W)
        x = self.projection(word_features)
        x, _ = self.lstm(x)
        emissions = self.classifier(x)  # (B,W,C)

        return self.crf.decode(emissions, word_mask), word_mask


# -------------------------
# Submission Model
# -------------------------
class Model:
    def __init__(self):
        self.device = torch.device("cpu")

        self.id2label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
        }

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.encoder_dir = os.path.join(base_dir, "weights", "encoder")
        self.head_path = os.path.join(base_dir, "weights", "head.pt")

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_dir)
        self.encoder = AutoModel.from_pretrained(self.encoder_dir).to(self.device)
        self.encoder.eval()

        # If you saved encoder in FP16 on disk, run in float32 on CPU safely:
        self.encoder = self.encoder.float()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = NERHeadWithCRF(input_dim=4096, hidden_dim=256, num_labels=7, dropout=0.5).to(self.device)
        state = torch.load(self.head_path, map_location=self.device)

        # IMPORTANT: allow loading of all head + crf params from your saved state
        missing, unexpected = self.head.load_state_dict(state, strict=False)
        # (No prints in submission required, but safe to keep silent)

        self.head.eval()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        tokens = x_test.tolist()
        if len(tokens) == 0:
            return np.array([], dtype=object)

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        word_ids = enc.word_ids(0)
        word_indices = torch.tensor(
            [(-1 if w is None else int(w)) for w in word_ids],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            features = torch.cat(out.hidden_states[-4:], dim=-1)  # (1,S,4096)

            pred_ids_list, word_mask = self.head.decode(features, word_indices)
            pred_ids = pred_ids_list[0]  # list[int] length == num_words

        labels = [self.id2label[i] for i in pred_ids]

        # Ensure output length matches original tokens length
        n = len(tokens)
        if len(labels) < n:
            labels += ["O"] * (n - len(labels))
        elif len(labels) > n:
            labels = labels[:n]

        return np.array(labels, dtype=object)
