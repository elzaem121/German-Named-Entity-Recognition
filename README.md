# German Named Entity Recognition (Task 4)

This project implements a complete pipeline for **German NER** with three target entity types:

- `PER` (Person)
- `ORG` (Organization)
- `LOC` (Location)

The solution is designed for the Task 4 submission format and includes training, evaluation, export, and submission validation artifacts.

## 1. Task Definition

Given tokenized German text, the model predicts BIO tags for the three target entities.

Evaluation metric:

- `F1_PER`
- `F1_ORG`
- `F1_LOC`

Final score:

`Macro-F1 = (F1_PER + F1_ORG + F1_LOC) / 3`

The task description mentions strong systems around ~85% F1.

## 2. Constraint Compliance

The implementation complies with the stated rules:

- Uses a pretrained language model as backbone (`deepset/gelectra-large`).
- The pretrained encoder is used as a **frozen feature extractor** (not fine-tuned for NER).
- Total parameters are under the 1B limit.

Parameter count from saved weights:

- Encoder: `334,686,208`
- NER head: `3,678,278`
- Total: `338,364,486`

## 3. Reported Validation Results

From the notebook evaluation run:

| Metric | Score |
|---|---:|
| Macro-F1 | **0.9006** |
| PER F1 | 0.9549 |
| ORG F1 | 0.8212 |
| LOC F1 | 0.9257 |

This exceeds the ~85% reference range from the task statement.

## 4. Model Architecture

### 4.1 Frozen Backbone

- Model: `deepset/gelectra-large`
- Hidden size: `1024`
- Feature representation: concatenation of last 4 hidden layers (`4 x 1024 = 4096`)
- Backbone remains frozen during NER head training.

### 4.2 Trainable NER Head (Notebook)

Pipeline:

1. Subword features from frozen encoder
2. Word-level compression (first subword per word id)
3. Projection: `Linear(4096 -> 512)` + `GELU` + `Dropout(0.5)`
4. BiLSTM (`hidden_dim=256`, bidirectional, 1 layer)
5. Classifier to 7 BIO tags
6. CRF for sequence modeling / decoding

Implementation notes:

- Multi-sample dropout is used during training (`5` dropout heads averaged).
- Code includes weighted CE + CRF hybrid-loss scaffolding.
- In the captured notebook configuration, `crf_weight=1` and `ce_weight=0` (effectively CRF loss in that run).

### 4.3 Submission Inference Head (`model.py`)

For portability, submission code uses a minimal custom CRF decode implementation that loads CRF transition parameters from `weights/head.pt`, avoiding runtime dependency on `torchcrf`.

## 5. Data Format and Preprocessing

Data files are TSV-based (train/val/test) under `public_data-4/`.

Parsing strategy:

- Sentence boundaries are detected by empty lines.
- Metadata/comment lines are skipped.
- Valid token lines are read from tab-separated rows.

Label handling:

- Target labels kept: `O, B/I-PER, B/I-ORG, B/I-LOC`
- Non-target labels (e.g., `OTH`, `part`, `deriv`) are mapped to `O`.

This aligns labels with the official evaluation scope.

## 6. Training Recipe (from `task_4.ipynb`)

Device selection:

- `cuda` if available, else `mps`, else `cpu`.

Key hyperparameters:

| Component | Value |
|---|---|
| Max sequence length (training) | `128` |
| Train batch size | `8` |
| Validation batch size | `16` |
| Epochs | `10` |
| Optimizer | `AdamW` |
| Learning rate | `3e-4` |
| Weight decay | `0.01` |
| Scheduler | `OneCycleLR(max_lr=3e-4)` |
| Head dropout | `0.5` |

Data augmentation used in notebook:

- Entity swapping with same entity type and same token length (`swap_prob=0.15`)
- Context token dropout to `[UNK]` on non-entity words (`dropout_prob=0.1`)

Training outputs:

- `best_head_weighted.pt` (best checkpoint by validation Macro-F1)
- `training_history_weighted.pt`

## 7. Repository Layout

```text
task-4/
├── task_4.ipynb                  # full training/evaluation/export workflow
├── model.py                      # submission model class with predict API
├── requirements.txt
├── public_data-4/
│   ├── train.tsv
│   ├── val.tsv
│   ├── test_x.tsv
│   └── test_y.tsv
├── submission/                   # packaged submission directory
│   ├── model.py
│   └── weights/
│       ├── head.pt
│       └── encoder/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── special_tokens_map.json
│           └── vocab.txt
├── final_submission/             # final variant of submission folder
├── best_head_weighted.pt
├── complete_model_weighted.pt
└── submission.zip
```

## 8. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torchcrf
```

`torchcrf` is required for notebook training. The final submission `model.py` does not require it.

## 9. Reproduce Training and Evaluation

Main workflow:

```bash
jupyter notebook task_4.ipynb
```

Run notebook sections in order:

1. Data loading and label cleaning
2. Frozen feature extractor setup
3. Dataset/collate and augmentation
4. Head training loop
5. Validation reporting and checkpoint save
6. Export encoder/head into `submission/weights/`
7. Package and validate `submission.zip`

## 10. Submission Packaging

Expected minimal submission structure:

```text
model.py
weights/head.pt
weights/encoder/config.json
weights/encoder/model.safetensors
weights/encoder/tokenizer.json
```

Notebook checks include:

- ZIP content verification
- Dynamic import test from extracted zip
- Runtime `Model().predict(...)` sanity checks

Recorded package size:

- `submission.zip`: ~`603.20 MB`

## 11. Inference API Contract

`model.py` exposes:

- Class: `Model`
- Method: `predict(x_test: np.ndarray) -> np.ndarray`

Input:

- `x_test`: 1D NumPy array of tokens for one sentence

Output:

- 1D NumPy array of predicted BIO labels (same length as input tokens)

Example:

```python
import numpy as np
from model import Model

m = Model()
tokens = np.array(["Angela", "Merkel", "besucht", "Berlin", "."])
pred_labels = m.predict(tokens)
print(pred_labels)
```

## 12. Practical Notes

- Inference in submission `model.py` is set to CPU.
- Encoder is converted to FP32 at load time for stable CPU execution.
- Tokenization truncates long sequences at `max_length=256` in inference.

## 13. Known Limitations

- Long sentences beyond max length are truncated.
- ORG performance is lower than PER/LOC in the reported run.
- Training is notebook-centric (no standalone CLI training script yet).

## 14. Suggested Next Improvements

- Add a script-based training entry point (`train.py`) for reproducibility.
- Add deterministic seed control and logging config exports.
- Tune for ORG entity recall (class weighting and augmentation policy).
