# 🇩🇪 German Named Entity Recognition  
### Frozen Transformer + BiLSTM + CRF  


---

## 🚀 Project Summary

This project implements a complete, competition-grade German Named Entity Recognition (NER) system under strict architectural constraints.

The system:

- Uses a large pretrained German Transformer as a **frozen feature extractor**
- Trains a lightweight structured prediction head
- Achieves **0.87 Macro-F1** in competitive evaluation
- Maintains full reproducibility and submission compliance

This repository demonstrates practical ML engineering skills in:

- Transformer feature extraction
- Sequence modeling
- Structured decoding (CRF)
- Robust preprocessing pipelines
- Submission packaging & deployment constraints

---

## 🏆 Competition Result

| Metric | Value |
|--------|--------|
| Final Macro-F1 | **0.87** |
| Reference Range | ~0.85 |
| Participants | 28 |
| Score | 100 / 100 |

📌 Performance exceeded the expected 85% range while respecting all constraints.

![Leaderboard Result](./leaderboard1.jpeg)

---

# 🎯 Problem Definition

Given tokenized German text, predict BIO entity labels for:

- **PER** (Person)
- **ORG** (Organization)
- **LOC** (Location)

### Constraints

- Pretrained models allowed (≤1B parameters)
- Encoder **must not be fine-tuned**
- Reproducible code submission required
- Individual implementation

Evaluation:

\[
Macro-F1 = \frac{F1_{PER} + F1_{ORG} + F1_{LOC}}{3}
\]

---

# 🧩 System Architecture Overview

```
TSV Input
   ↓
Robust Sentence Parsing
   ↓
Subword Alignment
   ↓
Frozen Transformer (Gelectra-Large)
   ↓
Layer Concatenation (4096-dim)
   ↓
Projection + GELU + Dropout
   ↓
BiLSTM
   ↓
Linear Layer
   ↓
CRF (Structured Decoding)
   ↓
BIO Tag Output
```

---

# 🧠 1️⃣ Feature Extraction Strategy

### Backbone

- Model: `deepset/gelectra-large`
- Hidden size: 1024
- Last 4 layers concatenated → 4096 features
- Fully frozen (no fine-tuning)

Total parameters: ~338M (<1B constraint)

### Engineering Rationale

- Preserve pretrained linguistic knowledge
- Prevent catastrophic forgetting
- Avoid overfitting
- Transfer high-quality contextual embeddings into task-specific head

This mirrors real-world production scenarios where large encoders are reused across tasks.

---

# 🏗 2️⃣ Sequence Modeling Head

### Architecture

- Linear(4096 → 512)
- GELU activation
- Dropout (0.5)
- BiLSTM (bidirectional, hidden=256)
- Linear → BIO logits
- CRF layer

### Why CRF?

- Enforces valid BIO transitions
- Prevents illegal sequences (e.g., I-ORG without B-ORG)
- Improves entity boundary consistency

### Why BiLSTM on top of Transformer?

Because encoder is frozen:

- Head must learn sequence adaptation
- LSTM captures bidirectional contextual refinement
- Improves boundary decisions

---

# 🧹 3️⃣ Data Engineering & Alignment

### Robust TSV Parsing

- Skips comment lines (`#`)
- Detects sentence boundaries via empty rows
- Reconstructs clean token sequences

### Label Cleaning Strategy

Retained:
- PER
- ORG
- LOC

Mapped to `O`:
- OTH
- part
- deriv

Ensures strict evaluation alignment.

### Subword Alignment

- Used `tokenizer.word_ids()`
- Only first subword used per token
- Constructed `word_mask` to:
  - Ignore padding
  - Ignore non-first subwords
  - Preserve 1:1 token-label mapping

This avoids label duplication artifacts.

---

# ⚙️ 4️⃣ Training Strategy

| Component | Value |
|------------|---------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Scheduler | OneCycleLR |
| Dropout | 0.5 |
| Gradient Clipping | 1.0 |
| Epochs | 10 |

### Loss

- CRF Loss (final model)

### Evaluation

- `seqeval`
- Strict span-based entity evaluation

### Early Stopping

- Based on Validation Macro-F1

---

# 🛡 5️⃣ Generalization & Robustness Design

This system intentionally favors **stability over aggressive recall boosting**.

Key decisions:

- Frozen encoder to reduce overfitting
- High dropout
- No oversampling
- Gradient clipping
- Conservative decision boundaries

Observed challenge:

Capitalized common nouns in German (e.g., *Arbeit*, *Liga*, *BA*) increase FP risk.

Model tuned to reduce false positives rather than inflate recall artificially.



---

# 📊 Performance Summary

Notebook Validation:

| Metric | Score |
|---|---:|
| Macro-F1 | **0.9006** |
| PER F1 | 0.9549 |
| ORG F1 | 0.8212 |
| LOC F1 | 0.9257 |

Competition Evaluation:

```
Macro-F1 = 0.87
```

Performance remains strong under hidden distribution.

---



# 👤 Author

Mohamed Elsayed
M.Sc. Data Science  

