# Memory-Based Cognitive Twin
### *Personalized Decision Prediction and Email Response Generation using LLMs*

> **University Mini-Project** — ML/Research Experiment  
> Research Question: *"Can historical memory help an AI predict a user's decision and generate a response that resembles how that user would actually respond?"*

---

## Project Overview

This project implements a **Memory-Based Cognitive Twin** — a system that learns from a user's email history to:

1. **Predict decisions** (ACCEPT / REJECT / MAYBE / DEFER / REQUEST_MORE_INFORMATION) using ML classifiers trained on historical email features.
2. **Generate personalized responses** by retrieving the 20 most relevant historical emails (memory retrieval via sentence embeddings + cosine similarity) and passing them to a Google Gemini LLM to draft a reply that matches the user's communication style.

```
Historical Emails
      ↓
Create Memory Embeddings  (SentenceTransformer all-MiniLM-L6-v2)
      ↓
Retrieve TOP 20 Relevant Historical Memories  (cosine similarity, timestamp < T)
      ↓
ML Model Predicts Decision  (Random Forest + TF-IDF + Embeddings)
      ↓
LLM Generates Personalized Response  (Google Gemini)
      ↓
Evaluate Prediction + Generated Response
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
# Windows
copy .env.example .env
# Then open .env and add your Gemini API key
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Generate the synthetic dataset

```bash
python scripts/generate_dataset.py
```

### 4. Run the full research experiment

```bash
python research_experiment.py
```

This trains both models, runs response generation, evaluates everything, and saves 5 figures + 3 metric JSON files.

### 5. Run the interactive demo

```bash
python response_demo.py                   # first test email
python response_demo.py --user alice      # Alice's test email
python response_demo.py --user bob --idx 2  # Bob's 3rd test email
```

### 6. Run the Jupyter notebook

```bash
jupyter notebook notebooks/cognitive_twin_experiment.ipynb
```

### 7. Run tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Project Structure

```
project/
├── data/
│   ├── raw/emails.csv              ← synthetic dataset (generated)
│   └── processed/                 ← cached embeddings (auto-generated)
├── notebooks/
│   └── cognitive_twin_experiment.ipynb
├── src/
│   ├── preprocessing.py           ← data loading, chronological split
│   ├── memory.py                  ← embedding model, TOP-20 retrieval
│   ├── features.py                ← TF-IDF + embedding features
│   ├── models.py                  ← Logistic Regression + Random Forest
│   ├── response_generator.py      ← Gemini API wrapper (+ mock mode)
│   └── evaluation.py              ← all metrics + 5 graphs
├── results/
│   ├── figures/                   ← 5 PNG plots
│   ├── metrics/                   ← JSON metric files
│   └── predictions/               ← test predictions CSV
├── scripts/
│   └── generate_dataset.py        ← creates synthetic dataset
├── tests/test_pipeline.py         ← pytest suite (25+ tests)
├── research_experiment.py         ← end-to-end experiment runner
├── response_demo.py               ← interactive single-email demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset

**Synthetic** — 300 emails across 5 user personas:

| User  | Style      | Description                              |
|-------|------------|------------------------------------------|
| alice | formal     | Structured emails, full sentences, professional sign-offs |
| bob   | casual     | Informal, short sentences, colloquial vocabulary |
| carol | friendly   | Warm, enthusiastic, exclamation marks     |
| dave  | concise    | Terse, minimal words, direct responses    |
| eve   | analytical | Data-driven language, structured reasoning |

**Decision classes** (balanced): ACCEPT (30%), REJECT (20%), MAYBE (20%), DEFER (15%), REQUEST_MORE_INFORMATION (15%)

> ⚠️ This is synthetic data. Results on this dataset do not generalize to real email behavior.

---

## Methodology

### Task 1 — Decision Prediction

**Split strategy**: Strictly chronological — no shuffling. All training records precede all validation records, which precede all test records. This mirrors real deployment where only past emails are available.

| Model | Features |
|-------|----------|
| Baseline | Logistic Regression on TF-IDF (3000 features, bigrams) |
| CognitiveTwin | Random Forest on TF-IDF + Sentence Embeddings (combined) |

**Leakage prevention**:
- TF-IDF fitted on training data only
- No val/test statistics used at any stage
- `user_id` excluded from feature set

### Task 2 — Memory-Based Response Generation

For each target email at timestamp **T**:
1. Embed the incoming email with `all-MiniLM-L6-v2`
2. Pool all past emails from the same user with timestamp **< T**
3. Rank by cosine similarity; retrieve TOP-20
4. Build a structured prompt with the 20 memories + predicted decision
5. Call Gemini API to generate a personalized response
6. Evaluate against the held-out actual response

**The actual response is NEVER passed to the LLM** — it is held out for evaluation only.

---

## Evaluation Metrics

### Decision Prediction
- Accuracy, Macro-F1, Macro-Precision, Macro-Recall
- Confusion matrix

### Response Generation
| Metric | How Measured |
|--------|-------------|
| Semantic Similarity | Cosine similarity between generated and actual response embeddings |
| Style Similarity | Comparison of avg sentence length, vocabulary richness, formality ratio |
| Relevance | Cosine similarity between generated response and incoming email |
| Decision Consistency | Keyword presence matching predicted decision |

---

## Graphs (in `results/figures/`)

| File | Description |
|------|-------------|
| `fig1_prediction_comparison.png` | Accuracy & Macro-F1: Baseline vs CognitiveTwin |
| `fig2_confusion_matrix.png` | Confusion matrix heatmap |
| `fig3_memory_similarity_distribution.png` | Distribution of retrieval similarity scores |
| `fig4_response_metrics.png` | 4 response evaluation metrics |
| `fig5_style_comparison.png` | Generated vs actual style features per user |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes (for real mode) | — | Google Gemini API key |
| `LLM_MODEL` | No | `gemini-1.5-flash` | Gemini model name |

If `LLM_API_KEY` is not set, the system runs in **MOCK MODE** automatically. Mock responses are clearly labelled `[MOCK]` and must not be used as research results.

---

## Limitations

1. **Synthetic data**: All emails and responses are computer-generated. Results do not reflect real human behavior or decision patterns.
2. **Evaluation metrics are proxies**: Semantic similarity and style metrics approximate — not prove — response quality.
3. **Small dataset**: 300 emails is sufficient for a university experiment but insufficient to train production-grade models.
4. **Fixed personas**: Real users have more complex, evolving communication styles.
5. **LLM evaluation**: The generated response is not evaluated by a human judge; automated metrics have known limitations.
6. **Decision labels**: In real email datasets, decisions are often implicit and multi-label.
7. **Embedding model**: `all-MiniLM-L6-v2` is lightweight but not specialized for email retrieval.

---

## Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn
- sentence-transformers (`all-MiniLM-L6-v2`, ~22 MB download on first run)
- google-generativeai
- matplotlib, seaborn
- python-dotenv
- pytest
