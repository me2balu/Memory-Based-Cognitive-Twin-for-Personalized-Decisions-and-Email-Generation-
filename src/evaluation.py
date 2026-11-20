"""
evaluation.py
--------------
All evaluation logic for both tasks + the 5 required graphs.

Task 1 — Decision Prediction:
  - Accuracy, Macro-F1, Precision, Recall, Confusion Matrix

Task 2 — Response Generation:
  1. Semantic Similarity  : cosine(generated_emb, actual_response_emb)
  2. Style Similarity     : avg sentence length, type-token ratio, formality proxy
  3. Relevance            : cosine(generated_emb, incoming_email_emb)
  4. Decision Consistency : keyword matching vs predicted decision

Graphs (saved to results/figures/):
  Fig 1 — Bar: Accuracy & Macro-F1 — Baseline vs CognitiveTwin
  Fig 2 — Heatmap: Confusion matrix (CognitiveTwin)
  Fig 3 — Histogram: memory retrieval similarity scores
  Fig 4 — Bar: 4 response evaluation metrics
  Fig 5 — Grouped bar: generated vs actual style features per user
"""

import pathlib
import json
import re
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from typing import List, Dict, Any, Tuple
from src.memory import embed_texts

# -- Paths ---------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

DECISION_CLASSES = ["ACCEPT", "DEFER", "MAYBE", "REJECT", "REQUEST_MORE_INFORMATION"]

# -- TASK 1: Decision prediction metrics ---------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute classification metrics for decision prediction."""
    acc    = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=DECISION_CLASSES)
    report = classification_report(y_true, y_pred, labels=DECISION_CLASSES,
                                   zero_division=0, output_dict=True)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": report,
    }
    print(f"\n[evaluation] {model_name}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Macro Prec : {prec:.4f}")
    print(f"  Macro Rec  : {rec:.4f}")
    return metrics


def save_metrics(metrics: Dict[str, Any], filename: str) -> None:
    path = MET_DIR / filename
    # Convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=convert)
    print(f"[evaluation] Metrics saved → {path.name}")


# -- TASK 2: Response generation metrics ---------------------------------------

def semantic_similarity(generated: str, reference: str) -> float:
    """
    Cosine similarity between embeddings of generated and actual response.
    Measures how semantically close the generated response is to the real one.
    """
    embs = embed_texts([generated, reference])
    return float(embs[0] @ embs[1])   # both are L2-normalised


def _style_features(text: str) -> Dict[str, float]:
    """Extract simple stylometric features."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b\w+\b", text.lower())
    avg_sent_len = np.mean([len(w.split()) for w in sentences]) if sentences else 0.0
    ttr = len(set(words)) / len(words) if words else 0.0   # type-token ratio

    # Very simple formality proxy: ratio of formal words vs total
    formal_words = {
        "sincerely", "regards", "dear", "please", "kindly", "thank",
        "appreciate", "accordingly", "therefore", "furthermore", "however",
    }
    formal_ratio = sum(1 for w in words if w in formal_words) / max(len(words), 1)

    return {
        "avg_sentence_length": round(float(avg_sent_len), 2),
        "type_token_ratio": round(ttr, 4),
        "formality_ratio": round(formal_ratio, 4),
    }


def style_similarity(generated: str, reference: str) -> float:
    """
    Compare stylometric features of generated vs reference text.
    Returns a scalar in [0, 1] (1 = identical style profile).
    """
    gen_f = _style_features(generated)
    ref_f = _style_features(reference)

    # Normalize differences for each feature and average
    keys = list(gen_f.keys())
    diffs = []
    for k in keys:
        g, r = gen_f[k], ref_f[k]
        max_val = max(abs(g), abs(r), 1e-9)
        diffs.append(abs(g - r) / max_val)

    return round(1.0 - float(np.mean(diffs)), 4)


def relevance_score(generated: str, incoming_email: str) -> float:
    """
    Cosine similarity between generated response and the incoming email.
    Measures whether the response actually addresses the email content.
    """
    embs = embed_texts([generated, incoming_email])
    return float(embs[0] @ embs[1])


# Decision-consistency keyword map
_DECISION_KEYWORDS: Dict[str, List[str]] = {
    "ACCEPT":   ["yes", "agree", "confirm", "happy", "works", "sure", "absolutely",
                 "pleased", "certainly", "great", "perfect", "attend", "register"],
    "REJECT":   ["no", "decline", "unable", "unfortunately", "regret", "cannot",
                 "sorry", "not possible", "can't", "nope", "reject"],
    "MAYBE":    ["maybe", "possibly", "uncertain", "consider", "think", "perhaps",
                 "not sure", "tbd", "possibly", "review", "evaluate"],
    "DEFER":    ["later", "defer", "postpone", "next", "future", "reschedule",
                 "table", "another time", "revisit", "upcoming"],
    "REQUEST_MORE_INFORMATION": [
        "more", "details", "clarify", "information", "context", "explain",
        "what", "why", "how", "specify", "data", "metrics",
    ],
}


def decision_consistency(generated: str, predicted_decision: str) -> float:
    """
    Check whether generated response contains keywords consistent with the
    predicted decision.  Returns score in [0, 1].
    """
    text_lower = generated.lower()
    keywords = _DECISION_KEYWORDS.get(predicted_decision, [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text_lower)
    return round(min(hits / max(len(keywords) * 0.3, 1), 1.0), 4)


def evaluate_response(
    generated: str,
    actual_response: str,
    incoming_email: str,
    predicted_decision: str,
) -> Dict[str, float]:
    """Run all 4 response evaluation metrics and return a dict."""
    sem = semantic_similarity(generated, actual_response)
    sty = style_similarity(generated, actual_response)
    rel = relevance_score(generated, incoming_email)
    dec = decision_consistency(generated, predicted_decision)
    return {
        "semantic_similarity": round(sem, 4),
        "style_similarity":    round(sty, 4),
        "relevance":           round(rel, 4),
        "decision_consistency": round(dec, 4),
    }


# -- GRAPH 1: Prediction performance comparison --------------------------------

def plot_prediction_comparison(
    baseline_metrics: Dict[str, Any],
    ct_metrics: Dict[str, Any],
) -> None:
    """Fig 1 — Bar chart: Accuracy & Macro-F1 for Baseline vs CognitiveTwin."""
    models = ["Baseline\n(LogReg+TF-IDF)", "CognitiveTwin\n(RF+Embeddings)"]
    accuracy = [baseline_metrics["accuracy"], ct_metrics["accuracy"]]
    macro_f1 = [baseline_metrics["macro_f1"], ct_metrics["macro_f1"]]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy", color="#4C72B0", alpha=0.9)
    bars2 = ax.bar(x + width/2, macro_f1, width, label="Macro F1", color="#DD8452", alpha=0.9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Decision Prediction Performance: Baseline vs CognitiveTwin", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig1_prediction_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- GRAPH 2: Confusion matrix -------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "CognitiveTwin Confusion Matrix",
) -> None:
    """Fig 2 — Heatmap: confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=DECISION_CLASSES)
    labels_short = ["ACCEPT", "DEFER", "MAYBE", "REJECT", "REQ_INFO"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_short, yticklabels=labels_short,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig2_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- GRAPH 3: Memory retrieval similarity distribution -------------------------

def plot_memory_similarity_distribution(
    all_memories: List[List[Dict[str, Any]]],
) -> None:
    """Fig 3 — Histogram: cosine similarity scores of all retrieved memories."""
    scores = [
        m["similarity_score"]
        for mem_list in all_memories
        for m in mem_list
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=30, color="#55A868", alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(scores), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(scores):.3f}")
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Memory Retrieval Similarity Scores", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig3_memory_similarity_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- GRAPH 4: Response evaluation metrics -------------------------------------

def plot_response_metrics(response_eval_list: List[Dict[str, float]]) -> None:
    """Fig 4 — Bar chart: mean of 4 response evaluation metrics."""
    keys   = ["semantic_similarity", "style_similarity", "relevance", "decision_consistency"]
    labels = ["Semantic\nSimilarity", "Style\nSimilarity", "Relevance", "Decision\nConsistency"]
    means  = [np.mean([r[k] for r in response_eval_list]) for k in keys]
    stds   = [np.std([r[k] for r in response_eval_list]) for k in keys]

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colours, alpha=0.88,
                  capsize=5, edgecolor="white")
    ax.set_ylabel("Score (mean ± std)", fontsize=12)
    ax.set_title("Response Generation Evaluation Metrics", fontsize=13)
    ax.set_ylim(0, 1.2)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig4_response_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- GRAPH 5: Generated vs Actual style comparison per user --------------------

def plot_style_comparison_per_user(
    results: List[Dict[str, Any]],
) -> None:
    """
    Fig 5 — Grouped bar: compare avg_sentence_length and type_token_ratio
    between generated and actual responses for each user.
    """
    user_ids = sorted(set(r["user_id"] for r in results))
    gen_asl, act_asl = [], []
    gen_ttr, act_ttr = [], []

    for uid in user_ids:
        user_res = [r for r in results if r["user_id"] == uid]
        gen_asl.append(np.mean([_style_features(r["generated"])["avg_sentence_length"] for r in user_res]))
        act_asl.append(np.mean([_style_features(r["actual"])["avg_sentence_length"]    for r in user_res]))
        gen_ttr.append(np.mean([_style_features(r["generated"])["type_token_ratio"]    for r in user_res]))
        act_ttr.append(np.mean([_style_features(r["actual"])["type_token_ratio"]       for r in user_res]))

    x = np.arange(len(user_ids))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Avg sentence length
    ax1.bar(x - width/2, gen_asl, width, label="Generated", color="#4C72B0", alpha=0.9)
    ax1.bar(x + width/2, act_asl, width, label="Actual",    color="#DD8452", alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(user_ids, fontsize=10)
    ax1.set_ylabel("Avg Sentence Length (words)", fontsize=11)
    ax1.set_title("Avg Sentence Length per User", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # Type-token ratio
    ax2.bar(x - width/2, gen_ttr, width, label="Generated", color="#55A868", alpha=0.9)
    ax2.bar(x + width/2, act_ttr, width, label="Actual",    color="#C44E52", alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(user_ids, fontsize=10)
    ax2.set_ylabel("Type-Token Ratio (vocabulary richness)", fontsize=11)
    ax2.set_title("Vocabulary Richness per User", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Generated vs Actual Response Style Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig5_style_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- GRAPH 6: Ablation study comparison ----------------------------------------

def plot_ablation_comparison(
    ablation_results: Dict[str, List[Dict[str, float]]],
) -> None:
    """
    Fig 6 — Grouped bar chart: ablation study across 3 conditions.

    Conditions (R2.4 / R3.2):
      - no_context  : no memories, no ML decision (pure LLM baseline)
      - memory_only : retrieved memories, but no ML-predicted decision
      - full        : retrieved memories + ML-predicted decision (full system)

    Shows the marginal contribution of each component to response quality.
    """
    conditions     = ["no_context", "memory_only", "full"]
    cond_labels    = ["No Context\n(LLM Baseline)", "Memory Only\n(+Retrieval)", "Full System\n(+Decision Pred.)"]
    metrics        = ["semantic_similarity", "style_similarity", "relevance", "decision_consistency"]
    metric_labels  = ["Semantic\nSimilarity", "Style\nSimilarity", "Relevance", "Decision\nConsistency"]
    colours        = ["#C44E52", "#DD8452", "#4C72B0"]   # red → orange → blue

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (cond, label, colour) in enumerate(zip(conditions, cond_labels, colours)):
        results = ablation_results.get(cond, [])
        if not results:
            continue
        means  = [np.mean([r[m] for r in results]) for m in metrics]
        stds   = [np.std( [r[m] for r in results]) for m in metrics]
        offset = (i - 1) * width
        bars   = ax.bar(
            x + offset, means, width,
            label=label, color=colour, alpha=0.88,
            capsize=4, yerr=stds, edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_ylabel("Score (mean ± std)", fontsize=12)
    ax.set_title(
        "Ablation Study: Contribution of Memory Retrieval and Decision Prediction",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig6_ablation_study.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- GRAPH 7: Persona diversity (evidence of individual preference learning) ----

def analyze_persona_diversity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute per-persona stylometric statistics from historical user_responses.

    Returns a dict keyed by user_id, each with mean ± std of:
      - avg_sentence_length : mean words per sentence
      - type_token_ratio    : lexical diversity (unique words / total words)
      - formality_ratio     : proportion of formal-register words

    Purpose (R3.1): Demonstrates that the system captures individual
    communication styles rather than a single generic pattern.  Statistically
    distinct per-user profiles are evidence that memory retrieval targets
    *individual* preferences.
    """
    diversity: Dict[str, Any] = {}
    for user_id in sorted(df["user_id"].unique()):
        user_responses = df[df["user_id"] == user_id]["user_response"].tolist()
        feats = [_style_features(resp) for resp in user_responses]
        diversity[user_id] = {
            "avg_sentence_length": {
                "mean": round(float(np.mean([f["avg_sentence_length"] for f in feats])), 2),
                "std":  round(float(np.std( [f["avg_sentence_length"] for f in feats])), 2),
            },
            "type_token_ratio": {
                "mean": round(float(np.mean([f["type_token_ratio"] for f in feats])), 4),
                "std":  round(float(np.std( [f["type_token_ratio"] for f in feats])), 4),
            },
            "formality_ratio": {
                "mean": round(float(np.mean([f["formality_ratio"] for f in feats])), 4),
                "std":  round(float(np.std( [f["formality_ratio"] for f in feats])), 4),
            },
            "n_samples": len(feats),
        }
    return diversity


def plot_persona_diversity(diversity_data: Dict[str, Any]) -> None:
    """
    Fig 7 — 3-panel bar chart: stylometric feature profile per persona.

    Each bar represents one user's mean score (± std) on a stylometric feature.
    Distinct bar heights confirm that each persona has a measurably different
    communication style — directly addressing Reviewer 3, Comment 1.
    """
    users   = sorted(diversity_data.keys())
    feature_keys    = ["avg_sentence_length", "type_token_ratio", "formality_ratio"]
    feature_labels  = ["Avg Sentence Length\n(words)", "Type-Token Ratio\n(lexical diversity)", "Formality Ratio"]
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, fkey, flabel, colour in zip(axes, feature_keys, feature_labels, colours):
        means = [diversity_data[u][fkey]["mean"] for u in users]
        stds  = [diversity_data[u][fkey]["std"]  for u in users]
        bars  = ax.bar(users, means, color=colour, alpha=0.88,
                       capsize=5, yerr=stds, edgecolor="white")
        ax.set_title(flabel, fontsize=11)
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis="x", labelsize=10, rotation=15)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Per-Persona Stylometric Profiles — Evidence of Individual Preference Learning (R3.1)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig7_persona_diversity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Saved {out.name}")


# -- Human evaluation sheet (R3.3) ---------------------------------------------

def generate_human_eval_sheet(
    results: List[Dict[str, Any]],
    filename: str = "human_eval_sheet.csv",
) -> None:
    """
    Generate a blinded human evaluation sheet (R3.3).

    Evaluators receive: incoming email + generated response (no ground-truth
    decision labels shown).  They answer 4 questions per sample:
      Q1 — What decision does the response reflect?
           (ACCEPT / REJECT / DEFER / MAYBE / REQUEST_MORE_INFORMATION)
      Q2 — Does the response adequately address the email? (1 = not at all, 5 = fully)
      Q3 — How natural does the language feel?            (1 = robotic,     5 = very natural)
      Q4 — Additional comments (free text)

    Saved to: results/human_eval/human_eval_sheet.csv
    """
    HUMAN_EVAL_DIR = MET_DIR.parent / "human_eval"
    HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = HUMAN_EVAL_DIR / filename

    fieldnames = [
        "eval_id",
        "incoming_email",
        "generated_response",
        "Q1_inferred_decision__ACCEPT_REJECT_DEFER_MAYBE_REQUEST_MORE_INFO",
        "Q2_addresses_email__1to5",
        "Q3_naturalness__1to5",
        "Q4_additional_comments",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "eval_id":            item["eval_id"],
                "incoming_email":     item["incoming_email"],
                "generated_response": item["generated_response"],
                "Q1_inferred_decision__ACCEPT_REJECT_DEFER_MAYBE_REQUEST_MORE_INFO": "",
                "Q2_addresses_email__1to5": "",
                "Q3_naturalness__1to5":     "",
                "Q4_additional_comments":   "",
            })
    print(f"[evaluation] Human eval sheet → {path}")
