"""
response_demo.py
-----------------
Interactive single-email demonstration of the Cognitive Twin pipeline.

Shows exactly:
  1. Input email
  2. ML-predicted decision
  3. TOP-20 retrieved memories (with similarity scores)
  4. LLM-generated personalized response
  5. Actual held-out response
  6. Evaluation metrics

This clearly demonstrates that the system uses historical memory
rather than generating a generic answer.

Usage:
    python response_demo.py                  # uses last test email
    python response_demo.py --user alice     # demo for a specific user
    python response_demo.py --idx 5          # 6th test email
"""

import sys
import argparse
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing      import load_raw, chronological_split
from src.features           import fit_tfidf, build_features, make_label_encoder
from src.models             import build_baseline, build_cognitive_twin, train_model, predict
from src.memory             import embed_texts, build_memory_store, retrieve_memories
from src.response_generator import generate_response, MOCK_MODE
from src.evaluation         import evaluate_response

DIVIDER = "-" * 70


def print_banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def run_demo(user_filter: str | None = None, test_idx: int = 0) -> None:
    print_banner("Memory-Based Cognitive Twin — Live Demo")
    if MOCK_MODE:
        print("\n  [!]  MOCK MODE  — No LLM_API_KEY found.")
        print("     Set LLM_API_KEY in .env to get real Gemini responses.\n")

    # -- Load & split ----------------------------------------------------------
    df = load_raw()
    train_df, val_df, test_df = chronological_split(df)

    # -- Train the CognitiveTwin ML model -------------------------------------
    le = make_label_encoder()
    y_train = le.transform(train_df["decision"].values)
    tfidf_vec = fit_tfidf(train_df["incoming_email"].tolist())
    X_train_ct, _ = build_features(train_df, tfidf_vec, use_embeddings=True)
    ct_model = build_cognitive_twin()
    train_model(ct_model, X_train_ct, y_train)

    # -- Build memory store ----------------------------------------------------
    all_embs = embed_texts(df["incoming_email"].tolist())
    memory_store = build_memory_store(df, all_embs)

    # -- Select demo target ----------------------------------------------------
    if user_filter:
        candidate_df = test_df[test_df["user_id"] == user_filter]
        if candidate_df.empty:
            print(f"  No test emails for user '{user_filter}'. Available: {list(test_df['user_id'].unique())}")
            return
    else:
        candidate_df = test_df

    if test_idx >= len(candidate_df):
        print(f"  Index {test_idx} out of range (max {len(candidate_df)-1}). Using last.")
        test_idx = len(candidate_df) - 1

    row = candidate_df.iloc[test_idx]
    global_pos = df[df["record_id"] == row["record_id"]].index[0]
    target_emb = all_embs[global_pos]

    # -- ML prediction ---------------------------------------------------------
    test_row_df = pd.DataFrame([row])
    X_demo, _ = build_features(test_row_df, tfidf_vec, use_embeddings=True)
    y_pred_encoded = predict(ct_model, X_demo)
    predicted_decision = le.inverse_transform(y_pred_encoded)[0]

    # -- Memory retrieval ------------------------------------------------------
    memories = retrieve_memories(
        target_record_id  = row["record_id"],
        target_user_id    = row["user_id"],
        target_timestamp  = row["timestamp"],
        target_embedding  = target_emb,
        memory_store      = memory_store,
        top_k             = 20,
    )

    # -- LLM Response generation -----------------------------------------------
    gen_result = generate_response(
        incoming_email     = row["incoming_email"],
        predicted_decision = predicted_decision,
        memories           = memories,
    )
    generated_text = gen_result["response"]

    # -- Evaluation ------------------------------------------------------------
    metrics = evaluate_response(
        generated          = generated_text,
        actual_response    = row["user_response"],
        incoming_email     = row["incoming_email"],
        predicted_decision = predicted_decision,
    )

    # ===========================================================================
    # OUTPUT
    # ===========================================================================

    print(f"\n  User ID   : {row['user_id']}")
    print(f"  Timestamp : {row['timestamp']}")
    print(f"  Subject   : {row['email_subject']}")

    print(f"\n{DIVIDER}")
    print("  INPUT EMAIL")
    print(f"{DIVIDER}")
    print(row["incoming_email"])

    print(f"\n{DIVIDER}")
    print("  ML PREDICTED DECISION")
    print(f"{DIVIDER}")
    print(f"  ➜  {predicted_decision}")
    print(f"     (Actual: {row['decision']}  |  Correct: {predicted_decision == row['decision']})")

    print(f"\n{DIVIDER}")
    print(f"  TOP {len(memories)} RETRIEVED HISTORICAL MEMORIES (chronologically valid)")
    print(f"{DIVIDER}")
    for i, m in enumerate(memories, 1):
        print(
            f"\n  Memory {i:>2}  |  {m['timestamp']}  |  sim={m['similarity_score']:.3f}"
            f"  |  decision={m['decision']}"
        )
        print(f"  Email   : {m['incoming_email'][:100].strip()}…")
        print(f"  My reply: {m['user_response'][:120].strip()}…")

    mode_label = f"[{gen_result['mode'].upper()} — model: {gen_result['model']}]"
    print(f"\n{DIVIDER}")
    print(f"  GENERATED PERSONALIZED RESPONSE  {mode_label}")
    print(f"{DIVIDER}")
    print(generated_text)

    print(f"\n{DIVIDER}")
    print("  ACTUAL HELD-OUT RESPONSE  (ground truth — never shown to LLM)")
    print(f"{DIVIDER}")
    print(row["user_response"])

    print(f"\n{DIVIDER}")
    print("  EVALUATION METRICS")
    print(f"{DIVIDER}")
    print(f"  Semantic Similarity  (generated ↔ actual)   : {metrics['semantic_similarity']:.4f}")
    print(f"  Style Similarity     (stylometric features) : {metrics['style_similarity']:.4f}")
    print(f"  Relevance            (generated ↔ email)    : {metrics['relevance']:.4f}")
    print(f"  Decision Consistency (response ↔ prediction): {metrics['decision_consistency']:.4f}")
    print(f"\n{'=' * 70}\n")

    if MOCK_MODE:
        print("  [!] NOTE: The response above is a MOCK. It is NOT a real LLM output.")
        print("     For a real demonstration, set LLM_API_KEY in your .env file.")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Twin single-email demo")
    parser.add_argument("--user", type=str, default=None,
                        help="Filter test emails by user_id (alice/bob/carol/dave/eve)")
    parser.add_argument("--idx",  type=int, default=0,
                        help="Index of the test email to demo (default: 0)")
    args = parser.parse_args()
    run_demo(user_filter=args.user, test_idx=args.idx)
