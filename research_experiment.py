"""
research_experiment.py
-----------------------
End-to-end research experiment runner.

Runs both tasks in sequence:
  1. Generate / load dataset
  2. Chronological split
  3. Build features (TF-IDF + embeddings)
  4. Train Baseline (LogReg) and CognitiveTwin (RF)
  5. Evaluate predictions on test set
  6. Build memory store for all records
  7. Run memory retrieval + LLM response generation on N_DEMO test emails
  8. Evaluate generated responses
  9. Save all metrics as JSON
  10. Generate all 5 figures

Usage:
    python research_experiment.py

Environment variables (set in .env):
    LLM_API_KEY  — Google Gemini API key (optional; uses mock mode if absent)
    LLM_MODEL    — Gemini model name (default: gemini-1.5-flash)
"""

import sys
import pathlib
import json
import numpy as np
import pandas as pd

# Allow imports from src/ and scripts/
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing     import load_raw, chronological_split, describe_split
from src.features          import fit_tfidf, build_features, make_label_encoder
from src.models            import build_baseline, build_cognitive_twin, train_model, predict
from src.memory            import embed_texts, build_memory_store, retrieve_memories
from src.response_generator import generate_response
from src.evaluation        import (
    evaluate_predictions, save_metrics,
    evaluate_response,
    plot_prediction_comparison,
    plot_confusion_matrix,
    plot_memory_similarity_distribution,
    plot_response_metrics,
    plot_style_comparison_per_user,
    plot_ablation_comparison,
    analyze_persona_diversity,
    plot_persona_diversity,
    generate_human_eval_sheet,
)

RANDOM_SEED = 42
N_DEMO      = 10   # number of test emails to run through response generation

PRED_DIR = ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment() -> None:
    print("=" * 60)
    print("  Memory-Based Cognitive Twin — Research Experiment")
    print("=" * 60)

    # -- 1. Load dataset ------------------------------------------------------
    print("\n[1/10] Loading dataset …")
    df = load_raw()

    # -- 2. Chronological split -----------------------------------------------
    print("\n[2/10] Splitting data chronologically …")
    train_df, val_df, test_df = chronological_split(df)
    describe_split(train_df, val_df, test_df)

    # -- 3. Label encoding ----------------------------------------------------
    le = make_label_encoder()
    y_train = le.transform(train_df["decision"].values)
    y_val   = le.transform(val_df["decision"].values)
    y_test  = le.transform(test_df["decision"].values)

    # -- 4. Feature extraction ------------------------------------------------
    print("\n[3/10] Extracting features …")
    tfidf_vec = fit_tfidf(train_df["incoming_email"].tolist())

    # Baseline features: TF-IDF only
    X_train_base, _ = build_features(train_df, tfidf_vec, use_embeddings=False)
    X_val_base,   _ = build_features(val_df,   tfidf_vec, use_embeddings=False)
    X_test_base,  _ = build_features(test_df,  tfidf_vec, use_embeddings=False)

    # CognitiveTwin features: TF-IDF + sentence embeddings
    X_train_ct, _ = build_features(train_df, tfidf_vec, use_embeddings=True)
    X_val_ct,   _ = build_features(val_df,   tfidf_vec, use_embeddings=True)
    X_test_ct,  _ = build_features(test_df,  tfidf_vec, use_embeddings=True)

    # -- 5. Train models ------------------------------------------------------
    print("\n[4/10] Training models …")
    baseline = build_baseline()
    ct_model = build_cognitive_twin()

    train_model(baseline, X_train_base, y_train)
    train_model(ct_model, X_train_ct,   y_train)
    print("  [OK] Baseline (LogReg) trained")
    print("  [OK] CognitiveTwin (RandomForest) trained")

    # -- 6. Evaluate on test set -----------------------------------------------
    print("\n[5/10] Evaluating on test set …")
    y_pred_base = predict(baseline, X_test_base)
    y_pred_ct   = predict(ct_model, X_test_ct)

    # Decode integer labels back to string labels
    y_test_str  = le.inverse_transform(y_test)
    y_pred_base_str = le.inverse_transform(y_pred_base)
    y_pred_ct_str   = le.inverse_transform(y_pred_ct)

    base_metrics = evaluate_predictions(y_test_str, y_pred_base_str, "Baseline (LogReg+TF-IDF)")
    ct_metrics   = evaluate_predictions(y_test_str, y_pred_ct_str,   "CognitiveTwin (RF+Embeddings)")

    save_metrics(base_metrics, "baseline_metrics.json")
    save_metrics(ct_metrics,   "cognitive_twin_metrics.json")

    # Save predictions CSV
    pred_df = test_df.copy()
    pred_df["pred_baseline"] = y_pred_base_str
    pred_df["pred_ct"]       = y_pred_ct_str
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)
    print(f"  [OK] Predictions saved → results/predictions/test_predictions.csv")

    # -- 7. Build memory store -------------------------------------------------
    print("\n[6/10] Building memory store (embeddings for all records) …")
    all_texts  = df["incoming_email"].tolist()
    all_embs   = embed_texts(all_texts)
    memory_store = build_memory_store(df, all_embs)
    print(f"  [OK] Memory store ready — {len(df)} records indexed")

    # -- 8. Response generation on N_DEMO test emails --------------------------
    print(f"\n[7/10] Running response generation on {N_DEMO} test emails …")

    # Select N_DEMO test examples (spread across users)
    demo_indices = []
    users = test_df["user_id"].unique()
    per_user = max(1, N_DEMO // len(users))
    for uid in users:
        uid_rows = test_df[test_df["user_id"] == uid].index.tolist()
        demo_indices.extend(uid_rows[:per_user])
    demo_indices = demo_indices[:N_DEMO]

    all_memories_lists = []
    response_eval_results = []
    style_comparison_data = []

    for idx in demo_indices:
        row = test_df.loc[idx]
        # Robust embedding lookup using record_id (works regardless of index state)
        global_pos = df[df["record_id"] == row["record_id"]].index[0]
        target_emb = all_embs[global_pos]

        # Use the CT model's predicted decision
        # Map idx back to test_df positional index for X_test_ct
        test_pos = list(test_df.index).index(idx)
        pred_decision = y_pred_ct_str[test_pos]

        # Retrieve TOP-20 memories (strictly chronological)
        memories = retrieve_memories(
            target_record_id  = row["record_id"],
            target_user_id    = row["user_id"],
            target_timestamp  = row["timestamp"],
            target_embedding  = target_emb,
            memory_store      = memory_store,
            top_k             = 20,
        )
        all_memories_lists.append(memories)

        # Generate personalized response
        gen_result = generate_response(
            incoming_email     = row["incoming_email"],
            predicted_decision = pred_decision,
            memories           = memories,
            user_id            = row["user_id"],
        )
        generated_text = gen_result["response"]

        # Evaluate
        eval_metrics = evaluate_response(
            generated       = generated_text,
            actual_response = row["user_response"],
            incoming_email  = row["incoming_email"],
            predicted_decision = pred_decision,
        )
        response_eval_results.append(eval_metrics)
        style_comparison_data.append({
            "user_id":   row["user_id"],
            "generated": generated_text,
            "actual":    row["user_response"],
        })

        mode_tag = f"[{gen_result['mode'].upper()}]"
        print(f"  {mode_tag} user={row['user_id']} decision={pred_decision} "
              f"sem_sim={eval_metrics['semantic_similarity']:.3f} "
              f"rel={eval_metrics['relevance']:.3f}")

    # -- 9. Save response metrics ----------------------------------------------
    print("\n[8/10] Saving response evaluation metrics …")
    agg = {k: round(float(np.mean([r[k] for r in response_eval_results])), 4)
           for k in response_eval_results[0]}
    save_metrics({"per_sample": response_eval_results, "aggregate": agg},
                 "response_metrics.json")
    print(f"  Aggregate response metrics: {agg}")

    # -- 9b. ABLATION STUDY (R2.4 / R3.2) -------------------------------------
    # Three conditions compared on the same N_DEMO test emails:
    #   no_context  : no memories, no ML decision  (pure LLM)
    #   memory_only : retrieved memories, no ML decision
    #   full        : retrieved memories + ML decision (main experiment)
    print(f"\n[8b/10] Running ablation study ({N_DEMO} emails × 2 extra conditions) …")

    ablation_results = {
        "no_context":  [],
        "memory_only": [],
        "full":        list(response_eval_results),   # reuse main-loop results
    }

    for cond in ["no_context", "memory_only"]:
        print(f"  Condition: {cond} …")
        for i, idx in enumerate(demo_indices):
            row       = test_df.loc[idx]
            test_pos  = list(test_df.index).index(idx)
            pred_dec  = y_pred_ct_str[test_pos]
            memories  = all_memories_lists[i]
            try:
                gen_result = generate_response(
                    incoming_email     = row["incoming_email"],
                    predicted_decision = pred_dec,
                    memories           = memories,
                    ablation_mode      = cond,
                    user_id            = row["user_id"],
                )
                eval_m = evaluate_response(
                    generated          = gen_result["response"],
                    actual_response    = row["user_response"],
                    incoming_email     = row["incoming_email"],
                    predicted_decision = pred_dec,
                )
                ablation_results[cond].append(eval_m)
                print(f"    [{cond}] user={row['user_id']} "
                      f"sem={eval_m['semantic_similarity']:.3f} "
                      f"cons={eval_m['decision_consistency']:.3f}")
            except Exception as exc:
                print(f"    [WARN] {cond} failed for idx={idx}: {exc}")

    # Aggregate and print ablation table
    ablation_agg = {}
    for cond, res_list in ablation_results.items():
        if res_list:
            ablation_agg[cond] = {
                k: round(float(np.mean([r[k] for r in res_list])), 4)
                for k in res_list[0]
            }
    save_metrics({"aggregate": ablation_agg, "per_sample": {
        c: v for c, v in ablation_results.items()
    }}, "ablation_metrics.json")
    print(f"\n  {'Condition':<22} {'Sem.Sim':>8} {'Style':>8} {'Rel':>8} {'Consist':>8}")
    print(f"  {'-'*58}")
    _cond_names = {"no_context": "No Context", "memory_only": "Memory Only", "full": "Full System"}
    for cond, agg_m in ablation_agg.items():
        print(f"  {_cond_names[cond]:<22} "
              f"{agg_m['semantic_similarity']:>8.4f} "
              f"{agg_m['style_similarity']:>8.4f} "
              f"{agg_m['relevance']:>8.4f} "
              f"{agg_m['decision_consistency']:>8.4f}")
    print(f"  [OK] Ablation metrics saved → results/metrics/ablation_metrics.json")

    # -- 9c. PERSONA DIVERSITY ANALYSIS (R3.1) ---------------------------------
    print("\n[8c/10] Analysing per-persona style diversity …")
    diversity_data = analyze_persona_diversity(train_df)
    save_metrics(diversity_data, "persona_diversity.json")
    print(f"\n  {'Persona':<10} {'AvgSentLen':>12} {'TTR':>8} {'Formality':>10}")
    print(f"  {'-'*44}")
    for uid, stats in diversity_data.items():
        print(f"  {uid:<10} "
              f"{stats['avg_sentence_length']['mean']:>8.1f} ± {stats['avg_sentence_length']['std']:<4.1f} "
              f"{stats['type_token_ratio']['mean']:>8.3f} "
              f"{stats['formality_ratio']['mean']:>10.3f}")
    print(f"  [OK] Persona diversity saved → results/metrics/persona_diversity.json")

    # -- 9d. HUMAN EVALUATION SHEET (R3.3) ------------------------------------
    print("\n[8d/10] Generating human evaluation sheet …")
    human_eval_data = [
        {
            "eval_id":            f"E{i+1:02d}",
            "incoming_email":     test_df.loc[idx, "incoming_email"],
            "generated_response": style_comparison_data[i]["generated"],
        }
        for i, idx in enumerate(demo_indices)
    ]
    generate_human_eval_sheet(human_eval_data)
    print(f"  [OK] Human eval sheet → results/human_eval/human_eval_sheet.csv")

    # -- 10. Generate all 7 figures --------------------------------------------
    print("\n[9/10] Generating figures …")
    plot_prediction_comparison(base_metrics, ct_metrics)
    plot_confusion_matrix(y_test_str, y_pred_ct_str)
    plot_memory_similarity_distribution(all_memories_lists)
    plot_response_metrics(response_eval_results)
    plot_style_comparison_per_user(style_comparison_data)
    plot_ablation_comparison(ablation_results)
    plot_persona_diversity(diversity_data)

    print("\n[10/10] Experiment complete!")
    print("=" * 60)
    print(f"  Figures     → results/figures/   (7 PNG files)")
    print(f"  Metrics     → results/metrics/   (5 JSON files)")
    print(f"  Human Eval  → results/human_eval/human_eval_sheet.csv")
    print(f"  Predictions → results/predictions/test_predictions.csv")
    print("=" * 60)

    # Final summary
    print(f"\n{'Metric':<30} {'Baseline':>12} {'CognitiveTwin':>14}")
    print("-" * 58)
    for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        print(f"  {k:<28} {base_metrics[k]:>12.4f} {ct_metrics[k]:>14.4f}")


if __name__ == "__main__":
    run_experiment()
