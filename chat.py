"""
chat.py
--------
Interactive chat with any Cognitive Twin persona.

You type a message → the system replies in Alice/Bob/Carol/Dave/Eve's style,
using their historical memory to personalize the response.

Usage:
    python chat.py --user alice
    python chat.py --user bob
    python chat.py --user eve
"""

import sys
import argparse
import pathlib
import textwrap
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing      import load_raw, chronological_split
from src.features           import fit_tfidf, build_features, make_label_encoder
from src.models             import build_cognitive_twin, train_model, predict
from src.memory             import embed_texts, build_memory_store, retrieve_memories
from src.response_generator import generate_response, MOCK_MODE

DIVIDER = "─" * 65

PERSONAS_INFO = {
    "alice": "Formal, professional, polished",
    "bob":   "Casual, laid-back, uses slang",
    "carol": "Friendly, warm, enthusiastic",
    "dave":  "Concise, blunt, minimal words",
    "eve":   "Analytical, data-driven, structured",
}


def setup(user_id: str):
    """Load data, train model, build memory store. Runs once at startup."""
    print(f"\n  Setting up Cognitive Twin for '{user_id}' …")
    df = load_raw()
    train_df, _, _ = chronological_split(df)

    le = make_label_encoder()
    y_train = le.transform(train_df["decision"].values)
    tfidf_vec = fit_tfidf(train_df["incoming_email"].tolist())
    X_train_ct, _ = build_features(train_df, tfidf_vec, use_embeddings=True)
    ct_model = build_cognitive_twin()
    train_model(ct_model, X_train_ct, y_train)

    all_embs = embed_texts(df["incoming_email"].tolist())
    memory_store = build_memory_store(df, all_embs)

    n = len(df[df["user_id"] == user_id])
    print(f"  [OK] Ready — {user_id}'s memory contains {n} past emails.\n")
    return df, ct_model, le, tfidf_vec, all_embs, memory_store


def reply(user_input, user_id, df, ct_model, le, tfidf_vec, all_embs, memory_store):
    """Run the full pipeline on a user-typed message and print the response."""

    # Embed the input
    input_emb = embed_texts([user_input])[0]

    # Predict decision
    input_df = pd.DataFrame([{"incoming_email": user_input, "user_id": user_id}])
    X_input, _ = build_features(input_df, tfidf_vec, use_embeddings=True)
    predicted_decision = le.inverse_transform(predict(ct_model, X_input))[0]

    # Retrieve memories (use a future timestamp so ALL past emails are eligible)
    future_ts = df[df["user_id"] == user_id]["timestamp"].max() + pd.Timedelta(days=1)
    memories = retrieve_memories(
        target_record_id  = "__chat__",
        target_user_id    = user_id,
        target_timestamp  = future_ts,
        target_embedding  = input_emb,
        memory_store      = memory_store,
        top_k             = 20,
    )

    # Generate response
    gen = generate_response(
        incoming_email     = user_input,
        predicted_decision = predicted_decision,
        memories           = memories,
        user_id            = user_id,
    )

    # Display
    print(f"\n{DIVIDER}")
    print(f"  🤖 Decision predicted : {predicted_decision}")
    print(f"  📚 Memories used      : {len(memories)}")
    print(f"  ⚙  Mode               : {gen['mode']} ({gen['model']})")
    print(f"{DIVIDER}")
    response_text = gen["response"].strip()
    for line in response_text.split("\n"):
        if line.strip():
            print(textwrap.fill(line, width=63, initial_indent="  ",
                                subsequent_indent="  "))
        else:
            print()
    print(f"\n{DIVIDER}\n")


def run_chat(user_id: str) -> None:
    if user_id not in PERSONAS_INFO:
        print(f"\n  Unknown persona '{user_id}'.")
        print(f"  Choose from: {', '.join(PERSONAS_INFO.keys())}\n")
        sys.exit(1)

    df, ct_model, le, tfidf_vec, all_embs, memory_store = setup(user_id)

    print("=" * 65)
    print(f"  💬  Cognitive Twin Chat  —  {user_id.upper()}")
    print(f"      Style: {PERSONAS_INFO[user_id]}")
    print("=" * 65)
    print("  Type anything and press Enter to get a reply.")
    print("  Commands: quit | personas | switch alice/bob/carol/dave/eve")
    print("=" * 65)

    if MOCK_MODE:
        print("\n  ⚠️  MOCK MODE — Set LLM_API_KEY in .env for real responses.\n")

    while True:
        try:
            user_input = input(f"\n  You → {user_id}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            return

        if not user_input:
            continue

        cmd = user_input.lower()

        # Commands
        if cmd in ("quit", "exit"):
            print("\n  Goodbye!\n")
            return

        if cmd == "personas":
            print()
            for name, style in PERSONAS_INFO.items():
                marker = "◀" if name == user_id else " "
                print(f"  {marker} {name:<8} — {style}")
            continue

        if cmd.startswith("switch "):
            new_user = cmd.split(" ", 1)[1].strip()
            if new_user in PERSONAS_INFO:
                run_chat(new_user)
                return
            else:
                print(f"  Unknown persona. Options: {', '.join(PERSONAS_INFO.keys())}")
            continue

        # Generate reply
        print(f"\n  ⏳ {user_id.capitalize()} is typing …")
        try:
            reply(user_input, user_id,
                  df, ct_model, le, tfidf_vec, all_embs, memory_store)
        except ValueError as e:
            print(f"\n  ⚠️  {e}")
            print("  💡 Try adding more context, for example:")
            print("     'Can we schedule a meeting next Monday to review the budget?'\n")
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Cognitive Twin chat")
    parser.add_argument(
        "--user", type=str, default="alice",
        help="Persona to chat with: alice / bob / carol / dave / eve",
    )
    args = parser.parse_args()
    run_chat(args.user)
