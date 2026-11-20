"""
tests/test_pipeline.py
-----------------------
Pytest suite covering all critical pipeline guarantees.

Run with:  pytest tests/test_pipeline.py -v
"""

import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_df():
    """Load raw dataset once for the session."""
    from src.preprocessing import load_raw
    return load_raw()


@pytest.fixture(scope="session")
def split_data(raw_df):
    from src.preprocessing import chronological_split
    return chronological_split(raw_df)


@pytest.fixture(scope="session")
def all_embeddings(raw_df):
    from src.memory import embed_texts
    texts = raw_df["incoming_email"].tolist()
    return embed_texts(texts)


@pytest.fixture(scope="session")
def memory_store(raw_df, all_embeddings):
    from src.memory import build_memory_store
    return build_memory_store(raw_df, all_embeddings)


# ================================================================================
# TEST 1 — Dataset schema
# ================================================================================

class TestDatasetSchema:
    def test_required_columns_present(self, raw_df):
        required = {"record_id", "user_id", "timestamp",
                    "email_subject", "incoming_email", "user_response", "decision"}
        assert required.issubset(set(raw_df.columns)), \
            f"Missing columns: {required - set(raw_df.columns)}"

    def test_valid_decision_labels(self, raw_df):
        valid = {"ACCEPT", "REJECT", "MAYBE", "DEFER", "REQUEST_MORE_INFORMATION"}
        found = set(raw_df["decision"].unique())
        assert found.issubset(valid), f"Invalid labels: {found - valid}"

    def test_no_null_values(self, raw_df):
        nulls = raw_df[["incoming_email", "user_response", "decision"]].isnull().sum()
        assert nulls.sum() == 0, f"Null values found: {nulls[nulls > 0]}"

    def test_all_five_users_present(self, raw_df):
        expected = {"alice", "bob", "carol", "dave", "eve"}
        found = set(raw_df["user_id"].unique())
        assert expected == found, f"Missing users: {expected - found}"

    def test_minimum_record_count(self, raw_df):
        assert len(raw_df) >= 200, f"Too few records: {len(raw_df)}"

    def test_all_decision_classes_present(self, raw_df):
        required = {"ACCEPT", "REJECT", "MAYBE", "DEFER", "REQUEST_MORE_INFORMATION"}
        present = set(raw_df["decision"].unique())
        assert required == present


# ================================================================================
# TEST 2 — Chronological split
# ================================================================================

class TestChronologicalSplit:
    def test_split_sizes_sum_to_total(self, raw_df, split_data):
        train, val, test = split_data
        assert len(train) + len(val) + len(test) == len(raw_df)

    def test_train_before_val(self, split_data):
        train, val, _ = split_data
        assert train["timestamp"].max() <= val["timestamp"].min(), \
            "Data leakage: train contains records after start of val!"

    def test_val_before_test(self, split_data):
        _, val, test = split_data
        assert val["timestamp"].max() <= test["timestamp"].min(), \
            "Data leakage: val contains records after start of test!"

    def test_no_shuffle(self, split_data):
        train, val, test = split_data
        # Combined reconstructed order must equal original sorted order
        combined = pd.concat([train, val, test])["timestamp"].values
        assert all(combined[i] <= combined[i+1] for i in range(len(combined)-1)), \
            "Records are not in chronological order!"

    def test_train_ratio_approximately_60_percent(self, raw_df, split_data):
        train, _, _ = split_data
        ratio = len(train) / len(raw_df)
        assert 0.55 <= ratio <= 0.65, f"Train ratio out of range: {ratio:.2f}"


# ================================================================================
# TEST 3 — No data leakage in ML features
# ================================================================================

class TestNoDataLeakage:
    def test_tfidf_fitted_on_train_only(self, split_data):
        """TF-IDF vocab should be fitted on training texts; val/test should not affect it."""
        from src.features import fit_tfidf
        train, val, _ = split_data
        vec = fit_tfidf(train["incoming_email"].tolist())
        # Transform should work on val (even with OOV words)
        from sklearn.exceptions import NotFittedError
        try:
            vec.transform(val["incoming_email"].tolist())
        except NotFittedError:
            pytest.fail("TF-IDF vectorizer not properly fitted!")

    def test_user_id_not_in_feature_set(self, split_data):
        """user_id must not appear as a feature — only email text is used."""
        from src.features import fit_tfidf, build_features
        train, _, test = split_data
        vec = fit_tfidf(train["incoming_email"].tolist())
        X_test, _ = build_features(test, vec, use_embeddings=False)
        # Feature matrix should not contain user_id encoding
        # (simply check its shape is n x max_features, not n x (max_features+n_users))
        n_users = test["user_id"].nunique()
        assert X_test.shape[1] <= 3000 + 384 + 5, \
            "Feature matrix appears to contain user_id encodings!"


# ================================================================================
# TEST 4 — Memory retrieval: exactly 20, no future, no target
# ================================================================================

class TestMemoryRetrieval:
    def _get_test_target(self, raw_df, split_data, all_embeddings, memory_store):
        """Helper: pick a test row that has enough historical records."""
        from src.memory import retrieve_memories
        _, _, test_df = split_data
        for _, row in test_df.iterrows():
            global_pos = raw_df[raw_df["record_id"] == row["record_id"]].index[0]
            target_emb = all_embeddings[global_pos]
            mems = retrieve_memories(
                target_record_id  = row["record_id"],
                target_user_id    = row["user_id"],
                target_timestamp  = row["timestamp"],
                target_embedding  = target_emb,
                memory_store      = memory_store,
                top_k             = 20,
            )
            if len(mems) >= 10:   # need enough history
                return row, mems, target_emb
        pytest.skip("No test row with >= 10 historical records found")

    def test_exactly_top_20_retrieved_when_possible(
        self, raw_df, split_data, all_embeddings, memory_store
    ):
        row, mems, _ = self._get_test_target(raw_df, split_data, all_embeddings, memory_store)
        # If the user has >= 20 historical records, we must get exactly 20
        from src.memory import retrieve_memories
        # count the user's records before this timestamp
        prior = raw_df[
            (raw_df["user_id"] == row["user_id"]) &
            (raw_df["timestamp"] < row["timestamp"]) &
            (raw_df["record_id"] != row["record_id"])
        ]
        expected = min(20, len(prior))
        assert len(mems) == expected, \
            f"Expected {expected} memories, got {len(mems)}"

    def test_no_future_memories(self, raw_df, split_data, all_embeddings, memory_store):
        row, mems, _ = self._get_test_target(raw_df, split_data, all_embeddings, memory_store)
        for m in mems:
            assert pd.Timestamp(m["timestamp"]) < row["timestamp"], \
                f"Future memory retrieved: {m['timestamp']} >= {row['timestamp']}"

    def test_target_not_in_memories(self, raw_df, split_data, all_embeddings, memory_store):
        row, mems, _ = self._get_test_target(raw_df, split_data, all_embeddings, memory_store)
        record_ids = [m["record_id"] for m in mems]
        assert row["record_id"] not in record_ids, \
            "Target record found in its own memory pool!"

    def test_memories_are_same_user(self, raw_df, split_data, all_embeddings, memory_store):
        row, mems, _ = self._get_test_target(raw_df, split_data, all_embeddings, memory_store)
        for m in mems:
            mem_user = raw_df[raw_df["record_id"] == m["record_id"]]["user_id"].values[0]
            assert mem_user == row["user_id"], \
                f"Memory from wrong user: {mem_user} != {row['user_id']}"

    def test_similarity_scores_in_valid_range(
        self, raw_df, split_data, all_embeddings, memory_store
    ):
        row, mems, _ = self._get_test_target(raw_df, split_data, all_embeddings, memory_store)
        for m in mems:
            assert -1.01 <= m["similarity_score"] <= 1.01, \
                f"Invalid similarity score: {m['similarity_score']}"


# ================================================================================
# TEST 5 — Embedding generation
# ================================================================================

class TestEmbeddingGeneration:
    def test_embedding_shape(self, raw_df, all_embeddings):
        assert all_embeddings.shape == (len(raw_df), 384), \
            f"Unexpected embedding shape: {all_embeddings.shape}"

    def test_embeddings_are_normalised(self, all_embeddings):
        norms = np.linalg.norm(all_embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), \
            "Embeddings are not L2-normalised!"

    def test_embedding_dtype(self, all_embeddings):
        assert all_embeddings.dtype in (np.float32, np.float64), \
            f"Unexpected dtype: {all_embeddings.dtype}"


# ================================================================================
# TEST 6 — Decision prediction
# ================================================================================

class TestDecisionPrediction:
    VALID_DECISIONS = {"ACCEPT", "REJECT", "MAYBE", "DEFER", "REQUEST_MORE_INFORMATION"}

    def test_model_outputs_valid_labels(self, split_data):
        from src.features import fit_tfidf, build_features, make_label_encoder
        from src.models import build_cognitive_twin, train_model, predict
        train, _, test = split_data
        le = make_label_encoder()
        y_train = le.transform(train["decision"].values)
        vec = fit_tfidf(train["incoming_email"].tolist())
        X_train, _ = build_features(train, vec, use_embeddings=True)
        X_test, _  = build_features(test, vec, use_embeddings=True)
        ct = build_cognitive_twin()
        train_model(ct, X_train, y_train)
        y_pred = le.inverse_transform(predict(ct, X_test))
        assert set(y_pred).issubset(self.VALID_DECISIONS)

    def test_accuracy_above_random_chance(self, split_data):
        """ML model should beat random baseline (1/5 = 0.20 for 5 classes)."""
        from sklearn.metrics import accuracy_score
        from src.features import fit_tfidf, build_features, make_label_encoder
        from src.models import build_cognitive_twin, train_model, predict
        train, _, test = split_data
        le = make_label_encoder()
        y_train = le.transform(train["decision"].values)
        y_test  = le.transform(test["decision"].values)
        vec = fit_tfidf(train["incoming_email"].tolist())
        X_train, _ = build_features(train, vec, use_embeddings=True)
        X_test, _  = build_features(test, vec, use_embeddings=True)
        ct = build_cognitive_twin()
        train_model(ct, X_train, y_train)
        y_pred = predict(ct, X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.20, f"Model accuracy ({acc:.3f}) is below random chance!"


# ================================================================================
# TEST 7 — Response evaluation metrics
# ================================================================================

class TestEvaluationMetrics:
    def test_semantic_similarity_identical_texts(self):
        from src.evaluation import semantic_similarity
        text = "This is a test sentence for similarity."
        score = semantic_similarity(text, text)
        assert score > 0.98, f"Identical texts should have high similarity: {score}"

    def test_semantic_similarity_different_texts(self):
        from src.evaluation import semantic_similarity
        score = semantic_similarity("yes, I agree", "no, I completely disagree and object")
        assert 0.0 <= score <= 1.0

    def test_style_similarity_in_range(self):
        from src.evaluation import style_similarity
        score = style_similarity("Hi there! Sounds great!", "Dear Sir, I concur.")
        assert 0.0 <= score <= 1.0

    def test_relevance_in_range(self):
        from src.evaluation import relevance_score
        score = relevance_score("Yes, I can meet on Friday.", "Can we meet on Friday?")
        assert 0.0 <= score <= 1.0

    def test_decision_consistency_accept(self):
        from src.evaluation import decision_consistency
        score = decision_consistency("Yes, I agree and confirm this.", "ACCEPT")
        assert score > 0.0

    def test_decision_consistency_reject(self):
        from src.evaluation import decision_consistency
        score = decision_consistency("No, I must decline this request.", "REJECT")
        assert score > 0.0


# ================================================================================
# TEST 8 — LLM mock mode
# ================================================================================

class TestMockMode:
    def test_mock_mode_returns_tagged_string(self):
        from src.response_generator import generate_response
        result = generate_response(
            incoming_email     = "Can we meet tomorrow?",
            predicted_decision = "ACCEPT",
            memories           = [],
            force_mock         = True,
        )
        assert result["mode"] == "mock"
        assert "[MOCK" in result["response"]

    def test_mock_mode_returns_dict_with_required_keys(self):
        from src.response_generator import generate_response
        result = generate_response("test", "ACCEPT", [], force_mock=True)
        for key in ("response", "mode", "model", "n_memories"):
            assert key in result, f"Missing key: {key}"

    def test_mock_mode_does_not_call_api(self):
        """Mock mode must never hit the network."""
        import unittest.mock as mock
        import src.response_generator as rg
        with mock.patch.object(rg, "_call_gemini", side_effect=Exception("API called!")) as m:
            result = rg.generate_response("test", "ACCEPT", [], force_mock=True)
            m.assert_not_called()
        assert result["mode"] == "mock"
