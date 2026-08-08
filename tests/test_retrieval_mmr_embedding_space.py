from vectra.core import VectraClient


class TestMmrSelectEmbeddingSpace:
    def test_uses_cosine_similarity_when_every_candidate_has_an_embedding(self):
        # Two candidates share zero lexical tokens with the top pick, but one is
        # embedding-similar (should be penalized) and one is embedding-dissimilar
        # (should be preferred) — lexical Jaccard alone cannot distinguish these
        # (both would score 0 overlap).
        candidates = [
            {"content": "xyz abc def", "score": 0.9, "embedding": [1, 0, 0]},
            {"content": "qrs tuv wxy", "score": 0.85, "embedding": [0.99, 0.01, 0]},
            {"content": "lmn opq rst", "score": 0.8, "embedding": [0, 1, 0]},
        ]

        result = VectraClient._mmr_select(None, candidates, 2, 0.3)

        assert len(result) == 2
        assert result[0]["content"] == "xyz abc def"
        assert result[1]["content"] == "lmn opq rst"

    def test_falls_back_to_lexical_jaccard_when_no_embeddings(self):
        candidates = [
            {"content": "the quick brown fox jumps over the lazy dog", "score": 0.9},
            {"content": "the quick brown fox jumps over the lazy cat", "score": 0.85},
            {"content": "completely unrelated content about space travel", "score": 0.7},
        ]

        result = VectraClient._mmr_select(None, candidates, 2, 0.1)

        assert result[1]["content"] == "completely unrelated content about space travel"

    def test_falls_back_to_lexical_when_only_some_candidates_have_embeddings(self):
        candidates = [
            {"content": "alpha beta gamma delta", "score": 0.9, "embedding": [1, 0]},
            {"content": "epsilon zeta eta theta", "score": 0.8},
        ]

        result = VectraClient._mmr_select(None, candidates, 2, 0.5)

        assert len(result) == 2
