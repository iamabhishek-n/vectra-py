from vectra.core import VectraClient


class TestReciprocalRankFusion:
    def test_merges_two_ranked_lists_favoring_docs_in_both(self):
        list_a = [{"content": "alpha"}, {"content": "beta"}]
        list_b = [{"content": "beta"}, {"content": "gamma"}]
        result = VectraClient._reciprocal_rank_fusion(None, [list_a, list_b])
        assert [d["content"] for d in result] == ["beta", "alpha", "gamma"]

    def test_returns_empty_list_for_empty_input(self):
        assert VectraClient._reciprocal_rank_fusion(None, []) == []

    def test_deduplicates_by_content_keeping_first_seen(self):
        list_a = [{"content": "same", "tag": "first"}]
        list_b = [{"content": "same", "tag": "second"}]
        result = VectraClient._reciprocal_rank_fusion(None, [list_a, list_b])
        assert len(result) == 1
        assert result[0]["tag"] == "first"


class TestMmrSelect:
    def test_returns_empty_list_for_empty_candidates(self):
        assert VectraClient._mmr_select(None, [], 5, 0.5) == []

    def test_highest_scoring_candidate_selected_first(self):
        candidates = [
            {"content": "low score doc about cats", "score": 0.2},
            {"content": "high score doc about dogs", "score": 0.9},
        ]
        result = VectraClient._mmr_select(None, candidates, 2, 0.5)
        assert result[0]["content"] == "high score doc about dogs"

    def test_prefers_diverse_second_pick_over_near_duplicate(self):
        candidates = [
            {"content": "the quick brown fox jumps over the lazy dog", "score": 0.9},
            {"content": "the quick brown fox jumps over the lazy cat", "score": 0.85},
            {"content": "completely unrelated content about space travel", "score": 0.7},
        ]
        result = VectraClient._mmr_select(None, candidates, 2, 0.3)
        assert len(result) == 2
        assert result[1]["content"] == "completely unrelated content about space travel"

    def test_respects_the_k_limit(self):
        candidates = [
            {"content": f"doc number {i} with unique words {i}{i}{i}", "score": 1 - i * 0.05}
            for i in range(10)
        ]
        result = VectraClient._mmr_select(None, candidates, 3, 0.5)
        assert len(result) == 3
