"""Unit tests for measures utility functions."""

import pytest
from src.benchmarks.measures import (
    normalize_answer,
    f1_score,
    exact_match_score,
)


class TestNormalizeAnswer:
    """Test cases for normalize_answer function."""

    def test_lowercase(self):
        """Test lowercase conversion."""
        assert normalize_answer("PARIS") == "paris"

    def test_remove_articles(self):
        """Test article removal."""
        assert normalize_answer("the capital") == "capital"
        assert normalize_answer("a city") == "city"
        assert normalize_answer("an apple") == "apple"

    def test_remove_punctuation(self):
        """Test punctuation removal."""
        assert normalize_answer("Paris, France") == "paris france"
        assert normalize_answer("hello!") == "hello"

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_combined(self):
        """Test combined normalization."""
        assert normalize_answer("The capital of France is Paris!") == "capital of france is paris"


class TestExactMatchScore:
    """Test cases for exact_match_score function."""

    def test_exact_match(self):
        """Test exact match returns 1.0."""
        assert exact_match_score("Paris", "Paris") == 1.0

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        assert exact_match_score("paris", "PARIS") == 1.0

    def test_no_match(self):
        """Test no match returns 0.0."""
        assert exact_match_score("Paris", "London") == 0.0

    def test_article_ignored(self):
        """Test articles are ignored."""
        assert exact_match_score("The Paris", "Paris") == 1.0

    def test_punctuation_ignored(self):
        """Test punctuation is ignored."""
        assert exact_match_score("Paris!", "Paris") == 1.0


class TestF1Score:
    """Test cases for f1_score function."""

    def test_perfect_match(self):
        """Test perfect match returns 1.0."""
        assert f1_score("Paris France", "Paris France") == 1.0

    def test_partial_match(self):
        """Test partial match returns score between 0 and 1."""
        score = f1_score("Paris France", "Paris")
        assert 0 < score < 1

    def test_no_match(self):
        """Test no match returns 0.0."""
        assert f1_score("Paris", "London") == 0.0

    def test_empty_prediction(self):
        """Test empty prediction."""
        assert f1_score("", "Paris") == 0.0

    def test_empty_ground_truth(self):
        """Test empty ground truth."""
        assert f1_score("Paris", "") == 0.0

    def test_word_overlap(self):
        """Test word overlap calculation."""
        # "capital city" vs "city center"
        # Common: "city" (1 word)
        # Precision: 1/2, Recall: 1/2
        # F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        score = f1_score("capital city", "city center")
        assert score == pytest.approx(0.5, rel=0.01)
