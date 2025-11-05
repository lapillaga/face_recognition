"""Unit tests for FAISS matcher."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.matcher_faiss import FaissMatcher


@pytest.fixture
def matcher():
    """Create a FaissMatcher instance."""
    return FaissMatcher(dimension=512)


@pytest.fixture
def sample_embeddings():
    """Create sample L2-normalized embeddings for testing."""
    # Create 3 persons with 5 embeddings each
    np.random.seed(42)

    # Alice: embeddings centered around [1, 0, 0, ...]
    alice_base = np.zeros(512, dtype=np.float32)
    alice_base[0] = 1.0
    alice_embs = []
    for _ in range(5):
        emb = alice_base + np.random.randn(512) * 0.1
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        alice_embs.append(emb)
    alice_embs = np.stack(alice_embs)

    # Bob: embeddings centered around [0, 1, 0, ...]
    bob_base = np.zeros(512, dtype=np.float32)
    bob_base[1] = 1.0
    bob_embs = []
    for _ in range(5):
        emb = bob_base + np.random.randn(512) * 0.1
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        bob_embs.append(emb)
    bob_embs = np.stack(bob_embs)

    # Carol: embeddings centered around [0, 0, 1, ...]
    carol_base = np.zeros(512, dtype=np.float32)
    carol_base[2] = 1.0
    carol_embs = []
    for _ in range(5):
        emb = carol_base + np.random.randn(512) * 0.1
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        carol_embs.append(emb)
    carol_embs = np.stack(carol_embs)

    return {
        "Alice": alice_embs,
        "Bob": bob_embs,
        "Carol": carol_embs,
    }


def test_add_embeddings(matcher, sample_embeddings):
    """Test adding embeddings to matcher."""
    # Add Alice
    matcher.add("Alice", sample_embeddings["Alice"])

    assert "Alice" in matcher.embeddings_per_label
    assert len(matcher.embeddings_per_label["Alice"]) == 5

    # Add Bob
    matcher.add("Bob", sample_embeddings["Bob"])

    assert "Bob" in matcher.embeddings_per_label
    assert len(matcher.embeddings_per_label["Bob"]) == 5


def test_add_single_embedding(matcher):
    """Test adding a single embedding (1D array)."""
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)

    matcher.add("Test", emb)

    assert "Test" in matcher.embeddings_per_label
    assert len(matcher.embeddings_per_label["Test"]) == 1


def test_add_unnormalized_embeddings(matcher):
    """Test that unnormalized embeddings are automatically normalized."""
    # Create unnormalized embeddings (norm != 1)
    embs = np.random.randn(5, 512).astype(np.float32) * 10.0

    # Should auto-normalize without error
    matcher.add("Test", embs)

    # Check stored embeddings are normalized
    for emb in matcher.embeddings_per_label["Test"]:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01


def test_build_index(matcher, sample_embeddings):
    """Test building FAISS index."""
    # Add embeddings
    matcher.add("Alice", sample_embeddings["Alice"])
    matcher.add("Bob", sample_embeddings["Bob"])
    matcher.add("Carol", sample_embeddings["Carol"])

    # Build index
    matcher.build()

    # Check index is created
    assert matcher.index is not None
    assert matcher.centroids is not None

    # Check correct number of centroids
    assert matcher.centroids.shape == (3, 512)
    assert matcher.index.ntotal == 3

    # Check labels are sorted
    assert matcher.labels == ["Alice", "Bob", "Carol"]


def test_build_empty_matcher(matcher):
    """Test that building with no embeddings raises error."""
    with pytest.raises(RuntimeError, match="No embeddings added"):
        matcher.build()


def test_search_correct_match(matcher, sample_embeddings):
    """Test that search returns correct match."""
    # Add and build
    matcher.add("Alice", sample_embeddings["Alice"])
    matcher.add("Bob", sample_embeddings["Bob"])
    matcher.add("Carol", sample_embeddings["Carol"])
    matcher.build()

    # Query with Alice-like embedding
    query = sample_embeddings["Alice"][0]

    labels, scores = matcher.search(query, topk=1)

    # Should match Alice
    assert labels[0] == "Alice"
    assert scores[0] > 0.5  # Good similarity (comparing to centroid)

    # Query with Bob-like embedding
    query = sample_embeddings["Bob"][0]

    labels, scores = matcher.search(query, topk=1)

    # Should match Bob
    assert labels[0] == "Bob"
    assert scores[0] > 0.5  # Good similarity (comparing to centroid)


def test_search_topk(matcher, sample_embeddings):
    """Test search with topk > 1."""
    # Add and build
    matcher.add("Alice", sample_embeddings["Alice"])
    matcher.add("Bob", sample_embeddings["Bob"])
    matcher.add("Carol", sample_embeddings["Carol"])
    matcher.build()

    # Query with Alice-like embedding
    query = sample_embeddings["Alice"][0]

    labels, scores = matcher.search(query, topk=3)

    # Should return 3 results
    assert len(labels) == 3
    assert len(scores) == 3

    # Alice should be first
    assert labels[0] == "Alice"

    # Scores should be in descending order
    assert scores[0] >= scores[1]
    assert scores[1] >= scores[2]


def test_search_before_build(matcher):
    """Test that search before build raises error."""
    with pytest.raises(RuntimeError, match="Index not built"):
        query = np.random.randn(512).astype(np.float32)
        matcher.search(query)


def test_save_and_load(matcher, sample_embeddings):
    """Test saving and loading index."""
    # Build matcher
    matcher.add("Alice", sample_embeddings["Alice"])
    matcher.add("Bob", sample_embeddings["Bob"])
    matcher.build()

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        index_path = tmpdir / "test.faiss"
        labels_path = tmpdir / "test_labels.json"
        stats_path = tmpdir / "test_stats.pkl"

        # Save
        matcher.save(index_path, labels_path, stats_path)

        # Check files exist
        assert index_path.exists()
        assert labels_path.exists()
        assert stats_path.exists()

        # Load into new matcher
        matcher2 = FaissMatcher(dimension=512)
        matcher2.load(index_path, labels_path)

        # Check loaded correctly
        assert matcher2.labels == ["Alice", "Bob"]
        assert matcher2.index.ntotal == 2

        # Query should return same results
        query = sample_embeddings["Alice"][0]

        labels1, scores1 = matcher.search(query, topk=1)
        labels2, scores2 = matcher2.search(query, topk=1)

        assert labels1 == labels2
        assert abs(scores1[0] - scores2[0]) < 0.001


def test_similar_embeddings_high_score():
    """Test that similar embeddings produce high similarity scores."""
    matcher = FaissMatcher(dimension=512)

    # Create two very similar embeddings
    base = np.random.randn(512).astype(np.float32)
    base = base / np.linalg.norm(base)

    emb1 = base + np.random.randn(512) * 0.01
    emb1 = emb1 / np.linalg.norm(emb1)

    emb2 = base + np.random.randn(512) * 0.01
    emb2 = emb2 / np.linalg.norm(emb2)

    # Add and build
    matcher.add("Person", emb1.reshape(1, -1))
    matcher.build()

    # Search with similar embedding
    labels, scores = matcher.search(emb2, topk=1)

    # Should have high similarity (>0.9)
    assert scores[0] > 0.9


def test_different_embeddings_low_score():
    """Test that different embeddings produce low similarity scores."""
    matcher = FaissMatcher(dimension=512)

    # Create two very different embeddings (orthogonal)
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0  # [1, 0, 0, ...]

    emb2 = np.zeros(512, dtype=np.float32)
    emb2[256] = 1.0  # [0, ..., 1, ..., 0]

    # Add and build
    matcher.add("Person1", emb1.reshape(1, -1))
    matcher.add("Person2", emb2.reshape(1, -1))
    matcher.build()

    # Search Person1 embedding
    labels, scores = matcher.search(emb1, topk=2)

    # Person1 should match itself with high score
    assert labels[0] == "Person1"
    assert scores[0] > 0.99

    # Person2 should have very low score (orthogonal vectors)
    assert labels[1] == "Person2"
    assert scores[1] < 0.1


def test_repr(matcher, sample_embeddings):
    """Test string representation of matcher."""
    # Before build
    repr_before = repr(matcher)
    assert "not built" in repr_before

    # After build
    matcher.add("Alice", sample_embeddings["Alice"])
    matcher.build()

    repr_after = repr(matcher)
    assert "labels=1" in repr_after
    assert "index_size=1" in repr_after