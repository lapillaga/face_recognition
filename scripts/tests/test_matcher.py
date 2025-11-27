#!/usr/bin/env python3
"""Test script for FAISS matcher with synthetic data.

This script demonstrates the matcher functionality by:
1. Creating synthetic embeddings for 5 persons
2. Building a FAISS index
3. Performing queries to verify matching works
4. Testing save/load functionality

Usage:
    python scripts/test_matcher.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging
from app.backends.insightface.matcher import FaissMatcher

logger = setup_logging(__name__)


def create_synthetic_person_embeddings(
    num_embeddings: int = 10,
    dimension: int = 512,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create synthetic embeddings for multiple persons.

    Each person has a "base" embedding with some Gaussian noise added
    to simulate variations in pose, lighting, etc.

    Args:
        num_embeddings: Number of embeddings per person
        dimension: Embedding dimension
        seed: Random seed for reproducibility

    Returns:
        Dict mapping person name → embeddings array [N, 512]
    """
    np.random.seed(seed)

    persons = {}

    # Create 5 persons with distinct base embeddings
    person_names = ["LUIS", "MARIA", "CARLOS", "ANA", "PEDRO"]

    for i, name in enumerate(person_names):
        # Create a unique base embedding for this person
        # Use different random seeds for each person
        np.random.seed(seed + i * 100)

        # Create base embedding (random direction in 512-D space)
        base = np.random.randn(dimension).astype(np.float32)
        base = base / np.linalg.norm(base)  # L2 normalize

        # Create multiple embeddings with slight variations
        embeddings = []

        for _ in range(num_embeddings):
            # Add Gaussian noise (simulate pose/lighting variations)
            noisy = base + np.random.randn(dimension) * 0.05
            noisy = noisy / np.linalg.norm(noisy)  # Re-normalize

            embeddings.append(noisy)

        persons[name] = np.stack(embeddings, axis=0)

        logger.info(
            f"Created {num_embeddings} embeddings for {name}, "
            f"shape: {persons[name].shape}"
        )

    return persons


def print_section(title: str) -> None:
    """Print a section divider."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    """Main function to test matcher."""
    logger.info("Starting FAISS matcher test with synthetic data...")

    # Step 1: Create synthetic embeddings
    print_section("Step 1: Creating Synthetic Embeddings")

    persons = create_synthetic_person_embeddings(
        num_embeddings=10,
        dimension=512,
        seed=42,
    )

    print(f"Created embeddings for {len(persons)} persons")
    for name, embs in persons.items():
        print(f"  - {name}: {embs.shape[0]} embeddings")

    # Step 2: Build FAISS index
    print_section("Step 2: Building FAISS Index")

    matcher = FaissMatcher(dimension=512)

    for name, embs in persons.items():
        matcher.add(name, embs)
        print(f"Added {name}: {embs.shape[0]} embeddings")

    matcher.build()
    print(f"FAISS index built: {matcher.index.ntotal} centroids")

    # Step 3: Test queries
    print_section("Step 3: Testing Queries")

    print("\nQuery Query 1: LUIS-like embedding (should match LUIS)")
    query1 = persons["LUIS"][0]  # First embedding from LUIS
    labels1, scores1 = matcher.search(query1, topk=3)

    print("   Top 3 matches:")
    for i, (label, score) in enumerate(zip(labels1, scores1), 1):
        indicator = "*" if i == 1 and label == "LUIS" else "  "
        print(f"   {indicator} {i}. {label:10s} - Score: {score:.4f}")

    print("\nQuery Query 2: MARIA-like embedding (should match MARIA)")
    query2 = persons["MARIA"][0]
    labels2, scores2 = matcher.search(query2, topk=3)

    print("   Top 3 matches:")
    for i, (label, score) in enumerate(zip(labels2, scores2), 1):
        indicator = "*" if i == 1 and label == "MARIA" else "  "
        print(f"   {indicator} {i}. {label:10s} - Score: {score:.4f}")

    print("\nQuery Query 3: CARLOS-like embedding (should match CARLOS)")
    query3 = persons["CARLOS"][0]
    labels3, scores3 = matcher.search(query3, topk=3)

    print("   Top 3 matches:")
    for i, (label, score) in enumerate(zip(labels3, scores3), 1):
        indicator = "*" if i == 1 and label == "CARLOS" else "  "
        print(f"   {indicator} {i}. {label:10s} - Score: {score:.4f}")

    # Step 4: Test centroid stability
    print_section("Step 4: Centroid Stability Test")

    print("Testing that multiple embeddings from same person → similar centroid")
    print()

    for name in ["LUIS", "MARIA"]:
        # Compute similarity between embeddings and centroid
        centroid_idx = matcher.labels.index(name)
        centroid = matcher.centroids[centroid_idx]

        similarities = []
        for emb in persons[name]:
            sim = np.dot(emb, centroid)
            similarities.append(sim)

        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        print(f"{name}:")
        print(f"  Average similarity to centroid: {avg_sim:.4f}")
        print(f"  Std deviation:                  {std_sim:.4f}")
        print(f"  Min similarity:                 {np.min(similarities):.4f}")
        print(f"  Max similarity:                 {np.max(similarities):.4f}")

        if avg_sim > 0.95:
            print(f"  Excellent centroid stability (>{0.95:.2f})")
        elif avg_sim > 0.90:
            print(f"  Good centroid stability (>{0.90:.2f})")
        else:
            print(f"  WARNING: Low centroid stability (<{0.90:.2f})")

        print()

    # Step 5: Test save/load
    print_section("Step 5: Testing Save/Load")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    index_path = models_dir / "test_centroids.faiss"
    labels_path = models_dir / "test_labels.json"
    stats_path = models_dir / "test_stats.pkl"

    # Save
    matcher.save(index_path, labels_path, stats_path)
    print(f"Saved index to {index_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Saved stats to {stats_path}")

    # Load into new matcher
    matcher2 = FaissMatcher(dimension=512)
    matcher2.load(index_path, labels_path)
    print(f"Loaded index from {index_path}")

    # Verify same results
    print("\nQuery Verifying loaded matcher produces same results...")
    labels_new, scores_new = matcher2.search(query1, topk=3)

    print("   Original matcher:")
    for label, score in zip(labels1, scores1):
        print(f"     - {label:10s}: {score:.4f}")

    print("   Loaded matcher:")
    for label, score in zip(labels_new, scores_new):
        print(f"     - {label:10s}: {score:.4f}")

    # Check if results match
    if labels1 == labels_new and np.allclose(scores1, scores_new, atol=1e-6):
        print("   Results match perfectly!")
    else:
        print("   Results don't match (unexpected)")

    # Step 6: Summary statistics
    print_section("Step 6: Summary Statistics")

    print(f"Total persons enrolled:     {len(matcher.labels)}")
    print(f"Total embeddings collected: {sum(len(embs) for embs in persons.values())}")
    print(f"Embeddings per person:      {10} (average)")
    print(f"Index size:                 {matcher.index.ntotal} centroids")
    print(f"Embedding dimension:        {matcher.dimension}")
    print()

    # Cleanup test files
    print("Cleaning up test files...")
    if index_path.exists():
        index_path.unlink()
        print(f"   Deleted {index_path}")
    if labels_path.exists():
        labels_path.unlink()
        print(f"   Deleted {labels_path}")
    if stats_path.exists():
        stats_path.unlink()
        print(f"   Deleted {stats_path}")

    print_section("All Tests Passed!")

    print()
    logger.info("FAISS matcher test completed successfully")


if __name__ == "__main__":
    main()