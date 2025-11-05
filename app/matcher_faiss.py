"""FAISS-based matcher for face recognition.

This module provides a FAISS index for efficient nearest-neighbor search
over face embeddings, enabling fast identification of enrolled persons.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


class FaissMatcher:
    """FAISS-based matcher for face identification.

    This matcher builds a FAISS index over face embeddings (centroids per person)
    and supports efficient nearest-neighbor search using cosine similarity.

    The workflow is:
    1. add() - Add embeddings for each person (can add multiple per person)
    2. build() - Compute centroids and build FAISS index
    3. search() - Query the index to find most similar person
    4. save() / load() - Persist and restore the index

    Attributes:
        labels: List of person names/IDs (ordered)
        embeddings_per_label: Dict mapping label → list of embeddings
        centroids: Average embedding per label, shape [N, 512]
        index: FAISS index for similarity search
        dimension: Embedding dimension (512 for ArcFace)

    Example:
        >>> matcher = FaissMatcher()
        >>> matcher.add("Alice", alice_embeddings)  # shape [10, 512]
        >>> matcher.add("Bob", bob_embeddings)      # shape [15, 512]
        >>> matcher.build()
        >>> labels, scores = matcher.search(query_embedding)
        >>> print(f"Best match: {labels[0]} with score {scores[0]:.2f}")
    """

    def __init__(self, dimension: int = 512):
        """Initialize FAISS matcher.

        Args:
            dimension: Embedding dimension (default: 512 for ArcFace)
        """
        self.dimension = dimension
        self.labels: List[str] = []
        self.embeddings_per_label: dict[str, list[np.ndarray]] = {}
        self.centroids: np.ndarray | None = None
        self.index: faiss.Index | None = None

        logger.debug(f"Initialized FaissMatcher with dimension={dimension}")

    def add(self, label: str, embeddings: np.ndarray) -> None:
        """Add embeddings for a person.

        Args:
            label: Person name or ID
            embeddings: Embeddings array, shape [N, D] or [D] for single embedding.
                       Must be L2-normalized.

        Raises:
            ValueError: If embeddings have wrong shape or are not normalized.

        Example:
            >>> embeddings = embedder.embed_batch(aligned_faces)  # [15, 512]
            >>> matcher.add("Luis", embeddings)
        """
        # Handle single embedding (1D array)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Validate shape
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings shape [N, {self.dimension}] or [{self.dimension}], "
                f"got {embeddings.shape}"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        # Validate normalization (should be close to 1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=0.1):
            logger.warning(
                f"Embeddings for '{label}' are not L2-normalized "
                f"(norms: {norms.min():.4f} - {norms.max():.4f}). "
                f"Re-normalizing..."
            )
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # Store embeddings
        if label not in self.embeddings_per_label:
            self.embeddings_per_label[label] = []

        for emb in embeddings:
            self.embeddings_per_label[label].append(emb.astype(np.float32))

        logger.info(
            f"Added {len(embeddings)} embedding(s) for '{label}' "
            f"(total: {len(self.embeddings_per_label[label])})"
        )

    def build(self) -> None:
        """Build FAISS index from stored embeddings.

        Computes centroid (average) embedding for each label and builds
        an IndexFlatIP (inner product) index for cosine similarity search.

        Raises:
            RuntimeError: If no embeddings have been added.

        Example:
            >>> matcher.add("Luis", embeddings1)
            >>> matcher.add("Maria", embeddings2)
            >>> matcher.build()
            >>> print(f"Index built with {matcher.index.ntotal} persons")
        """
        if not self.embeddings_per_label:
            raise RuntimeError("No embeddings added. Use add() before build().")

        logger.info(f"Building FAISS index with {len(self.embeddings_per_label)} labels...")

        # Sort labels for deterministic ordering
        self.labels = sorted(self.embeddings_per_label.keys())

        # Compute centroids (average embedding per label)
        centroids = []

        for label in self.labels:
            embs = np.array(self.embeddings_per_label[label])  # [N, 512]

            # Average and re-normalize
            centroid = embs.mean(axis=0)  # [512]
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

            centroids.append(centroid)

            logger.debug(
                f"Computed centroid for '{label}' from {len(embs)} embedding(s), "
                f"norm={np.linalg.norm(centroid):.4f}"
            )

        # Stack into array [N_labels, 512]
        self.centroids = np.stack(centroids, axis=0).astype(np.float32)

        # Build FAISS index
        # Use IndexFlatIP (inner product) for cosine similarity
        # Since embeddings are L2-normalized, inner product = cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)

        # Add centroids to index
        self.index.add(self.centroids)

        logger.info(
            f"FAISS index built successfully: {self.index.ntotal} centroids, "
            f"dimension={self.dimension}"
        )

    def search(
        self,
        embedding: np.ndarray,
        topk: int = 1,
    ) -> Tuple[List[str], List[float]]:
        """Search for most similar persons.

        Args:
            embedding: Query embedding, shape [512] or [1, 512], must be L2-normalized.
            topk: Number of top matches to return (default: 1)

        Returns:
            Tuple of (labels, scores):
                - labels: List of person names, ordered by similarity
                - scores: List of similarity scores (0-1 range, higher = more similar)

        Raises:
            RuntimeError: If index has not been built yet.
            ValueError: If embedding has wrong shape.

        Example:
            >>> labels, scores = matcher.search(query_embedding, topk=3)
            >>> for label, score in zip(labels, scores):
            ...     print(f"{label}: {score:.2f}")
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Handle 1D embedding
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate shape
        if embedding.shape != (1, self.dimension):
            raise ValueError(
                f"Expected embedding shape [1, {self.dimension}] or [{self.dimension}], "
                f"got {embedding.shape}"
            )

        # Validate normalization
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > 0.1:
            logger.warning(
                f"Query embedding norm {norm:.4f} is not close to 1.0. "
                f"Re-normalizing..."
            )
            embedding = embedding / (norm + 1e-10)

        # Ensure float32
        embedding = embedding.astype(np.float32)

        # Limit topk to index size
        topk = min(topk, self.index.ntotal)

        # Search FAISS index
        # distances are inner products (cosine similarity for normalized vectors)
        distances, indices = self.index.search(embedding, topk)

        # Extract results
        distances = distances[0]  # [topk]
        indices = indices[0]      # [topk]

        # Map indices to labels and clamp scores to [0, 1]
        result_labels = [self.labels[idx] for idx in indices]
        result_scores = [float(np.clip(dist, 0.0, 1.0)) for dist in distances]

        logger.debug(
            f"Search results (topk={topk}): "
            f"{list(zip(result_labels, result_scores))}"
        )

        return result_labels, result_scores

    def save(
        self,
        index_path: str | Path,
        labels_path: str | Path,
        stats_path: str | Path | None = None,
    ) -> None:
        """Save FAISS index and labels to disk.

        Args:
            index_path: Path to save FAISS index (.faiss file)
            labels_path: Path to save labels (.json file)
            stats_path: Optional path to save additional stats (.pkl file)

        Raises:
            RuntimeError: If index has not been built yet.

        Example:
            >>> matcher.save(
            ...     "models/centroids.faiss",
            ...     "models/labels.json",
            ...     "models/stats.pkl"
            ... )
        """
        if self.index is None or self.centroids is None:
            raise RuntimeError("Index not built. Call build() before save().")

        index_path = Path(index_path)
        labels_path = Path(labels_path)

        # Create parent directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save labels as JSON
        with open(labels_path, "w") as f:
            json.dump(self.labels, f, indent=2)
        logger.info(f"Saved {len(self.labels)} labels to {labels_path}")

        # Save additional stats (optional)
        if stats_path is not None:
            stats_path = Path(stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            stats = {
                "num_labels": len(self.labels),
                "dimension": self.dimension,
                "embeddings_per_label": {
                    label: len(embs)
                    for label, embs in self.embeddings_per_label.items()
                },
            }

            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)

            logger.info(f"Saved stats to {stats_path}")

    def load(
        self,
        index_path: str | Path,
        labels_path: str | Path,
    ) -> None:
        """Load FAISS index and labels from disk.

        Args:
            index_path: Path to FAISS index (.faiss file)
            labels_path: Path to labels (.json file)

        Raises:
            FileNotFoundError: If files don't exist.
            RuntimeError: If index dimension doesn't match.

        Example:
            >>> matcher = FaissMatcher()
            >>> matcher.load("models/centroids.faiss", "models/labels.json")
            >>> labels, scores = matcher.search(query_embedding)
        """
        index_path = Path(index_path)
        labels_path = Path(labels_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path} ({self.index.ntotal} entries)")

        # Validate dimension
        if self.index.d != self.dimension:
            raise RuntimeError(
                f"Index dimension {self.index.d} doesn't match "
                f"expected dimension {self.dimension}"
            )

        # Load labels
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        logger.info(f"Loaded {len(self.labels)} labels from {labels_path}")

        # Validate consistency
        if len(self.labels) != self.index.ntotal:
            logger.warning(
                f"Mismatch: {len(self.labels)} labels but {self.index.ntotal} "
                f"index entries. This may cause incorrect results."
            )

    def __repr__(self) -> str:
        """String representation of matcher."""
        if self.index is None:
            return f"FaissMatcher(dim={self.dimension}, not built)"
        else:
            return (
                f"FaissMatcher(dim={self.dimension}, "
                f"labels={len(self.labels)}, "
                f"index_size={self.index.ntotal})"
            )