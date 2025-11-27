"""Dlib-based matcher for face recognition using Euclidean distance.

This module provides a simple matcher using Euclidean distance and voting
mechanism as described in the PyImageSearch tutorial. Uses pickle for storage.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import face_recognition
import numpy as np

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DlibMatcher:
    """Dlib-based matcher for face identification using Euclidean distance.

    This matcher follows the PyImageSearch tutorial approach:
    - Store all embeddings (not just centroids)
    - Use Euclidean distance for comparison
    - Voting mechanism when multiple embeddings per person

    The workflow is:
    1. add() - Add embeddings for each person
    2. build() - Finalize the internal data structures
    3. search() - Query using distance + voting
    4. save() / load() - Persist and restore encodings

    Attributes:
        tolerance: Distance threshold for face matching (lower = stricter)
        encodings: List of all stored embeddings
        labels: List of corresponding labels (same length as encodings)
        dimension: Embedding dimension (128 for dlib)

    Example:
        >>> matcher = DlibMatcher(tolerance=0.6)
        >>> matcher.add("Alice", alice_embeddings)  # shape [10, 128]
        >>> matcher.add("Bob", bob_embeddings)      # shape [15, 128]
        >>> matcher.build()
        >>> labels, scores = matcher.search(query_embedding)
        >>> print(f"Best match: {labels[0]} with score {scores[0]:.2f}")
    """

    def __init__(self, tolerance: float = 0.6, dimension: int = 128):
        """Initialize dlib matcher.

        Args:
            tolerance: Distance threshold for considering a match.
                      Default 0.6 is standard for face_recognition library.
                      Lower values = stricter matching.
            dimension: Embedding dimension (default: 128 for dlib)
        """
        self.tolerance = tolerance
        self.dimension = dimension

        # Storage for embeddings and labels
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []

        # Grouped storage (for building)
        self._embeddings_per_label: dict[str, list[np.ndarray]] = {}
        self._is_built = False

        logger.debug(
            f"Initialized DlibMatcher with tolerance={tolerance}, "
            f"dimension={dimension}"
        )

    def add(self, label: str, embeddings: np.ndarray) -> None:
        """Add embeddings for a person.

        Args:
            label: Person name or ID
            embeddings: Embeddings array, shape [N, D] or [D] for single embedding.

        Raises:
            ValueError: If embeddings have wrong shape.

        Example:
            >>> embeddings = embedder.embed_batch(face_crops)  # [15, 128]
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

        # Store embeddings
        if label not in self._embeddings_per_label:
            self._embeddings_per_label[label] = []

        for emb in embeddings:
            self._embeddings_per_label[label].append(emb.astype(np.float32))

        logger.info(
            f"Added {len(embeddings)} embedding(s) for '{label}' "
            f"(total: {len(self._embeddings_per_label[label])})"
        )

    def build(self) -> None:
        """Build the matcher from stored embeddings.

        Unlike FAISS matcher, this stores all embeddings (not centroids)
        to support the voting mechanism from PyImageSearch tutorial.

        Raises:
            RuntimeError: If no embeddings have been added.

        Example:
            >>> matcher.add("Luis", embeddings1)
            >>> matcher.add("Maria", embeddings2)
            >>> matcher.build()
        """
        if not self._embeddings_per_label:
            raise RuntimeError("No embeddings added. Use add() before build().")

        logger.info(
            f"Building matcher with {len(self._embeddings_per_label)} labels..."
        )

        # Flatten embeddings and labels into parallel lists
        self.encodings = []
        self.names = []

        for label in sorted(self._embeddings_per_label.keys()):
            embs = self._embeddings_per_label[label]
            for emb in embs:
                self.encodings.append(emb)
                self.names.append(label)

        self._is_built = True

        logger.info(
            f"Matcher built successfully: {len(self.encodings)} encodings, "
            f"{len(set(self.names))} unique persons"
        )

    def search(
        self,
        embedding: np.ndarray,
        topk: int = 1,
    ) -> Tuple[List[str], List[float]]:
        """Search for most similar persons using voting mechanism.

        This implements the PyImageSearch tutorial approach:
        1. Compare query against all stored encodings
        2. Find all matches within tolerance distance
        3. Use voting to determine the most likely person
        4. Convert distance to similarity score (1 - distance)

        Args:
            embedding: Query embedding, shape [128] or [1, 128].
            topk: Number of top matches to return (default: 1)

        Returns:
            Tuple of (labels, scores):
                - labels: List of person names, ordered by votes/similarity
                - scores: List of similarity scores (0-1 range, higher = more similar)

        Raises:
            RuntimeError: If matcher has not been built yet.

        Example:
            >>> labels, scores = matcher.search(query_embedding, topk=3)
            >>> if scores[0] > 0.4:  # threshold
            ...     print(f"Identified: {labels[0]}")
            ... else:
            ...     print("Unknown person")
        """
        if not self._is_built:
            raise RuntimeError("Matcher not built. Call build() first.")

        if len(self.encodings) == 0:
            return [], []

        # Handle 1D embedding
        if embedding.ndim == 2:
            embedding = embedding.flatten()

        # Validate shape
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embedding.shape[0]}"
            )

        # Ensure float32
        embedding = embedding.astype(np.float32)

        # If embeddings are L2-normalized, we can use face_recognition directly
        # or compute distances manually
        # face_recognition.compare_faces returns boolean matches
        # face_recognition.face_distance returns actual distances

        # Compute distances to all stored encodings
        distances = face_recognition.face_distance(self.encodings, embedding)

        # Find matches within tolerance
        matches = list(distances <= self.tolerance)

        # Voting mechanism (from PyImageSearch tutorial)
        if True in matches:
            # Get indices of all matches
            matched_indices = [i for i, match in enumerate(matches) if match]

            # Count votes for each person
            votes = Counter(self.names[i] for i in matched_indices)

            # Sort by votes (descending), then by average distance (ascending)
            candidates = []
            for name, vote_count in votes.most_common():
                # Calculate average distance for this person
                person_distances = [
                    distances[i]
                    for i in matched_indices
                    if self.names[i] == name
                ]
                avg_distance = np.mean(person_distances)
                min_distance = np.min(person_distances)

                # Convert distance to similarity (1 - distance), clamped to [0, 1]
                similarity = float(np.clip(1.0 - min_distance, 0.0, 1.0))

                candidates.append((name, similarity, vote_count))

            # Sort by votes (desc), then similarity (desc)
            candidates.sort(key=lambda x: (-x[2], -x[1]))

            # Extract top-k results
            result_labels = [c[0] for c in candidates[:topk]]
            result_scores = [c[1] for c in candidates[:topk]]

        else:
            # No matches within tolerance
            # Return best match anyway but with low score
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            best_label = self.names[best_idx]

            # Convert distance to similarity
            similarity = float(np.clip(1.0 - best_distance, 0.0, 1.0))

            result_labels = [best_label]
            result_scores = [similarity]

        # Pad results if needed
        while len(result_labels) < topk and len(result_labels) < len(set(self.names)):
            # Add remaining persons with their best scores
            remaining = set(self.names) - set(result_labels)
            if not remaining:
                break

            for name in remaining:
                person_indices = [i for i, n in enumerate(self.names) if n == name]
                person_distances = [distances[i] for i in person_indices]
                min_dist = np.min(person_distances)
                similarity = float(np.clip(1.0 - min_dist, 0.0, 1.0))
                result_labels.append(name)
                result_scores.append(similarity)

                if len(result_labels) >= topk:
                    break

        logger.debug(
            f"Search results (topk={topk}): "
            f"{list(zip(result_labels[:topk], result_scores[:topk]))}"
        )

        return result_labels[:topk], result_scores[:topk]

    def save(
        self,
        encodings_path: str | Path,
        labels_path: str | Path | None = None,
        stats_path: str | Path | None = None,
    ) -> None:
        """Save encodings and labels to disk (pickle format).

        Args:
            encodings_path: Path to save encodings (.pkl file)
                           This file contains both encodings and names.
            labels_path: Optional path to save unique labels (.json file)
            stats_path: Optional path to save additional stats (.pkl file)

        Raises:
            RuntimeError: If matcher has not been built yet.

        Example:
            >>> matcher.save(
            ...     "models/encodings.pkl",
            ...     "models/labels.json",
            ...     "models/stats.pkl"
            ... )
        """
        if not self._is_built:
            raise RuntimeError("Matcher not built. Call build() before save().")

        encodings_path = Path(encodings_path)
        encodings_path.parent.mkdir(parents=True, exist_ok=True)

        # Save encodings and names in pickle format (like PyImageSearch tutorial)
        data = {
            "encodings": self.encodings,
            "names": self.names,
        }

        with open(encodings_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            f"Saved {len(self.encodings)} encodings to {encodings_path}"
        )

        # Save unique labels as JSON (optional)
        if labels_path is not None:
            labels_path = Path(labels_path)
            labels_path.parent.mkdir(parents=True, exist_ok=True)

            unique_labels = sorted(set(self.names))
            with open(labels_path, "w") as f:
                json.dump(unique_labels, f, indent=2)

            logger.info(f"Saved {len(unique_labels)} labels to {labels_path}")

        # Save additional stats (optional)
        if stats_path is not None:
            stats_path = Path(stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            # Count embeddings per label
            embeddings_per_label = Counter(self.names)

            stats = {
                "num_labels": len(set(self.names)),
                "num_encodings": len(self.encodings),
                "dimension": self.dimension,
                "tolerance": self.tolerance,
                "embeddings_per_label": dict(embeddings_per_label),
            }

            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)

            logger.info(f"Saved stats to {stats_path}")

    def load(
        self,
        encodings_path: str | Path,
        labels_path: str | Path | None = None,
    ) -> None:
        """Load encodings and labels from disk.

        Args:
            encodings_path: Path to encodings pickle file
            labels_path: Optional path to labels (not used, kept for API compatibility)

        Raises:
            FileNotFoundError: If file doesn't exist.

        Example:
            >>> matcher = DlibMatcher()
            >>> matcher.load("models/encodings.pkl")
            >>> labels, scores = matcher.search(query_embedding)
        """
        encodings_path = Path(encodings_path)

        if not encodings_path.exists():
            raise FileNotFoundError(f"Encodings file not found: {encodings_path}")

        # Load pickle file
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)

        self.encodings = data["encodings"]
        self.names = data["names"]
        self._is_built = True

        logger.info(
            f"Loaded {len(self.encodings)} encodings for "
            f"{len(set(self.names))} persons from {encodings_path}"
        )

    def __repr__(self) -> str:
        """String representation of matcher."""
        if not self._is_built:
            return f"DlibMatcher(tolerance={self.tolerance}, not built)"
        else:
            return (
                f"DlibMatcher(tolerance={self.tolerance}, "
                f"encodings={len(self.encodings)}, "
                f"persons={len(set(self.names))})"
            )