"""Dlib embedder for face feature extraction using face_recognition library.

This module provides face embedding extraction using dlib's ResNet-34 model
via the face_recognition library (PyImageSearch tutorial approach).
It converts face images into 128-dimensional feature vectors.
"""

from __future__ import annotations

from typing import Literal

import cv2
import face_recognition
import numpy as np

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DlibEmbedder:
    """Dlib embedder for extracting 128-D face features.

    This embedder uses dlib's ResNet-34 model (trained on ~3 million faces)
    to convert face images into 128-dimensional feature vectors.

    The model achieves 99.38% accuracy on the Labeled Faces in the Wild (LFW)
    benchmark using deep metric learning (triplet loss training).

    Attributes:
        model: Model size ("large" or "small")
        num_jitters: Number of times to re-sample face for encoding
        embedding_dim: Dimension of output embeddings (128 for dlib)

    Example:
        >>> embedder = DlibEmbedder(model="large")
        >>> embedding = embedder.embed(face_crop)
        >>> assert embedding.shape == (128,)
    """

    def __init__(
        self,
        model: Literal["large", "small"] = "large",
        num_jitters: int = 1,
    ):
        """Initialize dlib embedder.

        Args:
            model: Model size to use.
                   "large" - More accurate, slower (default)
                   "small" - Faster, slightly less accurate
            num_jitters: Number of times to re-sample the face when calculating
                        encoding. Higher values are more accurate but slower.
                        Default: 1 (no jittering)

        Example:
            >>> embedder = DlibEmbedder(model="large", num_jitters=1)
            >>> embedder = DlibEmbedder(model="small", num_jitters=5)  # More accurate
        """
        if model not in ("large", "small"):
            raise ValueError(f"model must be 'large' or 'small', got '{model}'")

        self.model = model
        self.num_jitters = num_jitters
        self.embedding_dim = 128

        logger.info(
            f"Initializing dlib embedder (model={model}, num_jitters={num_jitters})"
        )
        logger.info("Dlib embedder initialized successfully")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Extract 128-D embedding from a face image.

        Args:
            face_bgr: Face image in BGR format (OpenCV convention).
                     Can be any size - the model will handle it.
                     Typically, 112x112 aligned or original crop.

        Returns:
            Embedding vector, shape [128], dtype float32.
            Note: dlib embeddings are NOT L2-normalized by default.
            We normalize them for consistency with ArcFace backend.

        Raises:
            ValueError: If the input image is invalid or no face encoding could be computed.
            RuntimeError: If embedding extraction fails.

        Example:
            >>> face_crop = frame[y1:y2, x1:x2]
            >>> embedding = embedder.embed(face_crop)
            >>> distance = np.linalg.norm(embedding - other_embedding)
            >>> if distance < 0.6:
            ...     print("Same person")
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face image provided")

        if len(face_bgr.shape) != 3 or face_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel image, got shape {face_bgr.shape}"
            )

        try:
            # Convert BGR to RGB (face_recognition expects RGB)
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            # Get face encoding
            # We pass the whole image as a single face location
            h, w = face_rgb.shape[:2]
            # face_recognition expects (top, right, bottom, left) format
            face_locations = [(0, w, h, 0)]

            encodings = face_recognition.face_encodings(
                face_rgb,
                known_face_locations=face_locations,
                num_jitters=self.num_jitters,
                model=self.model,
            )

            if not encodings:
                raise ValueError(
                    "Could not compute face encoding. "
                    "The image may not contain a clear face."
                )

            # Get the first (and only) encoding
            embedding = np.array(encodings[0], dtype=np.float32)

            # Validate dimension
            if embedding.shape[0] != self.embedding_dim:
                raise RuntimeError(
                    f"Unexpected embedding dimension {embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )

            # L2 normalize for consistency with ArcFace backend
            # This allows using cosine similarity like ArcFace
            norm = np.linalg.norm(embedding)
            if norm < 1e-10:
                logger.warning("Embedding has near-zero norm, returning zero vector")
                return np.zeros(self.embedding_dim, dtype=np.float32)

            embedding = embedding / norm

            return embedding

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def embed_from_frame(
        self,
        frame_bgr: np.ndarray,
        face_location: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extract embedding directly from frame with face location.

        This method is more efficient when you have the original frame
        and face location, as it avoids an extra crop operation.

        Args:
            frame_bgr: Full frame in BGR format.
            face_location: Face location as (top, right, bottom, left).

        Returns:
            Embedding vector, shape [128], dtype float32.

        Example:
            >>> # From detector results
            >>> embedding = embedder.embed_from_frame(frame, (top, right, bottom, left))
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame provided")

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Get face encoding at specified location
            encodings = face_recognition.face_encodings(
                frame_rgb,
                known_face_locations=[face_location],
                num_jitters=self.num_jitters,
                model=self.model,
            )

            if not encodings:
                raise ValueError("Could not compute face encoding at given location")

            embedding = np.array(encodings[0], dtype=np.float32)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            return embedding

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to extract embedding from frame: {e}")
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def embed_batch(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings from multiple face images.

        Args:
            faces_bgr: List of face images in BGR format.

        Returns:
            Embeddings array, shape [N, 128], dtype float32.
            Each row is an L2-normalized 128-D embedding.

        Example:
            >>> crops = [frame[y1:y2, x1:x2] for det in detections]
            >>> embeddings = embedder.embed_batch(crops)
            >>> assert embeddings.shape == (len(crops), 128)
        """
        if not faces_bgr:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings = []

        for face in faces_bgr:
            try:
                emb = self.embed(face)
                embeddings.append(emb)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to embed face: {e}. Skipping.")
                continue

        if not embeddings:
            logger.warning("No embeddings extracted from batch.")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        return np.stack(embeddings, axis=0)

    def __repr__(self) -> str:
        """String representation of embedder."""
        return (
            f"DlibEmbedder(model='{self.model}', "
            f"num_jitters={self.num_jitters}, dim={self.embedding_dim})"
        )