"""ArcFace embedder for face feature extraction.

This module provides face embedding extraction using ArcFace model from InsightFace.
It converts aligned 112x112 face crops into 512-dimensional feature vectors.
"""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import Config
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ArcFaceEmbedder:
    """ArcFace embedder for extracting 512-D face features.

    This embedder uses InsightFace's ArcFace recognition model to convert
    aligned 112x112 face crops into L2-normalized 512-dimensional feature vectors.

    The embeddings are suitable for:
    - Face verification (cosine similarity > threshold)
    - Face identification (nearest neighbor search with FAISS)
    - Face clustering

    Attributes:
        app: InsightFace FaceAnalysis instance
        ctx_id: Compute context (-1=CPU, 0+=GPU)
        embedding_dim: Dimension of output embeddings (512 for ArcFace)

    Example:
        >>> embedder = ArcFaceEmbedder(config)
        >>> aligned = aligner.align(frame, detection.kps)
        >>> embedding = embedder.embed(aligned)
        >>> assert embedding.shape == (512,)
        >>> assert abs(np.linalg.norm(embedding) - 1.0) < 0.01
    """

    def __init__(self, config: Config):
        """Initialize ArcFace embedder.

        Args:
            config: Configuration object with model settings.

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.ctx_id = config.ctx_id
        self.embedding_dim = 512

        logger.info(
            f"Initializing ArcFace embedder with model_pack={config.model_pack}, "
            f"ctx_id={config.ctx_id}"
        )

        try:
            # Initialize FaceAnalysis
            # Note: We load the full model but will only use the recognition part
            self.app = FaceAnalysis(
                name=config.model_pack,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if config.ctx_id >= 0
                    else ["CPUExecutionProvider"]
                ),
            )

            # Prepare model with detection disabled (we only need recognition)
            # det_size is still required by prepare() even though we won't use detection
            self.app.prepare(ctx_id=config.ctx_id, det_size=(640, 640))

            logger.info(
                f"ArcFace embedder initialized successfully "
                f"(device: {'GPU' if config.ctx_id >= 0 else 'CPU'})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ArcFace embedder: {e}")
            raise RuntimeError(f"ArcFace embedder initialization failed: {e}") from e

    def embed(self, face_bgr_112: np.ndarray) -> np.ndarray:
        """Extract 512-D embedding from aligned face crop.

        Args:
            face_bgr_112: Aligned face crop in BGR format, shape [112, 112, 3], dtype uint8.
                         Must be pre-aligned using FivePointAligner.

        Returns:
            L2-normalized embedding vector, shape [512], dtype float32.
            Norm is approximately 1.0 (within 1e-6).

        Raises:
            ValueError: If input image has invalid shape or type.
            RuntimeError: If embedding extraction fails.

        Example:
            >>> aligned = aligner.align(frame, detection.kps)
            >>> embedding = embedder.embed(aligned)
            >>> similarity = np.dot(embedding, other_embedding)
            >>> if similarity > 0.7:
            ...     print("Same person")
        """
        # Validate input shape
        if face_bgr_112.shape != (112, 112, 3):
            raise ValueError(
                f"Expected face shape (112, 112, 3), got {face_bgr_112.shape}. "
                f"Use FivePointAligner to align faces before embedding."
            )

        if face_bgr_112.dtype != np.uint8:
            raise ValueError(
                f"Expected dtype uint8, got {face_bgr_112.dtype}. "
                f"Aligned faces should be in BGR uint8 format."
            )

        try:
            # Access the recognition model directly from FaceAnalysis
            # We don't need detection since the face is already aligned
            rec_model = None
            for taskname, model in self.app.models.items():
                if taskname == "recognition":
                    rec_model = model
                    break

            if rec_model is None:
                raise RuntimeError("Recognition model not found in FaceAnalysis")

            # Convert BGR to RGB (InsightFace models expect RGB)
            face_rgb = face_bgr_112[:, :, ::-1]

            # Get embedding directly from recognition model
            # The model's get_feat method expects aligned 112x112 RGB image
            embedding = rec_model.get_feat(face_rgb)

            # Ensure float32 type and proper shape
            embedding = embedding.flatten().astype(np.float32)

            # Validate embedding dimension
            if embedding.shape[0] != self.embedding_dim:
                raise RuntimeError(
                    f"Unexpected embedding dimension {embedding.shape[0]}, "
                    f"expected {self.embedding_dim}"
                )

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm < 1e-10:
                logger.warning("Embedding has near-zero norm, returning zero vector")
                return np.zeros(self.embedding_dim, dtype=np.float32)

            embedding = embedding / norm

            # Final validation
            final_norm = np.linalg.norm(embedding)
            if abs(final_norm - 1.0) > 0.01:
                logger.warning(
                    f"Embedding norm {final_norm:.4f} is not close to 1.0 after normalization"
                )

            return embedding

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def embed_batch(self, faces_bgr_112: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings from multiple aligned faces.

        Args:
            faces_bgr_112: List of aligned face crops, each [112, 112, 3] uint8.

        Returns:
            Embeddings array, shape [N, 512], dtype float32.
            Each row is an L2-normalized 512-D embedding.

        Example:
            >>> aligned_faces = aligner.align_batch(frame, kps_list)
            >>> embeddings = embedder.embed_batch(aligned_faces)
            >>> assert embeddings.shape == (len(aligned_faces), 512)
        """
        if not faces_bgr_112:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings = []

        for face in faces_bgr_112:
            try:
                emb = self.embed(face)
                embeddings.append(emb)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to embed face: {e}. Skipping.")
                continue

        if not embeddings:
            logger.warning("No embeddings extracted from batch.")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Stack embeddings into array
        embeddings_arr = np.stack(embeddings, axis=0)  # shape: [N, 512]

        return embeddings_arr

    def __repr__(self) -> str:
        """String representation of embedder."""
        device = "GPU" if self.ctx_id >= 0 else "CPU"
        return f"ArcFaceEmbedder(dim={self.embedding_dim}, device={device})"