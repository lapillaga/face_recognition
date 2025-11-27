"""Backend factory for face recognition pipeline.

This module provides a unified way to create face recognition components
(detector, embedder, matcher) for different backends:
- InsightFace: SCRFD detector + ArcFace embeddings (512-D) + FAISS
- dlib: HOG/CNN detector + ResNet-34 embeddings (128-D) + Euclidean

Usage:
    # Default backend (InsightFace)
    components = create_backend("insightface", config)

    # Alternative backend (dlib)
    components = create_backend("dlib", config, detector_model="hog")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.config import Config
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Backend type alias
BackendType = Literal["insightface", "dlib"]


@dataclass
class BackendComponents:
    """Container for backend components.

    Attributes:
        detector: Face detector instance
        aligner: Face aligner instance (may be None for dlib)
        embedder: Face embedder instance
        matcher: Face matcher instance
        backend_type: Name of the backend ("insightface" or "dlib")
        embedding_dim: Dimension of embeddings (512 or 128)
    """

    detector: object
    aligner: object | None
    embedder: object
    matcher: object
    backend_type: str
    embedding_dim: int


def create_backend(
    backend_type: BackendType = "insightface",
    config: Config | None = None,
    *,
    # dlib-specific options
    detector_model: Literal["hog", "cnn"] = "hog",
    embedder_model: Literal["large", "small"] = "large",
    tolerance: float = 0.6,
) -> BackendComponents:
    """Create face recognition components for specified backend.

    Args:
        backend_type: Backend to use ("insightface" or "dlib")
        config: Configuration object. If None, loads from .env
        detector_model: For dlib backend - "hog" (faster) or "cnn" (more accurate)
        embedder_model: For dlib backend - "large" or "small"
        tolerance: For dlib matcher - distance threshold (default 0.6)

    Returns:
        BackendComponents with detector, aligner, embedder, and matcher.

    Example:
        >>> from app.core.config import get_config
        >>> config = get_config()

        >>> # Use InsightFace (default)
        >>> components = create_backend("insightface", config)

        >>> # Use dlib with HOG detector
        >>> components = create_backend("dlib", config, detector_model="hog")

        >>> # Use dlib with CNN detector (more accurate)
        >>> components = create_backend("dlib", config, detector_model="cnn")
    """
    if config is None:
        from app.core.config import get_config
        config = get_config()

    if backend_type == "insightface":
        return _create_insightface_backend(config)
    elif backend_type == "dlib":
        return _create_dlib_backend(
            config,
            detector_model=detector_model,
            embedder_model=embedder_model,
            tolerance=tolerance,
        )
    else:
        raise ValueError(
            f"Unknown backend: '{backend_type}'. "
            f"Supported backends: 'insightface', 'dlib'"
        )


def _create_insightface_backend(config: Config) -> BackendComponents:
    """Create InsightFace backend components.

    Uses:
    - SCRFD detector (via InsightFace FaceAnalysis)
    - 5-point aligner (similarity transform to 112x112)
    - ArcFace embedder (512-D embeddings)
    - FAISS matcher (cosine similarity)
    """
    logger.info("Creating InsightFace backend...")

    from app.backends.insightface.aligner import FivePointAligner
    from app.backends.insightface.detector import SCRFDDetector
    from app.backends.insightface.embedder import ArcFaceEmbedder
    from app.backends.insightface.matcher import FaissMatcher

    detector = SCRFDDetector(config)
    aligner = FivePointAligner()
    embedder = ArcFaceEmbedder(config)
    matcher = FaissMatcher(dimension=512)

    logger.info("InsightFace backend created successfully")

    return BackendComponents(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        matcher=matcher,
        backend_type="insightface",
        embedding_dim=512,
    )


def _create_dlib_backend(
    config: Config,
    detector_model: str,
    embedder_model: str,
    tolerance: float,
) -> BackendComponents:
    """Create dlib backend components.

    Uses:
    - dlib HOG or CNN detector (via face_recognition)
    - No explicit aligner (face_recognition handles internally)
    - dlib ResNet-34 embedder (128-D embeddings)
    - Euclidean distance matcher with voting
    """
    logger.info(f"Creating dlib backend (detector={detector_model})...")

    from app.backends.dlib.detector import DlibDetector
    from app.backends.dlib.embedder import DlibEmbedder
    from app.backends.dlib.matcher import DlibMatcher

    detector = DlibDetector(model=detector_model)
    embedder = DlibEmbedder(model=embedder_model)
    matcher = DlibMatcher(tolerance=tolerance, dimension=128)

    # Note: dlib doesn't need explicit aligner - face_recognition
    # handles face alignment internally during encoding
    aligner = None

    logger.info("dlib backend created successfully")

    return BackendComponents(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        matcher=matcher,
        backend_type="dlib",
        embedding_dim=128,
    )


def get_model_paths(backend_type: BackendType, models_dir: str | Path) -> dict[str, Path]:
    """Get model file paths for a specific backend.

    Args:
        backend_type: Backend type ("insightface" or "dlib")
        models_dir: Base directory for model files

    Returns:
        Dictionary with paths for index, labels, and stats files.

    Example:
        >>> paths = get_model_paths("insightface", "models")
        >>> paths["index"]   # Path("models/centroids.faiss")
        >>> paths["labels"]  # Path("models/labels.json")

        >>> paths = get_model_paths("dlib", "models")
        >>> paths["index"]   # Path("models/encodings.pkl")
        >>> paths["labels"]  # Path("models/labels_dlib.json")
    """
    models_dir = Path(models_dir)

    if backend_type == "insightface":
        return {
            "index": models_dir / "centroids.faiss",
            "labels": models_dir / "labels.json",
            "stats": models_dir / "stats.pkl",
        }
    elif backend_type == "dlib":
        return {
            "index": models_dir / "encodings.pkl",
            "labels": models_dir / "labels_dlib.json",
            "stats": models_dir / "stats_dlib.pkl",
        }
    else:
        raise ValueError(f"Unknown backend: '{backend_type}'")


def load_matcher(
    backend_type: BackendType,
    models_dir: str | Path,
    tolerance: float = 0.6,
) -> object:
    """Load a pre-built matcher from disk.

    Args:
        backend_type: Backend type ("insightface" or "dlib")
        models_dir: Directory containing model files
        tolerance: For dlib matcher - distance threshold

    Returns:
        Loaded matcher instance (FaissMatcher or DlibMatcher)

    Raises:
        FileNotFoundError: If model files don't exist.

    Example:
        >>> matcher = load_matcher("insightface", "models")
        >>> labels, scores = matcher.search(embedding)
    """
    paths = get_model_paths(backend_type, models_dir)

    if backend_type == "insightface":
        from app.backends.insightface.matcher import FaissMatcher

        matcher = FaissMatcher(dimension=512)
        matcher.load(paths["index"], paths["labels"])

    elif backend_type == "dlib":
        from app.backends.dlib.matcher import DlibMatcher

        matcher = DlibMatcher(tolerance=tolerance, dimension=128)
        matcher.load(paths["index"])

    else:
        raise ValueError(f"Unknown backend: '{backend_type}'")

    logger.info(f"Loaded {backend_type} matcher from {models_dir}")
    return matcher