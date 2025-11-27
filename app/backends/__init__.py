"""Backend implementations for face recognition.

This package contains different backend implementations:
- insightface: SCRFD detector + ArcFace embeddings (512-D) + FAISS
- dlib: HOG/CNN detector + ResNet-34 embeddings (128-D) + Euclidean

Use the factory module to create backend components.
"""

from app.backends.factory import (
    create_backend,
    load_matcher,
    get_model_paths,
    BackendComponents,
    BackendType,
)

__all__ = [
    "create_backend",
    "load_matcher",
    "get_model_paths",
    "BackendComponents",
    "BackendType",
]