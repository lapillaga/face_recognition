"""dlib backend for face recognition (PyImageSearch tutorial approach).

Components:
- DlibDetector: Face detection using HOG or CNN via face_recognition
- DlibEmbedder: 128-D face embeddings using ResNet-34
- DlibMatcher: Euclidean distance matching with a voting mechanism
"""

from app.backends.dlib.detector import DlibDetector
from app.backends.dlib.embedder import DlibEmbedder
from app.backends.dlib.matcher import DlibMatcher

__all__ = [
    "DlibDetector",
    "DlibEmbedder",
    "DlibMatcher",
]