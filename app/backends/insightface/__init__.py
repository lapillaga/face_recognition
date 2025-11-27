"""InsightFace backend for face recognition.

Components:
- SCRFDDetector: Face detection using SCRFD model
- FivePointAligner: Face alignment using 5-point landmarks
- ArcFaceEmbedder: 512-D face embeddings using ArcFace
- FaissMatcher: Fast similarity search using FAISS
"""

from app.backends.insightface.detector import SCRFDDetector
from app.backends.insightface.aligner import FivePointAligner
from app.backends.insightface.embedder import ArcFaceEmbedder
from app.backends.insightface.matcher import FaissMatcher

__all__ = [
    "SCRFDDetector",
    "FivePointAligner",
    "ArcFaceEmbedder",
    "FaissMatcher",
]