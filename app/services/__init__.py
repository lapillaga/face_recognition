"""High-level services for face recognition pipeline.

This package contains the main services that orchestrate
detection, alignment, embedding, and matching.
"""

from app.services.enrollment import EnrollmentService
from app.services.recognition import RecognitionService, RecognitionResult

__all__ = [
    "EnrollmentService",
    "RecognitionService",
    "RecognitionResult",
]