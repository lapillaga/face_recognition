"""Dlib face detector using face_recognition library.

This module provides a face detector based on dlib's HOG or CNN models
via the face_recognition library (PyImageSearch tutorial approach).
"""

from __future__ import annotations

from typing import List, Literal

import cv2
import face_recognition
import numpy as np

from app.core.interfaces import BBox, Detection
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DlibDetector:
    """Face detector using dlib via a face_recognition library.

    Supports two detection models:
    - HOG: Faster, suitable for CPU, less accurate
    - CNN: More accurate, requires GPU for real-time performance

    Attributes:
        model: Detection model ("hog" or "cnn")
        upsample: Number of times to upsample image (higher = detect smaller faces)

    Example:
        >>> detector = DlibDetector(model="hog")
        >>> detections = detector.detect(frame)
        >>> print(f"Found {len(detections)} faces")
    """

    def __init__(
        self,
        model: Literal["hog", "cnn"] = "hog",
        upsample: int = 1,
    ):
        """Initialize dlib detector.

        Args:
            model: Detection model to use.
                   "hog" - Histogram of Oriented Gradients (faster, CPU-friendly)
                   "cnn" - Convolutional Neural Network (more accurate, GPU preferred)
            upsample: Number of times to upsample image before detection.
                      Higher values detect smaller faces but are slower.
                      Default: 1

        Example:
            >>> detector = DlibDetector(model="hog", upsample=1)
            >>> detector = DlibDetector(model="cnn", upsample=2)  # For smaller faces
        """
        if model not in ("hog", "cnn"):
            raise ValueError(f"model must be 'hog' or 'cnn', got '{model}'")

        self.model = model
        self.upsample = upsample

        logger.info(
            f"Initializing dlib detector (model={model}, upsample={upsample})"
        )
        logger.info("Dlib detector initialized successfully")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect faces in an image.

        Args:
            frame_bgr: Input image in BGR format (OpenCV convention), shape [H, W, 3]

        Returns:
            List of Detection objects, sorted by area (descending).
            Empty list if no faces are detected.

        Note:
            dlib/face_recognition doesn't provide confidence scores for HOG detector,
            so we use a fixed score of 0.99 for detected faces.
            The CNN model provides scores, but face_recognition doesn't expose them.

        Example:
            >>> detections = detector.detect(frame)
            >>> for i, det in enumerate(detections):
            ...     print(f"Face {i}: bbox={det.bbox}")
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Empty frame provided to detector")
            return []

        try:
            # Convert BGR to RGB (face_recognition expects RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Detect face locations
            # Returns list of tuples: (top, right, bottom, left)
            face_locations = face_recognition.face_locations(
                frame_rgb,
                number_of_times_to_upsample=self.upsample,
                model=self.model,
            )

            # Get image dimensions for clamping
            h, w = frame_bgr.shape[:2]

            # Convert to our Detection format
            detections = []
            for top, right, bottom, left in face_locations:
                # Clamp to image bounds
                x1 = max(0, min(left, w - 1))
                y1 = max(0, min(top, h - 1))
                x2 = max(0, min(right, w - 1))
                y2 = max(0, min(bottom, h - 1))

                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

                # dlib HOG doesn't provide landmarks in detection
                # We'll get them separately if needed for alignment
                # For now, return None (embedder_dlib handles this differently)
                kps = None

                # dlib/face_recognition doesn't expose confidence scores
                # Use fixed high score for detected faces
                score = 0.99

                detection = Detection(bbox=bbox, kps=kps, score=score)
                detections.append(detection)

            # Sort by area (largest first) since we don't have confidence scores
            detections.sort(key=lambda d: d.bbox.area, reverse=True)

            if len(detections) > 0:
                logger.debug(f"Detected {len(detections)} faces (model={self.model})")

            return detections

        except Exception as e:
            logger.error(f"Error during face detection: {e}", exc_info=True)
            return []

    def __repr__(self) -> str:
        """String representation of detector."""
        return f"DlibDetector(model='{self.model}', upsample={self.upsample})"