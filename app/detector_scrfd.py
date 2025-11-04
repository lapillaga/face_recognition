"""SCRFD face detector using InsightFace.

This module provides a face detector based on SCRFD (Sample and Computation
Redistribution for Efficient Face Detection) via InsightFace's FaceAnalysis API.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import Config
from app.interfaces import BBox, Detection
from app.logging_config import get_logger

logger = get_logger(__name__)


class SCRFDDetector:
    """Face detector using InsightFace SCRFD model.

    SCRFD is a state-of-the-art face detection model that provides both
    bounding boxes and 5-point facial landmarks (eyes, nose, mouth corners).

    Attributes:
        app: InsightFace FaceAnalysis instance
        ctx_id: Device context (-1=CPU, 0+=GPU)
        det_size: Detection input size (default: (640, 640))

    Example:
        >>> from app.config import get_config
        >>> config = get_config()
        >>> detector = SCRFDDetector(config)
        >>> detections = detector.detect(frame)
        >>> print(f"Found {len(detections)} faces")
    """

    def __init__(self, config: Config, det_size: tuple[int, int] = (640, 640)):
        """Initialize SCRFD detector.

        Args:
            config: Configuration object with ctx_id and model_pack
            det_size: Detection input size as (width, height).
                     Larger sizes = better accuracy but slower.
                     Common values: (320, 320), (640, 640)

        Raises:
            RuntimeError: If model fails to load.
        """
        self.ctx_id = config.ctx_id
        self.det_size = det_size

        logger.info(
            f"Initializing SCRFD detector (model={config.model_pack}, "
            f"device={'GPU:' + str(config.ctx_id) if config.ctx_id >= 0 else 'CPU'}, "
            f"det_size={det_size})"
        )

        try:
            # Initialize InsightFace FaceAnalysis
            # allowed_modules=['detection'] means only load detector, not recognition
            self.app = FaceAnalysis(
                name=config.model_pack,
                allowed_modules=["detection"],
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if config.ctx_id >= 0
                    else ["CPUExecutionProvider"]
                ),
            )

            # Prepare model with detection size
            self.app.prepare(ctx_id=config.ctx_id, det_size=det_size)

            logger.info("SCRFD detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SCRFD detector: {e}", exc_info=True)
            raise RuntimeError(f"Could not load SCRFD detector: {e}") from e

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect faces in an image.

        Args:
            frame_bgr: Input image in BGR format (OpenCV convention), shape [H, W, 3]

        Returns:
            List of Detection objects, sorted by confidence (descending).
            Empty list if no faces detected.

        Example:
            >>> detections = detector.detect(frame)
            >>> for i, det in enumerate(detections):
            ...     print(f"Face {i}: bbox={det.bbox}, score={det.score:.3f}")
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Empty frame provided to detector")
            return []

        try:
            # Get detections from InsightFace
            # Returns list of Face objects with bbox, kps, det_score
            faces = self.app.get(frame_bgr)

            # Convert to our Detection format
            detections = []
            for face in faces:
                # Extract bounding box [x1, y1, x2, y2]
                bbox_arr = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox_arr

                # Clamp to image bounds
                h, w = frame_bgr.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

                # Extract 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
                # Shape: (5, 2) with (x, y) coordinates
                kps = face.kps.astype(np.float32) if hasattr(face, "kps") else None

                # Detection score
                score = float(face.det_score)

                detection = Detection(bbox=bbox, kps=kps, score=score)
                detections.append(detection)

            # Sort by confidence (highest first)
            detections.sort(key=lambda d: d.score, reverse=True)

            if len(detections) > 0:
                logger.debug(f"Detected {len(detections)} faces")

            return detections

        except Exception as e:
            logger.error(f"Error during face detection: {e}", exc_info=True)
            return []

    def __repr__(self) -> str:
        """String representation of detector."""
        return (
            f"SCRFDDetector(ctx_id={self.ctx_id}, "
            f"det_size={self.det_size})"
        )