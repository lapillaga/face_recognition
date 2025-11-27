"""Recognition service for face identification in images and video.

This module provides the main recognition service that combines detection,
alignment, embedding, and matching to identify faces in real-time.

Supports two workflows:
1. InsightFace: detect → align (5-point) → embed → match (FAISS)
2. dlib: detect → crop bbox → embed → match (Euclidean distance)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.core.interfaces import Aligner, Detector, Embedder, Matcher, BBox
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RecognitionResult:
    """Result of face recognition for a single face.

    Attributes:
        bbox: Bounding box of detected face
        label: Identified person name or "unknown"
        score: Similarity score (0.0 to 1.0), higher = more confident
        is_known: True if person is identified (score >= threshold)

    Example:
        >>> result = RecognitionResult(
        ...     bbox=BBox(100, 100, 200, 200),
        ...     label="LUIS",
        ...     score=0.87,
        ...     is_known=True
        ... )
        >>> print(f"{result.label}: {result.score:.2f}")
        LUIS: 0.87
    """

    bbox: BBox
    label: str
    score: float
    is_known: bool

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RecognitionResult(bbox={self.bbox}, "
            f"label='{self.label}', score={self.score:.3f}, "
            f"is_known={self.is_known})"
        )


class RecognitionService:
    """Service for real-time face recognition.

    This service orchestrates the full recognition pipeline:
    1. Detect faces in frame
    2. Align each face to 112x112
    3. Extract 512-D embedding
    4. Search in FAISS index
    5. Apply threshold to determine if person is known

    Attributes:
        detector: Face detector
        aligner: Face aligner
        embedder: Embedding extractor
        matcher: FAISS matcher
        threshold: Minimum similarity score to consider a match (0.0 to 1.0)

    Example:
        >>> service = RecognitionService(
        ...     detector=detector,
        ...     aligner=aligner,
        ...     embedder=embedder,
        ...     matcher=matcher,
        ...     threshold=0.35
        ... )
        >>> results = service.recognize(frame)
        >>> for result in results:
        ...     print(f"{result.label}: {result.score:.2f}")
    """

    def __init__(
        self,
        detector: Detector,
        aligner: Optional[Aligner],
        embedder: Embedder,
        matcher: Matcher,
        threshold: float = 0.35,
    ):
        """Initialize recognition service.

        Args:
            detector: Face detector instance
            aligner: Face aligner instance (None for dlib backend)
            embedder: Embedding extractor instance
            matcher: Matcher instance (FAISS or dlib)
            threshold: Similarity threshold for positive identification (0.0 to 1.0)

        Raises:
            ValueError: If threshold is not in valid range.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.matcher = matcher
        self.threshold = threshold

        # Determine workflow based on aligner presence
        self._use_alignment = aligner is not None

        logger.info(
            f"Initialized RecognitionService with threshold={threshold:.2f}, "
            f"workflow={'alignment' if self._use_alignment else 'crop-based'}"
        )

    def recognize(self, frame: np.ndarray) -> List[RecognitionResult]:
        """Recognize all faces in a frame.

        Args:
            frame: Input image in BGR format [H, W, 3]

        Returns:
            List of RecognitionResult objects, one per detected face.
            List may be empty if no faces detected.

        Example:
            >>> results = service.recognize(frame)
            >>> print(f"Found {len(results)} faces")
            >>> for result in results:
            ...     if result.is_known:
            ...         print(f"Identified: {result.label} ({result.score:.2f})")
            ...     else:
            ...         print(f"Unknown face ({result.score:.2f})")
        """
        # Step 1: Detect faces
        detections = self.detector.detect(frame)

        if not detections:
            logger.debug("No faces detected in frame")
            return []

        logger.debug(f"Detected {len(detections)} face(s)")

        results = []

        # Step 2-5: Process each detected face
        for i, detection in enumerate(detections):
            try:
                # Get face image based on workflow
                if self._use_alignment:
                    # InsightFace workflow: use landmarks for alignment
                    if detection.kps is None:
                        logger.warning(
                            f"Detection {i} has no landmarks, skipping"
                        )
                        results.append(
                            RecognitionResult(
                                bbox=detection.bbox,
                                label="unknown",
                                score=0.0,
                                is_known=False,
                            )
                        )
                        continue
                    # Align face using 5-point landmarks
                    face_image = self.aligner.align(frame, detection.kps)
                else:
                    # dlib workflow: crop face region from bbox
                    bbox = detection.bbox
                    h, w = frame.shape[:2]
                    # Clamp bbox to image bounds
                    x1 = max(0, bbox.x1)
                    y1 = max(0, bbox.y1)
                    x2 = min(w, bbox.x2)
                    y2 = min(h, bbox.y2)
                    face_image = frame[y1:y2, x1:x2]

                # Step 3: Extract embedding
                embedding = self.embedder.embed(face_image)

                # Step 4: Search in index
                labels, scores = self.matcher.search(embedding, topk=1)

                # Step 5: Apply threshold
                best_label = labels[0]
                best_score = scores[0]

                if best_score >= self.threshold:
                    # Known person
                    label = best_label
                    is_known = True
                    logger.debug(
                        f"Recognized: {label} (score={best_score:.3f}, "
                        f"threshold={self.threshold:.3f})"
                    )
                else:
                    # Unknown person (below threshold)
                    label = "unknown"
                    is_known = False
                    logger.debug(
                        f"Unknown face (best match: {best_label} with score={best_score:.3f}, "
                        f"below threshold={self.threshold:.3f})"
                    )

                results.append(
                    RecognitionResult(
                        bbox=detection.bbox,
                        label=label,
                        score=best_score,
                        is_known=is_known,
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to process detection {i}: {e}")
                # Add unknown result with bbox
                results.append(
                    RecognitionResult(
                        bbox=detection.bbox,
                        label="unknown",
                        score=0.0,
                        is_known=False,
                    )
                )

        return results

    def recognize_single(
        self, frame: np.ndarray, bbox: BBox, kps: np.ndarray
    ) -> RecognitionResult:
        """Recognize a single face given its detection.

        This is useful when you already have a detection and want to
        skip the detection step.

        Args:
            frame: Input image in BGR format [H, W, 3]
            bbox: Bounding box of the face
            kps: 5-point facial landmarks, shape [5, 2]

        Returns:
            RecognitionResult for the face.

        Example:
            >>> detection = detector.detect(frame)[0]
            >>> result = service.recognize_single(frame, detection.bbox, detection.kps)
            >>> print(f"{result.label}: {result.score:.2f}")
        """
        try:
            # Align face
            aligned = self.aligner.align(frame, kps)

            # Extract embedding
            embedding = self.embedder.embed(aligned)

            # Search in index
            labels, scores = self.matcher.search(embedding, topk=1)

            # Apply threshold
            best_label = labels[0]
            best_score = scores[0]

            if best_score >= self.threshold:
                label = best_label
                is_known = True
            else:
                label = "unknown"
                is_known = False

            return RecognitionResult(
                bbox=bbox,
                label=label,
                score=best_score,
                is_known=is_known,
            )

        except Exception as e:
            logger.warning(f"Failed to recognize face: {e}")
            return RecognitionResult(
                bbox=bbox,
                label="unknown",
                score=0.0,
                is_known=False,
            )

    def set_threshold(self, threshold: float) -> None:
        """Update recognition threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0)

        Raises:
            ValueError: If threshold is not in valid range.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        old_threshold = self.threshold
        self.threshold = threshold

        logger.info(
            f"Updated recognition threshold: {old_threshold:.2f} → {threshold:.2f}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RecognitionService(threshold={self.threshold:.2f}, "
            f"matcher={self.matcher})"
        )
