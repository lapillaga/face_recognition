"""5-point face alignment for normalization to 112x112.

This module provides face alignment using 5 facial landmarks (eyes, nose, mouth)
to normalize faces to a standard pose and size for recognition.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


# Standard 5-point landmark positions for 112x112 aligned face
# These are the target positions for: left_eye, right_eye, nose, left_mouth, right_mouth
# Based on ArcFace alignment
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


class FivePointAligner:
    """Face aligner using 5-point landmarks.

    This aligner uses a similarity transformation to warp faces to a standard
    112x112 size with consistent pose, based on 5 facial landmarks.

    Attributes:
        dst_points: Target landmark positions in 112x112 space
        output_size: Output image size (default: 112x112)

    Example:
        >>> aligner = FivePointAligner()
        >>> aligned = aligner.align(frame, detection.kps)
        >>> assert aligned.shape == (112, 112, 3)
    """

    def __init__(
        self,
        dst_points: np.ndarray | None = None,
        output_size: tuple[int, int] = (112, 112),
    ):
        """Initialize 5-point face aligner.

        Args:
            dst_points: Target landmark positions, shape [5, 2].
                       If None, uses ArcFace standard positions.
            output_size: Output image size as (width, height).
                        Default is (112, 112) for ArcFace.
        """
        self.output_size = output_size
        self.dst_points = dst_points if dst_points is not None else ARCFACE_DST.copy()

        logger.debug(
            f"Initialized FivePointAligner with output_size={output_size}, "
            f"dst_points={self.dst_points.shape}"
        )

    def align(self, frame_bgr: np.ndarray, kps_5pt: np.ndarray) -> np.ndarray:
        """Align face to standard orientation and size using 5 landmarks.

        Args:
            frame_bgr: Input image in BGR format, shape [H, W, 3]
            kps_5pt: 5-point facial landmarks in absolute pixel coords, shape [5, 2]
                    Order: left_eye, right_eye, nose, left_mouth, right_mouth

        Returns:
            Aligned face crop in BGR format, shape [112, 112, 3], dtype uint8.

        Raises:
            ValueError: If landmarks are invalid or missing.

        Example:
            >>> aligned = aligner.align(frame, detection.kps)
            >>> cv2.imwrite("aligned_face.jpg", aligned)
        """
        if kps_5pt is None:
            raise ValueError("Landmarks (kps_5pt) cannot be None")

        if kps_5pt.shape != (5, 2):
            raise ValueError(f"Expected kps shape (5, 2), got {kps_5pt.shape}")

        # Ensure float32 type
        src_points = kps_5pt.astype(np.float32)

        # Estimate similarity transform (scale, rotation, translation)
        # Uses 5 point pairs to find optimal transformation
        tform = self._estimate_transform(src_points, self.dst_points)

        # Apply transformation
        aligned = cv2.warpAffine(
            frame_bgr,
            tform,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return aligned

    def _estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> np.ndarray:
        """Estimate similarity transformation matrix.

        Computes a 2x3 affine transformation matrix that maps src_points to dst_points
        using a similarity transform (scale + rotation + translation, no shear/skew).

        Args:
            src_points: Source landmarks, shape [5, 2]
            dst_points: Destination landmarks, shape [5, 2]

        Returns:
            2x3 affine transformation matrix for cv2.warpAffine.

        Note:
            Uses cv2.estimateAffinePartial2D for robust estimation.
            Falls back to cv2.getAffineTransform if fewer points.
        """
        # Use estimateAffinePartial2D for similarity transform
        # This finds optimal scale, rotation, and translation (no shear)
        tform, _ = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.LMEDS,  # Least-Median robust estimator
        )

        if tform is None:
            logger.warning(
                "estimateAffinePartial2D failed, falling back to 3-point affine"
            )
            # Fallback: use first 3 points for affine transform
            tform = cv2.getAffineTransform(src_points[:3], dst_points[:3])

        return tform

    def align_batch(
        self,
        frame_bgr: np.ndarray,
        kps_batch: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Align multiple faces from the same frame.

        Args:
            frame_bgr: Input image in BGR format
            kps_batch: List of 5-point landmark arrays, each shape [5, 2]

        Returns:
            List of aligned face crops, each shape [112, 112, 3].

        Example:
            >>> detections = detector.detect(frame)
            >>> kps_list = [det.kps for det in detections if det.kps is not None]
            >>> aligned_faces = aligner.align_batch(frame, kps_list)
        """
        aligned_faces = []

        for kps in kps_batch:
            try:
                aligned = self.align(frame_bgr, kps)
                aligned_faces.append(aligned)
            except ValueError as e:
                logger.warning(f"Failed to align face: {e}")
                continue

        return aligned_faces

    def __repr__(self) -> str:
        """String representation of aligner."""
        return f"FivePointAligner(output_size={self.output_size})"