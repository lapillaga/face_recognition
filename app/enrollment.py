"""Enrollment service for registering new persons.

This module provides functionality to capture face images from video sources,
apply quality filtering, align faces, and persist enrollment data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.interfaces import Aligner, Detector, VideoSource
from app.logging_config import get_logger
from app.utils import compute_sharpness

logger = get_logger(__name__)


class EnrollmentService:
    """Service for capturing and enrolling new persons.

    This service coordinates detection, alignment, and quality filtering
    to build a clean dataset of face images for each enrolled person.

    Workflow:
    1. Read frames from video source
    2. Detect faces in frame
    3. Align detected faces to 112x112
    4. Apply quality checks (sharpness, etc.)
    5. Save high-quality aligned crops

    Attributes:
        detector: Face detector
        aligner: Face aligner
        min_sharpness: Minimum Laplacian variance threshold
        save_dir: Root directory for saving enrollment images

    Example:
        >>> service = EnrollmentService(
        ...     detector=detector,
        ...     aligner=aligner,
        ...     save_dir="data/enroll"
        ... )
        >>> service.enroll_from_video(
        ...     source=webcam,
        ...     person_name="LUIS",
        ...     num_images=15
        ... )
    """

    def __init__(
        self,
        detector: Detector,
        aligner: Aligner,
        save_dir: str | Path = "data/enroll",
        min_sharpness: float = 100.0,
        min_detection_score: float = 0.5,
    ):
        """Initialize enrollment service.

        Args:
            detector: Face detector instance
            aligner: Face aligner instance
            save_dir: Root directory for saving enrollment images
            min_sharpness: Minimum Laplacian variance for quality filtering
            min_detection_score: Minimum detection confidence score
        """
        self.detector = detector
        self.aligner = aligner
        self.save_dir = Path(save_dir)
        self.min_sharpness = min_sharpness
        self.min_detection_score = min_detection_score

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized EnrollmentService: save_dir={self.save_dir}, "
            f"min_sharpness={min_sharpness}, min_detection_score={min_detection_score}"
        )

    def _get_person_dir(self, person_name: str) -> Path:
        """Get directory path for a person's enrollment images.

        Args:
            person_name: Person's name/label

        Returns:
            Path to person's directory (will be created if it doesn't exist)
        """
        person_dir = self.save_dir / person_name.upper()
        person_dir.mkdir(parents=True, exist_ok=True)
        return person_dir

    def _save_aligned_face(
        self,
        aligned_face: np.ndarray,
        person_name: str,
        index: int,
    ) -> Path:
        """Save aligned face crop to disk.

        Args:
            aligned_face: Aligned face image [112, 112, 3] BGR
            person_name: Person's name/label
            index: Image index (for filename)

        Returns:
            Path where image was saved.
        """
        person_dir = self._get_person_dir(person_name)
        filename = f"{index:04d}.jpg"
        save_path = person_dir / filename

        # Save with high quality JPEG (95%)
        cv2.imwrite(str(save_path), aligned_face, [cv2.IMWRITE_JPEG_QUALITY, 95])

        logger.debug(f"Saved aligned face to {save_path}")
        return save_path

    def process_frame(
        self,
        frame: np.ndarray,
        person_name: str,
        save_index: int,
    ) -> tuple[bool, Optional[np.ndarray], str]:
        """Process a single frame for enrollment.

        Args:
            frame: Input frame in BGR format [H, W, 3]
            person_name: Person's name/label
            save_index: Current save index

        Returns:
            Tuple of (success, aligned_face, message):
                - success: True if face was captured successfully
                - aligned_face: Aligned face [112, 112, 3] if success, None otherwise
                - message: Status message explaining the result

        Example:
            >>> success, face, msg = service.process_frame(frame, "LUIS", 0)
            >>> if success:
            ...     print(f"Captured face: {msg}")
            ... else:
            ...     print(f"Skipped: {msg}")
        """
        # Step 1: Detect faces
        detections = self.detector.detect(frame)

        if len(detections) == 0:
            return False, None, "No faces detected"

        if len(detections) > 1:
            return False, None, f"Multiple faces detected ({len(detections)})"

        detection = detections[0]

        # Step 2: Check detection confidence
        if detection.score < self.min_detection_score:
            return (
                False,
                None,
                f"Low detection confidence ({detection.score:.2f})",
            )

        # Step 3: Check if landmarks are available
        if detection.kps is None:
            return False, None, "No landmarks detected"

        # Step 4: Align face
        try:
            aligned_face = self.aligner.align(frame, detection.kps)
        except Exception as e:
            logger.warning(f"Alignment failed: {e}")
            return False, None, f"Alignment failed: {e}"

        # Step 5: Quality check - sharpness
        sharpness = compute_sharpness(aligned_face)

        if sharpness < self.min_sharpness:
            return (
                False,
                None,
                f"Low sharpness ({sharpness:.1f} < {self.min_sharpness})",
            )

        # Step 6: Save aligned face
        save_path = self._save_aligned_face(aligned_face, person_name, save_index)

        logger.info(
            f"Captured face {save_index}: "
            f"score={detection.score:.2f}, sharpness={sharpness:.1f}"
        )

        return True, aligned_face, f"Saved to {save_path.name}"

    def enroll_from_video(
        self,
        source: VideoSource,
        person_name: str,
        num_images: int = 15,
        display: bool = True,
        display_window: str = "Enrollment",
    ) -> int:
        """Enroll a person by capturing images from video source.

        This method reads frames from the video source, detects faces,
        applies quality filtering, and saves aligned crops.

        Args:
            source: Video source (webcam or video file)
            person_name: Person's name/label
            num_images: Target number of images to capture
            display: Whether to show preview window
            display_window: Window name for display

        Returns:
            Number of images successfully captured.

        Example:
            >>> from app.video_io import WebcamSource
            >>> source = WebcamSource(camera_id=0)
            >>> service = EnrollmentService(detector, aligner)
            >>> num_captured = service.enroll_from_video(
            ...     source=source,
            ...     person_name="LUIS",
            ...     num_images=15
            ... )
            >>> print(f"Captured {num_captured} images")
        """
        logger.info(
            f"Starting enrollment for '{person_name}' "
            f"(target: {num_images} images)"
        )

        person_name = person_name.upper()
        captured_count = 0
        frame_count = 0

        try:
            while captured_count < num_images:
                # Read frame
                success, frame = source.read()

                if not success:
                    logger.warning("Failed to read frame from source")
                    break

                frame_count += 1

                # Process frame
                success, aligned_face, message = self.process_frame(
                    frame, person_name, captured_count
                )

                # Display frame
                if display:
                    display_frame = frame.copy()

                    # Draw status text
                    status_text = (
                        f"Captured: {captured_count}/{num_images} - {message}"
                    )
                    cv2.putText(
                        display_frame,
                        status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if success else (0, 0, 255),
                        2,
                    )

                    # Draw guide box (center region where face should be)
                    h, w = display_frame.shape[:2]
                    guide_w, guide_h = w // 3, h // 2
                    guide_x1 = (w - guide_w) // 2
                    guide_y1 = (h - guide_h) // 2
                    guide_x2 = guide_x1 + guide_w
                    guide_y2 = guide_y1 + guide_h

                    cv2.rectangle(
                        display_frame,
                        (guide_x1, guide_y1),
                        (guide_x2, guide_y2),
                        (255, 255, 0),
                        2,
                    )

                    # Draw detections if any
                    detections = self.detector.detect(frame)
                    for det in detections:
                        bbox = det.bbox
                        color = (0, 255, 0) if len(detections) == 1 else (0, 0, 255)
                        cv2.rectangle(
                            display_frame,
                            (bbox.x1, bbox.y1),
                            (bbox.x2, bbox.y2),
                            color,
                            2,
                        )

                    cv2.imshow(display_window, display_frame)

                    # Check for ESC key to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        logger.info("User pressed ESC, stopping enrollment")
                        break

                # Increment counter if successful
                if success:
                    captured_count += 1

        finally:
            if display:
                cv2.destroyWindow(display_window)

        logger.info(
            f"Enrollment complete for '{person_name}': "
            f"{captured_count}/{num_images} images captured "
            f"({frame_count} frames processed)"
        )

        return captured_count

    def count_enrolled_images(self, person_name: str) -> int:
        """Count number of enrolled images for a person.

        Args:
            person_name: Person's name/label

        Returns:
            Number of images in person's enrollment directory.
        """
        person_dir = self._get_person_dir(person_name)

        # Count .jpg files
        images = list(person_dir.glob("*.jpg"))
        count = len(images)

        logger.debug(f"Person '{person_name}' has {count} enrolled images")
        return count

    def list_enrolled_persons(self) -> list[str]:
        """List all enrolled persons.

        Returns:
            List of person names (directory names in save_dir).
        """
        if not self.save_dir.exists():
            return []

        persons = [
            d.name
            for d in self.save_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        logger.debug(f"Found {len(persons)} enrolled persons: {persons}")
        return sorted(persons)

    def delete_person(self, person_name: str) -> bool:
        """Delete all enrollment data for a person.

        Args:
            person_name: Person's name/label

        Returns:
            True if person was deleted, False if person not found.
        """
        # Don't use _get_person_dir as it creates the directory
        person_dir = self.save_dir / person_name.upper()

        if not person_dir.exists():
            logger.warning(f"Person '{person_name}' not found")
            return False

        # Delete all images
        images = list(person_dir.glob("*.jpg"))
        for img in images:
            img.unlink()

        # Remove directory
        person_dir.rmdir()

        logger.info(f"Deleted person '{person_name}' ({len(images)} images)")
        return True

    def __repr__(self) -> str:
        """String representation."""
        num_persons = len(self.list_enrolled_persons())
        return (
            f"EnrollmentService(save_dir={self.save_dir}, "
            f"enrolled_persons={num_persons})"
        )
