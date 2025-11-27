"""Video input/output abstraction layer.

This module provides concrete implementations of video sources (webcam, file)
following the VideoSource protocol. It abstracts different video backends
to provide a consistent interface for frame reading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class WebcamSource:
    """Video source for reading from a webcam or USB camera.

    This class wraps OpenCV's VideoCapture for webcam access,
    providing a clean interface for real-time video processing.

    Attributes:
        camera_id: Camera device ID (0 for default camera)
        cap: OpenCV VideoCapture object
        _fps: Cached FPS value

    Example:
        >>> source = WebcamSource(camera_id=0)
        >>> if not source.is_opened:
        ...     raise RuntimeError("Failed to open webcam")
        >>> try:
        ...     while True:
        ...         success, frame = source.read()
        ...         if not success:
        ...             break
        ...         # Process frame...
        ... finally:
        ...     source.release()
    """

    def __init__(self, camera_id: int = 0):
        """Initialize webcam source.

        Args:
            camera_id: Camera device index (default: 0 for first camera)

        Raises:
            RuntimeError: If webcam cannot be opened.
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open webcam with camera_id={camera_id}. "
                f"Check if camera is connected and not in use by another application."
            )

        # Get FPS (may be 0 for some cameras)
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps == 0:
            logger.warning(
                f"Webcam FPS is 0 (unknown). Assuming 30 FPS for camera_id={camera_id}"
            )
            self._fps = 30.0

        # Try to set optimal resolution (720p) if possible
        # This may not work on all cameras, but won't cause errors
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get actual resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Opened webcam {camera_id}: {width}x{height} @ {self._fps:.1f} FPS"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from webcam.

        Returns:
            Tuple of (success, frame):
                - success: True if frame read successfully
                - frame: BGR image [H, W, 3] if success, None otherwise
        """
        if not self.cap.isOpened():
            logger.error("Webcam is not opened")
            return False, None

        success, frame = self.cap.read()

        if not success:
            logger.warning("Failed to read frame from webcam")
            return False, None

        return True, frame

    def release(self) -> None:
        """Release webcam resources."""
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"Released webcam {self.camera_id}")

    @property
    def fps(self) -> float:
        """Get webcam frame rate."""
        return self._fps

    @property
    def is_opened(self) -> bool:
        """Check if webcam is successfully opened."""
        return self.cap.isOpened()

    def __repr__(self) -> str:
        """String representation."""
        status = "opened" if self.is_opened else "closed"
        return f"WebcamSource(camera_id={self.camera_id}, status={status}, fps={self._fps:.1f})"

    def __enter__(self) -> WebcamSource:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit (auto-release)."""
        self.release()


class VideoFileSource:
    """Video source for reading from a video file.

    This class wraps OpenCV's VideoCapture for file playback,
    supporting common formats like MP4, AVI, MOV, etc.

    Attributes:
        video_path: Path to video file
        cap: OpenCV VideoCapture object
        _fps: Video file frame rate
        _frame_count: Total number of frames
        _current_frame: Current frame index

    Example:
        >>> source = VideoFileSource("path/to/video.mp4")
        >>> if not source.is_opened:
        ...     raise RuntimeError("Failed to open video")
        >>> try:
        ...     while True:
        ...         success, frame = source.read()
        ...         if not success:
        ...             break
        ...         # Process frame...
        ...         print(f"Progress: {source.progress:.1%}")
        ... finally:
        ...     source.release()
    """

    def __init__(self, video_path: str | Path):
        """Initialize video file source.

        Args:
            video_path: Path to video file

        Raises:
            FileNotFoundError: If video file does not exist.
            RuntimeError: If video file cannot be opened.
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video file: {self.video_path}. "
                f"File may be corrupted or codec not supported."
            )

        # Get video properties
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self._frame_count / self._fps if self._fps > 0 else 0

        logger.info(
            f"Opened video file: {self.video_path.name} "
            f"({width}x{height}, {self._frame_count} frames, "
            f"{self._fps:.1f} FPS, {duration:.1f}s)"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video file.

        Returns:
            Tuple of (success, frame):
                - success: True if frame read successfully
                - frame: BGR image [H, W, 3] if success, None otherwise

        Note:
            Returns (False, None) when end of video is reached.
        """
        if not self.cap.isOpened():
            logger.error("Video file is not opened")
            return False, None

        success, frame = self.cap.read()

        if success:
            self._current_frame += 1
        else:
            if self._current_frame < self._frame_count:
                logger.warning(
                    f"Failed to read frame {self._current_frame}/{self._frame_count}"
                )
            else:
                logger.info("Reached end of video file")

        return success, frame

    def release(self) -> None:
        """Release video file resources."""
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"Released video file: {self.video_path.name}")

    @property
    def fps(self) -> float:
        """Get video file frame rate."""
        return self._fps

    @property
    def is_opened(self) -> bool:
        """Check if video file is successfully opened."""
        return self.cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Get total number of frames in video."""
        return self._frame_count

    @property
    def current_frame(self) -> int:
        """Get current frame index (0-based)."""
        return self._current_frame

    @property
    def progress(self) -> float:
        """Get playback progress as fraction (0.0 to 1.0)."""
        if self._frame_count == 0:
            return 0.0
        return min(self._current_frame / self._frame_count, 1.0)

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame.

        Args:
            frame_number: Target frame index (0-based)

        Returns:
            True if seek successful, False otherwise.
        """
        if not 0 <= frame_number < self._frame_count:
            logger.warning(
                f"Invalid frame number {frame_number}. "
                f"Must be in range [0, {self._frame_count})"
            )
            return False

        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        if success:
            self._current_frame = frame_number
            logger.debug(f"Seeked to frame {frame_number}")
        else:
            logger.warning(f"Failed to seek to frame {frame_number}")

        return success

    def __repr__(self) -> str:
        """String representation."""
        status = "opened" if self.is_opened else "closed"
        return (
            f"VideoFileSource(path={self.video_path.name}, "
            f"status={status}, "
            f"frames={self._frame_count}, "
            f"fps={self._fps:.1f})"
        )

    def __enter__(self) -> VideoFileSource:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit (auto-release)."""
        self.release()
