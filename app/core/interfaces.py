"""Core interfaces and data structures for face recognition pipeline.

This module defines the abstract interfaces (Protocols) and data classes
that allow for modular, swappable components in the face recognition system.

Following the Dependency Inversion Principle, components depend on these
abstractions rather than concrete implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class BBox:
    """Bounding box for a detected face.

    Attributes:
        x1: Left edge x-coordinate (pixels)
        y1: Top edge y-coordinate (pixels)
        x2: Right edge x-coordinate (pixels)
        y2: Bottom edge y-coordinate (pixels)
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Get bounding box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get bounding box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Get bounding box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point (x, y) of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def clamp(self, img_width: int, img_height: int) -> BBox:
        """Clamp bounding box coordinates to image boundaries.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            New BBox with clamped coordinates.
        """
        return BBox(
            x1=max(0, min(self.x1, img_width - 1)),
            y1=max(0, min(self.y1, img_height - 1)),
            x2=max(0, min(self.x2, img_width - 1)),
            y2=max(0, min(self.y2, img_height - 1)),
        )

    def __repr__(self) -> str:
        """String representation of bounding box."""
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


@dataclass
class Detection:
    """Face detection result with bounding box, landmarks, and confidence.

    Attributes:
        bbox: Bounding box around detected face
        kps: Optional 5-point facial landmarks (shape: [5, 2]) in absolute pixel coordinates.
             Order: left_eye, right_eye, nose, left_mouth, right_mouth
        score: Detection confidence score (0.0 to 1.0)
    """

    bbox: BBox
    kps: Optional[np.ndarray]  # shape (5, 2) - 5 keypoints with (x, y) coords
    score: float

    def __post_init__(self) -> None:
        """Validate detection data after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Detection score must be in [0, 1], got {self.score}")

        if self.kps is not None:
            if not isinstance(self.kps, np.ndarray):
                raise TypeError(f"kps must be numpy array, got {type(self.kps)}")
            if self.kps.shape != (5, 2):
                raise ValueError(f"kps must have shape (5, 2), got {self.kps.shape}")

    def __repr__(self) -> str:
        """String representation of detection."""
        kps_str = "None" if self.kps is None else f"array{self.kps.shape}"
        return f"Detection(bbox={self.bbox}, score={self.score:.3f}, kps={kps_str})"


@runtime_checkable
class Detector(Protocol):
    """Protocol for face detection models.

    A Detector takes an image and returns a list of detected faces with
    bounding boxes, landmarks, and confidence scores.
    """

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect faces in an image.

        Args:
            frame_bgr: Input image in BGR format (OpenCV convention), shape [H, W, 3]

        Returns:
            List of Detection objects, one per detected face.
            List may be empty if no faces detected.

        Example:
            >>> detector = SCRFDDetector()
            >>> detections = detector.detect(frame)
            >>> for det in detections:
            ...     print(f"Face at {det.bbox} with confidence {det.score}")
        """
        ...


@runtime_checkable
class Aligner(Protocol):
    """Protocol for face alignment/normalization.

    An Aligner takes an image and facial landmarks, then returns a normalized
    face crop (typically 112x112) with consistent orientation.
    """

    def align(self, frame_bgr: np.ndarray, kps_5pt: np.ndarray) -> np.ndarray:
        """Align face to standard orientation and size.

        Args:
            frame_bgr: Input image in BGR format, shape [H, W, 3]
            kps_5pt: 5-point facial landmarks in absolute pixel coords, shape [5, 2]

        Returns:
            Aligned face crop in BGR format, shape [112, 112, 3], dtype uint8.

        Raises:
            ValueError: If landmarks are invalid or missing.

        Example:
            >>> aligner = FivePointAligner()
            >>> aligned = aligner.align(frame, detection.kps)
            >>> assert aligned.shape == (112, 112, 3)
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for face embedding extraction.

    An Embedder takes an aligned face crop and returns a fixed-dimensional
    feature vector (embedding) suitable for face recognition.
    """

    def embed(self, face_bgr_112: np.ndarray) -> np.ndarray:
        """Extract face embedding from aligned face crop.

        Args:
            face_bgr_112: Aligned face crop in BGR format, shape [112, 112, 3]

        Returns:
            L2-normalized embedding vector, shape [512], dtype float32.
            The vector norm should be approximately 1.0.

        Example:
            >>> embedder = ArcFaceEmbedder()
            >>> embedding = embedder.embed(aligned_face)
            >>> assert embedding.shape == (512,)
            >>> assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01
        """
        ...


@runtime_checkable
class Matcher(Protocol):
    """Protocol for face matching/identification using vector search.

    A Matcher maintains an index of known identities and performs
    nearest-neighbor search to identify faces.
    """

    def add(self, label: str, embeddings: np.ndarray) -> None:
        """Add embeddings for a known identity.

        Args:
            label: Identity name/label (e.g., "Alice", "Bob")
            embeddings: One or more embeddings for this identity, shape [N, 512]

        Example:
            >>> matcher = FaissMatcher()
            >>> matcher.add("Alice", alice_embeddings)  # alice_embeddings: [15, 512]
            >>> matcher.add("Bob", bob_embeddings)      # bob_embeddings: [12, 512]
        """
        ...

    def build(self) -> None:
        """Build the search index from added embeddings.

        Must be called after all identities are added via add() and
        before performing any searches.

        Example:
            >>> matcher.add("Alice", embeddings_a)
            >>> matcher.add("Bob", embeddings_b)
            >>> matcher.build()  # Construct FAISS index
        """
        ...

    def search(self, embedding: np.ndarray, topk: int = 1) -> Tuple[List[str], List[float]]:
        """Search for most similar identities.

        Args:
            embedding: Query embedding, shape [512]
            topk: Number of top matches to return

        Returns:
            Tuple of (labels, scores):
                - labels: List of matched identity names, length topk
                - scores: List of similarity scores (cosine similarity), length topk
                         Higher scores = more similar (range 0.0 to 1.0)

        Example:
            >>> labels, scores = matcher.search(query_embedding, topk=3)
            >>> print(f"Top match: {labels[0]} with score {scores[0]:.3f}")
            Top match: Alice with score 0.892
        """
        ...


@runtime_checkable
class VideoSource(Protocol):
    """Protocol for video sources (webcam, file, RTSP stream).

    Provides a consistent interface for reading video frames from
    different sources.
    """

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video source.

        Returns:
            Tuple of (success, frame):
                - success: True if frame read successfully, False otherwise
                - frame: BGR image array [H, W, 3] if success, None otherwise

        Example:
            >>> source = WebcamSource(camera_id=0)
            >>> success, frame = source.read()
            >>> if success:
            ...     process_frame(frame)
        """
        ...

    def release(self) -> None:
        """Release video source resources.

        Should be called when done with the video source to free resources.

        Example:
            >>> source = WebcamSource(camera_id=0)
            >>> try:
            ...     while True:
            ...         success, frame = source.read()
            ...         if not success:
            ...             break
            ... finally:
            ...     source.release()
        """
        ...

    @property
    def fps(self) -> float:
        """Get video source frame rate.

        Returns:
            Frames per second (may be 0.0 if unknown).
        """
        ...

    @property
    def is_opened(self) -> bool:
        """Check if video source is successfully opened.

        Returns:
            True if source is open and ready to read frames.
        """
        ...