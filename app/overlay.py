"""Drawing utilities for visualizing detection results.

This module provides functions to draw bounding boxes, landmarks, and labels
on video frames for visualization purposes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.interfaces import BBox, Detection


# Color palette (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 2,
) -> None:
    """Draw bounding box on frame (in-place).

    Args:
        frame: Image to draw on (modified in-place)
        bbox: Bounding box to draw
        color: BGR color tuple (default: green)
        thickness: Line thickness in pixels

    Example:
        >>> draw_bbox(frame, detection.bbox, color=(0, 255, 0))
    """
    cv2.rectangle(
        frame,
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y2),
        color,
        thickness,
    )


def draw_landmarks(
    frame: np.ndarray,
    kps: np.ndarray,
    color: Tuple[int, int, int] = COLOR_RED,
    radius: int = 3,
) -> None:
    """Draw 5-point facial landmarks on frame (in-place).

    Args:
        frame: Image to draw on (modified in-place)
        kps: Keypoints array of shape (5, 2) with (x, y) coordinates
             Order: left_eye, right_eye, nose, left_mouth, right_mouth
        color: BGR color tuple (default: red)
        radius: Circle radius in pixels

    Example:
        >>> if detection.kps is not None:
        ...     draw_landmarks(frame, detection.kps)
    """
    if kps is None:
        return

    # Draw each landmark as a filled circle
    for i, (x, y) in enumerate(kps):
        cv2.circle(
            frame,
            (int(x), int(y)),
            radius,
            color,
            -1,  # -1 = filled circle
        )


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = COLOR_WHITE,
    font_scale: float = 0.6,
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw text with optional background on frame (in-place).

    Args:
        frame: Image to draw on (modified in-place)
        text: Text string to draw
        position: (x, y) position for bottom-left corner of text
        color: Text color in BGR (default: white)
        font_scale: Font size scale factor
        thickness: Text thickness in pixels
        bg_color: Optional background color for text box (BGR)

    Example:
        >>> draw_text(frame, "Face 0.95", (x1, y1 - 10), bg_color=(0, 0, 0))
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    # Draw background box if requested
    if bg_color is not None:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline),
            (x + text_width, y + baseline),
            bg_color,
            -1,  # Filled rectangle
        )

    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,  # Anti-aliased
    )


def draw_label(
    frame: np.ndarray,
    bbox: BBox,
    text: str,
    bg_color: Tuple[int, int, int] = COLOR_GREEN,
    text_color: Tuple[int, int, int] = COLOR_WHITE,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Draw label text above bounding box (in-place).

    Args:
        frame: Image to draw on (modified in-place)
        bbox: Bounding box to draw label above
        text: Label text to draw
        bg_color: Background color for label box (BGR)
        text_color: Text color (BGR)
        font_scale: Font size scale factor
        thickness: Text thickness in pixels

    Example:
        >>> draw_label(frame, bbox, "LUIS (0.87)", bg_color=(0, 255, 0))
    """
    # Position label above bbox
    x = bbox.x1
    y = bbox.y1 - 10

    # Ensure label is within frame bounds
    if y < 20:
        y = bbox.y2 + 20

    draw_text(
        frame,
        text,
        (x, y),
        color=text_color,
        font_scale=font_scale,
        thickness=thickness,
        bg_color=bg_color,
    )


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = COLOR_GREEN,
    show_landmarks: bool = True,
    show_score: bool = True,
) -> None:
    """Draw complete detection visualization (bbox + landmarks + label).

    Args:
        frame: Image to draw on (modified in-place)
        detection: Detection object to visualize
        label: Optional label text (e.g., person name or "unknown")
        color: Color for bounding box and label
        show_landmarks: Whether to draw facial landmarks
        show_score: Whether to show confidence score

    Example:
        >>> for detection in detections:
        ...     draw_detection(frame, detection, label="Alice", color=(0, 255, 0))
    """
    # Draw bounding box
    draw_bbox(frame, detection.bbox, color=color)

    # Draw landmarks if available
    if show_landmarks and detection.kps is not None:
        draw_landmarks(frame, detection.kps, color=COLOR_RED)

    # Draw label with score
    if label is not None or show_score:
        text_parts = []
        if label is not None:
            text_parts.append(label)
        if show_score:
            text_parts.append(f"{detection.score:.2f}")

        text = " ".join(text_parts)

        # Position text above bounding box
        text_x = detection.bbox.x1
        text_y = max(detection.bbox.y1 - 10, 20)  # Keep text on screen

        draw_text(
            frame,
            text,
            (text_x, text_y),
            color=color,
            bg_color=COLOR_BLACK,
        )


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = COLOR_GREEN,
    show_landmarks: bool = True,
    show_score: bool = True,
) -> None:
    """Draw multiple detections on frame (in-place).

    Args:
        frame: Image to draw on (modified in-place)
        detections: List of Detection objects
        labels: Optional list of labels (same length as detections)
        color: Default color for all detections
        show_landmarks: Whether to draw facial landmarks
        show_score: Whether to show confidence scores

    Example:
        >>> detections = detector.detect(frame)
        >>> draw_detections(frame, detections, show_landmarks=True)
    """
    for i, detection in enumerate(detections):
        label = labels[i] if labels and i < len(labels) else None
        draw_detection(
            frame,
            detection,
            label=label,
            color=color,
            show_landmarks=show_landmarks,
            show_score=show_score,
        )


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = COLOR_YELLOW,
) -> None:
    """Draw FPS counter on frame (in-place).

    Args:
        frame: Image to draw on (modified in-place)
        fps: Frames per second value
        position: (x, y) position for text
        color: Text color in BGR (default: yellow)

    Example:
        >>> draw_fps(frame, 30.5)
    """
    text = f"FPS: {fps:.1f}"
    draw_text(
        frame,
        text,
        position,
        color=color,
        font_scale=0.7,
        thickness=2,
        bg_color=COLOR_BLACK,
    )


def draw_info_panel(
    frame: np.ndarray,
    info: dict[str, str | int | float],
    position: Tuple[int, int] = (10, 60),
    color: Tuple[int, int, int] = COLOR_WHITE,
) -> None:
    """Draw information panel with multiple key-value pairs.

    Args:
        frame: Image to draw on (modified in-place)
        info: Dictionary of key-value pairs to display
        position: (x, y) starting position
        color: Text color in BGR

    Example:
        >>> info = {"Faces": 2, "Model": "SCRFD", "Device": "CPU"}
        >>> draw_info_panel(frame, info)
    """
    x, y = position
    line_height = 25

    for i, (key, value) in enumerate(info.items()):
        text = f"{key}: {value}"
        draw_text(
            frame,
            text,
            (x, y + i * line_height),
            color=color,
            font_scale=0.5,
            thickness=1,
            bg_color=COLOR_BLACK,
        )