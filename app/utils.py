"""Utility functions for face recognition pipeline.

This module provides helper functions for image quality assessment,
normalization, and other common operations.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


def compute_sharpness(image: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance.

    The Laplacian operator measures the second derivative of the image,
    which highlights regions of rapid intensity change (edges). A higher
    variance indicates more edges and thus a sharper image.

    Args:
        image: Input image in BGR or grayscale, shape [H, W] or [H, W, 3]

    Returns:
        Sharpness score (Laplacian variance). Higher = sharper.
        Typical values: 100-300 for sharp faces, <100 for blurry.

    Example:
        >>> sharpness = compute_sharpness(face_crop)
        >>> if sharpness < 150:
        ...     print("Image too blurry")
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Return variance
    variance = laplacian.var()
    return float(variance)


def is_sharp_enough(image: np.ndarray, threshold: float = 150.0) -> bool:
    """Check if image is sharp enough for face recognition.

    Args:
        image: Input image in BGR or grayscale
        threshold: Minimum sharpness threshold (Laplacian variance)

    Returns:
        True if image is sharp enough, False otherwise.

    Example:
        >>> if is_sharp_enough(face_crop, threshold=150):
        ...     save_crop(face_crop)
    """
    sharpness = compute_sharpness(image)
    return sharpness >= threshold


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to float32 range [0, 1].

    Args:
        image: Input image in uint8 [0, 255]

    Returns:
        Normalized image in float32 [0, 1]

    Example:
        >>> normalized = normalize_image(face_crop)
        >>> assert normalized.dtype == np.float32
        >>> assert 0 <= normalized.max() <= 1
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from float32 [0, 1] to uint8 [0, 255].

    Args:
        image: Normalized image in float32 [0, 1]

    Returns:
        Image in uint8 [0, 255]

    Example:
        >>> denormalized = denormalize_image(normalized_image)
        >>> assert denormalized.dtype == np.uint8
    """
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize image while maintaining aspect ratio.

    The image is resized so that the smaller dimension equals target_size,
    and the larger dimension is scaled proportionally.

    Args:
        image: Input image
        target_size: Target size for the smaller dimension
        interpolation: OpenCV interpolation method

    Returns:
        Resized image maintaining aspect ratio.

    Example:
        >>> resized = resize_with_aspect_ratio(image, target_size=640)
    """
    h, w = image.shape[:2]

    # Determine scaling factor
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized


def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bbox as (x1, y1, x2, y2)
        bbox2: Second bbox as (x1, y1, x2, y2)

    Returns:
        IoU score in range [0, 1]. Higher = more overlap.

    Example:
        >>> iou = compute_iou((10, 10, 50, 50), (30, 30, 70, 70))
        >>> print(f"IoU: {iou:.2f}")
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    iou = intersection / union
    return float(iou)


def crop_with_padding(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Crop region from image with padding if bbox extends beyond boundaries.

    If the crop region extends beyond image boundaries, the output is padded
    with the specified color.

    Args:
        image: Input image in BGR
        x1, y1, x2, y2: Crop coordinates
        padding_color: BGR color for padding

    Returns:
        Cropped image with padding if needed.

    Example:
        >>> crop = crop_with_padding(image, -10, -10, 50, 50)  # Extends beyond image
    """
    h, w = image.shape[:2]

    # Compute padding needed
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Adjust crop coordinates to image bounds
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)
    x2_clipped = min(w, x2)
    y2_clipped = min(h, y2)

    # Crop region
    crop = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    # Add padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=padding_color,
        )

    return crop


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Assumes vectors are already L2-normalized. If not normalized,
    the result is still valid cosine similarity.

    Args:
        vec1: First vector, shape [D]
        vec2: Second vector, shape [D]

    Returns:
        Cosine similarity in range [-1, 1]. Higher = more similar.
        For L2-normalized vectors, this is simply the dot product.

    Example:
        >>> sim = compute_cosine_similarity(embedding1, embedding2)
        >>> if sim > 0.7:
        ...     print("Same person")
    """
    # Compute dot product (assumes normalized vectors)
    dot_product = np.dot(vec1, vec2)

    # Clamp to valid range (numerical stability)
    similarity = np.clip(dot_product, -1.0, 1.0)

    return float(similarity)


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """L2-normalize a vector or batch of vectors.

    Args:
        vec: Vector(s) to normalize, shape [D] or [N, D]
        eps: Small constant to avoid division by zero

    Returns:
        L2-normalized vector(s) with same shape.

    Example:
        >>> normalized = l2_normalize(embedding)
        >>> assert abs(np.linalg.norm(normalized) - 1.0) < 0.01
    """
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return vec / norm