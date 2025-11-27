"""Core modules for face recognition pipeline.

This package contains common utilities and interfaces used by all backends.
"""

from app.core.config import Config, get_config
from app.core.interfaces import BBox, Detection, Detector, Aligner, Embedder, Matcher, VideoSource
from app.core.logging_config import setup_logging, get_logger
from app.core.utils import (
    compute_sharpness,
    is_sharp_enough,
    normalize_image,
    denormalize_image,
    l2_normalize,
    resize_with_aspect_ratio,
    crop_with_padding,
    compute_iou,
    compute_cosine_similarity,
)
from app.core.video_io import WebcamSource, VideoFileSource
from app.core.overlay import (
    draw_bbox,
    draw_landmarks,
    draw_text,
    draw_label,
    draw_detection,
    draw_detections,
    draw_fps,
    draw_info_panel,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    # Interfaces
    "BBox",
    "Detection",
    "Detector",
    "Aligner",
    "Embedder",
    "Matcher",
    "VideoSource",
    # Logging
    "setup_logging",
    "get_logger",
    # Utils
    "compute_sharpness",
    "is_sharp_enough",
    "normalize_image",
    "denormalize_image",
    "l2_normalize",
    "resize_with_aspect_ratio",
    "crop_with_padding",
    "compute_iou",
    "compute_cosine_similarity",
    # Video
    "WebcamSource",
    "VideoFileSource",
    # Overlay
    "draw_bbox",
    "draw_landmarks",
    "draw_text",
    "draw_label",
    "draw_detection",
    "draw_detections",
    "draw_fps",
    "draw_info_panel",
]
