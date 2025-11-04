"""Configuration management for face recognition pipeline.

This module loads configuration from environment variables (.env file) and
provides a centralized Config class for accessing application settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables.

    Attributes:
        ctx_id: Device context ID (-1 for CPU, 0+ for GPU)
        thresh: Cosine similarity threshold for face matching (0.0-1.0)
        camera_id: Camera device ID for video capture
        display: Whether to show preview window (1) or run headless (0)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        model_pack: InsightFace model pack name
        num_enrollment_images: Number of face crops to capture per person
        min_sharpness: Minimum Laplacian variance for sharpness check
        max_fps: FPS cap for video processing (0 = no limit)
        frame_skip: Skip N frames for faster processing
    """

    ctx_id: int
    thresh: float
    camera_id: int
    display: bool
    log_level: str
    model_pack: str
    num_enrollment_images: int
    min_sharpness: float
    max_fps: int
    frame_skip: int

    # Paths
    data_dir: Path
    enroll_dir: Path
    val_dir: Path
    models_dir: Path

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.

        Returns:
            Config instance with values from environment or defaults.

        Raises:
            ValueError: If required environment variables are invalid.
        """
        # Get project root (parent of app/)
        project_root = Path(__file__).parent.parent

        # Device configuration
        ctx_id = int(os.getenv("CTX_ID", "-1"))

        # Recognition threshold
        thresh = float(os.getenv("THRESH", "0.35"))
        if not 0.0 <= thresh <= 1.0:
            raise ValueError(f"THRESH must be between 0.0 and 1.0, got {thresh}")

        # Camera configuration
        camera_id = int(os.getenv("CAMERA_ID", "0"))
        if camera_id < 0:
            raise ValueError(f"CAMERA_ID must be >= 0, got {camera_id}")

        # Display configuration
        display = bool(int(os.getenv("DISPLAY", "1")))

        # Logging configuration
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}, got {log_level}")

        # Model configuration
        model_pack = os.getenv("MODEL_PACK", "buffalo_l")
        valid_packs = ["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"]
        if model_pack not in valid_packs:
            raise ValueError(f"MODEL_PACK must be one of {valid_packs}, got {model_pack}")

        # Enrollment configuration
        num_enrollment_images = int(os.getenv("NUM_ENROLLMENT_IMAGES", "15"))
        if num_enrollment_images < 1:
            raise ValueError(
                f"NUM_ENROLLMENT_IMAGES must be >= 1, got {num_enrollment_images}"
            )

        min_sharpness = float(os.getenv("MIN_SHARPNESS", "150"))
        if min_sharpness < 0:
            raise ValueError(f"MIN_SHARPNESS must be >= 0, got {min_sharpness}")

        # Video processing
        max_fps = int(os.getenv("MAX_FPS", "30"))
        if max_fps < 0:
            raise ValueError(f"MAX_FPS must be >= 0, got {max_fps}")

        frame_skip = int(os.getenv("FRAME_SKIP", "1"))
        if frame_skip < 1:
            raise ValueError(f"FRAME_SKIP must be >= 1, got {frame_skip}")

        # Paths
        data_dir = project_root / "data"
        enroll_dir = data_dir / "enroll"
        val_dir = data_dir / "val"
        models_dir = project_root / "models"

        # Ensure directories exist
        enroll_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            ctx_id=ctx_id,
            thresh=thresh,
            camera_id=camera_id,
            display=display,
            log_level=log_level,
            model_pack=model_pack,
            num_enrollment_images=num_enrollment_images,
            min_sharpness=min_sharpness,
            max_fps=max_fps,
            frame_skip=frame_skip,
            data_dir=data_dir,
            enroll_dir=enroll_dir,
            val_dir=val_dir,
            models_dir=models_dir,
        )

    def __repr__(self) -> str:
        """Return string representation of config."""
        return (
            f"Config(\n"
            f"  Device: {'GPU' if self.ctx_id >= 0 else 'CPU'}:{self.ctx_id},\n"
            f"  Threshold: {self.thresh},\n"
            f"  Camera: {self.camera_id},\n"
            f"  Display: {self.display},\n"
            f"  Log Level: {self.log_level},\n"
            f"  Model: {self.model_pack},\n"
            f"  Enrollment Images: {self.num_enrollment_images},\n"
            f"  Min Sharpness: {self.min_sharpness}\n"
            f")"
        )


# Global config instance (lazy-loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get global config instance (singleton pattern).

    Returns:
        Config instance loaded from environment.
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config