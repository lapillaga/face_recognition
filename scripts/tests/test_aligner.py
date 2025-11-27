#!/usr/bin/env python3
"""Test script for face aligner with webcam.

This script detects faces, aligns them to 112x112, checks quality,
and displays/saves aligned crops.

Usage:
    python scripts/test_aligner.py

Controls:
    ESC or 'q' - Quit
    SPACE - Pause/Resume
    's' - Save current aligned faces
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backends.insightface.aligner import FivePointAligner
from app.core.config import get_config
from app.backends.insightface.detector import SCRFDDetector
from app.core.logging_config import setup_logging
from app.core.overlay import draw_detections, draw_fps, draw_info_panel, draw_text
from app.core.utils import compute_sharpness, is_sharp_enough

logger = setup_logging(__name__)


def create_aligned_grid(aligned_faces: list[np.ndarray], grid_size: int = 3) -> np.ndarray:
    """Create grid display of aligned faces.

    Args:
        aligned_faces: List of aligned face crops (112x112)
        grid_size: Number of faces per row

    Returns:
        Grid image showing aligned faces.
    """
    if not aligned_faces:
        # Return blank grid
        return np.zeros((112 * 2, 112 * grid_size, 3), dtype=np.uint8)

    # Pad to fill grid
    n_faces = len(aligned_faces)
    n_rows = (n_faces + grid_size - 1) // grid_size

    # Create grid
    grid = np.zeros((112 * n_rows, 112 * grid_size, 3), dtype=np.uint8)

    for i, face in enumerate(aligned_faces):
        row = i // grid_size
        col = i % grid_size

        y1 = row * 112
        y2 = y1 + 112
        x1 = col * 112
        x2 = x1 + 112

        grid[y1:y2, x1:x2] = face

    return grid


def main() -> None:
    """Main function to test aligner with webcam."""
    logger.info("Starting aligner test with webcam...")

    # Load configuration
    config = get_config()

    # Create output directory
    output_dir = Path("data/test_crops")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Aligned faces will be saved to {output_dir}")

    # Initialize detector and aligner
    logger.info("Initializing detector and aligner...")
    try:
        detector = SCRFDDetector(config)
        aligner = FivePointAligner()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return

    # Open webcam
    logger.info(f"Opening camera {config.camera_id}...")
    cap = cv2.VideoCapture(config.camera_id)

    if not cap.isOpened():
        logger.error(f"Failed to open camera {config.camera_id}")
        return

    # Get camera properties
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS))

    logger.info(f"Camera opened: {cam_width}x{cam_height} @ {cam_fps}fps")

    # FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    paused = False

    # Stats
    total_aligned = 0
    total_sharp = 0

    logger.info(
        "Press ESC or 'q' to quit, SPACE to pause/resume, 's' to save aligned faces"
    )

    try:
        while True:
            if not paused:
                # Read frame
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                # Detect faces
                detections = detector.detect(frame)

                # Align faces
                aligned_faces = []
                sharpness_scores = []

                for det in detections:
                    if det.kps is None:
                        continue

                    try:
                        # Align face
                        aligned = aligner.align(frame, det.kps)
                        aligned_faces.append(aligned)

                        # Compute sharpness
                        sharpness = compute_sharpness(aligned)
                        sharpness_scores.append(sharpness)

                        # Check if sharp enough
                        is_sharp = is_sharp_enough(aligned, threshold=config.min_sharpness)

                        total_aligned += 1
                        if is_sharp:
                            total_sharp += 1

                    except Exception as e:
                        logger.warning(f"Failed to align face: {e}")
                        continue

                # Draw detections on original frame
                draw_detections(
                    frame,
                    detections,
                    show_landmarks=True,
                    show_score=True,
                )

                # Add sharpness info to each detection
                for i, det in enumerate(detections[:len(sharpness_scores)]):
                    if i < len(sharpness_scores):
                        sharpness = sharpness_scores[i]
                        is_sharp = sharpness >= config.min_sharpness

                        text = f"Sharpness: {sharpness:.1f}"
                        color = (0, 255, 0) if is_sharp else (0, 0, 255)

                        draw_text(
                            frame,
                            text,
                            (det.bbox.x1, det.bbox.y2 + 20),
                            color=color,
                            font_scale=0.5,
                            bg_color=(0, 0, 0),
                        )

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed

                # Draw FPS
                draw_fps(frame, fps)

                # Draw info panel
                sharp_rate = (total_sharp / total_aligned * 100) if total_aligned > 0 else 0
                info = {
                    "Faces": len(detections),
                    "Aligned": len(aligned_faces),
                    "Sharp": f"{total_sharp}/{total_aligned} ({sharp_rate:.0f}%)",
                    "Threshold": config.min_sharpness,
                }
                draw_info_panel(frame, info, position=(10, 60))

                # Create aligned faces grid
                if aligned_faces:
                    grid = create_aligned_grid(aligned_faces, grid_size=4)

                    # Add title
                    cv2.putText(
                        grid,
                        "Aligned Faces (112x112)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # Show grid
                    if config.display:
                        cv2.imshow("Aligned Faces Grid", grid)

                # Log periodically
                if frame_count % 30 == 0:
                    logger.debug(
                        f"Frame {frame_count}: {len(aligned_faces)} faces aligned, "
                        f"{total_sharp}/{total_aligned} sharp, FPS: {fps:.1f}"
                    )

            # Display main frame
            if config.display:
                cv2.imshow("Face Aligner Test", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):  # ESC or 'q'
                logger.info("Quit requested by user")
                break

            elif key == ord(" "):  # SPACE
                paused = not paused
                status = "paused" if paused else "resumed"
                logger.info(f"Playback {status}")

            elif key == ord("s"):  # 's' - Save aligned faces
                if aligned_faces:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    for i, face in enumerate(aligned_faces):
                        filename = f"aligned_{timestamp}_{i:02d}.jpg"
                        filepath = output_dir / filename
                        cv2.imwrite(str(filepath), face)

                    logger.info(f"Saved {len(aligned_faces)} aligned faces to {output_dir}")
                else:
                    logger.warning("No aligned faces to save")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        sharp_rate = (total_sharp / total_aligned * 100) if total_aligned > 0 else 0

        logger.info(
            f"Processed {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.1f})"
        )
        logger.info(
            f"Aligned {total_aligned} faces, {total_sharp} sharp ({sharp_rate:.1f}%)"
        )


if __name__ == "__main__":
    main()