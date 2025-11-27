#!/usr/bin/env python3
"""Test script for face detector with webcam.

This script opens the webcam and displays detected faces in real-time with
bounding boxes, landmarks, and confidence scores.

Usage:
    python scripts/test_detector.py

Controls:
    ESC or 'q' - Quit
    SPACE - Pause/Resume
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_config
from app.backends.insightface.detector import SCRFDDetector
from app.core.logging_config import setup_logging
from app.core.overlay import draw_detections, draw_fps, draw_info_panel

logger = setup_logging(__name__)


def main() -> None:
    """Main function to test detector with webcam."""
    logger.info("Starting detector test with webcam...")

    # Load configuration
    config = get_config()

    # Initialize detector
    logger.info("Initializing detector...")
    try:
        detector = SCRFDDetector(config)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
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

    logger.info("Press ESC or 'q' to quit, SPACE to pause/resume")

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

                # Draw detections
                draw_detections(
                    frame,
                    detections,
                    show_landmarks=True,
                    show_score=True,
                )

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed

                # Draw FPS
                draw_fps(frame, fps)

                # Draw info panel
                info = {
                    "Faces": len(detections),
                    "Resolution": f"{cam_width}x{cam_height}",
                    "Device": "GPU" if config.ctx_id >= 0 else "CPU",
                }
                draw_info_panel(frame, info, position=(10, 60))

                # Log detections periodically
                if frame_count % 30 == 0:
                    logger.debug(
                        f"Frame {frame_count}: {len(detections)} faces detected, FPS: {fps:.1f}"
                    )

            # Display frame
            if config.display:
                cv2.imshow("Face Detector Test", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):  # ESC or 'q'
                logger.info("Quit requested by user")
                break
            elif key == ord(" "):  # SPACE
                paused = not paused
                status = "paused" if paused else "resumed"
                logger.info(f"Playback {status}")

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
        logger.info(
            f"Processed {frame_count} frames in {total_time:.2f}s "
            f"(avg FPS: {avg_fps:.1f})"
        )


if __name__ == "__main__":
    main()