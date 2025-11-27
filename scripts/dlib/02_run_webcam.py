#!/usr/bin/env python3
"""Real-time face recognition via webcam using dlib backend.

This script provides real-time face recognition using the
dlib/face_recognition library (PyImageSearch approach).

Usage:
    python scripts/dlib/02_run_webcam.py
    python scripts/dlib/02_run_webcam.py --camera 1 --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.backends import create_backend, get_model_paths, load_matcher
from app.core.config import Config
from app.core.logging_config import setup_logging
from app.core.overlay import draw_bbox, draw_label, draw_fps
from app.services.recognition import RecognitionService
from app.core.video_io import WebcamSource

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time face recognition via webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (0 for default)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold (0-1, higher=stricter). Default: 0.4",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing FAISS index files",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display (headless mode)",
    )

    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional path to save output video",
    )

    # dlib-specific options
    parser.add_argument(
        "--detector-model",
        type=str,
        choices=["hog", "cnn"],
        default="hog",
        help="Face detector model (hog=faster, cnn=more accurate)",
    )

    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a section divider."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    """Main function."""
    args = parse_args()

    print_section("Real-Time Face Recognition - dlib Backend")
    print(f"Camera ID:     {args.camera}")
    print(f"Models dir:    {args.models_dir}")
    print(f"Detector:      {args.detector_model}")
    print(f"Display:       {'No (headless)' if args.no_display else 'Yes'}")
    print(f"Save video:    {args.save_video or 'No'}")
    print()

    # Load config
    config = Config.from_env()

    # Use threshold from args if provided, otherwise default for dlib
    # Note: This is a SIMILARITY threshold (higher = stricter)
    # For dlib: similarity = 1 - distance, so threshold 0.4 ≈ distance 0.6
    threshold = args.threshold if args.threshold is not None else 0.4
    print(f"Threshold:     {threshold:.2f}")
    print()

    # Step 1: Load models and index
    print_section("Step 1: Loading Models and Index")

    models_dir = Path(args.models_dir)
    paths = get_model_paths("dlib", models_dir)

    if not paths["index"].exists():
        logger.error(f"Index not found in {models_dir}")
        print(f"Error: Encodings not found at {paths['index']}")
        print()
        print("Please encode faces first:")
        print("  python scripts/dlib/01_encode_faces.py --dataset data/dataset")
        return

    print("Creating dlib backend...")
    components = create_backend(
        backend_type="dlib",
        config=config,
        detector_model=args.detector_model,
    )
    print(f"Detector loaded: {components.detector}")
    print(f"Embedder loaded: {components.embedder}")

    print("Loading encodings...")
    matcher = load_matcher("dlib", models_dir)
    unique_labels = sorted(set(matcher.names))
    print(f"Loaded {len(matcher.names)} encodings for {len(unique_labels)} persons")
    print(f"Persons: {', '.join(unique_labels)}")

    # Step 2: Initialize recognition service
    print_section("Step 2: Initializing Recognition Service")

    service = RecognitionService(
        detector=components.detector,
        aligner=components.aligner,
        embedder=components.embedder,
        matcher=matcher,
        threshold=threshold,
    )
    print("Recognition service initialized")

    # Step 3: Open webcam
    print_section("Step 3: Opening Webcam")

    try:
        source = WebcamSource(camera_id=args.camera)
        print(f"Webcam opened: {source}")
    except RuntimeError as e:
        logger.error(f"Failed to open webcam: {e}")
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check if camera is connected")
        print("  - Try a different camera ID (--camera 1, 2, etc.)")
        print("  - Close other applications using the camera")
        return

    # Optional: Setup video writer
    video_writer = None
    if args.save_video:
        success, first_frame = source.read()
        if success:
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                args.save_video,
                fourcc,
                source.fps,
                (w, h),
            )
            logger.info(f"Saving video to {args.save_video}")
            print(f"Saving output to {args.save_video}")

    # Step 4: Run recognition loop
    print_section("Step 4: Running Recognition")
    print("Press 'q' or ESC to quit")
    print()

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            # Read frame
            success, frame = source.read()

            if not success:
                logger.warning("Failed to read frame from webcam")
                break

            frame_count += 1

            # Recognize faces
            results = service.recognize(frame)

            # Draw results
            for result in results:
                # Draw bounding box
                color = (0, 255, 0) if result.is_known else (0, 0, 255)
                draw_bbox(frame, result.bbox, color=color, thickness=2)

                # Draw label with score
                label_text = f"{result.label} ({result.score:.2f})"
                draw_label(
                    frame,
                    result.bbox,
                    label_text,
                    bg_color=color,
                    text_color=(255, 255, 255),
                )

            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed

            # Draw FPS
            draw_fps(frame, fps)

            # Display frame
            if not args.no_display:
                cv2.imshow("Face Recognition - Webcam", frame)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    logger.info("User pressed quit key")
                    break

            # Save frame to video
            if video_writer is not None:
                video_writer.write(frame)

            # Print status every 30 frames
            if frame_count % 30 == 0:
                num_known = sum(1 for r in results if r.is_known)
                num_unknown = len(results) - num_known
                logger.info(
                    f"Frame {frame_count}: {len(results)} faces "
                    f"({num_known} known, {num_unknown} unknown), FPS: {fps:.1f}"
                )

    except KeyboardInterrupt:
        print()
        print("Interrupted by user (Ctrl+C)")

    finally:
        # Cleanup
        source.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Summary
    print_section("Recognition Complete")
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"Total frames:  {frame_count}")
    print(f"Total time:    {elapsed:.1f}s")
    print(f"Average FPS:   {avg_fps:.1f}")

    if args.save_video:
        print(f"Video saved:   {args.save_video}")

    print()
    logger.info(f"Recognition session complete: {frame_count} frames, {avg_fps:.1f} FPS")


if __name__ == "__main__":
    main()