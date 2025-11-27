#!/usr/bin/env python3
"""Face recognition on video file using dlib backend.

This script runs face recognition on a video file using the
dlib/face_recognition library (PyImageSearch approach).

Usage:
    python scripts/dlib/02_run_video.py --video path/to/video.mp4
    python scripts/dlib/02_run_video.py --video input.mp4 --output output.mp4
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
from app.core.video_io import VideoFileSource

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face recognition on video file (dlib backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output video",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Similarity threshold (0-1, higher=stricter). Default: 0.4",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing encodings.pkl",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display (headless mode)",
    )

    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (1=all, 2=every other, etc.)",
    )

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

    print_section("Face Recognition on Video - dlib Backend")
    print(f"Input video:   {args.video}")
    print(f"Output video:  {args.output or 'None (no save)'}")
    print(f"Models dir:    {args.models_dir}")
    print(f"Detector:      {args.detector_model}")
    print(f"Threshold:     {args.threshold}")
    print(f"Skip frames:   {args.skip_frames}")
    print(f"Display:       {'No (headless)' if args.no_display else 'Yes'}")
    print()

    # Load config
    config = Config.from_env()

    # Check if encodings exist
    models_dir = Path(args.models_dir)
    paths = get_model_paths("dlib", models_dir)

    if not paths["index"].exists():
        print(f"Error: Encodings not found at {paths['index']}")
        print()
        print("Please encode faces first:")
        print("  python scripts/dlib/01_encode_faces.py --dataset data/dataset")
        return

    # Load components
    print_section("Loading Models")

    print("Creating dlib backend...")
    components = create_backend(
        backend_type="dlib",
        config=config,
        detector_model=args.detector_model,
    )
    print(f"Detector: {components.detector}")
    print(f"Embedder: {components.embedder}")

    print("Loading encodings...")
    matcher = load_matcher("dlib", models_dir)
    unique_labels = sorted(set(matcher.names))
    print(f"Loaded {len(matcher.names)} encodings for {len(unique_labels)} persons")
    print(f"Persons: {', '.join(unique_labels)}")

    # Initialize recognition service
    service = RecognitionService(
        detector=components.detector,
        aligner=components.aligner,  # None for dlib
        embedder=components.embedder,
        matcher=matcher,
        threshold=args.threshold,
    )

    # Open video file
    print_section("Opening Video File")

    try:
        source = VideoFileSource(args.video)
        print(f"Video opened: {source}")
        print(f"  Total frames: {source.frame_count}")
        print(f"  Duration: {source.frame_count / source.fps:.1f}s")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    # Setup video writer if needed
    video_writer = None
    if args.output:
        success, first_frame = source.read()
        if success:
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(args.output, fourcc, source.fps, (w, h))
            print(f"Output: {args.output}")
            source.seek(0)

    # Run recognition
    print_section("Running Recognition")
    if not args.no_display:
        print("Press 'q' or ESC to quit, SPACE to pause/resume")
    print()

    frame_count = 0
    processed_count = 0
    start_time = time.time()
    paused = False

    try:
        while True:
            # Handle pause
            if paused and not args.no_display:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q") or key == 27:
                    break
                continue

            # Read frame
            success, frame = source.read()
            if not success:
                break

            frame_count += 1

            # Skip frames if requested
            if frame_count % args.skip_frames != 0:
                continue

            processed_count += 1

            # Recognize faces
            results = service.recognize(frame)

            # Draw results
            for result in results:
                color = (0, 255, 0) if result.is_known else (0, 0, 255)
                draw_bbox(frame, result.bbox, color=color, thickness=2)

                label_text = f"{result.label} ({result.score:.2f})"
                draw_label(frame, result.bbox, label_text, bg_color=color)

            # Draw FPS and progress
            elapsed = time.time() - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            draw_fps(frame, fps)

            progress_text = f"Frame: {frame_count}/{source.frame_count} ({source.progress:.1%})"
            cv2.putText(frame, progress_text, (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            if not args.no_display:
                cv2.imshow("Face Recognition - dlib", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    paused = True

            # Save frame
            if video_writer:
                video_writer.write(frame)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        source.release()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Summary
    print_section("Complete")
    elapsed = time.time() - start_time
    avg_fps = processed_count / elapsed if elapsed > 0 else 0

    print(f"Total frames:     {frame_count}/{source.frame_count}")
    print(f"Processed:        {processed_count}")
    print(f"Time:             {elapsed:.1f}s")
    print(f"FPS:              {avg_fps:.1f}")
    if args.output:
        print(f"Output saved:     {args.output}")


if __name__ == "__main__":
    main()