#!/usr/bin/env python3
"""Face recognition on video file.

This script runs face recognition on a video file and optionally
saves the annotated output.

Usage:
    python scripts/run_video.py --video path/to/video.mp4
    python scripts/run_video.py --video input.mp4 --output output.mp4 --threshold 0.4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.aligner_fivept import FivePointAligner
from app.config import Config
from app.detector_scrfd import SCRFDDetector
from app.embedder_arcface import ArcFaceEmbedder
from app.logging_config import setup_logging
from app.matcher_faiss import FaissMatcher
from app.overlay import draw_bbox, draw_label, draw_fps
from app.recognition import RecognitionService
from app.video_io import VideoFileSource

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face recognition on video file",
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
        help="Path to save annotated output video (optional)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Recognition threshold (overrides .env THRESH value)",
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
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (1=all frames, 2=every other frame, etc.)",
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

    print_section("Face Recognition - Video File")
    print(f"Input video:   {args.video}")
    print(f"Output video:  {args.output or 'None (no save)'}")
    print(f"Models dir:    {args.models_dir}")
    print(f"Display:       {'No (headless)' if args.no_display else 'Yes'}")
    print(f"Skip frames:   {args.skip_frames}")
    print()

    # Load config
    config = Config.from_env()
    logger.info(f"Loaded config: ctx_id={config.ctx_id}, thresh={config.thresh}")

    # Use threshold from args if provided, otherwise from config
    threshold = args.threshold if args.threshold is not None else config.thresh
    print(f"Threshold:     {threshold:.2f}")
    print()

    # Step 1: Load models and index
    print_section("Step 1: Loading Models and Index")

    models_dir = Path(args.models_dir)
    index_path = models_dir / "centroids.faiss"
    labels_path = models_dir / "labels.json"

    if not index_path.exists() or not labels_path.exists():
        logger.error(f"FAISS index not found in {models_dir}")
        print(f"❌ Error: FAISS index not found")
        print()
        print("Please build the index first:")
        print("  1. Capture enrollment images:")
        print("     python scripts/capture_enroll.py --name YOURNAME --num-images 15")
        print("  2. Build FAISS index:")
        print("     python scripts/build_index.py")
        return

    print("Loading detector (SCRFD)...")
    detector = SCRFDDetector(config)
    print(f"✓ Detector loaded")

    print("Loading aligner (5-point)...")
    aligner = FivePointAligner()
    print(f"✓ Aligner loaded")

    print("Loading embedder (ArcFace)...")
    embedder = ArcFaceEmbedder(config)
    print(f"✓ Embedder loaded")

    print("Loading FAISS index...")
    matcher = FaissMatcher(dimension=512)
    matcher.load(index_path, labels_path)
    print(f"✓ FAISS index loaded: {len(matcher.labels)} persons enrolled")
    print(f"  Enrolled persons: {', '.join(matcher.labels)}")

    # Step 2: Initialize recognition service
    print_section("Step 2: Initializing Recognition Service")

    service = RecognitionService(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        matcher=matcher,
        threshold=threshold,
    )
    print(f"✓ Recognition service initialized")

    # Step 3: Open video file
    print_section("Step 3: Opening Video File")

    try:
        source = VideoFileSource(args.video)
        print(f"✓ Video opened: {source}")
        print(f"  Total frames: {source.frame_count}")
        print(f"  Duration: {source.frame_count / source.fps:.1f}s")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to open video: {e}")
        print(f"❌ Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check if video file exists")
        print("  - Try a different video format (MP4, AVI, MOV)")
        print("  - Check if video codec is supported")
        return

    # Optional: Setup video writer
    video_writer = None
    if args.output:
        success, first_frame = source.read()
        if success:
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                args.output,
                fourcc,
                source.fps,
                (w, h),
            )
            logger.info(f"Saving output to {args.output}")
            print(f"✓ Output video writer initialized")
            # Reset to beginning
            source.seek(0)

    # Step 4: Run recognition loop
    print_section("Step 4: Running Recognition")
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
                if key == ord(" "):  # SPACE
                    paused = False
                    logger.info("Resumed")
                elif key == ord("q") or key == 27:  # 'q' or ESC
                    break
                continue

            # Read frame
            success, frame = source.read()

            if not success:
                logger.info("Reached end of video")
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

            # Calculate and draw FPS
            elapsed = time.time() - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            draw_fps(frame, fps)

            # Draw progress
            progress_text = f"Frame: {frame_count}/{source.frame_count} ({source.progress:.1%})"
            cv2.putText(
                frame,
                progress_text,
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Display frame
            if not args.no_display:
                cv2.imshow("Face Recognition - Video", frame)

                # Check for keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    logger.info("User pressed quit key")
                    break
                elif key == ord(" "):  # SPACE
                    paused = True
                    logger.info("Paused")

            # Save frame to output video
            if video_writer is not None:
                video_writer.write(frame)

            # Print progress every 30 processed frames
            if processed_count % 30 == 0:
                num_known = sum(1 for r in results if r.is_known)
                num_unknown = len(results) - num_known
                logger.info(
                    f"Frame {frame_count}/{source.frame_count} ({source.progress:.1%}): "
                    f"{len(results)} faces ({num_known} known, {num_unknown} unknown), "
                    f"Processing FPS: {fps:.1f}"
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
    avg_fps = processed_count / elapsed if elapsed > 0 else 0

    print(f"Total frames:      {frame_count}/{source.frame_count}")
    print(f"Processed frames:  {processed_count}")
    print(f"Skipped frames:    {frame_count - processed_count}")
    print(f"Total time:        {elapsed:.1f}s")
    print(f"Processing FPS:    {avg_fps:.1f}")

    if args.output:
        print(f"Output saved:      {args.output}")

    print()
    logger.info(
        f"Video recognition complete: {processed_count}/{frame_count} frames, "
        f"{avg_fps:.1f} FPS"
    )


if __name__ == "__main__":
    main()