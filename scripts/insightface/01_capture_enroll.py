#!/usr/bin/env python3
"""CLI script for capturing enrollment images via webcam.

This script provides an interactive way to capture face images
for enrolling new persons into the face recognition system.

Uses InsightFace backend (SCRFD detector + 5-point alignment) to capture
high-quality aligned face crops (112x112) for later embedding extraction.

NOTE: This script is for InsightFace workflow only. For dlib/face_recognition,
use encode_faces.py which follows the PyImageSearch tutorial approach.

Usage:
    python scripts/capture_enroll.py --name LUIS --num-images 15
    python scripts/capture_enroll.py --name MARIA --num-images 20 --camera 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backends import create_backend
from app.core.config import Config
from app.services.enrollment import EnrollmentService
from app.core.logging_config import setup_logging
from app.core.video_io import WebcamSource

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture enrollment images via webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Person's name (will be uppercased)",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=15,
        help="Number of images to capture",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (0 for default)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/enroll",
        help="Root directory for saving enrollment images",
    )

    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum Laplacian variance for quality filtering",
    )

    parser.add_argument(
        "--min-detection-score",
        type=float,
        default=0.5,
        help="Minimum detection confidence score",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable preview window (headless mode)",
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

    print_section("Face Enrollment - Capture Images (InsightFace)")
    print(f"Person name:       {args.name.upper()}")
    print(f"Target images:     {args.num_images}")
    print(f"Camera ID:         {args.camera}")
    print(f"Save directory:    {args.save_dir}")
    print(f"Min sharpness:     {args.min_sharpness}")
    print(f"Min score:         {args.min_detection_score}")
    print(f"Display:           {'No (headless)' if args.no_display else 'Yes'}")
    print()
    print("NOTE: For dlib/face_recognition, use encode_faces.py instead.")
    print()

    # Load config
    config = Config.from_env()
    logger.info(f"Loaded config: ctx_id={config.ctx_id}")

    # Initialize components using InsightFace backend
    print_section("Initializing Components")

    print("Creating InsightFace backend...")
    components = create_backend(
        backend_type="insightface",
        config=config,
    )
    print(f"Detector loaded: {components.detector}")
    print(f"Aligner loaded: {components.aligner}")

    # Initialize enrollment service
    print("Initializing enrollment service...")
    service = EnrollmentService(
        detector=components.detector,
        aligner=components.aligner,
        save_dir=args.save_dir,
        min_sharpness=args.min_sharpness,
        min_detection_score=args.min_detection_score,
    )
    print(f"Service initialized: {service}")

    # Check if person already has enrolled images
    existing_count = service.count_enrolled_images(args.name)
    if existing_count > 0:
        print()
        print(f"WARNING: Person '{args.name.upper()}' already has {existing_count} enrolled images")
        response = input("Do you want to add more images? (y/N): ").strip().lower()
        if response != "y":
            print("Enrollment cancelled")
            return

    # Open webcam
    print_section("Opening Webcam")
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

    # Instructions
    print_section("Instructions")
    print("1. Position your face inside the blue guide box")
    print("2. Ensure good lighting and focus")
    print("3. Vary your pose slightly (look left/right, up/down)")
    print("4. Images will be captured automatically when quality is good")
    print("5. Press ESC to stop early")
    print()
    input("Press ENTER to start capturing...")

    # Start enrollment
    print_section("Capturing Images")
    try:
        num_captured = service.enroll_from_video(
            source=source,
            person_name=args.name,
            num_images=args.num_images,
            display=not args.no_display,
        )
    except KeyboardInterrupt:
        print()
        print("Enrollment interrupted by user (Ctrl+C)")
        num_captured = service.count_enrolled_images(args.name) - existing_count
    finally:
        source.release()

    # Summary
    print_section("Enrollment Complete")
    total_images = service.count_enrolled_images(args.name)
    print(f"Captured {num_captured} new images")
    print(f"Total images for '{args.name.upper()}': {total_images}")
    print(f"Images saved to: {Path(args.save_dir) / args.name.upper()}")
    print()

    if total_images < 10:
        print("WARNING: Warning: Less than 10 images. Recommend capturing at least 12-15 images.")
        print("   Run this script again to add more images.")
    else:
        print("Good! You can now build the FAISS index:")
        print("   python scripts/build_index.py")
    print()

    logger.info(f"Enrollment session complete: {num_captured} images captured")


if __name__ == "__main__":
    main()
