#!/usr/bin/env python3
"""Face recognition on static images.

This script performs face recognition on a single image file and displays
or saves the result with bounding boxes and labels.

Usage:
    python scripts/run_image.py --image path/to/image.jpg
    python scripts/run_image.py --image photo.jpg --save result.jpg --threshold 0.4
"""

from __future__ import annotations

import argparse
import sys
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
from app.overlay import draw_bbox, draw_label
from app.recognition import RecognitionService

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face recognition on static images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file",
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
        "--save",
        type=str,
        default=None,
        help="Path to save output image (if not specified, displays only)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display image window (useful with --save)",
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

    print_section("Face Recognition - Static Image")

    # Verify image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        print(f"Error: Image file not found: {image_path}")
        return

    print(f"Input image:   {image_path}")
    print(f"Models dir:    {args.models_dir}")
    print(f"Save output:   {args.save or 'No'}")
    print(f"Display:       {'No' if args.no_display else 'Yes'}")
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
        print("Error: FAISS index not found")
        print()
        print("Please build the index first:")
        print("  1. Capture enrollment images:")
        print("     python scripts/capture_enroll.py --name YOURNAME --num-images 15")
        print("  2. Build FAISS index:")
        print("     python scripts/build_index.py")
        return

    print("Loading detector (SCRFD)...")
    detector = SCRFDDetector(config)
    print("Detector loaded")

    print("Loading aligner (5-point)...")
    aligner = FivePointAligner()
    print("Aligner loaded")

    print("Loading embedder (ArcFace)...")
    embedder = ArcFaceEmbedder(config)
    print("Embedder loaded")

    print("Loading FAISS index...")
    matcher = FaissMatcher(dimension=512)
    matcher.load(index_path, labels_path)
    print(f"FAISS index loaded: {len(matcher.labels)} persons enrolled")
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
    print("Recognition service initialized")

    # Step 3: Load and process image
    print_section("Step 3: Processing Image")

    # Read image
    print(f"Reading image: {image_path}")
    frame = cv2.imread(str(image_path))

    if frame is None:
        logger.error(f"Failed to read image: {image_path}")
        print("Error: Could not read image file")
        print("   Make sure the file is a valid image format (jpg, png, etc.)")
        return

    h, w = frame.shape[:2]
    print(f"Image loaded: {w}x{h} pixels")

    # Recognize faces
    print("Running face recognition...")
    results = service.recognize(frame)
    print(f"Detection complete: {len(results)} face(s) found")
    print()

    # Step 4: Display results
    print_section("Step 4: Recognition Results")

    if len(results) == 0:
        print("No faces detected in image")
        print()
        print("Tips:")
        print("  - Make sure faces are clearly visible")
        print("  - Try different lighting conditions")
        print("  - Ensure faces are not too small or too far")
    else:
        # Print results
        for i, result in enumerate(results, 1):
            status = "KNOWN" if result.is_known else "UNKNOWN"
            print(f"Face {i}:")
            print(f"  Label:  {result.label}")
            print(f"  Score:  {result.score:.3f}")
            print(f"  Status: {status}")
            print(f"  BBox:   ({result.bbox.x1}, {result.bbox.y1}) → ({result.bbox.x2}, {result.bbox.y2})")
            print()

        # Draw results on frame
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

    # Step 5: Save/Display output
    print_section("Step 5: Output")

    # Save image if requested
    if args.save:
        save_path = Path(args.save)
        cv2.imwrite(str(save_path), frame)
        print(f"Image saved: {save_path}")

    # Display image if requested
    if not args.no_display:
        print("Displaying image (press any key to close)...")
        cv2.imshow("Face Recognition - Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Display closed")

    print()
    logger.info(f"Image processing complete: {len(results)} faces detected")


if __name__ == "__main__":
    main()