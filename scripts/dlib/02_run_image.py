#!/usr/bin/env python3
"""Face recognition on static images using dlib backend.

This script performs face recognition on a single image file using
the dlib/face_recognition library (PyImageSearch approach).

Usage:
    python scripts/dlib/02_run_image.py --image path/to/image.jpg
    python scripts/dlib/02_run_image.py --image photo.jpg --save result.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.backends import create_backend, get_model_paths, load_matcher
from app.core.config import Config
from app.core.logging_config import setup_logging
from app.core.overlay import draw_bbox, draw_label
from app.services.recognition import RecognitionService

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face recognition on static images (dlib backend)",
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
        "--save",
        type=str,
        default=None,
        help="Path to save output image",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display image window",
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

    print_section("Face Recognition - dlib Backend")
    print(f"Image:         {args.image}")
    print(f"Threshold:     {args.threshold}")
    print(f"Models dir:    {args.models_dir}")
    print(f"Detector:      {args.detector_model}")
    print(f"Save to:       {args.save or 'Not saving'}")
    print()

    # Verify image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

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

    # Load and process image
    print_section("Processing Image")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Failed to read image: {image_path}")
        return

    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Recognize faces
    results = service.recognize(frame)

    print(f"Detected {len(results)} face(s)")

    # Draw results
    for i, result in enumerate(results):
        color = (0, 255, 0) if result.is_known else (0, 0, 255)
        draw_bbox(frame, result.bbox, color=color, thickness=2)

        label_text = f"{result.label} ({result.score:.2f})"
        draw_label(frame, result.bbox, label_text, bg_color=color)

        status = "KNOWN" if result.is_known else "UNKNOWN"
        print(f"  Face {i+1}: {result.label} (score: {result.score:.3f}) [{status}]")

    # Save if requested
    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"Saved to: {args.save}")

    # Display if requested
    if not args.no_display:
        print()
        print("Press any key to close...")
        cv2.imshow("Face Recognition - dlib", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print_section("Complete")


if __name__ == "__main__":
    main()