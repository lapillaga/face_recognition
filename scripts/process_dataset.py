#!/usr/bin/env python3
"""Process a dataset of photos and prepare them for enrollment.

This script takes a directory of raw images organized by person,
detects faces, aligns them, filters by quality, and saves them
in the enrollment directory format.

Expected input structure:
    data/my_dataset/
        PERSON1/
            photo1.jpg  # can be full-body, face will be detected
            photo2.jpg
        PERSON2/
            photo1.jpg
            ...

Output structure:
    data/enroll/
        PERSON1/
            000.jpg  # aligned 112x112 face crops
            001.jpg
            ...
        PERSON2/
            000.jpg
            ...

Usage:
    # Simple usage (uses data/my_dataset by default)
    python scripts/process_dataset.py

    # Custom dataset location
    python scripts/process_dataset.py --dataset-dir path/to/other/dataset

    # Replace existing enrollment data
    python scripts/process_dataset.py --replace

    # Adjust quality filters
    python scripts/process_dataset.py --min-sharpness 100
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
from app.logging_config import setup_logging
from app.utils import compute_sharpness

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process raw dataset and prepare for enrollment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/my_dataset",
        help="Directory containing raw images organized by person (PERSON/photo.jpg)",
    )

    parser.add_argument(
        "--enroll-dir",
        type=str,
        default="data/enroll",
        help="Output enrollment directory",
    )

    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=150.0,
        help="Minimum Laplacian variance for sharpness filter",
    )

    parser.add_argument(
        "--min-detection-score",
        type=float,
        default=0.5,
        help="Minimum detection confidence score",
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing enrollment data (WARNING: deletes existing data)",
    )

    parser.add_argument(
        "--max-faces-per-image",
        type=int,
        default=1,
        help="Maximum faces to extract per image (1 = only dominant face)",
    )

    parser.add_argument(
        "--extensions",
        type=str,
        default="jpg,jpeg,png,bmp",
        help="Comma-separated list of image extensions to process",
    )

    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a section divider."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_next_filename(output_dir: Path) -> str:
    """Get next sequential filename in output directory.

    Args:
        output_dir: Directory to check for existing files

    Returns:
        Next available filename (e.g., "015.jpg")
    """
    existing_files = list(output_dir.glob("*.jpg"))
    if not existing_files:
        return "000.jpg"

    # Find highest number
    max_num = -1
    for file in existing_files:
        try:
            num = int(file.stem)
            max_num = max(max_num, num)
        except ValueError:
            continue

    return f"{max_num + 1:03d}.jpg"


def process_person_directory(
    person_dir: Path,
    output_dir: Path,
    detector: SCRFDDetector,
    aligner: FivePointAligner,
    min_sharpness: float,
    min_detection_score: float,
    max_faces_per_image: int,
    extensions: list[str],
) -> tuple[int, int, int]:
    """Process all images for one person.

    Args:
        person_dir: Input directory with raw images
        output_dir: Output directory for aligned faces
        detector: Face detector
        aligner: Face aligner
        min_sharpness: Minimum sharpness threshold
        min_detection_score: Minimum detection score
        max_faces_per_image: Max faces to extract per image
        extensions: List of valid image extensions

    Returns:
        Tuple of (total_processed, faces_extracted, faces_saved)
    """
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(person_dir.glob(f"*.{ext}"))
        image_files.extend(person_dir.glob(f"*.{ext.upper()}"))

    if not image_files:
        logger.warning(f"No images found in {person_dir}")
        return 0, 0, 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    faces_extracted = 0
    faces_saved = 0

    for img_path in sorted(image_files):
        logger.debug(f"Processing {img_path.name}")

        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Failed to read {img_path}")
            continue

        total_processed += 1

        # Detect faces
        detections = detector.detect(frame)

        if not detections:
            logger.debug(f"No faces detected in {img_path.name}")
            continue

        # Filter by detection score
        detections = [d for d in detections if d.score >= min_detection_score]

        if not detections:
            logger.debug(f"No faces passed detection score threshold in {img_path.name}")
            continue

        # Sort by detection score (highest first) and limit
        detections = sorted(detections, key=lambda d: d.score, reverse=True)
        detections = detections[:max_faces_per_image]

        faces_extracted += len(detections)

        # Process each detected face
        for det in detections:
            if det.kps is None:
                logger.warning(f"No landmarks for face in {img_path.name}")
                continue

            try:
                # Align face
                aligned = aligner.align(frame, det.kps)

                # Check sharpness
                sharpness = compute_sharpness(aligned)
                if sharpness < min_sharpness:
                    logger.debug(
                        f"Face too blurry in {img_path.name} "
                        f"(sharpness: {sharpness:.1f} < {min_sharpness})"
                    )
                    continue

                # Save aligned face
                output_path = output_dir / get_next_filename(output_dir)
                cv2.imwrite(str(output_path), aligned)
                faces_saved += 1

                logger.debug(
                    f"Saved face from {img_path.name} → {output_path.name} "
                    f"(score: {det.score:.2f}, sharpness: {sharpness:.1f})"
                )

            except Exception as e:
                logger.warning(f"Failed to process face in {img_path.name}: {e}")
                continue

    return total_processed, faces_extracted, faces_saved


def main() -> None:
    """Main function."""
    args = parse_args()

    print_section("Dataset Processor - Raw Photos to Enrollment Format")
    print(f"Dataset dir:       {args.dataset_dir}")
    print(f"Enrollment dir:    {args.enroll_dir}")
    print(f"Min sharpness:     {args.min_sharpness}")
    print(f"Min detection:     {args.min_detection_score}")
    print(f"Max faces/image:   {args.max_faces_per_image}")
    print(f"Replace existing:  {'Yes (WARNING: WILL DELETE)' if args.replace else 'No (merge)'}")
    print()

    # Verify the dataset directory exists
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    # Find person directories
    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not person_dirs:
        logger.error(f"No person directories found in {dataset_dir}")
        print(f"Error: No subdirectories found in {dataset_dir}")
        print()
        print("Expected structure:")
        print(f"  {dataset_dir}/")
        print("    PERSON1/")
        print("      photo1.jpg")
        print("      photo2.jpg")
        print("    PERSON2/")
        print("      photo1.jpg")
        print("      ...")
        return

    print(f"Found {len(person_dirs)} person directories:")
    for person_dir in sorted(person_dirs):
        print(f"  - {person_dir.name}")
    print()

    # Load config
    config = Config.from_env()
    logger.info(f"Loaded config: ctx_id={config.ctx_id}")

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]

    # Step 1: Initialize models
    print_section("Step 1: Loading Models")

    print("Loading detector (SCRFD)...")
    detector = SCRFDDetector(config)
    print("Detector loaded")

    print("Loading aligner (5-point)...")
    aligner = FivePointAligner()
    print("Aligner loaded")

    # Step 2: Process dataset
    print_section("Step 2: Processing Dataset")

    enroll_dir = Path(args.enroll_dir)

    # Warn if replacing
    if args.replace and enroll_dir.exists():
        import shutil

        print(f"WARNING: WARNING: Deleting existing enrollment directory: {enroll_dir}")
        response = input("Are you sure? Type 'yes' to continue: ")
        if response.lower() != "yes":
            print("Aborted.")
            return
        shutil.rmtree(enroll_dir)
        print(f"Deleted {enroll_dir}")

    # Statistics
    stats = {}
    total_images = 0
    total_faces_extracted = 0
    total_faces_saved = 0

    # Process each person
    for person_dir in sorted(person_dirs):
        person_name = person_dir.name
        output_dir = enroll_dir / person_name

        print(f"\nProcessing '{person_name}'...")

        processed, extracted, saved = process_person_directory(
            person_dir=person_dir,
            output_dir=output_dir,
            detector=detector,
            aligner=aligner,
            min_sharpness=args.min_sharpness,
            min_detection_score=args.min_detection_score,
            max_faces_per_image=args.max_faces_per_image,
            extensions=extensions,
        )

        stats[person_name] = {
            "processed": processed,
            "extracted": extracted,
            "saved": saved,
        }

        total_images += processed
        total_faces_extracted += extracted
        total_faces_saved += saved

        print(f"  {person_name}:")
        print(f"      Images processed: {processed}")
        print(f"      Faces extracted:  {extracted}")
        print(f"      Faces saved:      {saved}")

        if saved == 0:
            print("      WARNING: No faces saved (check quality filters)")

    # Summary
    print_section("Processing Complete")

    print(f"Persons processed:      {len(stats)}")
    print(f"Total images processed: {total_images}")
    print(f"Total faces extracted:  {total_faces_extracted}")
    print(f"Total faces saved:      {total_faces_saved}")
    print()

    print("Detailed statistics:")
    for person_name, person_stats in sorted(stats.items()):
        print(f"  {person_name:20s}: {person_stats['saved']:3d} faces saved")

    print()

    if total_faces_saved == 0:
        print("No faces were saved!")
        print()
        print("Troubleshooting:")
        print("  - Check image quality (not too blurry)")
        print("  - Try lowering --min-sharpness (current: {})".format(args.min_sharpness))
        print("  - Try lowering --min-detection-score (current: {})".format(args.min_detection_score))
        print("  - Check that images contain clear frontal faces")
    else:
        print("Dataset processed successfully!")
        print()
        print("Next step: Build FAISS index")
        print("  python scripts/build_index.py")
        print()
        print("Or continue adding more people:")
        print("  python scripts/capture_enroll.py --name NEWPERSON")
        print("  python scripts/build_index.py")

    logger.info(f"Dataset processing complete: {total_faces_saved} faces saved")


if __name__ == "__main__":
    main()