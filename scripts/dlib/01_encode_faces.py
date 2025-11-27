#!/usr/bin/env python3
"""Encode faces from a dataset using dlib/face_recognition (PyImageSearch approach).

This script follows the PyImageSearch tutorial for face recognition:
https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

It processes a dataset of images organized by person, detects faces,
extracts 128-D embeddings using dlib's ResNet-34 model, and saves
everything to a pickle file.

Expected input structure:
    data/dataset/
        PERSON1/
            photo1.jpg
            photo2.jpg
        PERSON2/
            photo1.jpg
            ...

Output:
    models/encodings.pkl  (contains encodings + names)

Usage:
    python scripts/encode_faces.py
    python scripts/encode_faces.py --dataset data/my_photos
    python scripts/encode_faces.py --detection-method cnn  # More accurate
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import face_recognition

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Encode faces from dataset (PyImageSearch approach)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset",
        help="Path to input dataset directory (organized by person)",
    )

    parser.add_argument(
        "--encodings",
        type=str,
        default="models/encodings.pkl",
        help="Path to output encodings pickle file",
    )

    parser.add_argument(
        "--detection-method",
        type=str,
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model (hog=faster, cnn=more accurate)",
    )

    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="Number of times to re-sample face when computing encoding (higher=more accurate but slower)",
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


def get_image_paths(dataset_dir: Path, extensions: list[str]) -> list[Path]:
    """Get all image paths from dataset directory.

    Args:
        dataset_dir: Root dataset directory
        extensions: List of valid image extensions

    Returns:
        List of image paths
    """
    image_paths = []

    for ext in extensions:
        image_paths.extend(dataset_dir.glob(f"**/*.{ext}"))
        image_paths.extend(dataset_dir.glob(f"**/*.{ext.upper()}"))

    return sorted(set(image_paths))


def main() -> None:
    """Main function following PyImageSearch tutorial approach."""
    args = parse_args()

    print_section("Face Encoding - dlib/face_recognition (PyImageSearch)")
    print(f"Dataset:          {args.dataset}")
    print(f"Output:           {args.encodings}")
    print(f"Detection method: {args.detection_method}")
    print(f"Num jitters:      {args.num_jitters}")
    print()

    # Verify dataset exists
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        print(f"Error: Dataset directory not found: {dataset_dir}")
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

    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]

    # Get all image paths
    print_section("Step 1: Finding Images")
    image_paths = get_image_paths(dataset_dir, extensions)

    if not image_paths:
        logger.error(f"No images found in {dataset_dir}")
        print(f"Error: No images found in {dataset_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Count images per person
    persons = {}
    for img_path in image_paths:
        # Person name is the parent directory name
        person_name = img_path.parent.name
        if person_name not in persons:
            persons[person_name] = 0
        persons[person_name] += 1

    print(f"Found {len(persons)} persons:")
    for person_name, count in sorted(persons.items()):
        print(f"  - {person_name}: {count} images")

    # Process images and extract encodings
    print_section("Step 2: Encoding Faces")
    print("This may take a while...")
    print()

    known_encodings = []
    known_names = []

    processed = 0
    faces_found = 0
    faces_skipped = 0

    for i, image_path in enumerate(image_paths, 1):
        # Get person name from parent directory
        person_name = image_path.parent.name

        # Progress indicator
        if i % 10 == 0 or i == len(image_paths):
            print(f"Processing image {i}/{len(image_paths)}: {image_path.name}")

        # Load image (OpenCV loads as BGR)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue

        # Convert BGR to RGB (face_recognition expects RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Detect face locations
        # Returns list of (top, right, bottom, left) tuples
        boxes = face_recognition.face_locations(
            image_rgb,
            model=args.detection_method,
        )

        if len(boxes) == 0:
            logger.debug(f"No faces found in {image_path.name}")
            faces_skipped += 1
            continue

        # Compute facial embeddings for each face
        encodings = face_recognition.face_encodings(
            image_rgb,
            known_face_locations=boxes,
            num_jitters=args.num_jitters,
        )

        # Store encodings and names
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)
            faces_found += 1

        processed += 1

    print()
    print(f"Images processed: {processed}")
    print(f"Faces found:      {faces_found}")
    print(f"Images skipped:   {faces_skipped} (no face detected)")

    if faces_found == 0:
        logger.error("No faces found in dataset")
        print()
        print("Error: No faces were found in the dataset.")
        print("Please check that your images contain clear, frontal faces.")
        return

    # Count encodings per person
    print()
    print("Encodings per person:")
    encoding_counts = {}
    for name in known_names:
        encoding_counts[name] = encoding_counts.get(name, 0) + 1

    for person_name, count in sorted(encoding_counts.items()):
        print(f"  - {person_name}: {count} encodings")

    # Save encodings to pickle file
    print_section("Step 3: Saving Encodings")

    output_path = Path(args.encodings)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create data structure (same as PyImageSearch tutorial)
    data = {
        "encodings": known_encodings,
        "names": known_names,
    }

    # Write to pickle file
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(known_encodings)} encodings to {output_path}")

    # Also save labels as JSON for compatibility
    import json
    labels_path = output_path.parent / "labels_dlib.json"
    unique_labels = sorted(set(known_names))
    with open(labels_path, "w") as f:
        json.dump(unique_labels, f, indent=2)
    print(f"Saved {len(unique_labels)} labels to {labels_path}")

    # Summary
    print_section("Encoding Complete")
    print(f"Total persons:    {len(unique_labels)}")
    print(f"Total encodings:  {len(known_encodings)}")
    print(f"Output file:      {output_path}")
    print()
    print("You can now run face recognition with dlib backend:")
    print("   python scripts/run_webcam.py --backend dlib")
    print("   python scripts/run_video.py --video path/to/video.mp4 --backend dlib")
    print()

    logger.info(
        f"Encoding complete: {len(known_encodings)} encodings for "
        f"{len(unique_labels)} persons saved to {output_path}"
    )


if __name__ == "__main__":
    main()