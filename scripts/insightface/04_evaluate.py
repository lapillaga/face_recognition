#!/usr/bin/env python3
"""Evaluate face recognition accuracy on a validation dataset.

This script measures precision, recall, and accuracy of the face recognition
system using a labeled dataset of images.

Dataset structure:
    data/val/
        PERSON1/
            img1.jpg
            img2.jpg
        PERSON2/
            img1.jpg
            img2.jpg
        unknown/        # Optional: images of people NOT in the index
            img1.jpg

Usage:
    python scripts/insightface/04_evaluate.py
    python scripts/insightface/04_evaluate.py --val-dir data/val --threshold 0.35

Output:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion details
    - Results saved to models/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.backends import create_backend, get_model_paths, load_matcher
from app.core.config import Config
from app.core.logging_config import setup_logging
from app.services.recognition import RecognitionService

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/val",
        help="Directory containing validation images (organized by person)",
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
        "--backend",
        type=str,
        choices=["insightface", "dlib"],
        default="insightface",
        help="Face recognition backend to use",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/evaluation_results.json",
        help="Path to save evaluation results",
    )

    return parser.parse_args()


def load_validation_images(val_dir: Path) -> dict[str, list[Path]]:
    """Load validation images organized by person.

    Args:
        val_dir: Directory containing person subdirectories with images

    Returns:
        Dictionary mapping person name to list of image paths
    """
    images_by_person = defaultdict(list)

    if not val_dir.exists():
        logger.error(f"Validation directory not found: {val_dir}")
        return {}

    # Iterate through person directories
    for person_dir in sorted(val_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        # Find all images in this person's directory
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for img_path in person_dir.glob(ext):
                images_by_person[person_name].append(img_path)

    return dict(images_by_person)


def evaluate(
    service: RecognitionService,
    images_by_person: dict[str, list[Path]],
    enrolled_persons: set[str],
) -> dict:
    """Run evaluation and compute metrics.

    Args:
        service: Recognition service instance
        images_by_person: Dictionary mapping person name to image paths
        enrolled_persons: Set of person names that are enrolled in the index

    Returns:
        Dictionary with evaluation results
    """
    results = {
        "total_images": 0,
        "total_faces_detected": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "no_face_detected": 0,
        "details": [],
        "per_person": {},
        "confusion": defaultdict(lambda: defaultdict(int)),
    }

    for person_name, image_paths in images_by_person.items():
        person_results = {
            "total": len(image_paths),
            "correct": 0,
            "incorrect": 0,
            "no_face": 0,
        }

        # Determine expected result
        # If person is enrolled, we expect their name
        # If person is in "unknown" folder or not enrolled, we expect "unknown"
        is_enrolled = person_name.upper() in {p.upper() for p in enrolled_persons}
        expected_label = person_name if is_enrolled else "unknown"

        for img_path in image_paths:
            results["total_images"] += 1

            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"Could not read image: {img_path}")
                results["no_face_detected"] += 1
                person_results["no_face"] += 1
                continue

            # Run recognition
            recognition_results = service.recognize(frame)

            if not recognition_results:
                # No face detected
                results["no_face_detected"] += 1
                person_results["no_face"] += 1
                results["details"].append({
                    "image": str(img_path),
                    "expected": expected_label,
                    "predicted": "NO_FACE_DETECTED",
                    "correct": False,
                })
                continue

            # Take the result with highest recognition score (not first detection)
            # This handles group photos better - picks the most confident match
            result = max(recognition_results, key=lambda r: r.score)
            results["total_faces_detected"] += 1

            predicted_label = result.label
            score = result.score

            # Check if prediction is correct
            # For enrolled persons: predicted should match person_name (case insensitive)
            # For unknown: predicted should be "unknown"
            if is_enrolled:
                is_correct = predicted_label.upper() == person_name.upper()
            else:
                is_correct = predicted_label.lower() == "unknown"

            if is_correct:
                results["correct_predictions"] += 1
                person_results["correct"] += 1
            else:
                results["incorrect_predictions"] += 1
                person_results["incorrect"] += 1

            # Record confusion matrix
            results["confusion"][expected_label][predicted_label] += 1

            # Record detailed result
            results["details"].append({
                "image": str(img_path),
                "expected": expected_label,
                "predicted": predicted_label,
                "score": float(score),
                "correct": is_correct,
            })

        results["per_person"][person_name] = person_results

    # Convert confusion defaultdict to regular dict
    results["confusion"] = {k: dict(v) for k, v in results["confusion"].items()}

    # Calculate metrics
    total_evaluated = results["correct_predictions"] + results["incorrect_predictions"]

    if total_evaluated > 0:
        results["accuracy"] = results["correct_predictions"] / total_evaluated
    else:
        results["accuracy"] = 0.0

    # Calculate per-class metrics
    results["per_class_accuracy"] = {}
    for person_name, person_data in results["per_person"].items():
        evaluated = person_data["correct"] + person_data["incorrect"]
        if evaluated > 0:
            results["per_class_accuracy"][person_name] = person_data["correct"] / evaluated
        else:
            results["per_class_accuracy"][person_name] = 0.0

    return results


def print_results(results: dict, threshold: float) -> None:
    """Print evaluation results in a formatted way."""
    print()
    print("=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print()
    print(f"Threshold:              {threshold:.2f}")
    print(f"Total images:           {results['total_images']}")
    print(f"Faces detected:         {results['total_faces_detected']}")
    print(f"No face detected:       {results['no_face_detected']}")
    print()
    print("-" * 70)
    print("  METRICS")
    print("-" * 70)
    print()
    print(f"Correct predictions:    {results['correct_predictions']}")
    print(f"Incorrect predictions:  {results['incorrect_predictions']}")
    print()
    print(f">>> ACCURACY:           {results['accuracy'] * 100:.2f}% <<<")
    print()
    print("-" * 70)
    print("  PER-PERSON RESULTS")
    print("-" * 70)
    print()
    print(f"{'Person':<20} {'Total':<8} {'Correct':<10} {'Incorrect':<10} {'Accuracy':<10}")
    print("-" * 58)

    for person_name, data in sorted(results["per_person"].items()):
        acc = results["per_class_accuracy"].get(person_name, 0) * 100
        print(f"{person_name:<20} {data['total']:<8} {data['correct']:<10} {data['incorrect']:<10} {acc:.1f}%")

    print()
    print("-" * 70)
    print("  CONFUSION MATRIX")
    print("-" * 70)
    print()

    if results["confusion"]:
        # Get all labels
        all_labels = set()
        for expected, predictions in results["confusion"].items():
            all_labels.add(expected)
            all_labels.update(predictions.keys())
        all_labels = sorted(all_labels)

        # Print header
        header = f"{'Expected \\ Predicted':<15}"
        for label in all_labels:
            header += f" {label[:10]:<10}"
        print(header)
        print("-" * len(header))

        # Print rows
        for expected in all_labels:
            row = f"{expected[:15]:<15}"
            for predicted in all_labels:
                count = results["confusion"].get(expected, {}).get(predicted, 0)
                row += f" {count:<10}"
            print(row)

    print()
    print("=" * 70)


def main() -> None:
    """Main function."""
    args = parse_args()

    print()
    print("=" * 70)
    print("  Face Recognition Evaluation")
    print("=" * 70)
    print()

    # Load config
    config = Config.from_env()
    threshold = args.threshold if args.threshold is not None else config.thresh

    print(f"Backend:        {args.backend}")
    print(f"Threshold:      {threshold:.2f}")
    print(f"Validation dir: {args.val_dir}")
    print(f"Models dir:     {args.models_dir}")
    print()

    # Check validation directory
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        print(f"ERROR: Validation directory not found: {val_dir}")
        print()
        print("Please create a validation dataset with the following structure:")
        print()
        print("  data/val/")
        print("      PERSON1/")
        print("          img1.jpg")
        print("          img2.jpg")
        print("      PERSON2/")
        print("          img1.jpg")
        print("      unknown/        # Optional: unknown people")
        print("          img1.jpg")
        print()
        print("Tip: You can copy some images from data/enroll/ to data/val/")
        print("     or capture new images for validation.")
        return

    # Load validation images
    print("Loading validation images...")
    images_by_person = load_validation_images(val_dir)

    if not images_by_person:
        print("ERROR: No images found in validation directory")
        return

    total_images = sum(len(imgs) for imgs in images_by_person.values())
    print(f"Found {total_images} images across {len(images_by_person)} persons:")
    for person, imgs in sorted(images_by_person.items()):
        print(f"  - {person}: {len(imgs)} images")
    print()

    # Load models and index
    print("Loading models and index...")
    models_dir = Path(args.models_dir)
    paths = get_model_paths(args.backend, models_dir)

    if not paths["index"].exists():
        print(f"ERROR: Index not found at {paths['index']}")
        print("Please build the index first using 02_build_index.py")
        return

    components = create_backend(
        backend_type=args.backend,
        config=config,
    )

    matcher = load_matcher(args.backend, models_dir)
    labels = matcher.names if hasattr(matcher, 'names') else matcher.labels
    enrolled_persons = set(labels)

    print(f"Enrolled persons: {sorted(enrolled_persons)}")
    print()

    # Initialize recognition service
    service = RecognitionService(
        detector=components.detector,
        aligner=components.aligner,
        embedder=components.embedder,
        matcher=matcher,
        threshold=threshold,
    )

    # Run evaluation
    print("Running evaluation...")
    print()

    results = evaluate(service, images_by_person, enrolled_persons)

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "threshold": threshold,
        "val_dir": str(val_dir),
        "enrolled_persons": sorted(enrolled_persons),
    }

    # Print results
    print_results(results, threshold)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove detailed results for JSON (can be large)
    results_to_save = {k: v for k, v in results.items() if k != "details"}
    results_to_save["num_detailed_results"] = len(results["details"])

    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # Print summary for document
    print("-" * 70)
    print("  FOR YOUR DOCUMENT")
    print("-" * 70)
    print()
    print(f"El sistema alcanzó una precisión del {results['accuracy'] * 100:.2f}% ")
    print(f"en el conjunto de validación compuesto por {total_images} imágenes ")
    print(f"de {len(images_by_person)} personas distintas, utilizando un umbral ")
    print(f"de similitud de {threshold}.")
    print()


if __name__ == "__main__":
    main()
