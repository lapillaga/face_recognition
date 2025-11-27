#!/usr/bin/env python3
"""CLI script for building FAISS index from enrollment images (InsightFace).

This script loads all aligned face images from data/enroll/, extracts
ArcFace embeddings (512-D), and builds a FAISS index for recognition.

NOTE: This script is for InsightFace workflow only. For dlib/face_recognition,
use encode_faces.py which follows the PyImageSearch tutorial approach
(processes raw images and saves embeddings directly to pickle).

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --enroll-dir data/enroll --models-dir models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backends import create_backend, get_model_paths
from app.core.config import Config
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index from enrollment images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--enroll-dir",
        type=str,
        default="data/enroll",
        help="Directory containing enrollment images (organized by person)",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for saving FAISS index and labels",
    )

    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Minimum number of images required per person",
    )

    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a section divider."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_enrollment_images(enroll_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load enrollment images organized by person.

    Args:
        enroll_dir: Root enrollment directory

    Returns:
        Dict mapping person_name → list of aligned face images [112, 112, 3]

    Raises:
        FileNotFoundError: If enroll_dir doesn't exist
        ValueError: If no persons found
    """
    if not enroll_dir.exists():
        raise FileNotFoundError(f"Enrollment directory not found: {enroll_dir}")

    images_per_person: dict[str, list[np.ndarray]] = {}

    # Find all person directories
    person_dirs = [d for d in enroll_dir.iterdir() if d.is_dir()]

    if not person_dirs:
        raise ValueError(f"No person directories found in {enroll_dir}")

    for person_dir in person_dirs:
        person_name = person_dir.name

        # Find all .jpg images
        image_paths = sorted(person_dir.glob("*.jpg"))

        if not image_paths:
            logger.warning(f"No images found for person '{person_name}', skipping")
            continue

        # Load images
        images = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))

            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Validate shape (should be 112x112x3)
            if img.shape != (112, 112, 3):
                logger.warning(
                    f"Image {img_path} has wrong shape {img.shape}, expected (112, 112, 3)"
                )
                continue

            images.append(img)

        if images:
            images_per_person[person_name] = images
            logger.info(f"Loaded {len(images)} images for '{person_name}'")

    return images_per_person


def main() -> None:
    """Main function."""
    args = parse_args()

    print_section("FAISS Index Builder (InsightFace)")
    print(f"Enrollment dir:    {args.enroll_dir}")
    print(f"Models dir:        {args.models_dir}")
    print(f"Min images/person: {args.min_images}")
    print()
    print("NOTE: For dlib/face_recognition, use encode_faces.py instead.")
    print()

    # Load config
    config = Config.from_env()
    logger.info(f"Loaded config: ctx_id={config.ctx_id}")

    # Step 1: Load enrollment images
    print_section("Step 1: Loading Enrollment Images")

    enroll_dir = Path(args.enroll_dir)

    try:
        images_per_person = load_enrollment_images(enroll_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Run 'python scripts/capture_enroll.py --name YOURNAME' first")
        print(f"  - Check that '{args.enroll_dir}' contains person directories")
        return

    # Check min images requirement
    persons_to_skip = []
    for person_name, images in images_per_person.items():
        if len(images) < args.min_images:
            logger.warning(
                f"Person '{person_name}' has only {len(images)} images "
                f"(min: {args.min_images}), skipping"
            )
            persons_to_skip.append(person_name)

    for person_name in persons_to_skip:
        del images_per_person[person_name]

    if not images_per_person:
        logger.error("No persons with sufficient images")
        print(f"Error: No persons have at least {args.min_images} images")
        print()
        print("Please capture more images:")
        print("  python scripts/capture_enroll.py --name YOURNAME --num-images 15")
        return

    print(f"Loaded images for {len(images_per_person)} persons:")
    total_images = 0
    for person_name, images in images_per_person.items():
        print(f"  - {person_name:15s}: {len(images):3d} images")
        total_images += len(images)
    print(f"  {'Total':15s}: {total_images:3d} images")

    # Step 2: Initialize InsightFace backend
    print_section("Step 2: Initializing InsightFace Backend")

    components = create_backend(backend_type="insightface", config=config)
    embedder = components.embedder
    matcher = components.matcher
    print(f"Embedder loaded: {embedder}")
    print(f"Matcher loaded: {matcher}")
    print(f"Embedding dimension: {components.embedding_dim}")

    # Step 3: Extract embeddings
    print_section("Step 3: Extracting Embeddings")

    embeddings_per_person: dict[str, np.ndarray] = {}

    for person_name, images in images_per_person.items():
        print(f"Processing '{person_name}'... ", end="", flush=True)

        # Extract embeddings for all images
        embeddings = []

        for img in images:
            try:
                embedding = embedder.embed(img)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to extract embedding: {e}")
                continue

        if not embeddings:
            logger.error(f"No embeddings extracted for '{person_name}', skipping")
            continue

        # Stack into array [N, embedding_dim]
        embeddings_array = np.stack(embeddings, axis=0)
        embeddings_per_person[person_name] = embeddings_array

        # Compute average intra-person similarity
        if len(embeddings) > 1:
            sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    sims.append(sim)
            avg_sim = np.mean(sims)
            print(f"{len(embeddings)} embeddings (avg similarity: {avg_sim:.3f})")
        else:
            print(f"{len(embeddings)} embedding")

    if not embeddings_per_person:
        logger.error("No embeddings extracted")
        print("Error: Failed to extract embeddings")
        return

    print(f"Extracted embeddings for {len(embeddings_per_person)} persons")

    # Step 4: Build FAISS index
    print_section("Step 4: Building FAISS Index")

    for person_name, embeddings in embeddings_per_person.items():
        matcher.add(person_name, embeddings)
        print(f"Added '{person_name}': {embeddings.shape[0]} embeddings")

    print("Building index...")
    matcher.build()
    print("Index built successfully")

    # Step 5: Save index
    print_section("Step 5: Saving Index")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get InsightFace paths
    paths = get_model_paths("insightface", models_dir)

    matcher.save(paths["index"], paths["labels"], paths["stats"])

    print(f"Saved index to {paths['index']}")
    print(f"Saved labels to {paths['labels']}")
    print(f"Saved stats to {paths['stats']}")

    # Summary
    print_section("Index Built Successfully")
    print(f"Persons enrolled:    {len(set(matcher.labels))}")
    print(f"Total embeddings:    {sum(len(e) for e in embeddings_per_person.values())}")
    print(f"Embedding dimension: {components.embedding_dim}")
    print()
    print("You can now run face recognition:")
    print("   python scripts/run_webcam.py")
    print("   python scripts/run_video.py --video path/to/video.mp4")
    print()

    logger.info("InsightFace FAISS index built successfully")


if __name__ == "__main__":
    main()
