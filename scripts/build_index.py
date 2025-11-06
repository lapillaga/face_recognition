#!/usr/bin/env python3
"""CLI script for building FAISS index from enrollment images.

This script loads all aligned face images from data/enroll/, extracts
embeddings using ArcFace, and builds a FAISS index for recognition.

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

from app.config import Config
from app.embedder_arcface import ArcFaceEmbedder
from app.logging_config import setup_logging
from app.matcher_faiss import FaissMatcher

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

    print_section("FAISS Index Builder - Build from Enrollment Images")
    print(f"Enrollment dir:    {args.enroll_dir}")
    print(f"Models dir:        {args.models_dir}")
    print(f"Min images/person: {args.min_images}")
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
        print(f"❌ Error: {e}")
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
        print(f"❌ Error: No persons have at least {args.min_images} images")
        print()
        print("Please capture more images:")
        print("  python scripts/capture_enroll.py --name YOURNAME --num-images 15")
        return

    print(f"✓ Loaded images for {len(images_per_person)} persons:")
    total_images = 0
    for person_name, images in images_per_person.items():
        print(f"  - {person_name:15s}: {len(images):3d} images")
        total_images += len(images)
    print(f"  {'Total':15s}: {total_images:3d} images")

    # Step 2: Initialize embedder
    print_section("Step 2: Initializing ArcFace Embedder")

    embedder = ArcFaceEmbedder(config)
    print(f"✓ Embedder loaded: {embedder}")

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

        # Stack into array [N, 512]
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
        print("❌ Error: Failed to extract embeddings")
        return

    print(f"✓ Extracted embeddings for {len(embeddings_per_person)} persons")

    # Step 4: Build FAISS index
    print_section("Step 4: Building FAISS Index")

    matcher = FaissMatcher(dimension=512)

    for person_name, embeddings in embeddings_per_person.items():
        matcher.add(person_name, embeddings)
        print(f"✓ Added '{person_name}': {embeddings.shape[0]} embeddings")

    print("Building index...")
    matcher.build()
    print(f"✓ FAISS index built: {matcher.index.ntotal} centroids")

    # Step 5: Save index
    print_section("Step 5: Saving FAISS Index")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    index_path = models_dir / "centroids.faiss"
    labels_path = models_dir / "labels.json"
    stats_path = models_dir / "stats.pkl"

    matcher.save(index_path, labels_path, stats_path)

    print(f"✓ Saved FAISS index to {index_path}")
    print(f"✓ Saved labels to {labels_path}")
    print(f"✓ Saved stats to {stats_path}")

    # Summary
    print_section("Index Built Successfully")
    print(f"Persons enrolled:    {len(matcher.labels)}")
    print(f"Total embeddings:    {sum(len(e) for e in embeddings_per_person.values())}")
    print(f"Centroids in index:  {matcher.index.ntotal}")
    print(f"Embedding dimension: {matcher.dimension}")
    print()
    print("✅ You can now run face recognition:")
    print("   python scripts/run_webcam.py")
    print("   python scripts/run_video.py --video path/to/video.mp4")
    print()

    logger.info("FAISS index built successfully")


if __name__ == "__main__":
    main()
