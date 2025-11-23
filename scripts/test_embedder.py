#!/usr/bin/env python3
"""Test script for ArcFace embedder with webcam.

This script detects faces, aligns them, extracts embeddings, and computes
similarity metrics to verify embedding quality.

Usage:
    python scripts/test_embedder.py

Controls:
    ESC or 'q' - Quit
    SPACE - Pause/Resume
    's' - Save current embeddings with a label
    'c' - Show comparison matrix of saved embeddings

Verification Criteria:
    - Embeddings have shape (512,)
    - L2 norm ≈ 1.0 (normalized)
    - Same person: cosine similarity > 0.9
    - Different people: cosine similarity < 0.5
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.aligner_fivept import FivePointAligner
from app.config import get_config
from app.detector_scrfd import SCRFDDetector
from app.embedder_arcface import ArcFaceEmbedder
from app.logging_config import setup_logging
from app.overlay import draw_detections, draw_fps, draw_info_panel, draw_text
from app.utils import compute_cosine_similarity, is_sharp_enough

logger = setup_logging(__name__)


class EmbeddingStore:
    """Store and compare embeddings with labels."""

    def __init__(self):
        """Initialize empty embedding store."""
        self.embeddings: Dict[str, List[np.ndarray]] = {}

    def add(self, label: str, embedding: np.ndarray) -> None:
        """Add embedding with label.

        Args:
            label: Person name or identifier
            embedding: 512-D embedding vector
        """
        if label not in self.embeddings:
            self.embeddings[label] = []

        self.embeddings[label].append(embedding.copy())
        logger.info(f"Added embedding for '{label}' (total: {len(self.embeddings[label])})")

    def get_average(self, label: str) -> np.ndarray | None:
        """Get average embedding for a label.

        Args:
            label: Person name

        Returns:
            Average L2-normalized embedding, or None if label not found.
        """
        if label not in self.embeddings or not self.embeddings[label]:
            return None

        # Average and re-normalize
        avg = np.mean(self.embeddings[label], axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-10)

        return avg

    def compute_similarity_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Compute pairwise similarity matrix between all labels.

        Returns:
            Similarity matrix (NxN) and list of labels.
        """
        labels = sorted(self.embeddings.keys())

        if not labels:
            return np.array([]), []

        # Get average embeddings for each label
        avg_embeddings = [self.get_average(label) for label in labels]

        # Compute pairwise similarities
        n = len(labels)
        sim_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if avg_embeddings[i] is not None and avg_embeddings[j] is not None:
                    sim = compute_cosine_similarity(avg_embeddings[i], avg_embeddings[j])
                    sim_matrix[i, j] = sim

        return sim_matrix, labels


def draw_embedding_info(
    frame: np.ndarray,
    embeddings: list[np.ndarray],
    position: tuple[int, int] = (10, 100),
) -> None:
    """Draw embedding statistics on frame.

    Args:
        frame: Frame to draw on
        embeddings: List of embeddings extracted this frame
        position: Top-left position for info panel
    """
    if not embeddings:
        info = {
            "Embeddings": 0,
            "Avg Norm": "N/A",
            "Dimensions": "N/A",
        }
    else:
        norms = [np.linalg.norm(emb) for emb in embeddings]
        avg_norm = np.mean(norms)

        info = {
            "Embeddings": len(embeddings),
            "Avg Norm": f"{avg_norm:.4f}",
            "Dimensions": embeddings[0].shape[0],
        }

    draw_info_panel(frame, info, position=position)


def create_similarity_visualization(
    sim_matrix: np.ndarray,
    labels: list[str],
    size: int = 400,
) -> np.ndarray:
    """Create heatmap visualization of similarity matrix.

    Args:
        sim_matrix: NxN similarity matrix
        labels: List of N labels
        size: Output image size

    Returns:
        Heatmap image with labels.
    """
    if sim_matrix.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    n = len(labels)

    # Create heatmap
    # Map similarity [0,1] to color (0=blue, 1=red)
    heatmap = np.zeros((n, n, 3), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            sim = sim_matrix[i, j]
            # Color mapping: blue (low) -> green (mid) -> red (high)
            if sim < 0.5:
                # Blue to cyan
                heatmap[i, j] = [int(255 * (1 - sim * 2)), 0, int(255 * sim * 2)]
            else:
                # Cyan to red
                heatmap[i, j] = [0, int(255 * (2 - sim * 2)), int(255 * ((sim - 0.5) * 2))]

    # Resize to target size
    cell_size = size // max(n, 1)
    heatmap_large = cv2.resize(
        heatmap,
        (cell_size * n, cell_size * n),
        interpolation=cv2.INTER_NEAREST,
    )

    # Add padding for labels
    vis = np.zeros((size, size, 3), dtype=np.uint8)
    vis[:heatmap_large.shape[0], :heatmap_large.shape[1]] = heatmap_large

    # Draw labels and values
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255)

    for i in range(n):
        # Row labels
        cv2.putText(
            vis,
            labels[i][:10],  # Truncate long labels
            (5, i * cell_size + cell_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

        # Column labels
        cv2.putText(
            vis,
            labels[i][:10],
            (i * cell_size + 5, heatmap_large.shape[0] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

        # Similarity values
        for j in range(n):
            sim_text = f"{sim_matrix[i, j]:.2f}"
            text_size = cv2.getTextSize(
                sim_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )[0]

            # Center text in cell
            text_x = i * cell_size + (cell_size - text_size[0]) // 2
            text_y = j * cell_size + (cell_size + text_size[1]) // 2

            cv2.putText(
                vis,
                sim_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return vis


def main() -> None:
    """Main function to test embedder with webcam."""
    logger.info("Starting embedder test with webcam...")

    # Load configuration
    config = get_config()

    # Initialize components
    logger.info("Initializing detector, aligner, and embedder...")
    try:
        detector = SCRFDDetector(config)
        aligner = FivePointAligner()
        embedder = ArcFaceEmbedder(config)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return

    # Open webcam
    logger.info(f"Opening camera {config.camera_id}...")
    cap = cv2.VideoCapture(config.camera_id)

    if not cap.isOpened():
        logger.error(f"Failed to open camera {config.camera_id}")
        return

    # Get camera properties
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS))

    logger.info(f"Camera opened: {cam_width}x{cam_height} @ {cam_fps}fps")

    # FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    paused = False

    # Embedding store
    store = EmbeddingStore()

    # Stats
    total_embeddings = 0

    logger.info(
        "Controls: ESC/q (quit), SPACE (pause), 's' (save), 'c' (compare)"
    )

    try:
        while True:
            if not paused:
                # Read frame
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                # Detect faces
                detections = detector.detect(frame)

                # Process each detection
                current_embeddings = []

                for det in detections:
                    if det.kps is None:
                        continue

                    try:
                        # Align face
                        aligned = aligner.align(frame, det.kps)

                        # Check quality
                        if not is_sharp_enough(aligned, threshold=config.min_sharpness):
                            logger.debug("Skipping blurry face")
                            continue

                        # Extract embedding
                        embedding = embedder.embed(aligned)
                        current_embeddings.append(embedding)

                        total_embeddings += 1

                        # Draw embedding norm
                        norm = np.linalg.norm(embedding)
                        norm_text = f"Norm: {norm:.4f}"
                        color = (0, 255, 0) if abs(norm - 1.0) < 0.01 else (0, 165, 255)

                        draw_text(
                            frame,
                            norm_text,
                            (det.bbox.x1, det.bbox.y2 + 20),
                            color=color,
                            font_scale=0.5,
                            bg_color=(0, 0, 0),
                        )

                    except Exception as e:
                        logger.warning(f"Failed to process face: {e}")
                        continue

                # Draw detections
                draw_detections(
                    frame,
                    detections,
                    show_landmarks=True,
                    show_score=True,
                )

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed

                # Draw FPS
                draw_fps(frame, fps)

                # Draw embedding info
                draw_embedding_info(frame, current_embeddings, position=(10, 90))

                # Draw store info
                store_info = {
                    "Saved Labels": len(store.embeddings),
                    "Total Saved": sum(len(embs) for embs in store.embeddings.values()),
                }
                draw_info_panel(frame, store_info, position=(10, 180))

                # Log periodically
                if frame_count % 30 == 0:
                    logger.debug(
                        f"Frame {frame_count}: {len(current_embeddings)} embeddings, "
                        f"FPS: {fps:.1f}"
                    )

            # Display main frame
            if config.display:
                cv2.imshow("Embedder Test", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):  # ESC or 'q'
                logger.info("Quit requested by user")
                break

            elif key == ord(" "):  # SPACE
                paused = not paused
                status = "paused" if paused else "resumed"
                logger.info(f"Playback {status}")

            elif key == ord("s"):  # 's' - Save embeddings
                if current_embeddings:
                    # Prompt for label (simple version - use first embedding)
                    label = input("Enter label for current face(s): ").strip()

                    if label:
                        for emb in current_embeddings:
                            store.add(label, emb)

                        logger.info(f"Saved {len(current_embeddings)} embedding(s) as '{label}'")
                    else:
                        logger.warning("No label provided, skipping save")
                else:
                    logger.warning("No embeddings to save in current frame")

            elif key == ord("c"):  # 'c' - Compare embeddings
                if len(store.embeddings) >= 2:
                    sim_matrix, labels = store.compute_similarity_matrix()

                    logger.info("Similarity Matrix:")
                    logger.info(f"Labels: {labels}")
                    logger.info(f"\n{sim_matrix}")

                    # Show visualization
                    if config.display:
                        vis = create_similarity_visualization(sim_matrix, labels)
                        cv2.imshow("Similarity Matrix", vis)
                        cv2.waitKey(3000)  # Show for 3 seconds

                    # Validate criteria
                    n = len(labels)
                    for i in range(n):
                        for j in range(n):
                            sim = sim_matrix[i, j]
                            if i == j:
                                # Same person (diagonal)
                                if sim < 0.99:
                                    logger.warning(
                                        f"Intra-person similarity {labels[i]}: {sim:.4f} < 0.99"
                                    )
                            else:
                                # Different people (off-diagonal)
                                if sim > 0.5:
                                    logger.warning(
                                        f"Inter-person similarity {labels[i]}-{labels[j]}: "
                                        f"{sim:.4f} > 0.5 (too similar!)"
                                    )
                                else:
                                    logger.info(
                                        f"Inter-person similarity {labels[i]}-{labels[j]}: "
                                        f"{sim:.4f} OK"
                                    )

                else:
                    logger.warning("Need at least 2 labels to compare. Use 's' to save embeddings.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        logger.info(
            f"Processed {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.1f})"
        )
        logger.info(f"Extracted {total_embeddings} embeddings")

        # Final comparison
        if len(store.embeddings) >= 2:
            sim_matrix, labels = store.compute_similarity_matrix()
            logger.info("\n=== Final Similarity Matrix ===")
            logger.info(f"Labels: {labels}")
            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels):
                    logger.info(f"{label_i} <-> {label_j}: {sim_matrix[i, j]:.4f}")


if __name__ == "__main__":
    main()