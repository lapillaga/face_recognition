
# Claude Code Prompt — Modern Face Recognition Pipeline

> **Objective**  
Implement a **modern, production‑oriented face recognition pipeline** in Python using **object‑oriented design**, clean architecture, and best practices (PEP 8 style, type hints, docstrings, tests, linting). The goal is to support **real‑time identification** (“This is LUIS”) via webcam or video file, and to keep the code **modular and scalable** so we can later add attendance (tracking + rules), liveness, and other features.

---

## Hard Requirements (Read Carefully)

1. Please carefully read all details below.
2. Use **InsightFace** to access **SCRFD (detector + 5 landmarks)** and **ArcFace (512‑D embeddings)** via `insightface.app.FaceAnalysis`, or via separate detector/recognition components as needed.
3. Use **FAISS** for nearest‑neighbor search over enrolled identities.
4. Provide: **webcam mode** and **video‑file mode** (OpenCV), with on‑screen bounding boxes and predicted names (or `unknown`).
5. Keep the pipeline **modular**: we must be able to swap the detector in the future without rewriting embeddings/matching logic.
6. Produce **clean, well‑tested, OOP** Python code, with **type hints**, **docstrings**, and **logging**. Follow PEP 8 conventions.
7. Organize the project for **future attendance** features (tracking, cooldown, event logging), but **in this task only implement detection + identification**.
8. Use **`uv`** for environment management and dependency installation (document commands).

---

## Tech Stack

- Python 3.12+
- [InsightFace](https://github.com/deepinsight/insightface) — use **SCRFD** detector and **ArcFace** embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — vector search (CPU; GPU optional)
- OpenCV (video I/O and display)
- `python-dotenv` for config
- Testing: `pytest`
- Lint/Format: `ruff`, `black`
- Type checking: `mypy` (optional but preferred)

---

## Project Structure (Create Exactly)

```
metodos_final/
  app/
    __init__.py
    config.py
    interfaces.py          # ABCs for Detector, Aligner, Embedder, Matcher, VideoSource
    detector_scrfd.py      # Detector & landmarks via SCRFD (InsightFace)
    aligner_fivept.py      # 5-point alignment
    embedder_arcface.py    # ArcFace embedder (InsightFace recognition model)
    matcher_faiss.py       # FAISS index (build/search)
    enrollment.py          # Enrollment service (capture, encode, persist)
    recognition.py         # Recognition service (image/video inference)
    video_io.py            # Webcam/RTSP/VideoFile sources
    overlay.py             # Drawing boxes/labels on frames
    utils.py               # Helpers: quality checks, normalization, image ops
    logging_config.py      # Structured logging configuration
  scripts/
    capture_enroll.py      # CLI: capture aligned faces per person
    build_index.py         # CLI: extract embeddings, build FAISS index
    run_webcam.py          # CLI: live recognition via webcam
    run_video.py           # CLI: recognition on a video file
  tests/
    test_embeds.py
    test_matcher.py
    test_enrollment.py
  data/
    enroll/                # Per-person aligned crops (112x112) for enrollment
    val/                   # Optional validation images for threshold tuning
  models/
    centroids.faiss        # (after build)
    labels.json            # (after build)
    stats.pkl              # (after build)
  .env.example
  pyproject.toml
  README.md
```

> **Note**: Future modules (not implemented now) could include `tracking/` (ByteTrack wrapper), `liveness/`, `api/` (FastAPI), `storage/` (DB), etc.

---

## Configuration

Create `.env` from `.env.example`:

```
# .env.example
CTX_ID=-1        # -1 = CPU, 0 = GPU
THRESH=0.35      # cosine similarity threshold for a positive match
CAMERA_ID=0      # default webcam
DISPLAY=1        # 1=show window, 0=headless
```

---

## OOP Design (Key Interfaces)

Create `app/interfaces.py` with these abstract interfaces (and dataclasses). Keep signatures stable.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class Detection:
    bbox: BBox
    kps: Optional[np.ndarray]  # shape (5, 2) absolute pixel coords
    score: float

@runtime_checkable
class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]: ...

@runtime_checkable
class Aligner(Protocol):
    def align(self, frame_bgr: np.ndarray, kps_5pt: np.ndarray) -> np.ndarray: ...  # returns aligned 112x112 BGR

@runtime_checkable
class Embedder(Protocol):
    def embed(self, face_bgr_112: np.ndarray) -> np.ndarray: ...  # (512,) float32 normalized

@runtime_checkable
class Matcher(Protocol):
    def add(self, label: str, embeddings: np.ndarray) -> None: ...
    def build(self) -> None: ...
    def search(self, embedding: np.ndarray, topk: int = 1) -> Tuple[List[str], List[float]]: ...
```

### Implementation Notes
- **Detector**: `detector_scrfd.py` uses InsightFace SCRFD via `FaceAnalysis` to produce `Detection` objects (bbox + 5 landmarks).  
- **Aligner**: `aligner_fivept.py` – perform 5‑point similarity transform to 112×112 (use InsightFace utilities or your own matrix code).  
- **Embedder**: `embedder_arcface.py` – call InsightFace recognition model to produce a 512‑D L2‑normalized vector.  
- **Matcher**: `matcher_faiss.py` – build FAISS index over **class centroids** (per‑label mean) or all embeddings; support cosine similarity (normalize L2 and use inner product).  

Document each class with docstrings and type hints.

---

## Enrollment Flow

- **Goal**: capture 12–20 aligned face crops per person (112×112), compute embeddings, and build FAISS index & `labels.json`.
- **Quality**: apply a simple sharpness check (Laplacian variance) and discard blurry crops.
- **Files**: store aligned crops under `data/enroll/<PersonName>/*.jpg`.

### CLI: `scripts/capture_enroll.py`
- Open webcam (`CAMERA_ID`), run detector → if exactly **one** face with good quality, align and save (`N` target images).  
- Show a preview with a guide box; allow `ESC` to quit.

### CLI: `scripts/build_index.py`
- Load all aligned images from `data/enroll/**`  
- Compute ArcFace embeddings  
- Compute per‑label centroid(s)  
- Build a FAISS index (inner‑product after L2‑norm)  
- Persist `models/centroids.faiss`, `models/labels.json`, `models/stats.pkl`

---

## Recognition Flow

### CLI: `scripts/run_webcam.py`
- Read `THRESH`, `CTX_ID` from `.env`  
- For each frame: detect → align → embed → FAISS search → draw bbox + label (if sim ≥ `THRESH`, else `unknown`)  
- Print FPS to console; show window if `DISPLAY=1`

### CLI: `scripts/run_video.py <path>`
- Same as webcam but reading from a video file.

---

## Core Implementations (Guidance)

### `detector_scrfd.py`
- Initialize `insightface.app.FaceAnalysis` with `name="buffalo_l"` and `ctx_id` from env.  
- In `detect(frame)`, call `app.get(frame)` and convert results into `Detection` with 5 keypoints and a confidence score.  
- Ensure consistent `BBox` integer rounding; clamp to image bounds.

### `aligner_fivept.py`
- Use the 5‑point landmarks to warp to 112×112 (InsightFace’s `norm_crop` or equivalent).  
- Validate input shapes; raise a clear exception if `kps` is missing.

### `embedder_arcface.py`
- Reuse the `FaceAnalysis` recognition model or load a dedicated ArcFace model.  
- Ensure returned vectors are `np.float32`, L2‑normalized.

### `matcher_faiss.py`
- Build a label list (ordered) and a FAISS index on the centroids.  
- Provide `search(embedding)` returning `(labels, scores)` sorted by similarity.  
- Support loading/saving the index and labels.

---

## Dependency & Environment (with `uv`)

```bash
# Create project & install deps
uv init face_attendance && cd face_attendance

uv add opencv-python insightface faiss-cpu python-dotenv numpy
uv add pytest ruff black mypy  # dev tools

# Example runtime flags
export CTX_ID=-1
export THRESH=0.35
```

**Windows note**: If FAISS wheels are an issue, fall back to `faiss-cpu` from conda-forge or prebuilt wheels; document for the user.

---

## Linting / Formatting / Tests

- Add `pyproject.toml` with `ruff` + `black` settings; run `ruff check .` and `black .` in CI.
- Use `pytest` for unit tests on: embedding shape/normalization, FAISS search, enrollment pipeline (mock frames).

**Example `tests/test_matcher.py`:**
```python
import numpy as np
from app.matcher_faiss import FaissMatcher

def test_matcher_basic():
    m = FaissMatcher()
    m.add("Alice", np.stack([np.ones(512, dtype=np.float32)]))
    m.add("Bob",   np.stack([-np.ones(512, dtype=np.float32)]))
    m.build()

    q = np.ones(512, dtype=np.float32)
    labels, scores = m.search(q, topk=1)
    assert labels[0] == "Alice"
    assert scores[0] > 0.9
```

---

## Logging & Error Handling

- Configure `logging` with timestamps and module names (`logging_config.py`).
- Fail fast with clear messages if: no camera, empty enrollment set, FAISS index not found, etc.

---

## Performance Hints

- Normalize embeddings once.  
- Use **vectorized** operations (`np.stack`, `faiss.normalize_L2`).  
- If GPU available, switch `CTX_ID=0` (and optionally use `onnxruntime-gpu` in a future iteration).  
- Avoid expensive per‑frame I/O; pre‑load FAISS index & labels.

---

## Data & Privacy Notes

- Store only aligned crops and embeddings locally.  
- Provide a way to **delete** a person’s data (`scripts` clean command).  
- Document retention policy if deployed.

---

## Deliverables (for this task)

1. All modules under `app/` implemented with type hints, docstrings, and logging.  
2. CLI scripts under `scripts/` working end‑to‑end (enroll → build index → webcam/video recognition).  
3. Tests under `tests/` with at least basic coverage for matcher & embedding.  
4. `README.md` explaining setup, commands, and troubleshooting.  
5. `.env.example` filled as above; `.env` ignored by VCS.

---

## Acceptance Tests (Manual)

1. **Enrollment**: capture 12–20 images of “Luis”.  
2. **Build index**: produces `models/centroids.faiss` and `labels.json`.  
3. **Webcam**: shows “Luis {score}” when Luis appears; “unknown” otherwise.  
4. **Video**: labels faces frame‑by‑frame in a sample clip.  
5. **Threshold tuning**: changing `THRESH` affects unknown/positive trade‑off.

---

## Future (Not in This Task)

- **Attendance**: integrate a tracker (ByteTrack) with cooldown and event persistence.  
- **Liveness**: passive RGB anti‑spoof model before confirming identity.  
- **Serving**: expose as a FastAPI service and/or migrate to ONNX/TensorRT/Triton for multi‑stream throughput.

---

## Tutorial Variant (Later)

We will add a parallel “academic” path: same architecture but **replace the embedder** with `face_recognition` (dlib 128‑D) and a basic file‑based store — without changing Detector/Aligner/Matcher interfaces. The codebase remains modular, and both paths can be toggled by a config flag.

---

**Now, Claude:**
- Scaffold the project structure above.  
- Implement all modules and scripts as specified.  
- Keep code in **English**, follow **PEP 8**, add **docstrings** and **type hints**.  
- Provide a concise `README.md` with run instructions and screenshots.  
- Make minimal platform‑specific assumptions; prefer portable code.
