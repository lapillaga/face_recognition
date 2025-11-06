"""Unit tests for enrollment service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from app.enrollment import EnrollmentService
from app.interfaces import BBox, Detection


@pytest.fixture
def mock_detector():
    """Create a mock detector."""
    detector = Mock()
    return detector


@pytest.fixture
def mock_aligner():
    """Create a mock aligner."""
    aligner = Mock()
    return aligner


@pytest.fixture
def enrollment_service(mock_detector, mock_aligner):
    """Create an enrollment service with mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        service = EnrollmentService(
            detector=mock_detector,
            aligner=mock_aligner,
            save_dir=tmpdir,
            min_sharpness=100.0,
            min_detection_score=0.5,
        )
        yield service


@pytest.fixture
def test_frame():
    """Create a test frame (640x480 RGB)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_aligned_face():
    """Create a test aligned face (112x112 RGB)."""
    # Create a sharp face (high Laplacian variance)
    face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Add some edges to increase sharpness
    face[50:60, :] = 255
    face[:, 50:60] = 0

    return face


def test_get_person_dir(enrollment_service):
    """Test getting person directory."""
    person_dir = enrollment_service._get_person_dir("test_person")

    assert person_dir.exists()
    assert person_dir.name == "TEST_PERSON"  # Should be uppercased


def test_process_frame_no_faces(enrollment_service, mock_detector, test_frame):
    """Test process_frame with no faces detected."""
    mock_detector.detect.return_value = []

    success, face, message = enrollment_service.process_frame(
        test_frame, "TEST", 0
    )

    assert not success
    assert face is None
    assert "No faces" in message


def test_process_frame_multiple_faces(enrollment_service, mock_detector, test_frame):
    """Test process_frame with multiple faces detected."""
    # Create two mock detections
    detection1 = Detection(
        bbox=BBox(10, 10, 100, 100),
        kps=np.array([[20, 30], [40, 30], [30, 50], [25, 70], [35, 70]], dtype=np.float32),
        score=0.9,
    )
    detection2 = Detection(
        bbox=BBox(200, 10, 290, 100),
        kps=np.array([[210, 30], [240, 30], [225, 50], [220, 70], [230, 70]], dtype=np.float32),
        score=0.8,
    )

    mock_detector.detect.return_value = [detection1, detection2]

    success, face, message = enrollment_service.process_frame(
        test_frame, "TEST", 0
    )

    assert not success
    assert face is None
    assert "Multiple" in message


def test_process_frame_low_confidence(enrollment_service, mock_detector, test_frame):
    """Test process_frame with low detection confidence."""
    detection = Detection(
        bbox=BBox(10, 10, 100, 100),
        kps=np.array([[20, 30], [40, 30], [30, 50], [25, 70], [35, 70]], dtype=np.float32),
        score=0.3,  # Below threshold (0.5)
    )

    mock_detector.detect.return_value = [detection]

    success, face, message = enrollment_service.process_frame(
        test_frame, "TEST", 0
    )

    assert not success
    assert face is None
    assert "Low detection confidence" in message


def test_process_frame_no_landmarks(enrollment_service, mock_detector, test_frame):
    """Test process_frame with no landmarks."""
    detection = Detection(
        bbox=BBox(10, 10, 100, 100),
        kps=None,  # No landmarks
        score=0.9,
    )

    mock_detector.detect.return_value = [detection]

    success, face, message = enrollment_service.process_frame(
        test_frame, "TEST", 0
    )

    assert not success
    assert face is None
    assert "No landmarks" in message


def test_process_frame_success(
    enrollment_service,
    mock_detector,
    mock_aligner,
    test_frame,
    test_aligned_face,
):
    """Test process_frame with successful capture."""
    detection = Detection(
        bbox=BBox(10, 10, 100, 100),
        kps=np.array([[20, 30], [40, 30], [30, 50], [25, 70], [35, 70]], dtype=np.float32),
        score=0.9,
    )

    mock_detector.detect.return_value = [detection]
    mock_aligner.align.return_value = test_aligned_face

    success, face, message = enrollment_service.process_frame(
        test_frame, "TEST", 0
    )

    assert success
    assert face is not None
    assert face.shape == (112, 112, 3)
    assert "Saved" in message

    # Check that aligner was called
    mock_aligner.align.assert_called_once()


def test_count_enrolled_images(enrollment_service):
    """Test counting enrolled images."""
    # Initially should be 0
    count = enrollment_service.count_enrolled_images("TEST")
    assert count == 0

    # Create a fake aligned face and save it
    face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    enrollment_service._save_aligned_face(face, "TEST", 0)

    # Should now be 1
    count = enrollment_service.count_enrolled_images("TEST")
    assert count == 1


def test_list_enrolled_persons(enrollment_service):
    """Test listing enrolled persons."""
    # Initially should be empty
    persons = enrollment_service.list_enrolled_persons()
    assert len(persons) == 0

    # Create directories for two persons
    enrollment_service._get_person_dir("ALICE")
    enrollment_service._get_person_dir("BOB")

    # Should now list both
    persons = enrollment_service.list_enrolled_persons()
    assert len(persons) == 2
    assert "ALICE" in persons
    assert "BOB" in persons


def test_delete_person(enrollment_service):
    """Test deleting a person's enrollment data."""
    # Create some enrollment data
    face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    enrollment_service._save_aligned_face(face, "TEST", 0)
    enrollment_service._save_aligned_face(face, "TEST", 1)

    # Verify data exists
    count = enrollment_service.count_enrolled_images("TEST")
    assert count == 2

    # Delete person
    result = enrollment_service.delete_person("TEST")
    assert result is True

    # Verify data is gone
    count = enrollment_service.count_enrolled_images("TEST")
    assert count == 0

    # Try deleting non-existent person
    result = enrollment_service.delete_person("NONEXISTENT")
    assert result is False


def test_repr(enrollment_service):
    """Test string representation."""
    repr_str = repr(enrollment_service)
    assert "EnrollmentService" in repr_str
    assert "save_dir" in repr_str
