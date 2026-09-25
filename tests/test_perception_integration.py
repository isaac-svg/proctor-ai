"""
Scenario tests through the whole stack (real MediaPipe / YOLO / YuNet+SFace /
Silero) using real face photos.

The photos are not checked in -- provide them yourself:

    PROCTOR_TEST_FACES_DIR=/path/to/dir pytest tests/test_perception_integration.py

where the directory holds `person_a.jpg` and `person_b.jpg` (two *different*
people, face filling a good part of a 4:3 frame). Skipped when unset, and
skipped if the heavy dependencies aren't installed.
"""

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("mediapipe")
pytest.importorskip("ultralytics")
pytest.importorskip("silero_vad")

FACES = os.environ.get("PROCTOR_TEST_FACES_DIR")
pytestmark = pytest.mark.skipif(not FACES, reason="set PROCTOR_TEST_FACES_DIR to run perception scenarios")


def load(name):
    img = cv2.imread(str(Path(FACES) / name))
    assert img is not None, f"missing {name} in PROCTOR_TEST_FACES_DIR"
    return cv2.resize(img, (640, 480))


def empty_room():
    """A plausible backdrop with structure (gradient, shelves, a window) but
    no person -- unlike blurred noise, which is nearly featureless."""
    yy, xx = np.mgrid[0:480, 0:640]
    room = np.stack([90 + xx * 0.08, 110 + yy * 0.05, 130 + (xx + yy) * 0.02], axis=-1).astype(np.uint8)
    cv2.rectangle(room, (40, 60), (220, 200), (200, 210, 230), -1)   # window
    cv2.rectangle(room, (300, 250), (600, 262), (40, 60, 90), -1)     # shelf
    for i in range(6):
        cv2.rectangle(room, (310 + i * 45, 200), (340 + i * 45, 250), (30 * i + 40, 90, 160 - 15 * i), -1)
    return cv2.GaussianBlur(room, (0, 0), 0.8)


def jpeg(img, seed):
    """Add tiny per-frame sensor noise so the feed doesn't read as 'frozen'."""
    rng = np.random.default_rng(seed)
    noisy = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def shared():
    from session_manager import SharedModels

    s = SharedModels()
    s.warm_up()
    return s


@pytest.fixture
def session(shared):
    from session_manager import ProctorSession

    clock = Clock()
    s = ProctorSession("scenario", shared, clock=clock)
    yield s, clock
    s.close()


def play(session_clock, img, seconds, seed=0):
    session, clock = session_clock
    out = []
    for i in range(seconds):
        clock.t += 1.0
        out += session.on_video_frame(jpeg(img, seed + i))
    return out


def kinds(emissions):
    return [e.event.alert_type for e in emissions]


def test_a_calm_exam_with_the_enrolled_person_raises_nothing(session):
    s, _ = session
    a = load("person_a.jpg")
    assert s.enroll([jpeg(a, 1), jpeg(a, 2), jpeg(a, 3)]).ok
    got = play(session, a, 30)
    # The sample portrait's subject isn't looking straight into the lens, so a
    # LOW gaze advisory is legitimate; nothing at MEDIUM or above may fire.
    assert [e.event.alert_type for e in got if e.event.severity != "LOW"] == []
    assert s.pipeline.identity.last_similarity is not None and s.pipeline.identity.last_similarity > 0.6


def test_a_different_person_taking_over_is_flagged_critical_with_evidence(session):
    s, _ = session
    a, b = load("person_a.jpg"), load("person_b.jpg")
    assert s.enroll([jpeg(a, 1), jpeg(a, 2), jpeg(a, 3)]).ok
    play(session, a, 10)
    swapped = play(session, b, 20, seed=100)
    mismatch = [e for e in swapped if e.event.alert_type == "IDENTITY_MISMATCH"]
    assert len(mismatch) == 1
    assert mismatch[0].event.severity == "CRITICAL"
    assert mismatch[0].clip and mismatch[0].clip["frames"], "expected an evidence clip"


def test_two_people_in_frame_is_flagged(session):
    a = load("person_a.jpg")
    both = np.hstack([cv2.resize(a, (320, 240)), cv2.resize(a, (320, 240))])
    both = cv2.resize(np.vstack([both, both * 0]), (640, 480))  # two faces side by side
    assert "MULTIPLE_PERSONS" in kinds(play(session, both, 12))


def test_a_covered_lens_raises_camera_covered_and_nothing_else(session):
    dark = np.full((480, 640, 3), 6, np.uint8)
    got = kinds(play(session, dark, 20))
    assert got == ["CAMERA_COVERED"], got  # not also FROZEN / FACE_ABSENT


def test_an_empty_chair_raises_face_absent_then_face_returned(session):
    a = load("person_a.jpg")
    room = empty_room()
    got = kinds(play(session, a, 5)) + kinds(play(session, room, 12)) + kinds(play(session, a, 6))
    got = [k for k in got if k != "GAZE_AWAY"]  # the sample portrait isn't looking at the lens
    assert got == ["FACE_ABSENT", "FACE_RETURNED"], got


def test_a_still_frame_repeated_is_flagged_as_frozen(session):
    a = load("person_a.jpg")
    frozen = jpeg(a, 7)
    s, clock = session
    out = []
    for _ in range(20):
        clock.t += 1.0
        out += s.on_video_frame(frozen)
    assert "CAMERA_FROZEN" in kinds(out)
