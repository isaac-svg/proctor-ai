from helpers import audio, face, frame, run_frames, types
from observations import FrameQuality
from pipeline import ProctorPipeline

dark = lambda t: frame(t, faces=[], quality=FrameQuality(mean_luma=5, luma_std=2, sharpness=1, motion=2))


def run(p, start, n, make):
    out = []
    for i in range(n):
        out += p.on_frame(make(start + i))
    return out


def test_a_covered_camera_raises_one_alert_not_three():
    p = ProctorPipeline()
    ev = run(p, 0, 5, lambda t: frame(t)) + run(p, 5, 40, dark)
    assert types(ev) == ["CAMERA_COVERED"]


def test_rules_resume_after_the_lens_is_uncovered():
    p = ProctorPipeline()
    ev = run(p, 0, 20, dark) + run(p, 20, 10, lambda t: frame(t)) + run(p, 30, 20, lambda t: frame(t, faces=[]))
    assert "CAMERA_COVERED" in types(ev) and "FACE_ABSENT" in types(ev)


def test_a_calm_exam_is_silent():
    p = ProctorPipeline()
    ev = run(p, 0, 600, lambda t: frame(t, quality=FrameQuality(motion=2.0)))
    for t in range(600):
        ev += p.on_audio(audio(t, dbfs=-55))
    assert ev == []


def test_snapshot_shape_matches_the_ai_analysis_contract():
    p = ProctorPipeline()
    p.on_frame(frame(0, faces=[face(yaw=3, pitch=1)]))
    s = p.snapshot()
    assert s["faces_detected"] == 1 and s["gaze_direction"] == "CENTER"
    assert set(s["head_pose"]) == {"yaw", "pitch", "roll"} and 0 <= s["risk_score"] <= 1


def test_snapshot_with_no_face():
    p = ProctorPipeline()
    p.on_frame(frame(0, faces=[]))
    assert p.snapshot()["faces_detected"] == 0


def test_tick_reports_feeds_going_quiet():
    p = ProctorPipeline()
    p.on_frame(frame(0))
    p.on_audio(audio(0))
    assert sorted(types(p.on_tick(30))) == ["AUDIO_FEED_LOST", "CAMERA_FEED_LOST"]
