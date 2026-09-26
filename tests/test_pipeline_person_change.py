from helpers import face, frame, frame_with_embedding, person, sighting, types
from pipeline import ProctorPipeline
from pipeline_config import PipelineConfig

CFG = PipelineConfig()
A, B = person(11), person(12)


def run(pipeline, start, seconds, who=None, seed0=0, ref_sim=None):
    """`who=None` is an empty chair. Identity is scored every other second, like the service does."""
    out = []
    for i in range(seconds):
        t = float(start + i)
        if who is None:
            f = frame(t, faces=[])
        elif i % 2 == 0:
            f = frame_with_embedding(t, sighting(who, 0.3, seed0 + i), ref_sim=ref_sim)
        else:
            f = frame(t)
        out += pipeline.on_frame(f)
    return out


def test_someone_else_sitting_down_after_the_candidate_left_is_caught_quickly():
    p = ProctorPipeline(CFG)
    assert types(run(p, 0, 120, A)) == []
    leave = run(p, 120, 40, None)
    assert "FACE_ABSENT" in types(leave)
    back = run(p, 160, 20, B, seed0=500)
    assert "FACE_RETURNED" in types(back)
    changed = [e for e in back if e.alert_type == "PERSON_CHANGED"]
    assert len(changed) == 1
    assert changed[0].details["after_absence_seconds"] >= 30
    # Quickly: within a handful of seconds of returning, not after a long run.
    assert changed[0].ts - 160 <= 12


def test_the_candidate_coming_back_from_the_bathroom_is_not_a_change():
    p = ProctorPipeline(CFG)
    run(p, 0, 120, A)
    run(p, 120, 60, None)
    back = run(p, 180, 60, A, seed0=700)
    assert "FACE_RETURNED" in types(back) and "PERSON_CHANGED" not in types(back)


def test_a_swap_is_reported_once_with_the_checkin_reference_too():
    p = ProctorPipeline(CFG)
    run(p, 0, 120, A, ref_sim=0.6)
    ev = run(p, 120, 40, B, seed0=500, ref_sim=0.1)
    assert types([e for e in ev if e.alert_type in ("PERSON_CHANGED", "IDENTITY_MISMATCH")]) == ["IDENTITY_MISMATCH"]
    assert [e for e in ev if e.alert_type == "IDENTITY_MISMATCH"][0].details["changed_during_exam"] is True


def test_the_alert_asks_for_evidence_and_never_carries_a_made_up_confidence():
    p = ProctorPipeline(CFG)
    run(p, 0, 120, A)
    ev = [e for e in run(p, 120, 40, B, seed0=500) if e.alert_type == "PERSON_CHANGED"][0]
    assert ev.evidence and ev.confidence is None and ev.severity == "HIGH"


def test_a_covered_camera_pauses_identity_checks_like_every_other_rule():
    from observations import FrameQuality
    p = ProctorPipeline(CFG)
    run(p, 0, 120, A)
    dark = lambda t: frame(t, faces=[], quality=FrameQuality(mean_luma=5, luma_std=2, sharpness=1, motion=1))
    out = []
    for i in range(30):
        out += p.on_frame(dark(float(120 + i)))
    assert "PERSON_CHANGED" not in types(out)
