from helpers import face, frame, obj, run_frames, types
from pipeline_config import PipelineConfig
from rules.presence import FacePresenceRule, FramingRule, MultiplePersonsRule

CFG = PipelineConfig()


def test_three_seconds_out_of_frame_is_not_reported():
    rule = FacePresenceRule(CFG)
    ev = run_frames(rule, 0, 5, lambda t: frame(t))
    ev += run_frames(rule, 5, 3, lambda t: frame(t, faces=[]))
    ev += run_frames(rule, 8, 6, lambda t: frame(t))
    assert ev == []


def test_prolonged_absence_reports_once_then_returns_with_duration():
    rule = FacePresenceRule(CFG)
    ev = run_frames(rule, 0, 3, lambda t: frame(t))
    ev += run_frames(rule, 3, 20, lambda t: frame(t, faces=[]))
    ev += run_frames(rule, 23, 5, lambda t: frame(t))
    assert types(ev) == ["FACE_ABSENT", "FACE_RETURNED"]
    assert ev[0].severity == "MEDIUM" and ev[0].evidence
    assert ev[1].details["absent_seconds"] >= 15
    assert ev[1].severity in ("MEDIUM", "HIGH")


def test_repeatedly_leaving_the_frame_is_flagged():
    rule = FacePresenceRule(CFG)
    ev, t = [], 0
    for _ in range(3):
        ev += run_frames(rule, t, 4, lambda x: frame(x))
        ev += run_frames(rule, t + 4, 10, lambda x: frame(x, faces=[]))
        t += 14
    ev += run_frames(rule, t, 4, lambda x: frame(x))
    assert "FREQUENT_FRAME_EXIT" in types(ev)
    assert types(ev).count("FREQUENT_FRAME_EXIT") == 1


def test_second_face_of_comparable_size_raises_multiple_persons():
    rule = MultiplePersonsRule(CFG)
    ev = run_frames(rule, 0, 8, lambda t: frame(t, faces=[face(0.12), face(0.10, cx=0.85)]))
    assert types(ev) == ["MULTIPLE_PERSONS"]
    assert ev[0].details == {"person_count": 2, "source": "face"}


def test_tiny_background_face_such_as_a_poster_is_ignored():
    rule = MultiplePersonsRule(CFG)
    ev = run_frames(rule, 0, 10, lambda t: frame(t, faces=[face(0.12), face(0.004, cx=0.9, cy=0.1)]))
    assert ev == []


def test_single_frame_second_face_is_ignored():
    rule = MultiplePersonsRule(CFG)
    ev = run_frames(rule, 0, 10, lambda t: frame(
        t, faces=[face(0.12), face(0.1, cx=0.85)] if t == 4 else [face(0.12)]))
    assert ev == []


def test_person_with_turned_away_face_is_caught_by_body_detection():
    rule = MultiplePersonsRule(CFG)
    ev = run_frames(rule, 0, 8, lambda t: frame(
        t, faces=[face(0.12)], objects=[obj("person", 0.9, 0.2), obj("person", 0.8, 0.15)]))
    assert types(ev) == ["MULTIPLE_PERSONS"]
    assert ev[0].details["source"] == "body"


def test_low_confidence_or_tiny_person_boxes_do_not_count():
    rule = MultiplePersonsRule(CFG)
    ev = run_frames(rule, 0, 10, lambda t: frame(
        t, objects=[obj("person", 0.9, 0.2), obj("person", 0.3, 0.2), obj("person", 0.9, 0.005)]))
    assert ev == []


def test_face_too_far_reported_once_with_cooldown():
    rule = FramingRule(CFG)
    ev = run_frames(rule, 0, 40, lambda t: frame(t, faces=[face(0.01)]))
    assert types(ev) == ["FACE_POSITION_ISSUE"]
    assert ev[0].details["reason"] == "too_far" and ev[0].severity == "LOW"


def test_face_partly_out_of_frame_is_reported():
    rule = FramingRule(CFG)
    ev = run_frames(rule, 0, 15, lambda t: frame(t, faces=[face(0.12, cx=0.98)]))
    assert [e.details["reason"] for e in ev] == ["partially_out_of_frame"]


def test_well_framed_face_produces_nothing():
    rule = FramingRule(CFG)
    assert run_frames(rule, 0, 60, lambda t: frame(t)) == []
