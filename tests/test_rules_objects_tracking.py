import pytest
from helpers import face, frame, obj_box, types
from pipeline_config import PipelineConfig
from rules.object_tracker import ObjectTracker, iou
from rules.objects import ProhibitedObjectRule
from observations import BoundingBox

CFG = PipelineConfig()


def feed(rule, script):
    """script: {second: [objects]}; every second from 0 to max is a frame."""
    out = []
    for t in range(0, max(script) + 1):
        out += rule.on_frame(frame(float(t), objects=script.get(t, [])))
    return out


def phone(conf=0.5, x=0.5, y=0.6):
    return obj_box("cell phone", conf, x, y)


# ---- tracker -----------------------------------------------------------------------------------------

def test_iou_of_identical_and_disjoint_boxes():
    a = BoundingBox(0.1, 0.1, 0.2, 0.2)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, BoundingBox(0.7, 0.7, 0.1, 0.1)) == 0.0


def test_the_same_object_in_the_next_frame_is_the_same_track():
    tr = ObjectTracker()
    first = tr.update(0.0, [phone(x=0.50)])
    second = tr.update(1.0, [phone(x=0.52)])
    assert first[0].id == second[0].id and second[0].frames_seen == 2


def test_a_moving_small_object_is_still_followed_even_with_little_overlap():
    tr = ObjectTracker()
    tr.update(0.0, [obj_box("cell phone", 0.6, 0.30, 0.5, 0.06, 0.09)])
    moved = tr.update(1.0, [obj_box("cell phone", 0.6, 0.38, 0.5, 0.06, 0.09)])  # a hand's width away, no overlap
    assert moved[0].frames_seen == 2


def test_two_phones_are_two_tracks_and_different_labels_never_merge():
    tr = ObjectTracker()
    seen = tr.update(0.0, [phone(x=0.1), phone(x=0.7), obj_box("book", 0.6, 0.1, 0.6)])
    assert len({t.id for t in seen}) == 3


def test_a_track_that_has_not_been_seen_for_a_while_is_forgotten_so_a_later_object_is_new():
    tr = ObjectTracker(max_gap_s=4.0)
    a = tr.update(0.0, [phone()])[0]
    b = tr.update(10.0, [phone()])[0]
    assert a.id != b.id


# ---- the rule ----------------------------------------------------------------------------------------

def test_one_low_confidence_sighting_is_a_ghost_and_raises_nothing():
    assert feed(ProhibitedObjectRule(CFG), {3: [phone(0.5)]}) == []


def test_a_single_very_confident_sighting_is_reported_at_once():
    ev = feed(ProhibitedObjectRule(CFG), {3: [phone(0.92)]})
    assert types(ev) == ["CELLPHONE_DETECTED"] and ev[0].severity == "HIGH"


def test_a_phone_seen_only_every_other_frame_is_still_confirmed():
    # Half hidden by a hand: the detector catches it in alternating frames, never two in a row.
    ev = feed(ProhibitedObjectRule(CFG), {1: [phone(0.4)], 3: [phone(0.42)], 5: [phone(0.4)]})
    assert types(ev)[0] == "CELLPHONE_DETECTED" and ev[0].severity == "HIGH" and ev[0].ts == 3.0  # confirmed at the second sighting


def test_two_weak_sightings_far_apart_are_not_enough():
    assert feed(ProhibitedObjectRule(CFG), {1: [phone(0.32)], 9: [phone(0.32)]}) == []


def test_two_sightings_in_different_places_are_two_ghosts_not_one_phone():
    assert feed(ProhibitedObjectRule(CFG), {1: [phone(0.4, x=0.05, y=0.05)], 2: [phone(0.4, x=0.85, y=0.85)]}) == []


def test_below_the_label_floor_is_ignored_entirely():
    assert feed(ProhibitedObjectRule(CFG), {t: [phone(0.2)] for t in range(10)}) == []


def test_a_phone_that_stays_escalates_to_critical_once():
    ev = feed(ProhibitedObjectRule(CFG), {t: [phone(0.55)] for t in range(12)})
    assert [e.severity for e in ev if e.alert_type == "CELLPHONE_DETECTED"] == ["HIGH", "CRITICAL"]
    assert ev[-1].details["sustained"] is True


def test_the_alert_says_where_the_phone_is_and_how_long_it_has_been_followed():
    ev = feed(ProhibitedObjectRule(CFG), {t: [phone(0.6)] for t in range(4)})
    d = ev[0].details
    assert d["object"] == "cell phone" and d["sightings"] >= 2 and set(d["box"]) == {"x", "y", "width", "height"}


def test_a_phone_held_to_the_face_is_marked_near_face_and_one_on_the_desk_is_not():
    near = ProhibitedObjectRule(CFG)
    f = face(area=0.12, cx=0.5, cy=0.4)  # face box height ~0.35
    ev = []
    for t in range(4):
        ev += near.on_frame(frame(float(t), faces=[f], objects=[obj_box("cell phone", 0.6, 0.55, 0.42)]))
    assert ev[0].details["near_face"] is True

    far = ProhibitedObjectRule(CFG)
    ev = []
    for t in range(4):
        ev += far.on_frame(frame(float(t), faces=[f], objects=[obj_box("cell phone", 0.6, 0.05, 0.85)]))
    assert ev[0].details["near_face"] is False


def test_it_repeats_only_after_the_cooldown():
    rule = ProhibitedObjectRule(CFG)
    ev = feed(rule, {t: [phone(0.6)] for t in range(0, 130)})
    assert types(ev).count("CELLPHONE_DETECTED") >= 3  # first, sustained, and a repeat after 60 s


def test_other_prohibited_labels_use_their_own_alert_type_and_severity():
    ev = feed(ProhibitedObjectRule(CFG), {t: [obj_box("book", 0.6, 0.2, 0.6)] for t in range(4)})
    assert types(ev) == ["PROHIBITED_OBJECT_DETECTED"] and ev[0].severity == "MEDIUM"


def test_labels_only_an_extra_model_knows_are_reported_with_their_own_severity():
    rule = ProhibitedObjectRule(CFG)
    ev = feed(rule, {t: [obj_box("earbuds", 0.6, 0.3, 0.4, 0.03, 0.03)] for t in range(4)})
    assert types(ev) == ["PROHIBITED_OBJECT_DETECTED"] and ev[0].severity == "HIGH" and ev[0].details["object"] == "earbuds"


def test_an_unknown_label_is_ignored():
    assert feed(ProhibitedObjectRule(CFG), {t: [obj_box("banana", 0.99, 0.2, 0.6)] for t in range(6)}) == []


def test_confirmed_boxes_are_available_for_the_evidence_frames():
    rule = ProhibitedObjectRule(CFG)
    feed(rule, {t: [phone(0.6)] for t in range(4)})
    assert [label for label, _ in rule.confirmed_boxes()] == ["cell phone"]
