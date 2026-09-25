from helpers import face, frame, obj, run_frames, types
from observations import FrameQuality
from pipeline_config import PipelineConfig
from rules.camera import CameraCoveredRule, CameraFeedWatchdog, CameraFrozenRule
from rules.identity import IdentityRule
from rules.objects import ProhibitedObjectRule

CFG = PipelineConfig()
dark = lambda t: frame(t, faces=[], quality=FrameQuality(mean_luma=6, luma_std=3, sharpness=1, motion=2))


def test_dark_frames_raise_camera_covered_once():
    rule = CameraCoveredRule(CFG)
    ev = run_frames(rule, 0, 30, dark)
    assert types(ev) == ["CAMERA_COVERED"]
    assert ev[0].details["reason"] == "dark" and ev[0].severity == "HIGH"


def test_one_dark_frame_is_ignored():
    rule = CameraCoveredRule(CFG)
    ev = run_frames(rule, 0, 10, lambda t: dark(t) if t == 5 else frame(t))
    assert ev == []


def test_blurry_frame_with_a_visible_face_is_a_bad_camera_not_tampering():
    rule = CameraCoveredRule(CFG)
    soft = lambda t: frame(t, quality=FrameQuality(mean_luma=120, luma_std=30, sharpness=4, motion=5))
    assert run_frames(rule, 0, 30, soft) == []


def test_a_cloth_or_finger_over_the_lens_is_covered_as_blocked():
    rule = CameraCoveredRule(CFG)
    blocked = lambda t: frame(t, faces=[], quality=FrameQuality(mean_luma=120, luma_std=2, sharpness=1, motion=1))
    ev = run_frames(rule, 0, 15, blocked)
    assert types(ev) == ["CAMERA_COVERED"] and ev[0].details["reason"] == "blocked"


def test_a_plain_wall_with_nobody_in_front_is_an_empty_chair_not_a_covered_lens():
    # Low detail and no face, but real contrast and some edges: the candidate
    # stepped away. That's FACE_ABSENT's job; accusing tampering would be a
    # false positive on an ordinary blank wall.
    rule = CameraCoveredRule(CFG)
    wall = lambda t: frame(t, faces=[], quality=FrameQuality(mean_luma=150, luma_std=9, sharpness=6, motion=1))
    assert run_frames(rule, 0, 60, wall) == []


def test_frozen_feed_is_flagged_after_the_window():
    rule = CameraFrozenRule(CFG)
    still = lambda t: frame(t, quality=FrameQuality(motion=0.0))
    ev = run_frames(rule, 0, 30, still)
    assert types(ev) == ["CAMERA_FROZEN"]


def test_live_noise_never_looks_frozen():
    rule = CameraFrozenRule(CFG)
    assert run_frames(rule, 0, 60, lambda t: frame(t, quality=FrameQuality(motion=1.5))) == []


def test_feed_lost_and_restored():
    rule = CameraFeedWatchdog(CFG)
    rule.on_frame(frame(0))
    assert rule.on_tick(5) == []
    lost = rule.on_tick(12)
    assert types(lost) == ["CAMERA_FEED_LOST"] and lost[0].severity == "HIGH"
    assert rule.on_tick(20) == []  # reported once
    assert types(rule.on_frame(frame(30))) == ["CAMERA_FEED_RESTORED"]


def test_watchdog_is_silent_before_the_first_frame():
    assert CameraFeedWatchdog(CFG).on_tick(1000) == []


# ------------------------------- identity -------------------------------
def ident(rule, sims, start=0):
    out = []
    for i, s in enumerate(sims):
        out += rule.on_frame(frame(start + i * 3, identity=s))
    return out


def test_a_different_person_for_several_checks_is_critical():
    ev = ident(IdentityRule(CFG), [0.12, 0.08, 0.15])
    assert types(ev) == ["IDENTITY_MISMATCH"] and ev[0].severity == "CRITICAL" and ev[0].evidence


def test_two_low_scores_are_not_enough():
    assert ident(IdentityRule(CFG), [0.1, 0.1]) == []


def test_a_good_score_resets_the_run():
    assert ident(IdentityRule(CFG), [0.1, 0.1, 0.7, 0.1, 0.1, 0.7, 0.1]) == []


def test_borderline_scores_between_the_thresholds_never_accuse():
    # 0.30..0.363 is "probably the same person in different light".
    assert ident(IdentityRule(CFG), [0.33] * 30) == []


def test_unscored_frames_neither_confirm_nor_refute():
    rule = IdentityRule(CFG)
    ev = ident(rule, [0.1, 0.1, None, None, 0.1])
    assert types(ev) == ["IDENTITY_MISMATCH"]


def test_identity_alert_has_a_cooldown():
    rule = IdentityRule(CFG)
    assert len(ident(rule, [0.1] * 12)) == 1


# ------------------------------- objects --------------------------------
def test_single_frame_phone_ghost_is_ignored():
    rule = ProhibitedObjectRule(CFG)
    ev = run_frames(rule, 0, 10, lambda t: frame(t, objects=[obj("cell phone", 0.6)] if t == 4 else []))
    assert ev == []


def test_phone_seen_twice_is_high_and_uses_the_legacy_alert_type():
    rule = ProhibitedObjectRule(CFG)
    ev = run_frames(rule, 0, 3, lambda t: frame(t, objects=[obj("cell phone", 0.6)]))
    assert types(ev) == ["CELLPHONE_DETECTED"] and ev[0].severity == "HIGH"


def test_phone_that_stays_escalates_to_critical_once():
    rule = ProhibitedObjectRule(CFG)
    ev = run_frames(rule, 0, 15, lambda t: frame(t, objects=[obj("cell phone", 0.6)]))
    assert [e.severity for e in ev] == ["HIGH", "CRITICAL"]


def test_below_confidence_floor_is_ignored():
    rule = ProhibitedObjectRule(CFG)
    assert run_frames(rule, 0, 10, lambda t: frame(t, objects=[obj("cell phone", 0.2)])) == []


def test_book_is_medium_and_a_distinct_alert_type():
    rule = ProhibitedObjectRule(CFG)
    ev = run_frames(rule, 0, 4, lambda t: frame(t, objects=[obj("book", 0.7)]))
    assert types(ev) == ["PROHIBITED_OBJECT_DETECTED"] and ev[0].severity == "MEDIUM"
    assert ev[0].details["object"] == "book"


def test_unlisted_objects_are_ignored():
    rule = ProhibitedObjectRule(CFG)
    assert run_frames(rule, 0, 10, lambda t: frame(t, objects=[obj("cup", 0.99), obj("keyboard", 0.9)])) == []
