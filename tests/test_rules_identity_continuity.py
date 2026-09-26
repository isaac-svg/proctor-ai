from helpers import cosine, frame_with_embedding, person, sighting, types
from pipeline_config import PipelineConfig
from rules.identity import IdentityRule

CFG = PipelineConfig()
A, B = person(1), person(2)


def feed(rule, start, seconds, who, ref_sim=None, noise=0.3, seed0=0):
    """One scored frame every 2 s (identity is scored on every other frame)."""
    out = []
    for i in range(seconds // 2):
        t = start + 2 * i
        out += rule.on_frame(frame_with_embedding(float(t), sighting(who, noise, seed0 + i), ref_sim=ref_sim))
    return out


def test_the_synthetic_people_behave_as_the_thresholds_assume():
    # Guards the fixtures: a person against themselves is far above the 'same' line, two people far below the 'changed' line.
    assert cosine(sighting(A, 0.3, 1), sighting(A, 0.3, 2)) > 0.85
    assert abs(cosine(A, B)) < 0.25


def test_the_same_person_all_exam_raises_nothing():
    rule = IdentityRule(CFG)
    assert feed(rule, 0, 600, A, ref_sim=0.6) == []
    assert rule.tracking and rule.tracked_person_verified


def test_it_takes_a_few_frames_to_learn_the_face_and_says_nothing_meanwhile():
    rule = IdentityRule(CFG)
    assert feed(rule, 0, 4, A) == [] and not rule.tracking
    feed(rule, 4, 6, A, seed0=10)
    assert rule.tracking


def test_a_different_person_sitting_down_mid_exam_is_reported_once_even_with_no_check_in_reference():
    rule = IdentityRule(CFG)
    assert feed(rule, 0, 120, A) == []
    ev = feed(rule, 120, 30, B, seed0=500)
    assert types(ev) == ["PERSON_CHANGED"]
    assert ev[0].severity == "HIGH" and ev[0].evidence and ev[0].details["continuity_mean"] < CFG.continuity_change_below


def test_the_swap_is_one_alert_not_one_per_frame_and_the_new_person_is_then_the_reference_point():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A)
    ev = feed(rule, 120, 300, B, seed0=500)
    assert types(ev) == ["PERSON_CHANGED"]
    assert rule.tracking  # it has learnt the new face, so a further change would be noticed too


def test_a_single_odd_frame_is_not_a_swap():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A)
    ev = rule.on_frame(frame_with_embedding(120.0, sighting(B, 0.3, 1)))
    ev += feed(rule, 122, 60, A, seed0=900)
    assert ev == []


def test_a_bad_frame_of_the_right_person_does_not_break_continuity():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A)
    # A blurry, half-turned frame of the same person scores lower but still well above the 'changed' line.
    assert feed(rule, 120, 40, A, noise=1.0, seed0=300) == []


def test_a_swap_right_after_leaving_the_frame_needs_less_evidence():
    slow = IdentityRule(CFG)
    feed(slow, 0, 120, A)
    two = [sighting(B, 0.3, 1), sighting(B, 0.3, 2)]
    assert [e for i, x in enumerate(two) for e in slow.on_frame(frame_with_embedding(120.0 + 2 * i, x))] == []  # ordinary: two frames are not enough

    quick = IdentityRule(CFG)
    feed(quick, 0, 120, A)
    quick.on_face_returned(119.0, 40.0)
    ev = [e for i, x in enumerate(two) for e in quick.on_frame(frame_with_embedding(120.0 + 2 * i, x))]
    assert types(ev) == ["PERSON_CHANGED"] and ev[0].details["after_absence_seconds"] == 40.0


def test_the_same_person_returning_after_a_break_is_fine():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A)
    rule.on_face_returned(130.0, 60.0)
    assert feed(rule, 130, 40, A, seed0=700) == []


def test_not_matching_the_checkin_photo_from_the_start_is_an_identity_mismatch():
    rule = IdentityRule(CFG)
    ev = feed(rule, 0, 30, B, ref_sim=0.1)
    assert types(ev) == ["IDENTITY_MISMATCH"] and ev[0].severity == "CRITICAL"
    assert ev[0].details["changed_during_exam"] is False and "confidence" not in ev[0].details


def test_a_swap_during_the_exam_is_one_critical_alert_that_says_it_changed_mid_exam():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A, ref_sim=0.6)
    ev = feed(rule, 120, 30, B, ref_sim=0.1, seed0=500)
    assert types(ev) == ["IDENTITY_MISMATCH"]  # not also PERSON_CHANGED
    assert ev[0].details["changed_during_exam"] is True


def test_the_same_candidate_who_put_on_glasses_is_flagged_for_review_not_accused():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A, ref_sim=0.6)  # tracked and verified against the photo
    ev = feed(rule, 120, 30, A, ref_sim=0.15, seed0=900)  # same face on screen, no longer like the photo
    assert types(ev) == ["APPEARANCE_CHANGED"] and ev[0].severity == "MEDIUM"
    assert "CRITICAL" not in [e.severity for e in ev]


def test_a_face_that_was_never_verified_against_the_photo_gets_no_such_benefit_of_the_doubt():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A, ref_sim=0.32)  # tracked, close enough not to be accused, but never a real match for the photo
    ev = feed(rule, 120, 30, A, ref_sim=0.15, seed0=900)
    assert types(ev) == ["IDENTITY_MISMATCH"]


def test_frames_with_nothing_scored_change_nothing():
    from helpers import frame
    rule = IdentityRule(CFG)
    assert [rule.on_frame(frame(float(t))) for t in range(20)] == [[]] * 20
    assert not rule.tracking


def test_the_two_alerts_respect_their_cooldowns():
    rule = IdentityRule(CFG)
    feed(rule, 0, 120, A)
    first = feed(rule, 120, 20, B, seed0=500)
    back = feed(rule, 140, 20, A, seed0=800)  # now A is 'different' from the tracked B, inside the cooldown
    assert types(first) == ["PERSON_CHANGED"] and back == []
