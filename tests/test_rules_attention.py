from helpers import face, frame, run_frames, types
from pipeline_config import PipelineConfig
from rules.attention import LookAwayRule

CFG = PipelineConfig()
away = lambda t: frame(t, faces=[face(yaw=40)])
ahead = lambda t: frame(t)


def test_a_two_second_glance_away_is_not_cheating():
    rule = LookAwayRule(CFG)
    ev = run_frames(rule, 0, 5, ahead) + run_frames(rule, 5, 2, away) + run_frames(rule, 7, 10, ahead)
    assert ev == []


def test_sustained_head_turn_raises_one_alert_not_one_per_frame():
    rule = LookAwayRule(CFG)
    ev = run_frames(rule, 0, 5, ahead) + run_frames(rule, 5, 30, away)
    assert types(ev) == ["SUSTAINED_LOOK_AWAY"]
    assert ev[0].severity == "MEDIUM"


def test_pitch_down_counts_as_looking_away():
    rule = LookAwayRule(CFG)
    ev = run_frames(rule, 0, 15, lambda t: frame(t, faces=[face(pitch=-35)]))
    assert types(ev) == ["SUSTAINED_LOOK_AWAY"]


def test_one_brief_glance_back_does_not_reset_a_long_look_away():
    rule = LookAwayRule(CFG)
    ev = run_frames(rule, 0, 10, away) + run_frames(rule, 10, 1, ahead) + run_frames(rule, 11, 10, away)
    assert types(ev) == ["SUSTAINED_LOOK_AWAY"]


def test_repeated_separate_look_aways_raise_frequent_once():
    rule = LookAwayRule(CFG)
    ev, t = [], 0
    for _ in range(7):
        ev += run_frames(rule, t, 8, away)
        ev += run_frames(rule, t + 8, 6, ahead)
        t += 14
    assert types(ev).count("SUSTAINED_LOOK_AWAY") == 7
    assert types(ev).count("FREQUENT_LOOK_AWAY") == 1
    assert next(e for e in ev if e.alert_type == "FREQUENT_LOOK_AWAY").severity == "HIGH"


def test_eyes_off_screen_with_head_still_is_a_low_severity_gaze_alert():
    rule = LookAwayRule(CFG)
    ev = run_frames(rule, 0, 12, lambda t: frame(t, faces=[face(gaze_x=0.8)]))
    assert types(ev) == ["GAZE_AWAY"] and ev[0].severity == "LOW"


def test_a_missing_face_is_not_treated_as_looking_away():
    rule = LookAwayRule(CFG)
    assert run_frames(rule, 0, 30, lambda t: frame(t, faces=[])) == []
