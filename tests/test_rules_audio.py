from helpers import audio, run_audio, types
from pipeline_config import PipelineConfig
from rules.audio import MicrophoneStateRule, MultipleVoicesRule, SpeechRule, WhisperRule

CFG = PipelineConfig()


def test_speech_is_reported_once_per_cooldown_not_every_second():
    rule = SpeechRule(CFG)
    ev = run_audio(rule, 0, 30, lambda t: audio(t, speech=1.0, dbfs=-25))
    assert types(ev).count("VOICE_DETECTED") == 1


def test_brief_speech_is_not_sustained_speaking():
    rule = SpeechRule(CFG)
    ev = run_audio(rule, 0, 3, lambda t: audio(t, speech=1.0)) + run_audio(rule, 3, 20, lambda t: audio(t))
    assert "SUSTAINED_SPEAKING" not in types(ev)


def test_sustained_speech_starts_medium_and_escalates_once_to_high():
    rule = SpeechRule(CFG)
    ev = run_audio(rule, 0, 60, lambda t: audio(t, speech=1.0))
    sustained = [e for e in ev if e.alert_type == "SUSTAINED_SPEAKING"]
    assert [e.severity for e in sustained] == ["MEDIUM", "HIGH"]
    assert all(e.evidence and e.evidence_kind == "audio" for e in sustained)


def quiet_room(t):
    return audio(t, dbfs=-55, flat=0.05)


def test_whisper_needs_a_learned_noise_floor_first():
    rule = WhisperRule(CFG)
    ev = run_audio(rule, 0, 15, lambda t: audio(t, dbfs=-30, flat=0.5))
    assert ev == []


def test_whisper_like_sound_above_the_floor_is_flagged_with_low_confidence():
    rule = WhisperRule(CFG)
    run_audio(rule, 0, 40, quiet_room)
    ev = run_audio(rule, 40, 20, lambda t: audio(t, dbfs=-38, flat=0.5))
    assert types(ev) == ["WHISPER_SUSPECTED"]
    assert ev[0].severity == "MEDIUM" and ev[0].confidence == 0.4


def test_loud_speech_is_not_mistaken_for_a_whisper():
    rule = WhisperRule(CFG)
    run_audio(rule, 0, 40, quiet_room)
    assert run_audio(rule, 40, 20, lambda t: audio(t, speech=1.0, dbfs=-20, flat=0.5)) == []


def test_a_dead_microphone_is_reported_and_recovery_noted():
    rule = MicrophoneStateRule(CFG)
    ev = run_audio(rule, 0, 30, lambda t: audio(t, dbfs=-96))
    ev += run_audio(rule, 30, 6, lambda t: audio(t, dbfs=-50))
    assert types(ev) == ["MICROPHONE_SILENT", "MICROPHONE_RESTORED"]


def test_a_normal_quiet_room_is_not_a_dead_microphone():
    rule = MicrophoneStateRule(CFG)
    assert run_audio(rule, 0, 120, lambda t: audio(t, dbfs=-58)) == []


def test_repeated_dropouts_are_flagged():
    rule = MicrophoneStateRule(CFG)
    ev, t = [], 0
    for _ in range(3):
        ev += run_audio(rule, t, 20, lambda x: audio(x, dbfs=-96))
        ev += run_audio(rule, t + 20, 8, lambda x: audio(x, dbfs=-50))
        t += 28
    assert "AUDIO_INTERRUPTIONS" in types(ev)


def test_audio_feed_lost_from_watchdog():
    rule = MicrophoneStateRule(CFG)
    rule.on_audio(audio(0))
    assert types(rule.on_tick(15)) == ["AUDIO_FEED_LOST"]
    assert rule.on_tick(30) == []
    assert types(rule.on_audio(audio(40))) == ["AUDIO_FEED_RESTORED"]


A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]


def test_two_distinct_voices_raise_multiple_voices():
    rule = MultipleVoicesRule(CFG)
    ev = []
    for t in range(10):
        ev += rule.on_audio(audio(t, speech=1.0, emb=A if t % 2 == 0 else B))
    assert types(ev) == ["MULTIPLE_VOICES"] and ev[0].severity == "HIGH"


def test_one_voice_with_one_odd_window_is_not_a_second_speaker():
    rule = MultipleVoicesRule(CFG)
    ev = []
    for t in range(15):
        ev += rule.on_audio(audio(t, speech=1.0, emb=B if t == 7 else A))
    assert ev == []


def test_without_an_embedder_the_rule_stays_inert():
    rule = MultipleVoicesRule(CFG)
    assert run_audio(rule, 0, 30, lambda t: audio(t, speech=1.0)) == []
