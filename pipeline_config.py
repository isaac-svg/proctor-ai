"""
Every tunable threshold in the proctoring rules, in one place.

Units: seconds for `*_s`, degrees for `*_deg`, 0..1 for ratios/confidences,
dBFS for `*_dbfs`. Defaults assume ~1 frame per second (shepherd-ai's
`snapshotIntervalMs: 1000`) and ~1-second audio windows.

Where a threshold trades false positives against false negatives, the
default leans towards **not accusing**: an exam-integrity signal that fires
on a sneeze gets ignored by proctors, and then misses the real thing. Every
"sustained" behaviour is a *ratio over a window* (see rules/episodes.py), so
one glance, one blink or one dropped frame can't trigger or cancel it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------
    # Face presence
    # ------------------------------------------------------------------
    # No face for most of a 5s window => FACE_ABSENT. Shorter absences
    # (leaning down to grab a pen) are not reported.
    face_absent_window_s: float = 5.0
    face_absent_on_ratio: float = 0.8
    face_absent_min_samples: int = 4
    # Leaving the frame repeatedly, even briefly each time, is its own signal.
    frame_exit_count: int = 3
    frame_exit_window_s: float = 600.0

    # ------------------------------------------------------------------
    # More than one person
    # ------------------------------------------------------------------
    multi_person_window_s: float = 4.0
    multi_person_on_ratio: float = 0.7
    multi_person_min_samples: int = 3
    # A secondary face must be at least this fraction of the primary face's
    # area to count -- filters posters, photos on the wall, and reflections.
    secondary_face_min_area_ratio: float = 0.25
    person_min_confidence: float = 0.5
    # A YOLO "person" box smaller than this fraction of the frame is a
    # distant/background figure, not someone at the desk.
    person_min_area: float = 0.04

    # ------------------------------------------------------------------
    # Attention (head + eyes)
    # ------------------------------------------------------------------
    yaw_threshold_deg: float = 25.0
    pitch_threshold_deg: float = 20.0
    look_away_window_s: float = 6.0
    look_away_on_ratio: float = 0.8
    look_away_min_samples: int = 4
    look_away_off_ratio: float = 0.3
    # This many separate look-away episodes inside the window => FREQUENT.
    frequent_look_away_count: int = 6
    frequent_look_away_window_s: float = 300.0
    # Eyes off-centre while the head stays still (reading a second screen).
    gaze_threshold: float = 0.5
    gaze_head_still_yaw_deg: float = 8.0
    gaze_window_s: float = 6.0
    gaze_on_ratio: float = 0.8

    # ------------------------------------------------------------------
    # Framing (advisory -- LOW severity, mostly for telling the student)
    # ------------------------------------------------------------------
    face_too_far_area: float = 0.02
    face_too_close_area: float = 0.55
    face_off_center: float = 0.32  # |centre - 0.5| on either axis
    face_min_visible_ratio: float = 0.8
    framing_window_s: float = 10.0
    framing_on_ratio: float = 0.8
    framing_cooldown_s: float = 120.0

    # ------------------------------------------------------------------
    # Camera integrity
    # ------------------------------------------------------------------
    camera_dark_luma: float = 20.0
    # "Blocked" needs BOTH near-zero contrast and near-zero edge energy. Kept
    # tight on purpose: a plain wall, an empty chair or a dim room must not
    # read as a covered lens. Tune against real webcam footage.
    camera_blocked_std: float = 4.0
    camera_blocked_sharpness: float = 4.0
    camera_covered_window_s: float = 4.0
    camera_covered_on_ratio: float = 0.8
    # Mean absolute frame-to-frame difference (0..255, 160x120 gray) below
    # this is "not a real sensor": a frozen driver or a still image fed through
    # a virtual camera yields *identical* frames (0.0), while any live sensor
    # -- even a still scene in good light -- sits well above it. A short
    # looped clip is not caught by this; it isn't identical frame to frame.
    camera_frozen_motion: float = 0.05
    camera_frozen_seconds: float = 12.0
    camera_feed_lost_s: float = 10.0

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    # SFace cosine similarity. OpenCV's recommended same-person cut-off is
    # 0.363; we only *accuse* below this lower value, so ordinary variation
    # in lighting/pose (which lands in 0.30-0.36) never raises an alert.
    identity_mismatch_below: float = 0.30
    identity_mismatch_consecutive: int = 3
    identity_cooldown_s: float = 300.0
    # Only score identity when the face is reasonably frontal and large
    # enough; otherwise the embedding is unreliable.
    identity_max_yaw_deg: float = 30.0
    identity_max_pitch_deg: float = 25.0
    identity_min_face_area: float = 0.03
    identity_min_sharpness: float = 25.0
    # Continuity: is the person at the desk the same one as a few minutes ago?
    # Needs no enrolment, and is a far stricter test than matching the check-in
    # photo, because it compares two moments of the *same session* (same camera,
    # same room, same lighting): the same person lands at 0.5-0.9 there, so a
    # score below `continuity_change_below` is a genuinely different face.
    continuity_bootstrap: int = 4
    continuity_same_above: float = 0.50
    continuity_change_below: float = 0.35
    continuity_consecutive: int = 3
    # Right after the candidate has been out of frame the swap is the classic
    # cheat, so a shorter run is enough.
    continuity_after_absence_consecutive: int = 2
    continuity_after_absence_window_s: float = 60.0
    continuity_update_alpha: float = 0.10
    person_change_cooldown_s: float = 120.0

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------
    # label -> (min confidence, severity). Anything not listed is ignored.
    # "laptop"/"book" are deliberately MEDIUM: an open-book exam legitimately
    # has books, and a second laptop in frame is suspicious but not proof.
    object_rules: Dict[str, tuple] = field(
        default_factory=lambda: {
            "cell phone": (0.30, "HIGH"),
            "book": (0.45, "MEDIUM"),
            "laptop": (0.50, "MEDIUM"),
            # The labels below are not in the stock COCO model. They are reported
            # only when an extra model that knows them is loaded (obd.py,
            # PROCTOR_EXTRA_MODEL); with the stock model they simply never appear.
            "tablet": (0.50, "HIGH"),
            "earbuds": (0.45, "HIGH"),
            "headphones": (0.50, "MEDIUM"),
            "smartwatch": (0.50, "MEDIUM"),
            "notes": (0.50, "MEDIUM"),
            "calculator": (0.50, "LOW"),
        }
    )
    # A detection is followed from frame to frame (rules/object_tracker.py) and
    # confirmed when its track has been seen in at least `object_confirm_hits`
    # of the last `object_confirm_window` frames AND the confidences add up to
    # `object_confirm_score` (two 0.35 sightings = 0.70, one 0.5 ghost is not
    # enough) -- OR when a single sighting is confident enough on its own.
    # Phones are often seen only intermittently (half hidden by a hand), which
    # is why the window is wider than the hits needed.
    object_confirm_hits: int = 2
    object_confirm_window: int = 5
    object_confirm_score: float = 0.70
    object_single_shot_confidence: float = 0.85
    object_track_iou: float = 0.20
    object_track_max_gap_s: float = 4.0
    object_cooldown_s: float = 60.0
    phone_sustained_s: float = 3.0
    # An object whose centre is within this many face-heights of the face is being
    # held up to it. Recorded in the alert; a phone at the ear is not the same as
    # a phone on the desk.
    near_face_distance: float = 1.6

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    speech_detected_cooldown_s: float = 60.0
    speaking_window_s: float = 8.0
    speaking_on_ratio: float = 0.6
    speaking_min_samples: int = 5
    speaking_escalate_after_s: float = 30.0
    # Whispering: audible above the room's noise floor but quiet, unvoiced,
    # and *not* picked up by the VAD. A heuristic -- reported at MEDIUM with
    # low confidence, never higher.
    whisper_max_dbfs: float = -28.0
    whisper_min_above_floor_db: float = 8.0
    whisper_min_flatness: float = 0.25
    whisper_window_s: float = 8.0
    whisper_on_ratio: float = 0.7
    # Digital silence: a real microphone always has a noise floor.
    mic_silent_dbfs: float = -75.0
    mic_silent_window_s: float = 15.0
    mic_silent_on_ratio: float = 0.95
    audio_feed_lost_s: float = 10.0
    audio_interruption_count: int = 3
    audio_interruption_window_s: float = 300.0
    # Other voices (needs a speaker embedder; see speaker_embedding.py).
    # Cosine similarity above which two windows are the same voice. Measured
    # with the shipped model on six distinct synthetic voices: same-voice
    # windows scored 0.62-0.92, different voices averaged 0.19 but a
    # look-alike pair reached 0.67. 0.5 errs towards *not* splitting one
    # person into two voices, at the cost of missing a very similar second
    # voice. Real microphones vary more than TTS -- tune on real recordings.
    voice_same_speaker_above: float = 0.5
    voice_other_needed: int = 3
    voice_window_s: float = 20.0
    voice_min_speech_ratio: float = 0.5

    # ------------------------------------------------------------------
    # Evidence clips
    # ------------------------------------------------------------------
    evidence_min_severity: str = "MEDIUM"
    evidence_max_per_session: int = 25
    evidence_frame_count: int = 3
    evidence_audio_seconds: float = 4.0
