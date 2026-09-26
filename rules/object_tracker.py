"""
Follows detected objects from frame to frame.

A detector answers "what is in *this* frame". A proctoring rule needs "is the
*same* phone still there": one sighting is often a ghost, while the same box
turning up again and again is a real object. Tracking also survives the
detector's misses -- a phone half hidden by a hand is seen in two frames out of
five, which is exactly what a strict "N frames in a row" rule cannot cope with.

Pure logic over boxes and timestamps; no model, no wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from observations import BoundingBox, ObjectObservation


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ix = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _centre_close(a: BoundingBox, b: BoundingBox) -> bool:
    """Small boxes barely overlap from one frame to the next when the object moves, so a second
    test: the centres are within the larger box's own size."""
    (ax, ay), (bx, by) = a.center, b.center
    reach = max(a.width, a.height, b.width, b.height)
    return abs(ax - bx) <= reach and abs(ay - by) <= reach


@dataclass
class Track:
    id: int
    label: str
    box: BoundingBox
    first_ts: float
    last_ts: float
    # (timestamp, confidence) of every sighting, oldest first.
    sightings: List[tuple] = field(default_factory=list)
    frames_seen: int = 0

    @property
    def max_confidence(self) -> float:
        return max((c for _, c in self.sightings), default=0.0)

    @property
    def age(self) -> float:
        return self.last_ts - self.first_ts

    def recent(self, since_ts: float) -> List[tuple]:
        return [s for s in self.sightings if s[0] >= since_ts]


class ObjectTracker:
    """Greedy association by box overlap, per label. Tracks that have not been
    seen for `max_gap_s` are dropped, so an object that leaves and a different
    one that arrives later are not merged into one long-lived track."""

    def __init__(self, iou_threshold: float = 0.2, max_gap_s: float = 4.0, keep_sightings_s: float = 30.0) -> None:
        self.iou_threshold = iou_threshold
        self.max_gap_s = max_gap_s
        self.keep_sightings_s = keep_sightings_s
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def update(self, ts: float, detections: List[ObjectObservation]) -> List[Track]:
        """Feeds one frame's detections. Returns the tracks that were seen *in this frame*."""
        for tid in [t.id for t in self._tracks.values() if ts - t.last_ts > self.max_gap_s]:
            del self._tracks[tid]

        seen: List[Track] = []
        unmatched = sorted(detections, key=lambda d: -d.confidence)  # the most confident claims a track first
        used: set = set()
        for det in unmatched:
            best: Optional[Track] = None
            best_score = 0.0
            for track in self._tracks.values():
                if track.label != det.label or track.id in used:
                    continue
                score = iou(track.box, det.box)
                if score < self.iou_threshold and _centre_close(track.box, det.box):
                    score = self.iou_threshold * 0.999  # matched on distance alone: weaker than a real overlap
                if score >= self.iou_threshold * 0.999 and score > best_score:
                    best, best_score = track, score
            if best is None:
                best = Track(self._next_id, det.label, det.box, ts, ts)
                self._next_id += 1
                self._tracks[best.id] = best
            used.add(best.id)
            best.box = det.box
            best.last_ts = ts
            best.frames_seen += 1
            best.sightings.append((ts, det.confidence))
            cutoff = ts - self.keep_sightings_s
            best.sightings = [s for s in best.sightings if s[0] >= cutoff]
            seen.append(best)
        return seen
