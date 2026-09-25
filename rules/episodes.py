"""
Temporal building blocks shared by every rule.

The single most important design constraint on these rules: **a momentary
glance is not cheating.** People look away while thinking, blink, shift in
their seat, and get a bad frame now and then. Frames also arrive at ~1/second
(shepherd-ai's default snapshot cadence), so a "sustained for 2.5 seconds"
threshold is really "2-3 samples in a row", which one noisy sample can break
or fake.

So rules never fire on an instantaneous condition. They feed boolean samples
into a `RatioEpisode`, which only starts an episode when the condition held
for most of a window, ends it only when the condition has genuinely gone away
(hysteresis, so it can't flap on/off), and reports the episode's duration.
Point events go through an `AlertGate` so one behaviour produces one alert,
not one per frame.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple


class RatioEpisode:
    """
    Detects an *episode* of a boolean condition over a sliding time window.

    - START when, over the last `window_s` seconds, at least `on_ratio` of the
      samples were active, there are at least `min_samples` of them, and the
      samples actually span most of the window (so a burst of frames right
      after a gap can't fake a long duration).
    - END when the active ratio over the last `off_window_s` seconds drops to
      `off_ratio` or below.

    `update()` returns "start", "end" or None. `active_since` / `duration()`
    describe the episode in progress; `last_duration` is the finished one.
    """

    def __init__(
        self,
        window_s: float,
        on_ratio: float = 0.8,
        min_samples: int = 3,
        off_ratio: float = 0.2,
        off_window_s: Optional[float] = None,
    ) -> None:
        if not 0 < on_ratio <= 1:
            raise ValueError("on_ratio must be in (0, 1]")
        self.window_s = window_s
        self.on_ratio = on_ratio
        self.min_samples = min_samples
        self.off_ratio = off_ratio
        self.off_window_s = off_window_s if off_window_s is not None else max(window_s / 2.0, 1.0)
        self._samples: Deque[Tuple[float, bool]] = deque()
        self._first_active_ts: Optional[float] = None
        self.active_since: Optional[float] = None
        self.last_duration: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.active_since is not None

    def duration(self, ts: float) -> float:
        return 0.0 if self.active_since is None else max(0.0, ts - self.active_since)

    def reset(self) -> None:
        self._samples.clear()
        self.active_since = None
        self._first_active_ts = None

    def _ratio(self, ts: float, span: float) -> Tuple[float, int]:
        cutoff = ts - span
        window = [active for (t, active) in self._samples if t >= cutoff]
        if not window:
            return 0.0, 0
        return sum(1 for a in window if a) / len(window), len(window)

    def update(self, active: bool, ts: float) -> Optional[str]:
        self._samples.append((ts, active))
        horizon = ts - max(self.window_s, self.off_window_s) - 1.0
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()

        if self.active_since is None:
            ratio, n = self._ratio(ts, self.window_s)
            if n < self.min_samples or ratio < self.on_ratio:
                return None
            # The samples must cover most of the window, and the *oldest one
            # counted* must itself be part of the run -- otherwise a lone
            # active sample after a long quiet gap would look "sustained".
            in_window = [(t, a) for (t, a) in self._samples if t >= ts - self.window_s]
            span = in_window[-1][0] - in_window[0][0]
            if span < self.window_s * 0.75:
                return None
            self.active_since = in_window[0][0]
            return "start"

        ratio, n = self._ratio(ts, self.off_window_s)
        if n >= 1 and ratio <= self.off_ratio:
            self.last_duration = max(0.0, ts - self.active_since)
            self.active_since = None
            return "end"
        return None


class AlertGate:
    """Cooldown per key: the first occurrence passes, repeats within
    `cooldown_s` are suppressed (and counted, so the next alert that gets
    through can say how many were swallowed)."""

    def __init__(self) -> None:
        self._last: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}

    def allow(self, key: str, ts: float, cooldown_s: float) -> bool:
        last = self._last.get(key)
        if last is not None and ts - last < cooldown_s:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False
        self._last[key] = ts
        return True

    def take_suppressed(self, key: str) -> int:
        """Number of repeats swallowed since the last alert that passed for
        `key`; resets the counter."""
        return self._suppressed.pop(key, 0)


class RollingCounter:
    """Counts timestamped occurrences inside a trailing window -- 'how many
    times did X start in the last 5 minutes'."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._times: Deque[float] = deque()

    def add(self, ts: float) -> int:
        self._times.append(ts)
        return self.count(ts)

    def count(self, ts: float) -> int:
        cutoff = ts - self.window_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        return len(self._times)

    def clear(self) -> None:
        self._times.clear()
