from rules.episodes import AlertGate, RatioEpisode, RollingCounter


def feed(ep, samples):
    out = []
    for ts, active in samples:
        r = ep.update(active, ts)
        if r:
            out.append((ts, r))
    return out


def test_single_active_sample_never_starts_an_episode():
    ep = RatioEpisode(window_s=5, on_ratio=0.8, min_samples=4)
    events = feed(ep, [(t, t == 3) for t in range(10)])
    assert events == []


def test_sustained_condition_starts_once_and_ends_once():
    ep = RatioEpisode(window_s=5, on_ratio=0.8, min_samples=4, off_window_s=2)
    events = feed(ep, [(t, True) for t in range(10)] + [(t, False) for t in range(10, 14)])
    assert [e for _, e in events] == ["start", "end"]


def test_one_glance_back_does_not_end_a_long_episode():
    ep = RatioEpisode(window_s=5, on_ratio=0.8, min_samples=4, off_ratio=0.2, off_window_s=3)
    samples = [(t, True) for t in range(8)] + [(8, False)] + [(t, True) for t in range(9, 14)]
    assert [e for _, e in feed(ep, samples)] == ["start"]


def test_lone_sample_after_long_gap_is_not_sustained():
    # Frames stop for a minute, then one arrives: the window holds a single
    # sample, which must not read as "active for the whole window".
    ep = RatioEpisode(window_s=5, on_ratio=0.8, min_samples=1)
    assert ep.update(True, 100.0) is None


def test_duration_reflects_episode_start():
    ep = RatioEpisode(window_s=4, on_ratio=0.8, min_samples=3, off_window_s=2)
    feed(ep, [(t, True) for t in range(6)])
    assert ep.is_active and ep.duration(5) >= 1


def test_alert_gate_suppresses_and_counts_repeats():
    g = AlertGate()
    assert g.allow("k", 0, 60)
    assert not g.allow("k", 10, 60)
    assert not g.allow("k", 20, 60)
    assert g.take_suppressed("k") == 2
    assert g.allow("k", 61, 60)


def test_rolling_counter_expires_old_entries():
    c = RollingCounter(100)
    c.add(0)
    c.add(50)
    assert c.add(120) == 2  # the t=0 entry aged out
