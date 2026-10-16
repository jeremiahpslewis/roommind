"""Learned AC sensor offset (device reading - room sensor)."""

from __future__ import annotations

import pytest

from custom_components.roommind.const import HEAD_OFFSET_MIN_SAMPLES, HEAD_OFFSET_SETTLE_S
from custom_components.roommind.managers.head_offset_tracker import HeadOffsetTracker

AC = "climate.ac"


def feed(tracker, action, device, room, *, start=0.0, n=HEAD_OFFSET_MIN_SAMPLES, step=30.0):
    """One settle period in *action*, then *n* readings 30 s apart; returns the time after the last."""
    tracker.observe(AC, action, device, room, start)
    t = start + HEAD_OFFSET_SETTLE_S
    for _ in range(n):
        tracker.observe(AC, action, device, room, t)
        t += step
    return t


def test_learns_the_offset_per_running_direction():
    tracker = HeadOffsetTracker()
    t = feed(tracker, "heating", 22.5, 21.0)
    feed(tracker, "cooling", 25.0, 24.0, start=t)
    assert tracker.offset(AC, "heat") == pytest.approx(1.5)
    assert tracker.offset(AC, "cool") == pytest.approx(1.0)
    assert tracker.offset(AC, "idle") == 0.0
    assert tracker.dirty


def test_idle_feeds_the_idle_offset_and_off_teaches_nothing():
    tracker = HeadOffsetTracker()
    feed(tracker, "off", 26.0, 24.0)
    assert tracker.offset(AC, "idle") == 0.0
    feed(tracker, "idle", 26.0, 24.0, start=10_000.0)
    assert tracker.offset(AC, "idle") == pytest.approx(2.0)


def test_readings_right_after_a_start_are_skipped():
    """The offset builds up with the airflow: the first minutes still carry the idle reading."""
    tracker = HeadOffsetTracker()
    tracker.observe(AC, "heating", 21.0, 21.0, 0.0)
    for t in range(30, int(HEAD_OFFSET_SETTLE_S), 30):
        tracker.observe(AC, "heating", 21.0, 21.0, float(t))
    for i in range(HEAD_OFFSET_MIN_SAMPLES):
        tracker.observe(AC, "heating", 22.5, 21.0, HEAD_OFFSET_SETTLE_S + 30.0 * i)
    assert tracker.get_state()[AC]["heat"][1] == HEAD_OFFSET_MIN_SAMPLES
    assert tracker.offset(AC, "heat") == pytest.approx(1.5)


def test_a_state_change_restarts_the_settle_period():
    tracker = HeadOffsetTracker()
    t = feed(tracker, "heating", 22.5, 21.0, n=3)
    tracker.observe(AC, "idle", 23.0, 21.0, t)
    tracker.observe(AC, "heating", 22.5, 21.0, t + 30.0)
    tracker.observe(AC, "heating", 22.5, 21.0, t + 60.0)  # inside the new settle period
    assert tracker.get_state()[AC]["heat"][1] == 3


def test_offset_stays_zero_until_enough_readings():
    tracker = HeadOffsetTracker()
    feed(tracker, "cooling", 25.0, 24.0, n=HEAD_OFFSET_MIN_SAMPLES - 1)
    assert tracker.offset(AC, "cool") == 0.0
    assert tracker.snapshot(AC)["cool"] is None


@pytest.mark.parametrize("action", ["fan", "drying", None])
def test_fan_dry_and_unknown_actions_teach_nothing(action):
    tracker = HeadOffsetTracker()
    feed(tracker, action, 25.0, 24.0)
    assert tracker.get_state() == {}


def test_missing_readings_teach_nothing():
    tracker = HeadOffsetTracker()
    feed(tracker, "cooling", None, 24.0)
    feed(tracker, "cooling", 25.0, None)
    assert tracker.get_state() == {}


def test_implausible_offset_is_ignored():
    """Four kelvin and more is a misplaced or broken sensor, not a bias to correct."""
    tracker = HeadOffsetTracker()
    feed(tracker, "heating", 27.0, 21.0)
    assert tracker.get_state() == {}


def test_follows_a_changing_offset():
    tracker = HeadOffsetTracker()
    t = feed(tracker, "cooling", 25.0, 24.0)
    feed(tracker, "cooling", 26.0, 24.0, start=t, n=100)
    assert tracker.offset(AC, "cool") == pytest.approx(2.0, abs=0.05)


def test_state_round_trips_and_skips_malformed_entries():
    tracker = HeadOffsetTracker()
    t = feed(tracker, "heating", 22.5, 21.0)
    feed(tracker, "idle", 23.0, 21.0, start=t)
    restored = HeadOffsetTracker()
    restored.load_state(
        {
            **tracker.get_state(),
            "climate.broken": {"heat": "x", "cool": [9.0, 50], "vent": [1.0, 50]},
            "climate.junk": 3,
        }
    )
    assert restored.offset(AC, "heat") == pytest.approx(1.5)
    assert restored.offset(AC, "idle") == pytest.approx(2.0)
    assert "climate.broken" not in restored.get_state()
    assert "climate.junk" not in restored.get_state()
    assert not restored.dirty
