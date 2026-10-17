"""Comfort metrics against synthetic timelines."""

from __future__ import annotations

import pytest
from helpers import scenario

from roommind_sim.analysis.load import RunData
from roommind_sim.analysis.metrics import (
    _cover_metrics,
    _device_metrics,
    _room_metrics,
    _truth_beta_device,
    _truth_beta_s,
)
from roommind_sim.scenario import load_scenario_dict


def _run(temps: list[float], heat: list[float], hold: list[bool] | None = None, step_s: int = 3600) -> RunData:
    """One sample per step; the first one only opens the interval and is not integrated."""
    scn = load_scenario_dict(scenario({"r": {}}))
    t0 = scn.start
    held = hold or [False] * len(temps)
    samples = [{"t": t0 + step_s * i, "rooms": {"r": {"t_air": temp}}} for i, temp in enumerate(temps)]
    observer = [
        {
            "t": t0 + step_s * i,
            "rooms": {
                "r": {
                    "heat_target": h,
                    "cool_target": None,
                    "mode": "idle" if held[i] else "heating",
                    "compressor_protection_active": held[i],
                }
            },
        }
        for i, h in enumerate(heat)
    ]
    return RunData(None, scn, samples, observer, [], t0 + step_s * len(temps), {})  # type: ignore[arg-type]


def test_heating_above_band_counts_as_overshoot():
    m = _room_metrics(_run([21.0, 22.0, 22.5, 22.0], [21.0] * 4), "r")
    assert m["heat_overshoot_kh"] == pytest.approx(0.5 + 1.0 + 0.5)
    assert m["heat_coast_kh"] == 0.0


def test_passive_cooldown_after_setback_is_coasting_not_overshoot():
    temps = [21.5, 21.5, 21.3, 21.0, 20.6, 20.2, 19.8, 19.4]
    heat = [21.5, 21.5, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_overshoot_kh"] == 0.0
    assert m["heat_coast_kh"] == pytest.approx(1.8 + 1.5 + 1.1 + 0.7 + 0.3)
    assert m["heat_above_band_kh"] == pytest.approx(m["heat_coast_kh"])


def test_coast_ends_in_band_so_later_overshoot_counts():
    temps = [21.5, 21.5, 20.0, 19.4, 20.5]
    heat = [21.5, 21.5, 19.0, 19.0, 19.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_coast_kh"] == pytest.approx(0.5)
    assert m["heat_overshoot_kh"] == pytest.approx(1.0)


def test_rewarming_during_setback_ends_the_coast():
    temps = [21.5, 21.5, 21.0, 20.6, 21.0, 21.4]
    heat = [21.5, 21.5, 19.0, 19.0, 19.0, 19.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_coast_kh"] == pytest.approx(1.5 + 1.1)
    assert m["heat_overshoot_kh"] == pytest.approx(1.5 + 1.9)


def test_preheat_ahead_of_comfort_block_is_not_overshoot():
    temps = [17.5, 17.5, 17.5, 19.5, 20.5, 21.0, 21.0]
    heat = [17.0, 17.0, 17.0, 17.0, 17.0, 21.0, 21.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_overshoot_kh"] == 0.0
    assert m["undershoot_kh"] == 0.0
    assert m["heat_above_band_kh"] == pytest.approx(2.0 + 3.0)


def test_preheat_lookahead_is_limited():
    temps = [18.0, 19.5, 19.5, 19.5, 19.5, 19.5]
    heat = [17.0, 17.0, 17.0, 17.0, 17.0, 21.0]
    m = _room_metrics(_run(temps, heat), "r")
    assert m["heat_overshoot_kh"] == pytest.approx(2.0)


def _with_sun(run: RunData, ghi: list[float], solar_w: list[float]) -> RunData:
    for s, g, w in zip(run.samples, ghi, solar_w, strict=True):
        s["outdoor"] = {"ghi": g}
        s["rooms"]["r"]["solar_w"] = w
    return run


def test_truth_beta_s_in_roommind_units():
    run = _with_sun(_run([20.0] * 3, [20.0] * 3), [0.0, 250.0, 125.0], [0.0, 500.0, 250.0])
    assert _truth_beta_s(run, "r", 3.6e6) == pytest.approx(2.0)
    night = _with_sun(_run([20.0] * 2, [20.0] * 2), [0.0, 0.0], [0.0, 0.0])
    assert _truth_beta_s(night, "r", 3.6e6) is None


def test_hold_overshoot_counts_only_while_the_min_run_holds():
    temps = [20.0, 21.0, 21.8, 22.6, 23.0, 22.0]
    hold = [False, False, True, True, False, False]
    m = _room_metrics(_run(temps, [21.0] * 6, hold, step_s=600), "r")
    assert m["hold_heat_over_max_k"] == pytest.approx(1.6)
    assert m["hold_heat_over_kh"] == pytest.approx((0.3 + 1.1) / 6, abs=1e-3)


def test_hold_running_into_a_setback_is_measured_against_the_comfort_target():
    temps = [20.0, 21.0, 21.4, 21.6]
    heat = [21.0, 21.0, 18.0, 18.0]
    hold = [False, False, True, True]
    m = _room_metrics(_run(temps, heat, hold, step_s=600), "r")
    assert m["hold_heat_over_max_k"] == pytest.approx(0.6)


def test_hold_while_preheating_is_measured_against_the_coming_target():
    temps = [18.0, 19.0, 20.5, 21.2]
    heat = [18.0, 18.0, 18.0, 21.0]
    hold = [False, False, True, False]
    m = _room_metrics(_run(temps, heat, hold, step_s=600), "r")
    assert m["hold_heat_over_max_k"] == 0.0


def _cover_run(hours: int, position: dict[int, int], ghi: dict[int, float], heating: set[int]) -> RunData:
    """Hourly samples from the scenario start (00:00 local, 2026-01-12); dicts map hour -> value."""
    rooms = {"r": {"windows": [{"orientation": "S", "area_m2": 2}], "covers": [{"entity_id": "cover.r"}]}}
    scn = load_scenario_dict(scenario(rooms))
    t0 = scn.start
    pos = 100
    samples, observer = [], []
    for h in range(hours + 1):
        pos = position.get(h, pos)
        t = t0 + 3600 * h
        samples.append({"t": t, "outdoor": {"ghi": ghi.get(h, 0.0)}, "covers": {"cover.r": {"position": pos}}})
        observer.append({"t": t, "rooms": {"r": {"mode": "heating" if h in heating else "idle"}}})
    events = [
        {
            "t": t0 + 3600 * h,
            "type": "command",
            "entity_id": "cover.r",
            "service": "set_cover_position",
            "data": {"position": p},
            "result": "accepted",
        }
        for h, p in position.items()
    ]
    events.append(
        {
            "t": t0 + 3600 * 2,
            "type": "command",
            "entity_id": "cover.r",
            "service": "set_cover_position",
            "data": {"position": 5},
            "result": "rejected",
        }
    )
    return RunData(None, scn, samples, observer, events, t0 + 3600 * hours, {})  # type: ignore[arg-type]


def test_cover_metrics_count_moves_and_shading_per_local_day():
    # day 1 sunny, shaded 12-15 (3 h, one of them while heating); day 2 sunny, never shaded in the afternoon
    run = _cover_run(48, {12: 40, 15: 100, 30: 60, 31: 100}, {13: 700.0, 37: 650.0}, heating={14})
    m = _cover_metrics(run, "cover.r", "r")
    assert m["commands"] == 4  # the rejected one does not count
    assert m["max_commands_per_day"] == 2
    assert m["max_commands_per_day_unattended"] == 2
    assert m["commands_per_day"] == pytest.approx(2.0)
    assert m["shaded_h"] == pytest.approx(4.0)
    assert m["mean_shaded_position"] == pytest.approx((3 * 40 + 60) / 4)
    assert m["earliest_first_move_h"] == pytest.approx(6.0)  # day 2, 06:00
    assert m["mean_first_move_h"] == pytest.approx(9.0)
    assert m["shaded_while_heating_h"] == pytest.approx(1.0)
    assert m["sunny_days"] == 2
    assert m["sunny_days_unshaded"] == 1


def test_cover_metrics_without_sun_or_moves():
    m = _cover_metrics(_cover_run(24, {}, {12: 300.0}, heating=set()), "cover.r", "r")
    assert m["commands"] == 0
    assert m["max_commands_per_day"] == 0
    assert m["shaded_h"] == 0.0
    assert m["sunny_days"] == 0
    assert m["earliest_first_move_h"] is None
    assert m["mean_shaded_position"] is None


def test_idle_below_target_measures_the_longest_idle_stretch():
    temps = [19.0, 19.5, 19.5, 19.5, 20.5, 19.5, 19.5]
    idle = [False, True, True, True, True, True, False]
    m = _room_metrics(_run(temps, [21.0] * 7, idle), "r")
    # idle 1.5 K below from t1 to t3 (2 h); t4 is within 1 K, t5 idle again for 0 h
    assert m["idle_below_target_max_h"] == pytest.approx(2.0)


def test_truth_beta_device_per_running_share():
    run = _run([20.0] * 4, [21.0] * 4)
    run.scenario.rooms["r"].devices = [type("D", (), {"entity_id": e})() for e in ("climate.a", "climate.b")]
    devices = [
        {"climate.a": {"action": "idle", "heat_w": 0.0}, "climate.b": {"action": "idle", "heat_w": 0.0}},
        {"climate.a": {"action": "heating", "heat_w": 1000.0}, "climate.b": {"action": "heating", "heat_w": 1000.0}},
        {"climate.a": {"action": "heating", "heat_w": 1000.0}, "climate.b": {"action": "idle", "heat_w": 0.0}},
        {"climate.a": {"action": "cooling", "heat_w": -500.0}, "climate.b": {"action": "idle", "heat_w": 0.0}},
    ]
    for s, d in zip(run.samples, devices, strict=True):
        s["devices"] = d
    # 3000 W over a running share of 1.5 -> 2000 W per full label, 3.6 MJ/K -> 2 K/h
    assert _truth_beta_device(run, "r", 3.6e6, "heating") == pytest.approx(2.0)
    assert _truth_beta_device(run, "r", 3.6e6, "cooling") == pytest.approx(1.0)
    run.samples = run.samples[:1]
    assert _truth_beta_device(run, "r", 3.6e6, "heating") is None


def test_cover_moves_on_a_day_with_a_manual_move_are_left_out_of_the_unattended_count():
    run = _cover_run(48, {12: 40, 15: 100, 30: 60}, {13: 700.0}, heating=set())
    run.events.append(
        {"t": run.start + 3600 * 13, "type": "action", "action": "cover.manual", "args": {"entity_id": "cover.r"}}
    )
    m = _cover_metrics(run, "cover.r", "r")
    assert m["max_commands_per_day"] == 2
    assert m["max_commands_per_day_unattended"] == 1


def test_night_blower_hours_count_the_fast_fan_separately():
    """Hourly samples from local midnight; each interval is booked at its end."""
    scn = load_scenario_dict(scenario({"r": {"devices": [{"entity_id": "climate.d", "profile": "generic_split_ac"}]}}))
    t0 = scn.start

    def dev(hour: int) -> dict:
        if hour in (0, 1, 2):
            return {"mode": "cool", "active": False, "fan_mode": "Auto"}
        if hour in (3, 4, 5):
            return {"mode": "cool", "active": True, "fan_mode": "Quiet"}
        if hour == 23:
            return {"mode": "heat", "active": False, "fan_mode": "Auto"}  # thermo-off: blower stopped
        return {"mode": "cool", "active": True, "fan_mode": "High"}  # daytime: not counted

    samples = [
        {"t": t0 + 3600 * i, "devices": {"climate.d": {"electric_w": 0, "heat_w": 0, **dev(i % 24)}}}
        for i in range(25)
    ]
    run = RunData(None, scn, samples, [], [], t0 + 3600 * 24, {})  # type: ignore[arg-type]
    m = _device_metrics(run, "climate.d", 24.0)
    assert m["night_blower_h"] == pytest.approx(6.0)  # intervals ending 01-05 and 00 of the next day
    assert m["night_fast_fan_h"] == pytest.approx(3.0)
