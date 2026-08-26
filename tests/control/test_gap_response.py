"""Tests for the learned AC gap→cooling-rate response."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.roommind.const import AC_MAX_HEAD_GAP_C
from custom_components.roommind.control.gap_response import (
    MIN_IDENTIFIED_KNOTS,
    MIN_TOTAL_SAMPLES,
    GapResponse,
    GapResponseManager,
    HeadOffset,
)
from custom_components.roommind.control.mpc_controller import MPCController
from custom_components.roommind.control.thermal_model import RCModel, RoomModelManager


def _train(curve: GapResponse, truth, gaps=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), rounds=12):
    for _ in range(rounds):
        for g in gaps:
            curve.observe(g, truth(g))
    return curve


def _saturating(gap: float) -> float:
    """A plausible inverter: near-linear then saturating at ~4 °C/h."""
    return 4.0 * (1.0 - 2.718281828 ** (-gap / 1.5))


def test_curve_starts_unconfident():
    curve = GapResponse()
    assert not curve.is_confident()
    assert curve.identified_knots == 0


def test_curve_learns_a_saturating_response():
    curve = _train(GapResponse(), _saturating)
    assert curve.is_confident()
    for gap in (0.75, 1.25, 2.5, 3.5, 5.0):
        assert abs(curve.rate_for_gap(gap) - _saturating(gap)) < 0.6, gap


def test_curve_stays_monotone_under_noisy_observations():
    """A dip in the data is noise: more gap can never buy less cooling."""
    curve = GapResponse()
    noisy = [3.0, 1.0, 3.5, 0.5, 3.8, 2.0, 4.0]
    for _ in range(10):
        for gap, rate in zip(curve.knots, noisy, strict=True):
            curve.observe(gap, rate)
    assert curve.values == sorted(curve.values)
    sampled = [curve.rate_for_gap(g / 4) for g in range(2, 25)]
    assert all(b >= a - 1e-9 for a, b in zip(sampled, sampled[1:], strict=False)), sampled


def test_zero_gap_means_zero_incremental_rate():
    curve = _train(GapResponse(), _saturating)
    assert curve.rate_for_gap(0.0) == 0.0


def test_gap_for_rate_inverts_the_curve():
    curve = _train(GapResponse(), _saturating)
    for rate in (0.5, 1.0, 2.0, 3.0):
        gap = curve.gap_for_rate(rate, max_gap=8.0)
        assert abs(curve.rate_for_gap(gap) - rate) < 0.05, rate


def test_gap_for_rate_stops_at_saturation_instead_of_extrapolating():
    """Past saturation, more gap buys noise — return where the curve flattens,
    not the last knot: an over-demanded rate must land at the knee."""
    curve = _train(GapResponse(), _saturating)
    gap = curve.gap_for_rate(50.0, max_gap=8.0)
    assert gap < curve.knots[-1]
    # The knee already delivers (nearly) everything the curve ever measured
    assert curve.rate_for_gap(gap) >= 0.9 * curve.rate_for_gap(curve.knots[-1])


def test_gap_for_rate_respects_the_max_gap_cap():
    curve = _train(GapResponse(), _saturating)
    gap = curve.gap_for_rate(50.0, max_gap=2.0)
    assert gap <= 2.0
    assert curve.rate_for_gap(gap) >= 0.9 * curve.rate_for_gap(2.0)


def test_small_rate_requests_yield_small_gaps():
    """The whole point: a gentle demand must not command a large gap."""
    curve = _train(GapResponse(), _saturating)
    assert curve.gap_for_rate(0.3, max_gap=8.0) < 1.0


def test_rejects_unphysical_observations():
    curve = GapResponse()
    assert not curve.observe(gap=-1.0, rate=2.0)
    assert not curve.observe(gap=1.0, rate=-5.0)
    assert not curve.observe(gap=1.0, rate=1e6)
    assert curve.n_observations == 0


def test_curve_survives_a_round_trip():
    curve = _train(GapResponse(), _saturating)
    restored = GapResponse.from_dict(curve.to_dict())
    assert restored.is_confident()
    assert restored.n_observations == curve.n_observations
    for gap in (0.5, 1.5, 3.0):
        assert abs(restored.rate_for_gap(gap) - curve.rate_for_gap(gap)) < 0.01


def test_confidence_needs_a_spread_of_gaps_not_just_samples():
    """Hammering one gap must not certify the curve.

    A controller in steady state sits at one gap forever; the kernel would
    certify the knots around it, but the resulting curve is flat and says
    nothing about which gap to choose.
    """
    curve = GapResponse()
    for _ in range(MIN_TOTAL_SAMPLES * 2):
        curve.observe(1.0, 2.0)
    assert curve.n_observations >= MIN_TOTAL_SAMPLES
    assert curve.identified_knots >= MIN_IDENTIFIED_KNOTS  # kernel spread alone
    assert not curve.is_confident(), "one operating point must not certify a slope"

    # A spread of gaps does certify it.
    for _ in range(12):
        for gap in (0.5, 1.0, 2.0, 3.0):
            curve.observe(gap, _saturating(gap))
    assert curve.is_confident()


def test_head_offset_prefers_the_running_estimate():
    off = HeadOffset()
    for _ in range(20):
        off.observe(head_temp=24.0, room_temp=22.0, is_running=False)  # stratified, idle
        off.observe(head_temp=22.8, room_temp=22.0, is_running=True)  # fan moving air
    assert abs(off.commanding_offset() - 0.8) < 0.1


def test_head_offset_falls_back_to_zero_when_unknown():
    assert HeadOffset().commanding_offset() == 0.0


def test_head_offset_rejects_implausible_readings():
    off = HeadOffset()
    assert not off.observe(head_temp=99.0, room_temp=22.0, is_running=True)
    assert off.running is None


def test_manager_keeps_curves_separate_per_device_and_mode():
    mgr = GapResponseManager()
    mgr.curve("climate.a", "cooling").observe(1.0, 2.0)
    assert mgr.curve("climate.a", "heating").n_observations == 0
    assert mgr.curve("climate.b", "cooling").n_observations == 0
    assert mgr.curve("climate.a", "cooling").n_observations == 1


def test_manager_derives_rate_from_passive_drift_residual():
    """Only cooling beyond what the room would have done on its own counts."""
    mgr = GapResponseManager()
    # Room fell 0.5°C in 30 min, but would have risen 0.1°C unaided:
    # the unit did 0.6°C in 30 min = 1.2°C/h.
    mgr.observe_response(
        "climate.ac",
        "cooling",
        gap=2.0,
        observed_temp_change=-0.5,
        predicted_passive_change=0.1,
        dt_minutes=30.0,
    )
    curve = mgr.curve("climate.ac", "cooling")
    assert curve.n_observations == 1
    # Repeat until the recursive mean converges: the knot must settle on the
    # residual rate (1.2 °C/h), not on the raw observed change (1.0 °C/h).
    for _ in range(60):
        mgr.observe_response(
            "climate.ac",
            "cooling",
            gap=2.0,
            observed_temp_change=-0.5,
            predicted_passive_change=0.1,
            dt_minutes=30.0,
        )
    assert abs(curve.rate_for_gap(2.0) - 1.2) < 0.05


def test_manager_survives_a_round_trip():
    mgr = GapResponseManager()
    _train(mgr.curve("climate.ac", "cooling"), _saturating)
    mgr.observe_offset("climate.ac", head_temp=22.8, room_temp=22.0, is_running=True)
    restored = GapResponseManager.from_dict(mgr.to_dict())
    assert restored.curve("climate.ac", "cooling").is_confident()
    assert restored.offset("climate.ac").n_running == 1


# --- integration with the controller -----------------------------------------


def _ac_hass(head_temp=22.6, setpoint=21.0):
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    ac = MagicMock()
    ac.state = "cool"
    ac.attributes = {
        "hvac_modes": ["cool", "off"],
        "temperature": setpoint,
        "current_temperature": head_temp,
        "min_temp": 16.0,
        "max_temp": 30.0,
        "target_temp_step": 0.5,
    }
    hass.states.get = MagicMock(return_value=ac)
    return hass


def _ac_room():
    return {
        "area_id": "living_room",
        "thermostats": [],
        "acs": ["climate.ac"],
        "climate_mode": "cool_only",
        "devices": [{"entity_id": "climate.ac", "type": "ac", "role": "auto", "setpoint_mode": "proportional"}],
    }


async def _commanded(gap_mgr, pf, current_temp=22.1, target=22.0):
    # Head reads the same as the room so the head-frame shift is zero and
    # these tests isolate the learned-curve gating from the bias translation.
    hass = _ac_hass(head_temp=current_temp)
    mgr = RoomModelManager()
    mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))
    ctrl = MPCController(
        hass,
        _ac_room(),
        model_manager=mgr,
        outdoor_temp=32.0,
        settings={},
        has_external_sensor=True,
        gap_manager=gap_mgr,
    )
    await ctrl.async_apply("cooling", target, power_fraction=pf, current_temp=current_temp)
    sent = [c[0][2]["temperature"] for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    return sent[-1] if sent else None


@pytest.mark.asyncio
async def test_unidentified_curve_leaves_the_heuristic_in_charge():
    """Until the curve is identified the previous behaviour must be untouched."""
    gap_mgr = GapResponseManager()
    assert not gap_mgr.curve("climate.ac", "cooling").is_confident()
    # Same value the heuristic path produces for a 0.5-step device.
    assert await _commanded(gap_mgr, pf=1.0) == 21.5


@pytest.mark.asyncio
async def test_identified_curve_sizes_the_gap_from_measurement():
    """Once identified, the setpoint is (room + offset) - gap_for_rate."""
    gap_mgr = GapResponseManager()
    _train(gap_mgr.curve("climate.ac", "cooling"), _saturating)
    for _ in range(20):
        gap_mgr.offset("climate.ac").observe(head_temp=22.6, room_temp=22.0, is_running=True)

    # Q_cool = 4.0 °C/h, so pf=0.15 asks for 0.6 °C/h. The learned curve reaches
    # that at a small gap, and offset ~0.6 shifts the anchor to the head sensor.
    gentle = await _commanded(gap_mgr, pf=0.15)
    hard = await _commanded(gap_mgr, pf=1.0)
    assert gentle > hard, "a smaller demand must command a smaller gap"
    assert 22.1 + 0.6 - gentle < 1.5, "gentle demand must not open a large head gap"


@pytest.mark.asyncio
async def test_learned_path_never_exceeds_the_single_gap_ceiling():
    """The one safety parameter: no curve may ask for more than AC_MAX_HEAD_GAP_C."""
    gap_mgr = GapResponseManager()
    # A curve that barely cools at any gap would otherwise demand an extreme setpoint.
    _train(gap_mgr.curve("climate.ac", "cooling"), lambda g: 0.05 * g)
    for _ in range(20):
        gap_mgr.offset("climate.ac").observe(head_temp=22.0, room_temp=22.0, is_running=True)
    sp = await _commanded(gap_mgr, pf=1.0)
    # Ceiling respected before the device's own 0.5°C step quantization.
    assert 22.1 - sp <= AC_MAX_HEAD_GAP_C + 0.5


# ---------------------------------------------------------------------------
# Saturation knee + release offset selection
# ---------------------------------------------------------------------------


def test_gap_for_rate_saturates_at_the_knee_not_the_cap():
    """An over-demanded rate lands where the curve stops climbing.

    With a plateau from 3 K onward, requesting an unattainable rate (e.g.
    from a model whose Q_cool is still inflated) must return ~the knee, not
    the 6 K cap — otherwise every saturated request slams the deepest
    possible setpoint.
    """
    curve = GapResponse.from_dict(
        {
            "knots": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            "values": [0.5, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5],
            "counts": [5, 5, 5, 5, 5, 5, 5],
            "n_observations": 40,
            "gap_min": 0.5,
            "gap_max": 4.0,
        }
    )
    knee = curve.gap_for_rate(10.0, AC_MAX_HEAD_GAP_C)
    assert knee < 3.5, f"saturated request returned {knee}, expected the ~3K knee"
    assert knee >= 2.0
    # Within the measured range the inversion is unchanged
    assert abs(curve.gap_for_rate(1.0, AC_MAX_HEAD_GAP_C) - 1.0) < 0.1


def test_release_offset_prefers_most_adverse_reading():
    ho = HeadOffset()
    ho.running, ho.n_running = -2.0, 10
    ho.idle, ho.n_idle = 1.0, 10
    assert ho.commanding_offset() == -2.0  # active: return-air value
    assert ho.release_offset("cool") == 1.0  # release must clear the idle reading
    assert ho.release_offset("heat") == -2.0
    assert HeadOffset().release_offset("cool") == 0.0


@pytest.mark.asyncio
async def test_release_clears_idle_offset_not_running():
    """A cooling release must use the idle offset, not the running one.

    Running (return-air) offset is -2 K while idle (stratified) is +1 K: the
    old commanding_offset preference under-shot the release by 3 K and the
    unit re-armed as soon as its fan slowed.
    """
    gap_mgr = GapResponseManager()
    ho = gap_mgr.offset("climate.ac")
    ho.running, ho.n_running = -2.0, 10
    ho.idle, ho.n_idle = 1.0, 10
    hass = _ac_hass(head_temp=20.9)
    mgr = RoomModelManager()
    mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))
    ctrl = MPCController(
        hass,
        _ac_room(),
        model_manager=mgr,
        outdoor_temp=32.0,
        settings={},
        has_external_sensor=True,
        gap_manager=gap_mgr,
    )
    # Room below target → release; base 21.0 + idle offset 1.0 = 22.0
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.5)
    sent = [c[0][2]["temperature"] for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert 22.0 in sent


@pytest.mark.asyncio
async def test_active_command_keeps_running_offset():
    """Active runs still translate with the running (return-air) offset."""
    gap_mgr = GapResponseManager()
    ho = gap_mgr.offset("climate.ac")
    ho.running, ho.n_running = -2.0, 10
    ho.idle, ho.n_idle = 1.0, 10
    hass = _ac_hass(head_temp=24.0)
    mgr = RoomModelManager()
    mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))
    ctrl = MPCController(
        hass,
        _ac_room(),
        model_manager=mgr,
        outdoor_temp=32.0,
        settings={},
        has_external_sensor=True,
        gap_manager=gap_mgr,
    )
    # Room 26, target 23, pf 0.5 → room-frame 21.0; running offset -2 → 19.0
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.5, current_temp=26.0)
    sent = [c[0][2]["temperature"] for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert 19.0 in sent
