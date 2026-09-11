"""Tests for proportional TRV setpoints, power calculations, AC proportional control, dynamic boost."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.roommind.control.mpc_controller import (
    MPCController,
    TargetTemps,
)
from custom_components.roommind.control.thermal_model import RCModel, RoomModelManager

from .conftest import build_hass, make_room


@pytest.mark.asyncio
async def test_proportional_power_far_from_target():
    """MPC mode, large error → power_fraction near 1.0."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    model_mgr.update("living_room", 15.0, 5.0, "heating", 5.0)
    model_mgr.update("living_room", 16.0, 5.0, "heating", 5.0)
    model_mgr.get_prediction_std = MagicMock(return_value=0.1)
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
    # Mock a realistic trained model (2 EKF updates give alpha=_ALPHA_MIN which is
    # too low for the optimizer to distinguish heating from idle via T_eq clamping)
    model_mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))
    ctrl = MPCController(
        hass,
        room,
        model_manager=model_mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=15.0, target_temp=21.0)
    assert mode == "heating"
    assert pf >= 0.7  # large error → high power


@pytest.mark.asyncio
async def test_proportional_power_near_target():
    """MPC mode, small error → reduced power_fraction."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    # Use a known model with moderate Q_heat so a small 0.3°C error yields frac < 1.
    # This tests MPC proportional behavior, not EKF learning.
    model_mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=8.0, Q_cool=10.0))
    model_mgr.get_prediction_std = MagicMock(return_value=0.1)
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 40, 0))
    ctrl = MPCController(
        hass,
        room,
        model_manager=model_mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=20.7, target_temp=21.0)
    assert mode is not None
    assert mode == "heating"
    assert pf < 1.0  # near target → less than full power


@pytest.mark.asyncio
async def test_proportional_trv_setpoint():
    """TRV setpoint is proportional between current_temp and 30°C."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=model_mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    # 50% power at 20°C → TRV = 20 + 0.5*(30-20) = 25°C
    await ctrl.async_apply("heating", 21.0, power_fraction=0.5, current_temp=20.0)
    calls = hass.services.async_call.call_args_list
    set_temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert set_temp_calls
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == 25.0


@pytest.mark.asyncio
async def test_proportional_mixed_trv_ac_half_power():
    """Mixed TRV+AC room at 50% power: both get correct proportional targets."""
    hass = build_hass()

    trv_state = MagicMock()
    trv_state.state = "heat"
    trv_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 21.0, "min_temp": 5.0, "max_temp": 30.0}

    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {
        "hvac_modes": ["heat", "cool", "off"],
        "temperature": 20.0,
        "min_temp": 16.0,
        "max_temp": 30.0,
    }

    def states_get(eid):
        if eid == "climate.trv":
            return trv_state
        if eid == "climate.ac":
            return ac_state
        return None

    hass.states.get = MagicMock(side_effect=states_get)

    room = make_room(thermostats=["climate.trv"], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=0.5, current_temp=18.0)

    calls = hass.services.async_call.call_args_list
    # TRV: 18 + 0.5*(30-18) = 24.0
    trv_temp = [c for c in calls if c[0][1] == "set_temperature" and c[0][2].get("entity_id") == "climate.trv"]
    assert trv_temp and trv_temp[0][0][2]["temperature"] == 24.0
    # AC: 18 + 0.5*(30-18) = 24.0
    ac_temp = [c for c in calls if c[0][1] == "set_temperature" and c[0][2].get("entity_id") == "climate.ac"]
    assert ac_temp and ac_temp[0][0][2]["temperature"] == 24.0


@pytest.mark.asyncio
async def test_proportional_ac_heating_half_power():
    """AC heating at 50% power gets proportional boost between current and 30°C."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat", "cool", "off"], "temperature": 20.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=0.5, current_temp=20.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    # 20 + 0.5*(30-20) = 25.0, capped by the error-scaled AC setpoint limit at
    # the default slider (gain 1.2 at comfort_weight 70): 21 + 1.2*1.0 = 22.2
    assert any(c[0][2]["temperature"] == 22.2 for c in temp_calls)


@pytest.mark.asyncio
async def test_proportional_ac_cooling_half_power():
    """AC cooling at 50% power gets proportional boost between current and 16°C."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 23.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=35.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.5, current_temp=26.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    # 26 - 0.5*(26-16) = 21.0
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_proportional_ac_heating_clamped_floor():
    """Very low power heating: AC target clamped to effective_target floor."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 20.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    # Holding regime (pf <= MIN): rung servo starts at the quiet end,
    # one step above the parked release (21.0 - 1.0 + 0.5, no head data).
    await ctrl.async_apply("heating", 21.0, power_fraction=0.01, current_temp=20.5)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 20.5 for c in temp_calls)


@pytest.mark.asyncio
async def test_proportional_ac_cooling_clamped_ceiling():
    """Very low power cooling: AC target clamped to effective_target ceiling."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 25.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=35.0,
        settings={},
        has_external_sensor=True,
    )
    # Holding regime (pf <= MIN): the rung servo starts at the quiet end,
    # one step below the parked release (23.0 + 1.0 - 0.5, no head data).
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.01, current_temp=23.5)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 23.5 for c in temp_calls)


@pytest.mark.asyncio
async def test_proportional_ac_heating_no_current_temp():
    """AC heating without current_temp falls back to effective_target."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat", "cool", "off"], "temperature": 20.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=0.8)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_proportional_ac_cooling_no_current_temp():
    """AC cooling without current_temp falls back to effective_target."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 25.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=35.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.8)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 23.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_proportional_ac_managed_mode_unchanged():
    """Managed mode AC gets actual target, NOT proportional boost (regression guard)."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat_cool", "heat", "cool", "off"], "temperature": 20.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(
        thermostats=[],
        acs=["climate.ac"],
        climate_mode="auto",
        temperature_sensor="",
    )
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=False,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=0.5, current_temp=18.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    # Managed mode: AC should get actual target (21.0), not proportional boost
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


# ---------------------------------------------------------------------------
# Dynamic boost target tests (#76)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_heating_boost_trv_full_power():
    """TRV at full power uses dynamic boost target (35) instead of default 30."""
    hass = build_hass()
    trv_state = MagicMock()
    trv_state.state = "off"
    trv_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 20.0, "max_temp": 35.0}
    hass.states.get = MagicMock(return_value=trv_state)

    room = make_room(thermostats=["climate.trv"], acs=[])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=20.0, heating_boost_target=35.0)

    temp_calls = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 35.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_dynamic_heating_boost_none_fallback():
    """When heating_boost_target is None, falls back to HEATING_BOOST_TARGET (30)."""
    hass = build_hass()
    trv_state = MagicMock()
    trv_state.state = "off"
    trv_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 20.0}
    hass.states.get = MagicMock(return_value=trv_state)

    room = make_room(thermostats=["climate.trv"], acs=[])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=20.0, heating_boost_target=None)

    temp_calls = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 30.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_dynamic_heating_boost_proportional():
    """TRV at 50% power with dynamic boost=35: 20 + 0.5*(35-20) = 27.5."""
    hass = build_hass()
    trv_state = MagicMock()
    trv_state.state = "off"
    trv_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 20.0, "max_temp": 35.0}
    hass.states.get = MagicMock(return_value=trv_state)

    room = make_room(thermostats=["climate.trv"], acs=[])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=0.5, current_temp=20.0, heating_boost_target=35.0)

    temp_calls = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 27.5 for c in temp_calls)


@pytest.mark.asyncio
async def test_dynamic_cooling_boost_full_power():
    """AC at full cooling power uses dynamic boost (18) instead of default 16."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 23.0, "min_temp": 18.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=35.0,
        settings={},
        has_external_sensor=True,
    )
    # Room 6°C over target so the error-scaled limit is slack and the device
    # boost target is the binding constraint.
    await ctrl.async_apply("cooling", 23.0, power_fraction=1.0, current_temp=29.0, cooling_boost_target=18.0)

    temp_calls = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 18.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_dynamic_cooling_boost_none_fallback():
    """When cooling_boost_target is None, falls back to AC_COOLING_BOOST_TARGET (16)."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 23.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=35.0,
        settings={},
        has_external_sensor=True,
    )
    # Room 6°C over target so the error-scaled limit is slack and the fallback
    # boost constant is the binding constraint: 29 - 1.0*(29-16) = 16.0
    await ctrl.async_apply("cooling", 23.0, power_fraction=1.0, current_temp=29.0, cooling_boost_target=None)

    temp_calls = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 16.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_dynamic_ac_heating_boost():
    """AC in heating mode uses ac_heating_boost_target instead of default 30."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat", "cool", "off"], "temperature": 20.0, "max_temp": 28.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    # Room 7°C below target so the error-scaled setpoint limit is slack and the
    # device boost target is the binding constraint.
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=14.0, ac_heating_boost_target=28.0)

    temp_calls = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 28.0 for c in temp_calls)


def _ctrl_with_cw(cw):
    hass = build_hass()
    room = make_room()
    settings = {} if cw is None else {"comfort_weight": cw}
    return MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings=settings,
        has_external_sensor=True,
    )


def test_slider_default_and_comfort_keep_approach_rate_one():
    assert _ctrl_with_cw(None)._approach_rate == 1.0  # default cw=70
    assert _ctrl_with_cw(70)._approach_rate == 1.0
    assert _ctrl_with_cw(100)._approach_rate == 1.0


def test_slider_efficiency_lowers_approach_rate():
    assert _ctrl_with_cw(0)._approach_rate == pytest.approx(0.2)
    assert _ctrl_with_cw(35)._approach_rate == pytest.approx(0.6)


def test_slider_default_and_comfort_keep_ac_cap_unbounded():
    assert _ctrl_with_cw(None)._ac_boost_delta == 50.0
    assert _ctrl_with_cw(70)._ac_boost_delta == 50.0
    assert _ctrl_with_cw(100)._ac_boost_delta == 50.0


def test_slider_efficiency_tightens_ac_cap():
    assert _ctrl_with_cw(0)._ac_boost_delta == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_ac_boost_cap_limits_setpoint_at_efficiency():
    """At full efficiency the AC heating setpoint is capped at target + 3°C."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "heat"
    ac_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 21.0, "min_temp": 16.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={"comfort_weight": 0},
        has_external_sensor=True,
    )
    # pf=1.0 would map to boost 30°C. Room 10°C under target, so the
    # error-scaled limit (gain 0.5 at comfort_weight 0 -> 5.0) is slack and the
    # slider cap binds: target(21) + 3 = 24°C.
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=11.0)
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp
    assert set_temp[-1][0][2]["temperature"] == 24.0


@pytest.mark.asyncio
async def test_ac_boost_cap_does_not_apply_at_comfort():
    """At comfort/default the cap is unbounded; AC reaches boost as today."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "heat"
    ac_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 21.0, "min_temp": 16.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    # Same room as the efficiency case above: at the default slider the cap is
    # unbounded and the AC reaches its boost target.
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=11.0)
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp
    assert set_temp[-1][0][2]["temperature"] == 30.0


@pytest.mark.asyncio
async def test_ac_cooling_boost_cap_floors_setpoint_at_efficiency():
    """At full efficiency the AC cooling setpoint is floored at target - 3°C."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "cool"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 23.0, "min_temp": 16.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=30.0,
        settings={"comfort_weight": 0},
        has_external_sensor=True,
    )
    # pf=1.0 would map to cool boost 16°C. Room 8°C over target, so the
    # error-scaled limit (gain 0.5 at comfort_weight 0 -> 4.0) is slack and the
    # slider cap binds: target(23) - 3 = 20°C.
    await ctrl.async_apply("cooling", 23.0, power_fraction=1.0, current_temp=31.0)
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp
    assert set_temp[-1][0][2]["temperature"] == 20.0


def settings_for(cw):
    return {} if cw is None else {"comfort_weight": cw}


def _make_controller(cw):
    hass = build_hass()
    room = make_room()
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings=settings_for(cw),
        has_external_sensor=True,
    )
    return hass, ctrl


def _mock_device(hass, setpoint):
    dev = MagicMock()
    dev.state = "heat"
    dev.attributes = {"hvac_modes": ["heat", "off"], "temperature": setpoint, "min_temp": 16.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=dev)
    return dev


def test_proportional_deadband_helper_disabled_at_comfort():
    _, ctrl = _make_controller(None)  # cw=70 default
    assert ctrl._proportional_deadband("climate.x", 18.0, 22.0) is None


def test_proportional_deadband_helper_values_at_efficiency():
    from custom_components.roommind.const import (
        PROPORTIONAL_DEADBAND_C,
        PROPORTIONAL_DEADBAND_NEAR_TARGET_C,
    )

    _, ctrl = _make_controller(0)  # full efficiency
    assert ctrl._proportional_deadband("climate.x", 18.0, 22.0) == PROPORTIONAL_DEADBAND_C
    assert ctrl._proportional_deadband("climate.x", 21.5, 22.0) == PROPORTIONAL_DEADBAND_NEAR_TARGET_C


def test_proportional_deadband_helper_none_for_direct_device():
    _, ctrl = _make_controller(0)
    ctrl._direct_eids = {"climate.direct"}
    assert ctrl._proportional_deadband("climate.direct", 18.0, 22.0) is None


def test_proportional_deadband_helper_none_when_current_temp_unknown():
    _, ctrl = _make_controller(0)  # full efficiency
    assert ctrl._proportional_deadband("climate.x", None, 22.0) is None


@pytest.mark.asyncio
async def test_call_deadband_suppresses_subthreshold_change():
    hass, ctrl = _make_controller(0)
    _mock_device(hass, setpoint=22.0)
    await ctrl._call(
        "set_temperature", {"entity_id": "climate.x", "temperature": 22.3}, temp_intent="heat", deadband=0.5
    )
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp == []  # 0.3 < 0.5 → suppressed


@pytest.mark.asyncio
async def test_call_deadband_sends_suprathreshold_change():
    hass, ctrl = _make_controller(0)
    _mock_device(hass, setpoint=22.0)
    await ctrl._call(
        "set_temperature", {"entity_id": "climate.x", "temperature": 22.6}, temp_intent="heat", deadband=0.5
    )
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert len(set_temp) == 1  # 0.6 >= 0.5 → sent


@pytest.mark.asyncio
async def test_call_without_deadband_preserves_exact_behavior():
    hass, ctrl = _make_controller(None)
    _mock_device(hass, setpoint=22.0)
    await ctrl._call("set_temperature", {"entity_id": "climate.x", "temperature": 22.3}, temp_intent="heat")
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert len(set_temp) == 1  # no deadband → today's behavior: round(22.0,1) != round(22.3,1) → sent


@pytest.mark.asyncio
async def test_call_without_deadband_skips_when_rounds_equal():
    hass, ctrl = _make_controller(None)
    _mock_device(hass, setpoint=22.0)
    await ctrl._call("set_temperature", {"entity_id": "climate.x", "temperature": 22.04}, temp_intent="heat")
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp == []  # round(22.04,1)==round(22.0,1) → skipped, exactly as before


@pytest.mark.asyncio
async def test_call_deadband_near_target_finer_band():
    hass, ctrl = _make_controller(0)
    _mock_device(hass, setpoint=22.0)
    # 0.3°C change with the finer 0.2 near-target deadband → sent (0.3 >= 0.2)
    await ctrl._call(
        "set_temperature", {"entity_id": "climate.x", "temperature": 22.3}, temp_intent="heat", deadband=0.2
    )
    sent = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert len(sent) == 1
    # 0.15°C change with the 0.2 deadband → suppressed
    hass.services.async_call.reset_mock()
    _mock_device(hass, setpoint=22.0)
    await ctrl._call(
        "set_temperature", {"entity_id": "climate.x", "temperature": 22.15}, temp_intent="heat", deadband=0.2
    )
    sent = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert sent == []


@pytest.mark.asyncio
async def test_call_deadband_converts_to_fahrenheit_units():
    from homeassistant.const import UnitOfTemperature

    hass, ctrl = _make_controller(0)
    hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT
    dev = MagicMock()
    dev.state = "heat"
    dev.attributes = {"hvac_modes": ["heat", "off"], "temperature": 72.0, "min_temp": 60.0, "max_temp": 86.0}
    hass.states.get = MagicMock(return_value=dev)
    # deadband 0.5°C = 0.9°F → a 0.5°F change must be suppressed
    await ctrl._call(
        "set_temperature", {"entity_id": "climate.x", "temperature": 72.5}, temp_intent="heat", deadband=0.5
    )
    sent = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert sent == []
    # a 1.0°F change (>= 0.9°F) must be sent
    hass.services.async_call.reset_mock()
    dev.attributes["temperature"] = 72.0
    await ctrl._call(
        "set_temperature", {"entity_id": "climate.x", "temperature": 73.0}, temp_intent="heat", deadband=0.5
    )
    sent = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# Release-position anchoring (cold-evening AC overshoot fix)
# ---------------------------------------------------------------------------


def _make_cooling_ctrl():
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "cool"
    ac_state.attributes = {"hvac_modes": ["cool", "off"], "temperature": 20.0, "min_temp": 16.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)
    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=18.0,
        settings={},
        has_external_sensor=True,
    )
    return hass, ctrl


@pytest.mark.asyncio
async def test_ac_cooling_releases_when_room_below_target():
    """Room below cool target with holding-level power: parked release.

    Cold-evening scenario: ventilation already cools the room below the target
    while demand is at/below the holding minimum. The old anchor (current_temp)
    kept the commanded setpoint at or below the falling room temperature, so
    the AC could never release and chased the room downward. Now: holding
    regime → parity value = target; room below target → release → setback
    parking (21.0 + 1.0 with no head data).
    """
    hass, ctrl = _make_cooling_ctrl()
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.1, current_temp=20.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert temp_calls
    sp = temp_calls[0][0][2]["temperature"]
    assert sp == 22.0
    # The commanded setpoint must sit well above the room temperature
    assert sp > 20.0


@pytest.mark.asyncio
async def test_ac_cooling_zero_power_releases_at_target():
    """Zero cooling demand while below target: setpoint == target (full release)."""
    hass, ctrl = _make_cooling_ctrl()
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    # Parked at the setback position: target + 1.0 (no head data)
    assert any(c[0][2]["temperature"] == 22.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_ac_cooling_full_power_below_target_is_error_bounded():
    """Full power below target: the signed-error limit caps the excursion.

    The release anchor alone would map pf=1.0 to the device minimum, but a
    room already below target has zero correcting error, so only the floor
    excursion applies (default slider: 0.6°C below target). Deep pre-cooling
    below target is deliberately not available to the heuristic path.
    """
    hass, ctrl = _make_cooling_ctrl()
    await ctrl.async_apply("cooling", 21.0, power_fraction=1.0, current_temp=20.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 20.4 for c in temp_calls)


@pytest.mark.asyncio
async def test_trv_heating_releases_when_room_above_target():
    """Heating mirror: room above heat target with zero power → setpoint == target."""
    hass = build_hass()
    trv_state = MagicMock()
    trv_state.state = "heat"
    trv_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 22.0, "min_temp": 5.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=trv_state)
    room = make_room()
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    # Old anchor kept the setpoint at the room temp (22.0), continuing to heat past target
    await ctrl.async_apply("heating", 21.0, power_fraction=0.0, current_temp=22.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_ac_heating_releases_when_room_above_target():
    """Heating mirror for ACs: room above heat target with zero power → setpoint == target."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "heat"
    ac_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 22.0, "min_temp": 16.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)
    room = make_room(thermostats=[], acs=["climate.ac"])
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=0.0, current_temp=22.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    # Parked at the setback position: target - 1.0 (no head data)
    assert any(c[0][2]["temperature"] == 20.0 for c in temp_calls)


@pytest.mark.asyncio
async def test_trv_heating_full_power_above_target_still_boosts():
    """Deliberate pre-heating (pf=1.0) above target still commands the boost setpoint.

    TRVs carry no error-scaled limit — pre-heating past the current schedule
    target is exactly the UFH use case the full-power path exists for.
    """
    hass = build_hass()
    trv_state = MagicMock()
    trv_state.state = "heat"
    trv_state.attributes = {"hvac_modes": ["heat", "off"], "temperature": 22.0, "min_temp": 5.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=trv_state)
    room = make_room()
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=22.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    # anchor + 1.0*(boost - anchor) = boost = device max_temp (30.0)
    assert any(c[0][2]["temperature"] == 30.0 for c in temp_calls)


# ---------------------------------------------------------------------------
# Error-bounded AC setpoint gap (near-target blast fix)
# ---------------------------------------------------------------------------


def _cool_only_ac_hass(min_temp=16.0, step=None):
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "cool"
    attrs = {
        "hvac_modes": ["cool", "off"],
        "temperature": 22.0,
        "min_temp": min_temp,
        "max_temp": 30.0,
    }
    if step is not None:
        attrs["target_temp_step"] = step
    ac_state.attributes = attrs
    hass.states.get = MagicMock(return_value=ac_state)
    return hass


async def _cooling_setpoint(cw, current_temp, target=22.0, pf=1.0, step=None):
    """Commanded AC setpoint for one (slider, room temperature) combination."""
    hass = _cool_only_ac_hass(step=step)
    ctrl = MPCController(
        hass,
        make_room(thermostats=[], acs=["climate.ac"]),
        model_manager=RoomModelManager(),
        outdoor_temp=32.0,
        settings={} if cw is None else {"comfort_weight": cw},
        has_external_sensor=True,
    )
    await ctrl.async_apply("cooling", target, power_fraction=pf, current_temp=current_temp)
    sent = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    return sent[-1][0][2]["temperature"] if sent else None


@pytest.mark.asyncio
@pytest.mark.parametrize("cw", [0, 20, 50, 70, 100])
async def test_cooling_gap_stays_small_near_target(cw):
    """A 0.1°C excursion must never command a gap the AC answers with full output.

    The gap between room and setpoint is what drives compressor and fan speed,
    so it — not the excursion below target — is the quantity to bound.
    """
    sp = await _cooling_setpoint(cw, 22.1)
    assert 22.1 - sp <= 1.0, f"comfort_weight={cw} commanded a {22.1 - sp:.1f}°C gap"
    assert sp < 22.0, "the unit still needs some demand to run at all"


@pytest.mark.asyncio
async def test_priority_slider_has_authority_near_target():
    """Efficiency must command a gentler setpoint than Comfort close to target.

    Regression: the excursion limit used to be slider-independent, so every
    position from Efficiency to Comfort produced a bit-identical setpoint in
    exactly the band where the user notices the AC being loud.
    """
    for room_temp in (22.1, 22.5, 23.0):
        by_slider = [await _cooling_setpoint(cw, room_temp) for cw in (0, 35, 70, 100)]
        assert by_slider == sorted(by_slider, reverse=True), by_slider
        assert by_slider[0] > by_slider[-1], f"slider inert at room {room_temp}: {by_slider}"


@pytest.mark.asyncio
async def test_cooling_setpoint_scales_with_error():
    """Larger error buys a proportionally larger gap; pull-down still reaches the device floor."""
    # Default slider (comfort_weight 70): gain 1.2, floor 0.6 after 0.1°C quantization
    assert await _cooling_setpoint(None, 22.1) == 21.4  # floor binds
    assert await _cooling_setpoint(None, 23.0) == 20.8  # 22 - 1.2*1.0
    assert await _cooling_setpoint(None, 24.0) == 19.6  # 22 - 1.2*2.0
    assert await _cooling_setpoint(None, 27.0) == 16.0  # device minimum, full pull-down


@pytest.mark.asyncio
async def test_cooling_setpoint_survives_coarse_device_step():
    """On a whole-degree AC the excursion must not round back onto the target.

    Half a degree of intent is worth nothing to a device that only accepts whole
    degrees: it snaps to the target, the redundancy check then suppresses the
    send, and the AC is handed no demand at all.
    """
    for cw in (0, 70, 100):
        sp = await _cooling_setpoint(cw, 22.1, step=1.0)
        assert sp == 21.0, f"comfort_weight={cw} snapped back to {sp}"
    # A finer step keeps finer authority
    assert await _cooling_setpoint(70, 22.1, step=0.5) == 21.5


@pytest.mark.asyncio
async def test_cooling_setpoint_limit_ignores_overshoot_past_target():
    """A room already below the cool target must not buy a larger excursion.

    The limit is scaled by the error in the direction the mode is correcting.
    Using abs() would hand the widest allowance to exactly the overshoot the
    limit exists to prevent.
    """
    # Room 1°C *below* target: |error| = 1.0, correcting error = 0, floor only.
    assert await _cooling_setpoint(None, 21.0) == 21.4  # floor only


@pytest.mark.asyncio
async def test_ac_heating_setpoint_bounded_by_control_error():
    """The error-scaled limit is symmetric: small deficit → small heat boost."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat", "cool", "off"], "temperature": 21.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)

    ctrl = MPCController(
        hass,
        make_room(thermostats=[], acs=["climate.ac"]),
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=20.9)
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp
    assert set_temp[-1][0][2]["temperature"] == 21.6  # floor only, default slider


@pytest.mark.asyncio
async def test_ac_heating_setpoint_limit_ignores_overshoot_past_target():
    """Symmetric: a room already above the heat target gets only the floor."""
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = "off"
    ac_state.attributes = {"hvac_modes": ["heat", "cool", "off"], "temperature": 21.0, "max_temp": 30.0}
    hass.states.get = MagicMock(return_value=ac_state)

    ctrl = MPCController(
        hass,
        make_room(thermostats=[], acs=["climate.ac"]),
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=22.0)
    set_temp = [c for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]
    assert set_temp
    assert set_temp[-1][0][2]["temperature"] == 21.6


# ---------------------------------------------------------------------------
# Head-frame setpoint translation (coarse/biased AC head sensor)
# ---------------------------------------------------------------------------


def _head_ctrl(head_temp, step=None, state_mode="cool", modes=("cool", "off")):
    hass = build_hass()
    ac_state = MagicMock()
    ac_state.state = state_mode
    attrs = {
        "hvac_modes": list(modes),
        "temperature": None,
        "min_temp": 16.0,
        "max_temp": 30.0,
        "current_temperature": head_temp,
    }
    if step is not None:
        attrs["target_temp_step"] = step
    ac_state.attributes = attrs
    hass.states.get = MagicMock(return_value=ac_state)
    ctrl = MPCController(
        hass,
        make_room(thermostats=[], acs=["climate.ac"]),
        model_manager=RoomModelManager(),
        outdoor_temp=18.0,
        settings={},
        has_external_sensor=True,
    )
    return hass, ctrl


def _sent_temps(hass):
    return [c[0][2]["temperature"] for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]


@pytest.mark.asyncio
async def test_cooling_release_clears_warm_head():
    """Release below target parks one step past the head's own reading.

    Room 20.0 (below the 21.0 target) with the head reading 23.0: the old
    release at 21.0 kept the compressor running until the head — and so the
    room — fell a degree past target (the 1 K sawtooth). The park is the
    first level past the reading's quantization interval, 24.0 — no wider, so
    the re-engage that follows stays within the adjacent-levels dither.
    """
    hass, ctrl = _head_ctrl(head_temp=23.0)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.0)
    assert 26.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_cooling_release_with_a_cool_head_still_clears_the_target():
    """A head reading cooler than the room cannot park below the room's target.

    Head 19, room 20, target 21. One step past the reading is 19.5 — and a
    unit left at 19.5 re-arms as soon as its reading hits 19.5, i.e. the room
    at 20.5, half a degree UNDER the target it was parked to respect. The
    parked unit becomes the binding thermostat at a temperature nobody chose,
    and the MPC, seeing no deficit, never intervenes to correct it. The park
    is floored at the target's own level, 21.0.
    """
    hass, ctrl = _head_ctrl(head_temp=19.0)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.0)
    assert 22.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_cooling_active_command_takes_full_shift():
    """Active commands translate fully so the delivered gap matches the intent."""
    hass, ctrl = _head_ctrl(head_temp=25.0)
    # Room 26, target 23, pf 0.5 → room-frame 21.0; head reads 1.0 cold → 20.0
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.5, current_temp=26.0)
    assert 20.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_cooling_hold_at_target_starts_gentle():
    """Room just above target with no demand: servo starts at the quiet end,
    one step below the observed park (head 22.2 → park 22.5 → start 22.0):
    the lower of the two levels adjacent to the reading."""
    hass, ctrl = _head_ctrl(head_temp=22.2)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=21.2)
    assert 22.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_cooling_release_near_target_winds_down_gradually():
    """Just below target the servo winds down one rung above parity instead
    of slamming straight to the parked release: park 22.8 (head bias +0.8,
    1.0 parking) minus one 1.0 step → 21.8 → snapped to 22.0, short of the
    park."""
    hass, ctrl = _head_ctrl(head_temp=21.4, step=1.0)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.6)
    assert 22.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_heating_release_clears_cold_head():
    """Heating mirror: park below the reading and its interval (20.0 → 19.0)."""
    hass, ctrl = _head_ctrl(head_temp=20.0, state_mode="heat", modes=("heat", "off"))
    await ctrl.async_apply("heating", 21.0, power_fraction=0.0, current_temp=22.0)
    assert 17.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_compressor_hold_translates_to_head_frame():
    """A min-run hold keeps the AC engaged at the target as the DEVICE
    perceives it — head-frame shift, but no setback parking."""
    hass, ctrl = _head_ctrl(head_temp=23.0)
    await ctrl.async_apply(
        "idle",
        TargetTemps(heat=None, cool=21.0),
        power_fraction=0.0,
        current_temp=20.0,
        compressor_forced_on={"climate.ac"},
    )
    assert 24.0 in _sent_temps(hass)  # 21.0 + (23.0 - 20.0), no parking


@pytest.mark.asyncio
async def test_active_command_quantized_by_controller_not_device():
    """Active commands land on exact device steps, ties toward demand.

    Sending 0.1°-precision values lets the device round them unpredictably,
    so consecutive commands a few tenths apart could jump two whole degrees.
    """
    hass, ctrl = _head_ctrl(head_temp=22.0, step=1.0)
    # Excursion path: room-frame 20.0 (error-limit floor) + 0.8 shift = 20.8 → 21.0
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.4, current_temp=21.2)
    assert 21.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_active_command_tie_rounds_toward_demand():
    hass, ctrl = _head_ctrl(head_temp=21.7, step=1.0)
    # Room-frame 20.0 + 0.5 shift = 20.5 → tie → 20.0 (deeper, the demand side)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.4, current_temp=21.2)
    assert 20.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_holding_power_engages_the_rung_servo():
    """Holding-level demand hands the setpoint to the rung servo.

    The servo starts one device-step gentler than parity — the rung a
    trickling unit most likely balances a small load at — and then steps
    closed-loop on the room, at most once per dwell.
    """
    hass, ctrl = _head_ctrl(head_temp=22.3, step=1.0)
    # park = 21.0 + 1.0 head bias + 1.0 setback = 23.0; start one step below → 22.0
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.15, current_temp=21.3)
    assert 22.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_rung_servo_steps_toward_demand_when_room_stays_warm():
    """A rung that underdelivers is lowered one step after the dwell."""
    import time as _time

    from custom_components.roommind.control.mpc_controller import _hold_rungs

    hass, ctrl = _head_ctrl(head_temp=22.3, step=1.0)
    _hold_rungs["climate.ac"] = (23.0, _time.time() - 1000.0)  # dwell elapsed
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.15, current_temp=21.3)
    assert 22.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_rung_servo_respects_the_dwell():
    """No adjustment before the dwell elapses — shifts stay slow."""
    import time as _time

    from custom_components.roommind.control.mpc_controller import _hold_rungs

    hass, ctrl = _head_ctrl(head_temp=22.3, step=1.0)
    _hold_rungs["climate.ac"] = (23.0, _time.time() - 10.0)  # just adjusted
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.15, current_temp=21.3)
    assert 23.0 in _sent_temps(hass)


# ---------------------------------------------------------------------------
# Observed-reading park: with a whole-degree actuator the only optimal steady
# states are a fixed level or a dither between two ADJACENT levels. The head
# reports the reading it regulates against, so the level it is satisfied at
# is directly observable from that reading — no estimate, learned offset or
# safety margin, each of which only widened the release/re-engage swing (the
# 3 K jumps users saw on the head unit). What the reading does NOT give for
# free is its own resolution: a head reporting on the step grid has published
# an interval, not a value, and a park inside that interval is only a park
# when the truth happens to sit at the bottom of it.
# ---------------------------------------------------------------------------


def test_observed_park_clears_the_readings_quantization_interval():
    """An on-grid reading is an interval; an off-grid one is a value.

    A head reporting 22 on a whole-degree ladder has said "somewhere in
    [22, 23)", so a park at 23 only parks if the truth is near 22.0. Field
    case: the living room commanded 23 against a reported 22 and was not
    meaningfully backed off. The park clears the whole interval, 24. A head
    with real resolution lands off the grid and keeps the lean one-step park.
    """
    from custom_components.roommind.control.mpc_controller import observed_park_level

    # Unlatched throughout: these are the base rule, and the sub-cases share
    # an entity id, so a latch carried between them would mask it.
    park = lambda h, intent="cool": observed_park_level(h, "climate.ac", intent, latch=False)  # noqa: E731

    assert park(_head_ctrl(head_temp=22.0, step=1.0)[0]) == 25.0
    assert park(_head_ctrl(head_temp=22.0, step=1.0, state_mode="heat", modes=("heat", "off"))[0], "heat") == 19.0
    # The margin is a temperature, not a count of steps: snapped up to the
    # ladder, 22.2 + 3.0 lands on 25.5 at half-degree steps and 26.0 at whole.
    assert park(_head_ctrl(head_temp=22.2, step=0.5)[0]) == 25.5
    assert park(_head_ctrl(head_temp=22.2, step=1.0)[0]) == 26.0
    # margin_c=0.0 gives back the lean bound the holding servo runs against,
    # still strictly past the reading even when the reading sits on a level.
    lean = lambda h: observed_park_level(h, "climate.ac", "cool", latch=False, margin_c=0.0)  # noqa: E731

    assert lean(_head_ctrl(head_temp=22.2, step=0.5)[0]) == 22.5
    assert lean(_head_ctrl(head_temp=22.0, step=1.0)[0]) == 23.0
    # No reading → no observed park (callers fall back to the estimate)
    assert park(_head_ctrl(head_temp=None, step=1.0)[0]) is None


def test_observed_park_latches_against_the_ratchet():
    """The park does NOT follow the head upward once parked.

    Field data (bedroom, 9-11 Sep): a single 5 h idle stretch saw the head
    reading cross 19 -> 20 -> 21 as the room warmed, and an unlatched park
    walked 20 -> 21 -> 22 with it. Together with the active ladder that put
    three levels between the extremes — the 2 K swings the user reported.
    """
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, _ = _head_ctrl(head_temp=21.0, step=1.0)
    assert mc.observed_park_level(hass, "climate.ac", "cool") == 24.0
    # Room warms while parked; the head crosses a whole degree.
    hass.states.get.return_value.attributes["current_temperature"] = 22.0
    assert mc.observed_park_level(hass, "climate.ac", "cool") == 24.0, "park must not ratchet"
    # It also must not sink when the head falls again — one level, held.
    hass.states.get.return_value.attributes["current_temperature"] = 20.0
    assert mc.observed_park_level(hass, "climate.ac", "cool") == 24.0
    # An active command supersedes the latch; the next park re-reads the head.
    mc.reset_park_latch("climate.ac")
    assert mc.observed_park_level(hass, "climate.ac", "cool") == 23.0
    mc.clear_command_cache()


@pytest.mark.asyncio
async def test_kitchen_park_does_not_undercut_the_cool_target():
    """Regression, kitchen 11 Sep: parked at 23 under a 23.5 target all day.

    Whole-degree head reading 22 against a room of 22.5 — a reported value
    that quantization alone can place a step off the room it shares — cool
    target 23.5, nothing running. The park was 22 + 1 = 23 — under the target
    — so the unit held the kitchen at ~23 and the MPC saw no deficit to act
    on. It also latched there: with the room below target no active command
    is ever sent, and only an active command drops the latch. The park is now
    24, the first level the unit cannot cool the room past.
    """
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, ctrl = _head_ctrl(head_temp=22.0, step=1.0)
    await ctrl.async_apply("cooling", 23.5, power_fraction=0.0, current_temp=22.5)
    sent = _sent_temps(hass)
    assert sent, "expected a parked command"
    assert max(sent) >= 24.0, f"park still undercuts the 23.5 target: {sent}"
    mc.clear_command_cache()


def test_park_is_floored_at_the_level_that_leaves_the_room_alone():
    """A park may never be a colder thermostat than the room's own target.

    Field case (kitchen, 11 Sep): head reading 22 on a whole-degree unit, room
    22.5, cool target 23.5. "One step past the reading" is 23 — half a degree
    UNDER the target — so the parked unit regulates the room to ~23 and the
    MPC, seeing no deficit, never intervenes. The floor lifts it to the first
    level at or past the target, 24.
    """
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, _ = _head_ctrl(head_temp=22.0, step=1.0)
    floor = mc.no_demand_level(23.5, "cool", head_shift=-0.5, step=1.0)
    assert floor == 24.0
    assert mc.observed_park_level(hass, "climate.ac", "cool", park_floor=floor) == 25.0
    mc.clear_command_cache()

    # Heating mirrors it: the park may not sit above the heat target.
    hass, _ = _head_ctrl(head_temp=22.0, step=1.0, state_mode="heat", modes=("heat", "off"))
    floor = mc.no_demand_level(20.5, "heat", head_shift=0.5, step=1.0)
    assert floor == 20.0
    assert mc.observed_park_level(hass, "climate.ac", "heat", park_floor=floor) == 19.0
    mc.clear_command_cache()


def test_park_floor_leaves_a_warm_reading_head_alone():
    """The floor only bites on the gentle side — it never pulls a park down.

    A head reading above the room is the case the observed park was built for;
    past 26 and its interval stays 28 even though the target's level is far
    below, so the floor never comes into it.
    """
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, _ = _head_ctrl(head_temp=26.0, step=1.0)
    floor = mc.no_demand_level(23.5, "cool", head_shift=3.5, step=1.0)
    assert mc.observed_park_level(hass, "climate.ac", "cool", park_floor=floor) == 29.0
    mc.clear_command_cache()


def test_park_floor_outranks_the_latch():
    """A latched level below the floor is raised; the head may still not raise it.

    The latch exists to stop the park ratcheting after the head reading, and
    that still holds. But a latch under the room's target pins the room there
    forever — the MPC sees no deficit, so no active command ever drops the
    latch. The floor moves with the target, not with the head, so lifting to
    it cannot ratchet.
    """
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, _ = _head_ctrl(head_temp=21.0, step=1.0)
    floor = mc.no_demand_level(21.5, "cool", head_shift=0.0, step=1.0)
    assert mc.observed_park_level(hass, "climate.ac", "cool", park_floor=floor) == 24.0
    # Room warms while parked and the head follows: the latch still holds.
    hass.states.get.return_value.attributes["current_temperature"] = 22.0
    assert mc.observed_park_level(hass, "climate.ac", "cool", park_floor=floor) == 24.0
    # The target moves up past the latched level: the floor lifts it.
    floor = mc.no_demand_level(25.5, "cool", head_shift=0.0, step=1.0)
    assert mc.observed_park_level(hass, "climate.ac", "cool", park_floor=floor) == 26.0
    mc.clear_command_cache()


def test_no_demand_level_carries_only_the_adverse_bias():
    """A favourably biased head must not drag the bound toward demand."""
    from custom_components.roommind.control.mpc_controller import no_demand_level

    # Cooling: a head reading 1.5 K below the room is favourable — ignored.
    assert no_demand_level(23.5, "cool", head_shift=-1.5, step=1.0) == 24.0
    # A head reading high is adverse and must be carried in full.
    assert no_demand_level(23.5, "cool", head_shift=1.5, step=1.0) == 25.0
    # Heating: a head reading high is the favourable direction — ignored.
    assert no_demand_level(20.5, "heat", head_shift=1.5, step=1.0) == 20.0
    assert no_demand_level(20.5, "heat", head_shift=-1.5, step=1.0) == 19.0
    # Snapped away from demand, never onto the demand side of the bound.
    assert no_demand_level(23.0, "cool", step=1.0) == 23.0
    assert no_demand_level(23.1, "cool", step=1.0) == 24.0
    # Unknown step: the bound itself.
    assert no_demand_level(23.5, "cool") == 23.5


def test_observed_park_unlatched_read_is_pure():
    """latch=False reads the head without setting or consuming the latch."""
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, _ = _head_ctrl(head_temp=21.0, step=1.0)
    assert mc.observed_park_level(hass, "climate.ac", "cool", latch=False) == 24.0
    hass.states.get.return_value.attributes["current_temperature"] = 22.0
    assert mc.observed_park_level(hass, "climate.ac", "cool", latch=False) == 25.0
    assert "climate.ac" not in mc._park_latch
    mc.clear_command_cache()


@pytest.mark.asyncio
async def test_park_does_not_escalate_when_ventilation_cools_the_room():
    """A parked room sinking below target must NOT push the setpoint up.

    The room falling while the setpoint already clears the head reading means
    something OTHER than the compressor is cooling it — on this house, the
    ventilation, which is the disturbance this whole control problem started
    with. Raising the park cannot slow ventilation; it only widens the swing
    (field data: parks at 24 and 25 on nights the room drifted 0.7 K low).
    """
    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, ctrl = _head_ctrl(head_temp=21.0, step=1.0)
    sent = []
    for room in (20.9, 20.6, 20.3, 20.0, 19.8):
        hass.services.async_call.reset_mock()
        with patch.object(mc.time, "time", return_value=1000.0 + 3600 * len(sent)):
            await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=room)
        temps = _sent_temps(hass)
        if temps:
            sent.append(temps[-1])
    assert sent, "expected at least one parked command"
    # Two regimes, not a drift: the servo works near the lean bound (21/22)
    # while the room is still close to target, then the full park (24) once it
    # is past the band. What matters is that neither level CLIMBS cycle after
    # cycle, which is where the old room-error escalation took it (field data:
    # parks at 24 and 25 on nights the room drifted 0.7 K low).
    assert set(sent) <= {21.0, 22.0, 24.0}, f"unexpected level while parked: {sent}"
    parked = sent[2:]
    assert parked == [parked[0]] * len(parked), f"park still moving once parked: {sent}"
    mc.clear_command_cache()


@pytest.mark.asyncio
async def test_release_and_reengage_stay_within_adjacent_levels():
    """End to end on a whole-degree head: the park and the holding re-engage
    are two adjacent levels around the reading — the optimal dither."""
    from unittest.mock import patch

    from custom_components.roommind.control import mpc_controller as mc

    mc.clear_command_cache()
    hass, ctrl = _head_ctrl(head_temp=22.0, step=1.0)
    with patch.object(mc.time, "time", return_value=1000.0):
        # Room well below target: full park past the reading's interval
        await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.2)
    assert _sent_temps(hass)[-1] == 25.0
    hass.services.async_call.reset_mock()
    with patch.object(mc.time, "time", return_value=1000.0 + mc.HOLD_RUNG_DWELL_S):
        # Room back just above target with holding-level demand, a dwell
        # later: the servo steps one level down — not a jump to demand.
        await ctrl.async_apply("cooling", 21.0, power_fraction=0.15, current_temp=21.3)
    assert _sent_temps(hass)[-1] == 23.0
    mc.clear_command_cache()
