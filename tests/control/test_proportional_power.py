"""Tests for proportional TRV setpoints, power calculations, AC proportional control, dynamic boost."""

from __future__ import annotations

from unittest.mock import MagicMock

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
    # Raw: 20.5 + 0.01*(30-20.5) = 20.595 → clamped to max(21.0, 20.6) = 21.0
    await ctrl.async_apply("heating", 21.0, power_fraction=0.01, current_temp=20.5)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


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
    # Raw: 23.5 - 0.01*(23.5-16) = 23.425 → clamped to min(23.0, 23.4) = 23.0
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.01, current_temp=23.5)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 23.0 for c in temp_calls)


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
    """Room below cool target with low power: setpoint anchors at target, above room temp.

    Cold-evening scenario: ventilation already cools the room below the target
    while the AC is held in a cooling run. The old anchor (current_temp) kept
    the commanded setpoint at or below the falling room temperature, so the AC
    could never release and chased the room downward.
    """
    hass, ctrl = _make_cooling_ctrl()
    # anchor = max(20.0, 21.0) = 21.0 → 21.0 - 0.1*(21.0-16) = 20.5
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.1, current_temp=20.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert temp_calls
    sp = temp_calls[0][0][2]["temperature"]
    assert sp == 20.5
    # The commanded setpoint must sit above the room temperature so the AC can stop
    assert sp > 20.0


@pytest.mark.asyncio
async def test_ac_cooling_zero_power_releases_at_target():
    """Zero cooling demand while below target: setpoint == target (full release)."""
    hass, ctrl = _make_cooling_ctrl()
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.0)

    calls = hass.services.async_call.call_args_list
    temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


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
    assert any(c[0][2]["temperature"] == 21.0 for c in temp_calls)


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
    """Release below target must land above the head's own reading.

    Room 20.0 (below the 21.0 target) with the head reading 23.0: the old
    release at 21.0 kept the compressor running until the head — and so the
    room — fell a degree past target (the 1 K sawtooth).
    """
    hass, ctrl = _head_ctrl(head_temp=23.0)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.0)
    assert 24.0 in _sent_temps(hass)  # 21.0 + (23.0 - 20.0)


@pytest.mark.asyncio
async def test_cooling_release_ignores_favourable_head():
    """A head reading cooler than the room already releases — no shift."""
    hass, ctrl = _head_ctrl(head_temp=19.0)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.0)
    assert 21.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_cooling_active_command_takes_full_shift():
    """Active commands translate fully so the delivered gap matches the intent."""
    hass, ctrl = _head_ctrl(head_temp=25.0)
    # Room 26, target 23, pf 0.5 → room-frame 21.0; head reads 1.0 cold → 20.0
    await ctrl.async_apply("cooling", 23.0, power_fraction=0.5, current_temp=26.0)
    assert 20.0 in _sent_temps(hass)


@pytest.mark.asyncio
async def test_cooling_hold_at_target_shifts_for_warm_head():
    """Room just above target, setpoint clamped to target: still an active
    command, so the warm head bias applies in full — without it the unit
    cools the room a degree past target chasing its own sensor."""
    hass, ctrl = _head_ctrl(head_temp=22.2)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=21.2)
    assert 22.0 in _sent_temps(hass)  # 21.0 + (22.2 - 21.2)


@pytest.mark.asyncio
async def test_cooling_release_snaps_up_on_coarse_step():
    """On a whole-degree device the release rounds AWAY from demand."""
    hass, ctrl = _head_ctrl(head_temp=21.4, step=1.0)
    await ctrl.async_apply("cooling", 21.0, power_fraction=0.0, current_temp=20.6)
    assert 22.0 in _sent_temps(hass)  # 21.0 + 0.8 → ceil to step


@pytest.mark.asyncio
async def test_heating_release_clears_cold_head():
    """Heating mirror: a head reading colder than the room lowers the release."""
    hass, ctrl = _head_ctrl(head_temp=20.0, state_mode="heat", modes=("heat", "off"))
    await ctrl.async_apply("heating", 21.0, power_fraction=0.0, current_temp=22.0)
    assert 19.0 in _sent_temps(hass)  # 21.0 + (20.0 - 22.0)


@pytest.mark.asyncio
async def test_compressor_hold_translates_to_head_frame():
    """A min-run hold parks the AC at the target as the DEVICE perceives it."""
    hass, ctrl = _head_ctrl(head_temp=23.0)
    await ctrl.async_apply(
        "idle",
        TargetTemps(heat=None, cool=21.0),
        power_fraction=0.0,
        current_temp=20.0,
        compressor_forced_on={"climate.ac"},
    )
    assert 24.0 in _sent_temps(hass)  # 21.0 + (23.0 - 20.0)
