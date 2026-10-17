"""A setback device switches off after an hour of idle when the MPC plan expects no run in its mode (#255)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import AC_RELEASE_PARK_C, TargetTemps
from custom_components.roommind.control.mpc_controller import (
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    SETBACK_HOLD_MINUTES,
    SETBACK_RELEASE_BLOCKS,
    MPCController,
    _last_commands,
    async_idle_device,
    clear_command_cache,
)
from custom_components.roommind.control.mpc_optimizer import MPCPlan
from custom_components.roommind.control.thermal_model import RoomModelManager

from .conftest import build_hass, make_room

# Fork: an AC's setback sits AC_RELEASE_PARK_C past the target (upstream: 2 K)
HEAT_SETBACK = 21.0 - AC_RELEASE_PARK_C
COOL_SETBACK = 24.0 + AC_RELEASE_PARK_C

EID = "climate.ac1"
HEAT_TARGETS = TargetTemps(heat=21.0, cool=24.0)


def _state(hvac: str, temperature: float):
    state = MagicMock()
    state.state = hvac
    state.attributes = {
        "hvac_modes": ["heat", "cool", "off"],
        "min_temp": 5.0,
        "max_temp": 35.0,
        "target_temp_step": 0.5,
        "temperature": temperature,
    }
    return state


def _devices(idle_action: str = "setback") -> list[dict]:
    return [
        {
            "entity_id": EID,
            "type": "ac",
            "role": "auto",
            "heating_system_type": "",
            "idle_action": idle_action,
            "idle_fan_mode": "",
        }
    ]


def _idle_for(minutes: float) -> float:
    return time.time() - minutes * 60


def _controller(
    hvac: str,
    temperature: float,
    actions: list[str],
    monkeypatch,
    *,
    has_external_sensor=True,
    idle_since: float | None = None,
    previous_mode: str = MODE_IDLE,
):
    hass = build_hass()
    hass.states.get = MagicMock(return_value=_state(hvac, temperature))
    room = make_room(thermostats=[], acs=[EID])
    room["devices"] = _devices()
    ctrl = MPCController(
        hass,
        room,
        model_manager=RoomModelManager(),
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=has_external_sensor,
        previous_mode=previous_mode,
        idle_since=_idle_for(SETBACK_HOLD_MINUTES + 1) if idle_since is None else idle_since,
    )
    plan = MPCPlan(
        actions=actions,
        temperatures=[21.0] * (len(actions) + 1),
        power_fractions=[0.0 if a == MODE_IDLE else 0.5 for a in actions],
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: plan,
    )
    return hass, ctrl


def _with_run_at(block: int, mode: str, length: int = 36) -> list[str]:
    actions = [MODE_IDLE] * length
    actions[block] = mode
    return actions


def _calls(hass, service: str) -> list[dict]:
    return [c[0][2] for c in hass.services.async_call.call_args_list if c[0][1] == service]


def _turned_off(hass) -> bool:
    return any(d.get("hvac_mode") == "off" for d in _calls(hass, "set_hvac_mode"))


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_command_cache()
    _last_commands.clear()
    yield
    clear_command_cache()


@pytest.mark.asyncio
async def test_heat_setback_turns_off_when_no_heating_is_planned(monkeypatch):
    hass, ctrl = _controller("heat", 21.0, [MODE_IDLE] * 36, monkeypatch)

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    assert mode == MODE_IDLE
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert _turned_off(hass)
    assert HEAT_SETBACK not in [d["temperature"] for d in _calls(hass, "set_temperature")]


@pytest.mark.asyncio
async def test_heat_setback_stays_when_heating_is_planned_within_the_hour(monkeypatch):
    actions = _with_run_at(SETBACK_RELEASE_BLOCKS - 1, MODE_HEATING)
    hass, ctrl = _controller("heat", 21.0, actions, monkeypatch)

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert not _turned_off(hass)
    assert [d["temperature"] for d in _calls(hass, "set_temperature")] == [HEAT_SETBACK]


@pytest.mark.asyncio
async def test_heat_setback_turns_off_when_the_next_run_is_beyond_the_hour(monkeypatch):
    actions = _with_run_at(SETBACK_RELEASE_BLOCKS, MODE_HEATING)
    hass, ctrl = _controller("heat", 21.0, actions, monkeypatch)

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert _turned_off(hass)


@pytest.mark.asyncio
async def test_cool_setback_follows_planned_cooling_not_heating(monkeypatch):
    """A cooling device only cares about cooling runs: planned heating does not keep it on."""
    hass, ctrl = _controller("cool", 24.0, _with_run_at(3, MODE_HEATING), monkeypatch)

    mode, _ = ctrl._evaluate_mpc(23.5, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=23.5)

    assert _turned_off(hass)


@pytest.mark.asyncio
async def test_cool_setback_stays_when_cooling_is_planned_within_the_hour(monkeypatch):
    hass, ctrl = _controller("cool", 24.0, _with_run_at(3, MODE_COOLING), monkeypatch)

    mode, _ = ctrl._evaluate_mpc(23.5, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=23.5)

    assert not _turned_off(hass)
    assert [d["temperature"] for d in _calls(hass, "set_temperature")] == [COOL_SETBACK]


@pytest.mark.asyncio
async def test_compressor_min_run_hold_outranks_the_release(monkeypatch):
    hass, ctrl = _controller("heat", 23.0, [MODE_IDLE] * 36, monkeypatch)

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3, compressor_forced_on={EID})

    assert not _turned_off(hass)


@pytest.mark.asyncio
async def test_managed_mode_keeps_the_setback(monkeypatch):
    """Managed Mode has no plan, so the setback stays as before."""
    hass, ctrl = _controller("heat", 21.0, [MODE_IDLE] * 36, monkeypatch, has_external_sensor=False)

    mode, _ = await ctrl.async_evaluate(None, TargetTemps(heat=21.0, cool=None))
    await ctrl.async_apply(MODE_IDLE, TargetTemps(heat=21.0, cool=None), power_fraction=0.0)

    assert not _turned_off(hass)
    assert [d["temperature"] for d in _calls(hass, "set_temperature")] == [HEAT_SETBACK]


@pytest.mark.asyncio
async def test_a_short_plan_keeps_the_setback(monkeypatch):
    """A plan shorter than the release window cannot rule out a run, so nothing is released."""
    hass, ctrl = _controller("heat", 21.0, [MODE_IDLE] * (SETBACK_RELEASE_BLOCKS - 1), monkeypatch)

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert not _turned_off(hass)


@pytest.mark.asyncio
async def test_setback_holds_through_a_short_pause_even_if_the_plan_sees_none(monkeypatch):
    """The plan saw an hour of idle after AC heating runs that restarted after 15-30 min (sim)."""
    hass, ctrl = _controller("heat", 21.0, [MODE_IDLE] * 36, monkeypatch, idle_since=_idle_for(20))

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert not _turned_off(hass)
    assert [d["temperature"] for d in _calls(hass, "set_temperature")] == [HEAT_SETBACK]


@pytest.mark.asyncio
async def test_unknown_idle_start_keeps_the_setback(monkeypatch):
    hass, ctrl = _controller("heat", 21.0, [MODE_IDLE] * 36, monkeypatch)
    ctrl._idle_since = None

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert not _turned_off(hass)


@pytest.mark.asyncio
async def test_a_run_that_just_ended_keeps_the_setback(monkeypatch):
    """The room was heating last cycle: its idle time starts now, whatever an old timestamp says."""
    hass, ctrl = _controller("heat", 21.0, [MODE_IDLE] * 36, monkeypatch, previous_mode=MODE_HEATING)

    mode, _ = ctrl._evaluate_mpc(21.3, HEAT_TARGETS)
    assert mode == MODE_IDLE
    await ctrl.async_apply(mode, HEAT_TARGETS, power_fraction=0.0, current_temp=21.3)

    assert not _turned_off(hass)


def test_nothing_is_released_when_the_controller_decides_to_run(monkeypatch):
    """The guard can start a run the plan did not have; no device may be switched off then."""
    _, ctrl = _controller("heat", 21.0, [MODE_IDLE] * 36, monkeypatch)
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.decide_mpc_action",
        lambda *a, **kw: (MODE_HEATING, 0.5),
    )

    assert ctrl._evaluate_mpc(20.5, HEAT_TARGETS) == (MODE_HEATING, 0.5)
    assert ctrl._setback_release == frozenset()


@pytest.mark.asyncio
async def test_async_idle_device_release_only_hits_the_released_mode():
    hass = build_hass()
    hass.states.get = MagicMock(return_value=_state("cool", 24.0))

    await async_idle_device(
        hass, EID, _devices(), area_id="living_room", targets=HEAT_TARGETS, release_setback=frozenset({"heat"})
    )

    assert not _turned_off(hass)
    assert [d["temperature"] for d in _calls(hass, "set_temperature")] == [COOL_SETBACK]


@pytest.mark.asyncio
async def test_async_idle_device_release_ignores_other_idle_actions():
    hass = build_hass()
    state = _state("heat", 21.0)
    state.attributes["hvac_modes"] = ["heat", "fan_only", "off"]
    state.attributes["fan_modes"] = ["low", "high"]
    hass.states.get = MagicMock(return_value=state)
    devices = _devices("fan_only")
    devices[0]["idle_fan_mode"] = "low"

    await async_idle_device(
        hass, EID, devices, area_id="living_room", targets=HEAT_TARGETS, release_setback=frozenset({"heat"})
    )

    assert [d["hvac_mode"] for d in _calls(hass, "set_hvac_mode")] == ["fan_only"]
