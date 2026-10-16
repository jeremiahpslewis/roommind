"""AC setpoints in the device's frame: the learned sensor offset moves what is sent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import MODE_COOLING, MODE_HEATING, MODE_IDLE, TargetTemps
from custom_components.roommind.control.mpc_controller import MPCController, clear_command_cache
from custom_components.roommind.control.thermal_model import RoomModelManager
from custom_components.roommind.managers.head_offset_tracker import HeadOffsetTracker

from .conftest import build_hass, make_room

AC = "climate.ac"


def _tracker(**offsets: float) -> HeadOffsetTracker:
    tracker = HeadOffsetTracker()
    tracker.load_state({AC: {state: [value, 50] for state, value in offsets.items()}})
    return tracker


def _ac_state(hvac="cool", **attrs):
    state = MagicMock()
    state.state = hvac
    state.attributes = {
        "hvac_modes": ["heat", "cool", "off"],
        "min_temp": 16.0,
        "max_temp": 30.0,
        "temperature": None,
        **attrs,
    }
    return state


def _ctrl(state, tracker=None, *, room=None, has_external_sensor=True):
    hass = build_hass()
    hass.states.get = MagicMock(return_value=state)
    ctrl = MPCController(
        hass,
        room or make_room(thermostats=[], acs=[AC]),
        model_manager=RoomModelManager(),
        outdoor_temp=20.0,
        settings={},
        has_external_sensor=has_external_sensor,
        head_offsets=tracker,
    )
    return hass, ctrl


def _sent(hass):
    return [c[0][2]["temperature"] for c in hass.services.async_call.call_args_list if c[0][1] == "set_temperature"]


async def _apply(tracker, mode, targets, *, state=None, room=None, has_external_sensor=True, **kwargs):
    clear_command_cache()
    hass, ctrl = _ctrl(
        state or _ac_state("cool" if mode == MODE_COOLING else "heat"),
        tracker,
        room=room,
        has_external_sensor=has_external_sensor,
    )
    await ctrl.async_apply(mode, targets, **kwargs)
    return _sent(hass)


@pytest.mark.asyncio
async def test_cooling_setpoint_moves_by_the_cooling_offset():
    """A sensor reading 1.5 K warm while cooling gets the proportional setpoint 1.5 K higher."""
    kwargs = {"power_fraction": 0.2, "current_temp": 24.5}
    plain = await _apply(None, MODE_COOLING, TargetTemps(heat=None, cool=24.0), **kwargs)
    shifted = await _apply(_tracker(cool=1.5, idle=3.0), MODE_COOLING, TargetTemps(heat=None, cool=24.0), **kwargs)
    assert len(plain) == len(shifted) == 1
    assert shifted[0] == pytest.approx(plain[0] + 1.5)
    # Above the room target in the room's frame: the unit holds the room at it, not 1.5 K below
    assert shifted[0] > 24.0


@pytest.mark.asyncio
async def test_heating_setpoint_moves_by_the_heating_offset():
    kwargs = {"power_fraction": 0.1, "current_temp": 20.5}
    plain = await _apply(None, MODE_HEATING, TargetTemps(heat=21.0, cool=None), **kwargs)
    shifted = await _apply(_tracker(heat=1.5), MODE_HEATING, TargetTemps(heat=21.0, cool=None), **kwargs)
    assert shifted[0] == pytest.approx(plain[0] + 1.5)


@pytest.mark.asyncio
async def test_cooling_setpoint_stays_inside_the_device_range():
    """A sensor reading cold while cooling cannot push the full-power setpoint below min_temp."""
    sent = await _apply(
        _tracker(cool=-2.0), MODE_COOLING, TargetTemps(heat=None, cool=24.0), power_fraction=1.0, current_temp=27.0
    )
    assert sent == [16.0]


@pytest.mark.asyncio
async def test_direct_device_keeps_its_own_offset():
    room = make_room(thermostats=[], acs=[AC])
    room["devices"][0].update(setpoint_mode="direct", setpoint_offset=1.0)
    kwargs = {"power_fraction": 0.2, "current_temp": 24.5, "room": room}
    plain = await _apply(None, MODE_COOLING, TargetTemps(heat=None, cool=24.0), **kwargs)
    shifted = await _apply(_tracker(cool=1.5), MODE_COOLING, TargetTemps(heat=None, cool=24.0), **kwargs)
    assert shifted == plain == [25.0]


@pytest.mark.asyncio
async def test_unlearned_offset_changes_nothing():
    kwargs = {"power_fraction": 0.2, "current_temp": 24.5}
    plain = await _apply(None, MODE_COOLING, TargetTemps(heat=None, cool=24.0), **kwargs)
    tracker = HeadOffsetTracker()
    tracker.load_state({AC: {"cool": [1.5, 3]}})  # too few readings
    assert await _apply(tracker, MODE_COOLING, TargetTemps(heat=None, cool=24.0), **kwargs) == plain


@pytest.mark.asyncio
async def test_setback_moves_by_the_idle_offset():
    """Idle on a setback, a sensor reading the warm layer would restart the unit inside the setback."""
    room = make_room(thermostats=[], acs=[AC])
    room["devices"][0]["idle_action"] = "setback"
    targets = TargetTemps(heat=None, cool=24.0)
    state = _ac_state("cool", temperature=20.0)
    plain = await _apply(None, MODE_IDLE, targets, state=state, room=room, current_temp=23.5)
    shifted = await _apply(_tracker(idle=2.0, cool=1.0), MODE_IDLE, targets, state=state, room=room, current_temp=23.5)
    assert len(plain) == len(shifted) == 1
    assert shifted[0] == pytest.approx(plain[0] + 2.0)


@pytest.mark.asyncio
async def test_held_unit_gets_a_lowered_target_in_its_frame():
    """The compressor hold passes a lowered target on; the unit gets it plus its heating offset."""
    held = {AC}
    out = []
    for tracker in (None, _tracker(heat=1.5)):
        clear_command_cache()
        state = _ac_state("heat")
        hass, ctrl = _ctrl(state, tracker)
        await ctrl.async_apply(MODE_HEATING, TargetTemps(heat=21.0, cool=None), power_fraction=1.0, current_temp=19.0)
        state.attributes["temperature"] = 29.5
        hass.services.async_call.reset_mock()
        await ctrl.async_apply(
            MODE_IDLE, TargetTemps(heat=17.0, cool=None), current_temp=19.5, compressor_forced_on=held
        )
        out.append(_sent(hass))
    assert out[0] == [17.0]
    assert out[1] == [18.5]


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [1.88, 1.6, 2.0])
async def test_setback_with_offset_is_not_resent_to_a_whole_degree_device(idle):
    """A device without target_temp_step that shows whole degrees gets the setback once."""
    room = make_room(thermostats=[], acs=[AC])
    room["devices"][0]["idle_action"] = "setback"
    targets = TargetTemps(heat=None, cool=24.0)
    state = _ac_state("cool", temperature=26.0)
    clear_command_cache()
    hass, ctrl = _ctrl(state, _tracker(idle=idle, cool=1.0), room=room)
    sent = []
    for _ in range(3):
        hass.services.async_call.reset_mock()
        await ctrl.async_apply(MODE_IDLE, targets, current_temp=23.5)
        sent.append(_sent(hass))
        if sent[-1]:
            # The device keeps whole degrees
            state.attributes["temperature"] = float(round(sent[-1][-1]))
    assert sent[0] == [float(round(24.0 + idle + 2.0))]
    assert sent[1:] == [[], []]


@pytest.mark.asyncio
async def test_setback_with_offset_snaps_to_half_degrees_without_a_step():
    """A device without a step that shows tenths gets the setback on the 0.5 °C grid."""
    room = make_room(thermostats=[], acs=[AC])
    room["devices"][0]["idle_action"] = "setback"
    state = _ac_state("cool", temperature=26.3)
    sent = await _apply(
        _tracker(idle=1.88), MODE_IDLE, TargetTemps(heat=None, cool=24.0), state=state, room=room, current_temp=23.5
    )
    assert sent == [28.0]
    state = _ac_state("cool", temperature=26.3)
    sent = await _apply(
        _tracker(idle=1.6), MODE_IDLE, TargetTemps(heat=None, cool=24.0), state=state, room=room, current_temp=23.5
    )
    assert sent == [27.5]
