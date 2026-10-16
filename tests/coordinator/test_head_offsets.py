"""Coordinator wiring of the learned AC sensor offset (observation, controller, persistence)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.roommind import coordinator as coordinator_module
from custom_components.roommind.const import HEAD_OFFSET_SETTLE_S

from .conftest import SAMPLE_ROOM, _create_coordinator, _make_store_mock, make_mock_states_get

AC = "climate.living_room"


def _ac_room(**device) -> dict:
    return {
        **SAMPLE_ROOM,
        "climate_mode": "cool_only",
        "devices": [{"entity_id": AC, "type": "ac", "role": "auto", "heating_system_type": "", **device}],
        "thermostats": [],
        "acs": [AC],
        "heating_system_type": "",
    }


def _states(hvac="cool", action="cooling", head=29.5, room="28.0"):
    return make_mock_states_get(
        temp=room,
        outdoor_temp="30.0",
        extra={
            AC: (
                hvac,
                {
                    "hvac_action": action,
                    "current_temperature": head,
                    "temperature": 24.0,
                    "hvac_modes": ["off", "cool", "heat"],
                    "min_temp": 16.0,
                    "max_temp": 30.0,
                },
            ),
        },
    )


def _setup(hass, mock_config_entry, room, states, settings=None):
    store = _make_store_mock({room["area_id"]: room}, settings=settings)
    hass.data = {"roommind": {"store": store}}
    hass.states.get = MagicMock(side_effect=states)
    hass.services.async_call = AsyncMock()
    return store, _create_coordinator(hass, mock_config_entry)


@pytest.mark.asyncio
async def test_cycles_feed_the_tracker_and_the_controller_gets_it(hass, mock_config_entry):
    room = _ac_room()
    store, coordinator = _setup(hass, mock_config_entry, room, _states())
    now = [1000.0]
    seen: dict = {}
    real_controller = coordinator_module.MPCController

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return real_controller(*args, **kwargs)

    with (
        patch.object(coordinator_module.time, "time", lambda: now[0]),
        patch("custom_components.roommind.coordinator.MPCController", side_effect=_capture),
    ):
        for _ in range(15):
            await coordinator._async_update_data()
            now[0] += HEAD_OFFSET_SETTLE_S / 2
    assert seen["head_offsets"] is coordinator._head_offsets
    assert coordinator._head_offsets.offset(AC, "cool") == pytest.approx(1.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("hvac", ["off", "fan_only", "dry"])
async def test_a_unit_that_is_not_on_teaches_nothing(hass, mock_config_entry, hvac):
    """The idle offset is for the setback, where the unit stays on; an off unit may not even update its reading."""
    room = _ac_room()
    _, coordinator = _setup(hass, mock_config_entry, room, _states(hvac=hvac, action="idle", head=30.0))
    tracker = MagicMock()
    coordinator._head_offsets = tracker
    coordinator._observe_head_offsets(room, 28.0)
    args = tracker.observe.call_args.args
    assert args[:4] == (AC, None, 30.0, 28.0)


@pytest.mark.asyncio
async def test_a_unit_on_its_setback_feeds_the_idle_offset(hass, mock_config_entry):
    room = _ac_room()
    _, coordinator = _setup(hass, mock_config_entry, room, _states(hvac="cool", action="idle", head=30.0))
    tracker = MagicMock()
    coordinator._head_offsets = tracker
    coordinator._observe_head_offsets(room, 28.0)
    assert tracker.observe.call_args.args[:4] == (AC, "idle", 30.0, 28.0)


@pytest.mark.asyncio
async def test_learned_offsets_are_loaded_and_saved(hass, mock_config_entry):
    room = _ac_room()
    saved = {AC: {"cool": [1.4, 40]}}
    store, coordinator = _setup(hass, mock_config_entry, room, _states(), settings={"ac_head_offsets": saved})
    await coordinator._async_update_data()
    assert coordinator._head_offsets.offset(AC, "cool") == pytest.approx(1.5)

    coordinator._head_offsets.dirty = True
    coordinator._thermal_save_count = coordinator_module.THERMAL_SAVE_CYCLES - 1
    await coordinator._async_update_data()
    calls = [c.args[0] for c in store.async_save_settings.call_args_list if "ac_head_offsets" in c.args[0]]
    assert calls and calls[-1]["ac_head_offsets"][AC]["cool"][0] == pytest.approx(1.4)
    assert coordinator._head_offsets.dirty is False


@pytest.mark.asyncio
async def test_offsets_of_removed_devices_are_dropped(hass, mock_config_entry):
    room = _ac_room()
    saved = {AC: {"cool": [1.4, 40]}, "climate.gone": {"cool": [2.0, 40]}}
    _, coordinator = _setup(hass, mock_config_entry, room, _states(), settings={"ac_head_offsets": saved})
    await coordinator._async_update_data()
    assert set(coordinator._head_offsets.get_state()) == {AC}


@pytest.mark.asyncio
async def test_no_room_sensor_means_no_learning(hass, mock_config_entry):
    """Without an external sensor the room temperature is the device's own reading."""
    room = {**_ac_room(), "temperature_sensor": ""}
    _, coordinator = _setup(hass, mock_config_entry, room, _states())
    coordinator._head_offsets.observe = MagicMock()
    await coordinator._async_update_data()
    coordinator._head_offsets.observe.assert_not_called()


def test_device_setpoint_display_is_in_the_acs_frame():
    from custom_components.roommind.coordinator import RoomMindCoordinator

    show = RoomMindCoordinator._compute_device_setpoint
    cool = {"device_min_temp": 16.0, "has_thermostats": False, "has_acs": True}
    plain = show("cooling", 0.2, 24.5, 24.0, True, **cool)
    assert show("cooling", 0.2, 24.5, 24.0, True, head_offset=1.5, **cool) == pytest.approx(plain + 1.5)
    heat_ac = {"device_max_temp": 30.0, "has_thermostats": False, "has_acs": True}
    plain = show("heating", 0.1, 20.5, 21.0, True, **heat_ac)
    assert show("heating", 0.1, 20.5, 21.0, True, head_offset=1.5, **heat_ac) == pytest.approx(plain + 1.5)
    # With TRVs in the room the shown value is the TRV's, which has no head offset
    heat_trv = {"device_max_temp": 30.0, "has_thermostats": True, "has_acs": True}
    assert show("heating", 0.1, 20.5, 21.0, True, head_offset=1.5, **heat_trv) == show(
        "heating", 0.1, 20.5, 21.0, True, **heat_trv
    )


def test_orchestrated_device_setpoint_display_is_in_the_acs_frame():
    from custom_components.roommind.coordinator import RoomMindCoordinator
    from custom_components.roommind.managers.heat_source_orchestrator import DeviceCommand, HeatSourcePlan

    show = RoomMindCoordinator._compute_device_setpoint_orchestrated

    def plan(eid, device_type):
        cmd = DeviceCommand(
            entity_id=eid, role="primary", device_type=device_type, active=True, power_fraction=0.1, reason="t"
        )
        return HeatSourcePlan(commands=[cmd], active_sources="primary", reason="test")

    offsets = {AC: 1.5}
    plain = show(plan(AC, "ac"), 20.5, 21.0, 30.0, 30.0)
    assert show(plan(AC, "ac"), 20.5, 21.0, 30.0, 30.0, head_offsets=offsets) == pytest.approx(plain + 1.5)
    # A TRV command keeps its own value
    trv = plan("climate.trv", "thermostat")
    assert show(trv, 20.5, 21.0, 30.0, 30.0, head_offsets=offsets) == show(trv, 20.5, 21.0, 30.0, 30.0)
