"""Tests for the gap-response observation interval accumulation.

The coordinator cycle (UPDATE_INTERVAL = 30 s) is shorter than the minimum
useful observation window (MIN_OBSERVATION_DT = 1.0 min).  The observation
baseline must therefore accumulate across cycles: advancing it every cycle
would make every interval 0.5 min and reject every observation forever —
which is exactly what starved the gap-response store in the field
(n_observations stuck at ~1 after days of continuous cooling).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.roommind.const import MODE_COOLING, MODE_IDLE

from .conftest import SAMPLE_ROOM, _create_coordinator, _make_store_mock

AC_EID = "climate.bedroom_ac"

AC_ROOM = {
    **SAMPLE_ROOM,
    "thermostats": [],
    "acs": [AC_EID],
    "devices": [
        {
            "entity_id": AC_EID,
            "type": "ac",
            "role": "auto",
            "heating_system_type": "",
            "idle_action": "setback",
            "idle_fan_mode": "low",
            "setpoint_mode": "proportional",
        }
    ],
}


def _ac_state(state: str = "cool", head: float = 21.0, setpoint: float = 21.8):
    s = MagicMock()
    s.state = state
    s.attributes = {"current_temperature": head, "temperature": setpoint}
    return s


def _make_gap_coordinator(hass, mock_config_entry):
    hass.data = {"roommind": {"store": _make_store_mock({AC_ROOM["area_id"]: AC_ROOM})}}
    hass.states.get = MagicMock(return_value=_ac_state())
    hass.services.async_call = AsyncMock()
    coordinator = _create_coordinator(hass, mock_config_entry)
    coordinator.outdoor_temp_effective = 17.0
    coordinator._gap_manager = MagicMock()
    model = MagicMock()
    model.predict.return_value = 20.0
    coordinator._model_manager = MagicMock()
    coordinator._model_manager.get_model.return_value = model
    return coordinator


def _observe_at(coordinator, t_seconds: float, *, temp: float = 20.5, mode: str = MODE_COOLING) -> None:
    with patch("custom_components.roommind.coordinator.time.monotonic", return_value=t_seconds):
        coordinator._observe_gap_response(
            area_id=AC_ROOM["area_id"],
            room=AC_ROOM,
            current_temp=temp,
            mode=mode,
            has_external_sensor=True,
            q_solar=0.0,
            q_residual=0.0,
            q_occupancy=0.0,
        )


class TestGapObservationAccumulation:
    def test_short_cycles_accumulate_to_an_observation(self, hass, mock_config_entry):
        """30 s cycles must not reset the baseline — two of them make 1 min."""
        coordinator = _make_gap_coordinator(hass, mock_config_entry)

        _observe_at(coordinator, 0.0)  # establishes the baseline
        assert not coordinator._gap_manager.observe_response.called

        _observe_at(coordinator, 30.0)  # 0.5 min — too short, keep baseline
        assert not coordinator._gap_manager.observe_response.called

        _observe_at(coordinator, 60.0)  # 1.0 min from the ORIGINAL baseline
        assert coordinator._gap_manager.observe_response.call_count == 1
        assert coordinator._gap_manager.observe_offset.call_count == 1
        kwargs = coordinator._gap_manager.observe_response.call_args.kwargs
        assert abs(kwargs["dt_minutes"] - 1.0) < 1e-9

    def test_baseline_advances_after_consuming_interval(self, hass, mock_config_entry):
        """After an observation the next interval starts fresh."""
        coordinator = _make_gap_coordinator(hass, mock_config_entry)
        _observe_at(coordinator, 0.0)
        _observe_at(coordinator, 60.0)  # consumed
        _observe_at(coordinator, 90.0)  # 0.5 min since new baseline — too short
        assert coordinator._gap_manager.observe_response.call_count == 1
        _observe_at(coordinator, 120.0)  # 1.0 min since t=60
        assert coordinator._gap_manager.observe_response.call_count == 2

    def test_mode_change_resets_baseline(self, hass, mock_config_entry):
        """A mode change mid-interval discards the mixed-dynamics interval."""
        coordinator = _make_gap_coordinator(hass, mock_config_entry)
        _observe_at(coordinator, 0.0)
        _observe_at(coordinator, 60.0, mode=MODE_IDLE)  # mode changed → reset, no obs
        assert not coordinator._gap_manager.observe_response.called
        _observe_at(coordinator, 120.0, mode=MODE_IDLE)  # 1 min of stable idle
        # AC still reports "cool" (engaged) → idle interval observed as cooling
        assert coordinator._gap_manager.observe_response.call_count == 1
        assert coordinator._gap_manager.observe_response.call_args.args[1] == MODE_COOLING

    def test_stale_interval_discarded(self, hass, mock_config_entry):
        """An interval past MAX_OBSERVATION_DT is dropped, baseline restarts."""
        coordinator = _make_gap_coordinator(hass, mock_config_entry)
        _observe_at(coordinator, 0.0)
        _observe_at(coordinator, 30 * 60.0)  # 30 min — beyond MAX (20)
        assert not coordinator._gap_manager.observe_response.called
        _observe_at(coordinator, 31 * 60.0)  # 1 min since restart
        assert coordinator._gap_manager.observe_response.call_count == 1
