"""Tests for EkfTrainingManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import EKF_UPDATE_MIN_DT
from custom_components.roommind.managers.ekf_training_manager import EkfTrainingManager


@pytest.fixture
def model_manager():
    mm = MagicMock()
    mm.update = MagicMock()
    mm.update_window_open = MagicMock()
    return mm


@pytest.fixture
def mgr(model_manager):
    return EkfTrainingManager(model_manager)


class TestFlush:
    def test_flush_with_accumulated_data_calls_update(self, mgr, model_manager):
        """Flush with accumulated data should call model_manager.update."""
        mgr._accumulated_dt["room1"] = 5.0
        mgr._accumulated_mode["room1"] = "heat"
        mgr._accumulated_pf["room1"] = 0.8
        mgr.flush(
            "room1",
            current_temp=20.0,
            T_outdoor=5.0,
            can_heat=True,
            can_cool=False,
            q_solar=0.0,
        )
        model_manager.update.assert_called_once_with(
            "room1",
            20.0,
            5.0,
            "heat",
            5.0,
            can_heat=True,
            can_cool=False,
            power_fraction=0.8,
            q_solar=0.0,
            q_residual=0.0,
            q_occupancy=0.0,
            airflow="normal",
        )
        assert "room1" not in mgr._accumulated_dt
        assert "room1" not in mgr._accumulated_mode
        assert "room1" not in mgr._accumulated_pf

    def test_flush_without_accumulated_data_no_update(self, mgr, model_manager):
        """Flush with no accumulated data should not call update."""
        mgr.flush(
            "room1",
            current_temp=20.0,
            T_outdoor=5.0,
            can_heat=True,
            can_cool=False,
            q_solar=0.0,
        )
        model_manager.update.assert_not_called()

    def test_flush_zero_dt_no_update(self, mgr, model_manager):
        """Flush with accumulated_dt=0 should not call update."""
        mgr._accumulated_dt["room1"] = 0.0
        mgr._accumulated_mode["room1"] = "heat"
        mgr.flush(
            "room1",
            current_temp=20.0,
            T_outdoor=5.0,
            can_heat=True,
            can_cool=False,
            q_solar=0.0,
        )
        model_manager.update.assert_not_called()

    def test_flush_applies_shading_to_solar(self, mgr, model_manager):
        """q_solar passed to update should be multiplied by shading_factor."""
        mgr._accumulated_dt["room1"] = 3.0
        mgr._accumulated_mode["room1"] = "idle"
        mgr._accumulated_pf["room1"] = 1.0
        mgr.flush(
            "room1",
            current_temp=20.0,
            T_outdoor=5.0,
            can_heat=True,
            can_cool=False,
            q_solar=100.0,
            shading_factor=0.5,
        )
        _, kwargs = model_manager.update.call_args
        assert kwargs["q_solar"] == pytest.approx(50.0)


class TestProcess:
    COMMON = dict(
        current_temp=20.0,
        T_outdoor=5.0,
        q_residual=0.0,
        shading_factor=1.0,
        q_solar=0.0,
        can_heat=True,
        can_cool=False,
        dt_minutes=0.5,
    )

    def test_window_open_flushes_and_calls_update_window(self, mgr, model_manager):
        """Window open should flush + call update_window_open with learn_k_window=True."""
        mgr._accumulated_dt["r1"] = 3.0
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 1.0
        mgr.process(
            "r1",
            **self.COMMON,
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=True,
            raw_open=False,
        )
        model_manager.update.assert_called_once()  # flush
        model_manager.update_window_open.assert_called_once()
        _, kwargs = model_manager.update_window_open.call_args
        assert kwargs["learn_k_window"] is True
        assert "r1" not in mgr._accumulated_dt

    def test_window_open_no_learn_with_residual(self, mgr, model_manager):
        """Window open with q_residual > 0 → learn_k_window=False."""
        mgr.process(
            "r1",
            **{**self.COMMON, "q_residual": 0.5},
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=True,
            raw_open=False,
        )
        _, kwargs = model_manager.update_window_open.call_args
        assert kwargs["learn_k_window"] is False

    def test_raw_open_flushes_and_calls_update_window(self, mgr, model_manager):
        """raw_open (within delay) should also flush + update_window_open."""
        mgr.process(
            "r1",
            **self.COMMON,
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=False,
            raw_open=True,
        )
        model_manager.update_window_open.assert_called_once()

    def test_none_mode_flushes_only(self, mgr, model_manager):
        """ekf_mode=None should flush but NOT call update_window_open."""
        mgr._accumulated_dt["r1"] = 3.0
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 1.0
        mgr.process(
            "r1",
            **self.COMMON,
            ekf_mode=None,
            ekf_pf=0.0,
            window_open=False,
            raw_open=False,
        )
        model_manager.update.assert_called_once()
        model_manager.update_window_open.assert_not_called()
        assert "r1" not in mgr._accumulated_dt

    def test_accumulation_below_threshold(self, mgr, model_manager):
        """Process below EKF_UPDATE_MIN_DT should accumulate, not update."""
        mgr.process(
            "r1",
            **{**self.COMMON, "dt_minutes": 0.1},
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=False,
            raw_open=False,
        )
        model_manager.update.assert_not_called()
        assert mgr._accumulated_dt["r1"] == pytest.approx(0.1)
        assert mgr._accumulated_mode["r1"] == "heat"

    def test_accumulation_crosses_threshold_triggers_update(self, mgr, model_manager):
        """Accumulating past threshold should trigger model update."""
        mgr._accumulated_dt["r1"] = EKF_UPDATE_MIN_DT - 0.1
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 1.0
        mgr.process(
            "r1",
            **{**self.COMMON, "dt_minutes": 0.2},
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=False,
            raw_open=False,
        )
        model_manager.update.assert_called_once()
        assert mgr._accumulated_dt["r1"] == 0.0

    def test_mode_transition_flushes_before_accumulating(self, mgr, model_manager):
        """Switching from heat to cool should flush heat data first."""
        mgr._accumulated_dt["r1"] = 3.0
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 0.8
        mgr.process(
            "r1",
            **{**self.COMMON, "dt_minutes": 0.5},
            ekf_mode="cool",
            ekf_pf=0.6,
            window_open=False,
            raw_open=False,
        )
        first_call = model_manager.update.call_args_list[0]
        assert first_call.args[3] == "heat"
        assert mgr._accumulated_mode["r1"] == "cool"

    def test_power_fraction_weighted_average(self, mgr, model_manager):
        """Power fraction should be weighted by dt_minutes."""
        mgr._accumulated_dt["r1"] = 1.0
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 0.5
        mgr.process(
            "r1",
            **{**self.COMMON, "dt_minutes": 1.0},
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=False,
            raw_open=False,
        )
        # (0.5 * 1.0 + 1.0 * 1.0) / 2.0 = 0.75
        assert mgr._accumulated_pf["r1"] == pytest.approx(0.75)

    def test_last_temps_always_updated(self, mgr, model_manager):
        """last_temps should be updated on every process call."""
        mgr.process(
            "r1",
            **{**self.COMMON, "current_temp": 22.5},
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=False,
            raw_open=False,
        )
        assert mgr.last_temps["r1"] == 22.5

    def test_occupancy_passed_through(self, mgr, model_manager):
        """q_occupancy should be passed to model_manager.update."""
        mgr._accumulated_dt["r1"] = EKF_UPDATE_MIN_DT
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 1.0
        mgr.process(
            "r1",
            **{**self.COMMON, "dt_minutes": 1.0},
            ekf_mode="heat",
            ekf_pf=1.0,
            window_open=False,
            raw_open=False,
            q_occupancy=0.5,
        )
        _, kwargs = model_manager.update.call_args
        assert kwargs["q_occupancy"] == 0.5


class TestClearAndRemove:
    def test_clear_removes_accumulated_state(self, mgr):
        mgr._accumulated_dt["r1"] = 5.0
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 0.8
        mgr.clear("r1")
        assert "r1" not in mgr._accumulated_dt
        assert "r1" not in mgr._accumulated_mode
        assert "r1" not in mgr._accumulated_pf

    def test_clear_preserves_last_temps(self, mgr):
        mgr.last_temps["r1"] = 20.0
        mgr.clear("r1")
        assert mgr.last_temps["r1"] == 20.0

    def test_remove_room_clears_everything(self, mgr):
        mgr._accumulated_dt["r1"] = 5.0
        mgr._accumulated_mode["r1"] = "heat"
        mgr._accumulated_pf["r1"] = 0.8
        mgr.last_temps["r1"] = 20.0
        mgr.remove_room("r1")
        assert "r1" not in mgr._accumulated_dt
        assert "r1" not in mgr.last_temps

    def test_clear_nonexistent_room_no_error(self, mgr):
        mgr.clear("nonexistent")

    def test_remove_nonexistent_room_no_error(self, mgr):
        mgr.remove_room("nonexistent")
