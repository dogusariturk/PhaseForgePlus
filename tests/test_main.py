"""Test cases for the PhaseForgePlus class."""

from pathlib import Path

import numpy as np
import pytest
from espei.utils import PickleableTinyDB
from pycalphad import Database
from scipy import optimize as scipy_optimize

from phaseforgeplus import PhaseForgePlus, main

PT_W_START_OBJECTIVE = 6627.6


class TestMain:
    """Unit tests for the PhaseForgePlus class."""

    def setup_method(self):
        """Set up the test environment with a sample TDB file and ZPF data."""
        self.db_path = Path(__file__).parent.joinpath("../example/data/pt-w.tdb")
        self.zpf_path = Path(__file__).parent.joinpath("../example/data/")

    def test_initializes_with_valid_inputs(self):
        """Test if PhaseForgePlus initializes correctly with valid inputs."""
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)
        assert isinstance(instance.db, Database)
        assert isinstance(instance.db_zpf, PickleableTinyDB)
        assert isinstance(instance.db_neq, PickleableTinyDB)
        assert instance.pressure == pressure
        assert instance.temperature == temperature

    def test_raises_error_for_invalid_tdb_path(self):
        """Test if PhaseForgePlus raises an error for an invalid TDB path."""
        db_path = "../example/data/invalid.tdb"
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        with pytest.raises(FileNotFoundError):
            PhaseForgePlus(db_path, self.zpf_path, points, pressure, temperature)

    def test_raises_error_for_invalid_zpf_path(self):
        """Test if PhaseForgePlus raises an error for an invalid ZPF path."""
        zpf_path = "invalid_data"
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, zpf_path, points, pressure, temperature)
        assert len(instance.db_zpf) == 0

    def test_handles_empty_points_list(self):
        """Test if PhaseForgePlus handles an empty points list correctly."""
        points = []
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)
        assert instance.points == points
        assert len(instance.db_neq) == 3

    def optimizes_with_valid_inputs(self):  # FIXME
        """Test if PhaseForgePlus optimizes correctly with valid inputs."""
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)
        result = instance.optimize()
        assert isinstance(result, scipy_optimize.OptimizeResult)
        assert result.success

    def test_raises_error_for_invalid_pressure(self):
        """Test if PhaseForgePlus raises an error for an invalid pressure value."""
        points = [0, 1]
        pressure = "invalid_pressure"
        temperature = 298.15
        with pytest.raises(ValueError):
            PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)

    def test_raises_error_for_invalid_temperature(self):
        """Test if PhaseForgePlus raises an error for an invalid temperature value."""
        points = [0, 1]
        pressure = 101325
        temperature = "invalid_temperature"
        with pytest.raises(ValueError):
            PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)

    def test_returns_correct_initial_values_with_valid_symbols(self):
        """Test if PhaseForgePlus returns correct initial values with valid symbols."""
        instance = PhaseForgePlus(self.db_path, self.zpf_path, [0, 1], 101325, 298.15)
        initial_values = instance.get_initial_values()
        assert isinstance(initial_values, np.ndarray)
        assert len(initial_values) > 0

    def setup_quadratic(self, monkeypatch):
        """Return an instance with a known quadratic objective, and the list of calls to scipy's minimize."""
        instance = PhaseForgePlus(self.db_path, self.zpf_path, [0, 1], 101325, 298.15)
        num_parameters = len(instance.get_initial_values())
        target = np.arange(num_parameters, dtype=float)

        def fake_objective(db, db_zpf, db_neq, hinge=None):
            return lambda params: (float(((params - target) ** 2).sum()), 2 * (params - target))

        calls = []
        minimize = scipy_optimize.minimize

        def recording_minimize(*args, **kwargs):
            calls.append((args, kwargs))
            return minimize(*args, **kwargs)

        monkeypatch.setattr(main, "objective_function", fake_objective)
        monkeypatch.setattr(main.scipy_optimize, "minimize", recording_minimize)
        return instance, target, calls

    def test_optimize_default_is_unchanged(self, monkeypatch):
        """Test that without arguments optimize runs conjugate gradient from the database values with its old options."""
        instance, _, calls = self.setup_quadratic(monkeypatch)
        result = instance.optimize()
        (func, start), kwargs = calls[0]
        assert callable(func)
        assert np.array_equal(start, instance.get_initial_values())
        assert kwargs == {"jac": True, "method": "CG", "options": {"disp": True}}
        assert len(calls) == 1
        assert isinstance(result, scipy_optimize.OptimizeResult)
        assert not hasattr(result, "runs")

    def test_optimize_passes_method_and_options(self, monkeypatch):
        """Test that the method and options reach scipy and replace the default options."""
        instance, target, calls = self.setup_quadratic(monkeypatch)
        result = instance.optimize(method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-12})
        assert calls[0][1]["method"] == "L-BFGS-B"
        assert calls[0][1]["options"] == {"maxiter": 100, "ftol": 1e-12}
        assert result.x == pytest.approx(target, abs=1e-4)

    def test_optimize_maxiter_limits_the_run(self, monkeypatch):
        """Test that maxiter stops the run."""
        instance, target, _ = self.setup_quadratic(monkeypatch)
        curvature = np.arange(1.0, len(target) + 1)

        def fake_objective(db, db_zpf, db_neq, hinge=None):
            return lambda params: (float((curvature * (params - target) ** 2).sum()), 2 * curvature * (params - target))

        monkeypatch.setattr(main, "objective_function", fake_objective)
        result = instance.optimize(options={"maxiter": 1})
        assert result.nit == 1
        assert not result.success

    def test_optimize_returns_the_best_of_the_starts(self, monkeypatch):
        """Test that each start runs, in order, and the lowest objective is returned with all runs."""
        instance, target, calls = self.setup_quadratic(monkeypatch)
        num_parameters = len(target)
        starts = [target + 1.0, target.tolist(), np.full(num_parameters, 1e3)]
        result = instance.optimize(options={"maxiter": 0}, starts=starts)
        assert len(calls) == len(starts)
        assert np.array_equal(calls[1][0][1], target)
        assert result.start_index == 1
        assert result.fun == 0.0
        assert [run.fun for run in result.runs] == pytest.approx([float(num_parameters), 0.0, result.runs[2].fun])
        assert result.runs[1] is result

    def test_optimize_ignores_a_nan_run_among_the_starts(self, monkeypatch):
        """Test that a run with a NaN objective is not returned as the best."""
        instance, target, _ = self.setup_quadratic(monkeypatch)
        minimize = main.scipy_optimize.minimize

        def fake_minimize(func, start, **kwargs):
            result = minimize(func, start, **kwargs)
            if start[0] == target[0] + 1.0:
                result.fun = np.nan
            return result

        monkeypatch.setattr(main.scipy_optimize, "minimize", fake_minimize)
        result = instance.optimize(options={"maxiter": 0}, starts=[target + 1.0, target + 2.0])
        assert result.start_index == 1

    @pytest.mark.parametrize("starts", [[], [[1.0, 2.0]]])
    def test_optimize_rejects_bad_starts(self, monkeypatch, starts):
        """Test that an empty list of starts and a start of the wrong length raise an error."""
        instance, _, calls = self.setup_quadratic(monkeypatch)
        with pytest.raises(ValueError, match="starts must be"):
            instance.optimize(starts=starts)
        assert not calls

    def test_optimize_with_options_runs_on_pt_w(self):
        """Test a short real run with options, without a fake objective."""
        instance = PhaseForgePlus(
            self.db_path, self.zpf_path, [1801, 1601, 1401, 1201, 1001, 802, 602, 402, 202], 101325, 298.15
        )
        result = instance.optimize(method="L-BFGS-B", options={"maxiter": 1})
        assert result.nit == 1
        assert result.fun < PT_W_START_OBJECTIVE
