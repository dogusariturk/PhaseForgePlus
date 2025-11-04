"""Test cases for the PhaseForgePlus class."""

from pathlib import Path

import numpy as np
import pytest
from espei.utils import PickleableTinyDB
from pycalphad import Database
from scipy import optimize as scipy_optimize

from phaseforgeplus import PhaseForgePlus


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
