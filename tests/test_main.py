"""Test cases for the PhaseForgePlus class."""

import unittest
from pathlib import Path

import numpy as np
from espei.utils import PickleableTinyDB
from pycalphad import Database
from scipy import optimize as scipy_optimize

from phaseforgeplus import PhaseForgePlus


class TestMain(unittest.TestCase):
    """Unit tests for the PhaseForgePlus class."""

    def setUp(self):
        """Set up the test environment with a sample TDB file and ZPF data."""
        self.db_path = Path(__file__).parent.joinpath("data/pt-w.tdb")
        self.zpf_path = Path(__file__).parent.joinpath("data/")

    def test_initializes_with_valid_inputs(self):
        """Test if PhaseForgePlus initializes correctly with valid inputs."""
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)
        self.assertIsInstance(instance.db, Database)
        self.assertIsInstance(instance.db_zpf, PickleableTinyDB)
        self.assertIsInstance(instance.db_neq, PickleableTinyDB)
        self.assertEqual(instance.pressure, pressure)
        self.assertEqual(instance.temperature, temperature)

    def test_raises_error_for_invalid_tdb_path(self):
        """Test if PhaseForgePlus raises an error for an invalid TDB path."""
        db_path = "data/invalid.tdb"
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        with self.assertRaises(FileNotFoundError):
            PhaseForgePlus(db_path, self.zpf_path, points, pressure, temperature)

    def test_raises_error_for_invalid_zpf_path(self):
        """Test if PhaseForgePlus raises an error for an invalid ZPF path."""
        zpf_path = "invalid_data"
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, zpf_path, points, pressure, temperature)
        self.assertEqual(len(instance.db_zpf), 0)

    def test_handles_empty_points_list(self):
        """Test if PhaseForgePlus handles an empty points list correctly."""
        points = []
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)
        self.assertEqual(instance.points, points)
        self.assertEqual(len(instance.db_neq), 3)

    def optimizes_with_valid_inputs(self):  # FIXME
        """Test if PhaseForgePlus optimizes correctly with valid inputs."""
        points = [0, 1]
        pressure = 101325
        temperature = 298.15
        instance = PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)
        result = instance.optimize()
        self.assertIsInstance(result, scipy_optimize.OptimizeResult)
        self.assertTrue(result.success)

    def test_raises_error_for_invalid_pressure(self):
        """Test if PhaseForgePlus raises an error for an invalid pressure value."""
        points = [0, 1]
        pressure = "invalid_pressure"
        temperature = 298.15
        with self.assertRaises(ValueError):
            PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)

    def test_raises_error_for_invalid_temperature(self):
        """Test if PhaseForgePlus raises an error for an invalid temperature value."""
        points = [0, 1]
        pressure = 101325
        temperature = "invalid_temperature"
        with self.assertRaises(ValueError):
            PhaseForgePlus(self.db_path, self.zpf_path, points, pressure, temperature)

    def test_returns_correct_initial_values_with_valid_symbols(self):
        """Test if PhaseForgePlus returns correct initial values with valid symbols."""
        instance = PhaseForgePlus(self.db_path, self.zpf_path, [0, 1], 101325, 298.15)
        initial_values = instance.get_initial_values()
        self.assertIsInstance(initial_values, np.ndarray)
        self.assertGreater(len(initial_values), 0)
