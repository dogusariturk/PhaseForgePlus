"""Test cases for the objective function in phaseforgeplus.utils.objective."""

from pathlib import Path

import numpy as np
import pytest
from espei.utils import PickleableTinyDB, database_symbols_to_fit, unpack_piecewise

from phaseforgeplus.io import load_tdb, search_and_load_yaml
from phaseforgeplus.utils import generate_neq_db
from phaseforgeplus.utils.objective import objective_function

EXAMPLE = Path(__file__).parent.joinpath("../example/data/")
POINTS = [1801, 1601, 1401, 1201, 1001, 802, 602, 402, 202]


def setup_pt_w():
    """Return a new objective of the Pt-W example, its database, its fit parameters and their values."""
    db = load_tdb(EXAMPLE / "pt-w.tdb")
    db_neq = generate_neq_db(db, list(db.phases), list(db.elements), POINTS)
    names = database_symbols_to_fit(db)
    values = np.array([unpack_piecewise(db.symbols[name]) for name in names])
    return objective_function(db, search_and_load_yaml(EXAMPLE), db_neq), db, names, values


class TestPtW:
    """Tests of the objective with the ESPEI of uv.lock on the Pt-W example."""

    def test_value_at_the_database_parameters(self):
        """Test the objective at the database parameters. The tolerance is above the rounding seen across pycalphad."""
        func, _, _, values = setup_pt_w()
        assert func(values)[0] == pytest.approx(6627.5843264, rel=1e-8)

    def test_database_keeps_its_symbols_and_a_second_objective_agrees(self):
        """Test that an evaluation leaves the database as it was, so a second objective in this process works."""
        func, db, names, values = setup_pt_w()
        symbols = set(db.symbols)
        first = func(values)[0]
        assert set(db.symbols) == symbols
        assert database_symbols_to_fit(db) == names
        assert setup_pt_w()[0](values)[0] == pytest.approx(first, rel=1e-12)

    def test_gradient_matches_central_differences(self):
        """Test the gradient along one direction, which all parameters contribute to."""
        func, _, _, values = setup_pt_w()
        direction = np.random.default_rng(0).standard_normal(len(values)) * np.maximum(np.abs(values), 1.0)
        step = 1e-6
        gradient = func(values)[1]
        difference = (func(values + step * direction)[0] - func(values - step * direction)[0]) / (2 * step)
        assert gradient @ direction == pytest.approx(difference, rel=1e-5)


class TestObjective:
    """Unit tests for the objective function in phaseforgeplus.utils.objective."""

    def setup_method(self):
        """Set up the test environment with a sample TDB file and ZPF data."""
        db_path = Path(__file__).parent.joinpath("../example/data/pt-w.tdb")
        self.db = load_tdb(db_path)
        zpf_path = Path(__file__).parent.joinpath("../example/data/")
        self.db_zpf = search_and_load_yaml(zpf_path)

    def calculates_negative_log_likelihood_correctly(self):  # FIXME
        """Test if the objective function calculates the negative log likelihood correctly."""
        db_neq = generate_neq_db(self.db, ["FCC_A1", "BCC_A2"], ["PT", "W"], [0, 1])
        func = objective_function(self.db, self.db_zpf, db_neq)
        params = np.arange(1, len(database_symbols_to_fit(self.db)) + 1)
        neg_likelihood, neg_gradient = func(params)
        assert isinstance(neg_likelihood, float)
        assert isinstance(neg_gradient, np.ndarray)

    def handles_empty_zpf_database(self):  # FIXME
        """Test if the objective function handles an empty ZPF database correctly."""
        db_zpf = PickleableTinyDB()
        db_neq = PickleableTinyDB("../example/data/neq_data.json")
        func = objective_function(self.db, db_zpf, db_neq)
        params = [1.0, 2.0, 3.0]
        neg_likelihood, neg_gradient = func(params)
        assert isinstance(neg_likelihood, float)
        assert isinstance(neg_gradient, np.ndarray)

    def handles_empty_neq_database(self):  # FIXME
        """Test if the objective function handles an empty NEQ database correctly."""
        db_zpf = PickleableTinyDB("../example/data/zpf_data.json")
        db_neq = PickleableTinyDB()
        func = objective_function(self.db, db_zpf, db_neq)
        params = [1.0, 2.0, 3.0]
        neg_likelihood, neg_gradient = func(params)
        assert isinstance(neg_likelihood, float)
        assert isinstance(neg_gradient, np.ndarray)

    def raises_error_for_invalid_parameters(self):  # FIXME
        """Test if the objective function raises an error for invalid parameters."""
        db_zpf = PickleableTinyDB("../example/data/zpf_data.json")
        db_neq = PickleableTinyDB("../example/data/neq_data.json")
        func = objective_function(self.db, db_zpf, db_neq)
        params = "invalid_params"
        with pytest.raises(TypeError):
            func(params)

    def handles_no_symbols_to_fit(self):  # FIXME
        """Test if the objective function handles no symbols to fit correctly."""
        db_zpf = PickleableTinyDB("../example/data/zpf_data.json")
        db_neq = PickleableTinyDB("../example/data/neq_data.json")
        func = objective_function(self.db, db_zpf, db_neq)
        params = []
        neg_likelihood, neg_gradient = func(params)
        assert neg_likelihood == 0.0
        assert np.all(neg_gradient == 0)
