"""Test cases for the objective function in phaseforgeplus.utils.objective."""

from pathlib import Path

import numpy as np
import pytest
from espei.utils import PickleableTinyDB, database_symbols_to_fit

from phaseforgeplus.io import load_tdb, search_and_load_yaml
from phaseforgeplus.utils import generate_neq_db
from phaseforgeplus.utils.objective import objective_function


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
