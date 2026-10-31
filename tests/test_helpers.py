"""Test cases for the helper functions in phaseforgeplus.utils.helpers."""

from pathlib import Path

import numpy as np
import pytest
from espei.error_functions.non_equilibrium_thermochemical_error import FixedConfigurationPropertyResidual
from espei.utils import PickleableTinyDB, database_symbols_to_fit, unpack_piecewise

from phaseforgeplus.io import load_tdb
from phaseforgeplus.utils import generate_neq_db, generate_neq_yaml

NEQ_SIGMA = 500.0 / 0.1  # ESPEI's standard deviation of HM in J/mol, divided by the weight of generate_neq_yaml


class TestHelpers:
    """Unit tests for helper functions in phaseforgeplus.utils.helpers."""

    def setup_method(self):
        """Set up the test environment with a sample TDB file."""
        db_path = Path(__file__).parent.joinpath("../example/data/pt-w.tdb")
        self.db = load_tdb(db_path)

    def test_generates_correct_neq_yaml_block(self):
        """Test if the generate_neq_yaml function creates a valid YAML block."""
        phase = "FCC_A1"
        components = ["PT", "W"]
        points = [0, 1]
        result = generate_neq_yaml(self.db, phase, components, points)
        assert isinstance(result, dict)
        assert "components" in result
        assert "phases" in result
        assert "solver" in result
        assert "conditions" in result
        assert "output" in result
        assert "values" in result

    def test_handles_empty_points_in_neq_yaml(self):
        """Test if the generate_neq_yaml function handles empty points correctly."""
        phase = "FCC_A1"
        components = ["PT", "W"]
        points = []
        result = generate_neq_yaml(self.db, phase, components, points)
        assert result["values"] == [[[]]]
        assert result["solver"]["sublattice_occupancies"] == []

    def test_generates_neq_db_with_multiple_phases(self):
        """Test if the generate_neq_db function creates a valid NEQ database."""
        phases = ["FCC_A1", "BCC_A2"]
        components = ["PT", "W"]
        points = [0, 1]
        result = generate_neq_db(self.db, phases, components, points)
        assert isinstance(result, PickleableTinyDB)
        assert len(result) > 0

    def test_handles_empty_phases_in_neq_db(self):
        """Test if the generate_neq_db function handles empty phases correctly."""
        phases = []
        components = ["PT", "W"]
        points = [0, 1]
        result = generate_neq_db(self.db, phases, components, points)
        assert isinstance(result, PickleableTinyDB)
        assert len(result) == 0

    def neq_term(self, components, shift=0.0):
        """Return the NEQ dataset, residuals and log-likelihood for FCC_A1, with the parameter VV0011 of L0 shifted."""
        db_neq = generate_neq_db(self.db, ["FCC_A1"], components, [1801, 1001, 202])
        names = database_symbols_to_fit(self.db)
        theta = np.array([unpack_piecewise(self.db.symbols[name]) for name in names])
        theta[names.index("VV0011")] += shift
        residual = FixedConfigurationPropertyResidual(self.db, db_neq, phase_models=None, symbols_to_fit=names)
        return db_neq, np.array(residual.get_residuals(theta)[0]), residual.get_likelihood(theta)

    @pytest.mark.parametrize("components", [["PT", "W"], ["W", "PT"]])
    def test_neq_term_reaches_espei(self, components):
        """Test that ESPEI reads every value, for both orders of the components, and the residuals vanish at the data."""
        _, residuals, likelihood = self.neq_term(components)
        assert residuals == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)
        assert likelihood == pytest.approx(3 * -np.log(NEQ_SIGMA * np.sqrt(2 * np.pi)))

    def test_neq_residual_matches_hand_computation(self):
        """Test the term with L0 raised by 2 kJ/mol: HM_MIX rises by x_PT * x_W * 2000 J/mol of atoms."""
        db_neq, residuals, likelihood = self.neq_term(["PT", "W"], shift=2.0)
        occupancies = db_neq.all()[0]["solver"]["sublattice_occupancies"]
        expected = np.array([x_pt * x_w * 2000.0 for ((x_pt, x_w),) in occupancies])
        assert np.all(expected > 0)
        assert residuals == pytest.approx(expected, rel=1e-8)
        hand = np.sum(-0.5 * (expected / NEQ_SIGMA) ** 2 - np.log(NEQ_SIGMA * np.sqrt(2 * np.pi)))
        assert likelihood == pytest.approx(hand)
