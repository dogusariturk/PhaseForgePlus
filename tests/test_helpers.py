"""Test cases for the helper functions in phaseforgeplus.utils.helpers."""

from pathlib import Path

from espei.utils import PickleableTinyDB

from phaseforgeplus.io import load_tdb
from phaseforgeplus.utils import generate_neq_db, generate_neq_yaml


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
        assert result["values"] == [[]]
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
