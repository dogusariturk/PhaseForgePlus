"""Test suite for the loaders module in phaseforgeplus.io.loaders."""

from pathlib import Path

import pytest
import yaml
from espei.utils import PickleableTinyDB
from pycalphad import Database

from phaseforgeplus.io.loaders import load_tdb, load_yaml, search_and_load_yaml


class TestLoaders:
    """Unit tests for the loaders module in phaseforgeplus.io.loaders."""

    def test_loads_tdb_file_into_database(self):
        """Test if the load_tdb function correctly loads a TDB file into a Database object."""
        db_path = Path(__file__).parent.joinpath("../example/data/pt-w.tdb")
        db = load_tdb(db_path)
        assert isinstance(db, Database)

    def test_raises_error_for_nonexistent_tdb_file(self):
        """Test if the load_tdb function raises an error for a non-existent TDB file."""
        with pytest.raises(FileNotFoundError):
            load_tdb("nonexistent.tdb")

    def test_parses_yaml_file_correctly(self):
        """Test if the load_yaml function correctly parses a YAML file."""
        zpf_path = Path(__file__).parent.joinpath("../example/data/zpf_1.yaml")
        data = load_yaml(zpf_path)
        assert isinstance(data, dict)
        assert "components" in data
        assert "output" in data
        assert "values" in data

    def test_parses_yaml_string_correctly(self):
        """Test if the load_yaml function correctly parses a YAML string."""
        yaml_string = "key: value"
        data = load_yaml(yaml_string)
        assert data == {"key": "value"}

    def test_raises_error_for_invalid_yaml(self):
        """Test if the load_yaml function raises an error for invalid YAML."""
        invalid_yaml = "key: value: another"
        with pytest.raises(yaml.YAMLError):
            load_yaml(invalid_yaml)

    def test_loads_yaml_files_from_directory(self):
        """Test if the search_and_load_yaml function loads YAML files from a directory."""
        data_path = Path(__file__).parent.joinpath("../example/data/")
        db = search_and_load_yaml(data_path)
        assert isinstance(db, PickleableTinyDB)
        assert len(db) > 0

    def test_handles_empty_directory_for_yaml_loading(self):
        """Test if the search_and_load_yaml function handles an empty directory."""
        db = search_and_load_yaml("empty_dir")
        assert isinstance(db, PickleableTinyDB)
        assert len(db) == 0
