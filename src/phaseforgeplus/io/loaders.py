"""Loaders for various file formats used in PhaseForge+."""

from pathlib import Path
from typing import Any

import yaml
from espei.datasets import recursive_glob
from espei.utils import MemoryStorage, PickleableTinyDB
from pycalphad import Database


def load_tdb(path: str) -> Database:
    """Load a TDB file from disk into a pycalphad Database.

    Args:
        path (str): The path to the TDB file.

    Returns:
        Database: A pycalphad Database object containing the thermodynamic data.
    """
    return Database(path)


def load_yaml(path: str) -> dict[str, Any]:
    """Parse a YAML file from disk or provided as a string and return its contents as a dict.

    Args:
        path (str): The path to the YAML file or a string containing YAML content.

    Returns:
        dict[str, Any]: A dictionary containing the parsed YAML data.
    """
    if Path(path).is_file():
        with open(path) as stream:
            return yaml.safe_load(stream)

    return yaml.safe_load(path)


def search_and_load_yaml(path: str) -> PickleableTinyDB:
    """Search for YAML files in a directory and load them into a PickleableTinyDB.

    Args:
        path (str): The directory path to search for YAML files.

    Returns:
        PickleableTinyDB: A PickleableTinyDB instance containing the data from the YAML files.
    """
    with PickleableTinyDB(storage=MemoryStorage) as tiny_db:
        for yaml in recursive_glob(path, pattern="*.yaml"):
            tiny_db.insert(load_yaml(yaml))
    return tiny_db
