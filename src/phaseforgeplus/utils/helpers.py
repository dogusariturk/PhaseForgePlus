"""Generate NEQ YAML blocks for ESPEI from pycalphad Database."""

from typing import Any

from espei.utils import MemoryStorage, PickleableTinyDB
from pycalphad import Database, calculate


def generate_neq_yaml(
    db: Database,
    phase: str,
    components: list[str],
    points: Any,
    pressure: float = 101325,
    temperature: float = 298.15,
) -> dict[str, Any]:
    """Dynamically generate an ESPEI NEQ YAML block for a single phase.

    Args:
        db (Database): The pycalphad Database object containing the thermodynamic data.
        phase (str): The name of the phase for which to generate the YAML block.
        components (list[str]): A list of component symbols for the phase.
        points (Any): A list of points (indices) for which to calculate the properties.
        pressure (float, optional): The pressure at which to calculate the properties, by default 101325 Pa.
        temperature (float, optional): The temperature at which to calculate the properties, by default 298.15 K.

    Returns:
        dict[str, Any]: A dictionary representing the NEQ YAML block for the specified phase.
    """
    calc_results = calculate(
        dbf=db,
        comps=components,
        phases=phase,
        output="HM_MIX",
        P=pressure,
        T=temperature,
    )

    hm_mix = [[calc_results.HM_MIX.values.squeeze().tolist()[point] for point in points]]

    occupancies = [[calc_results.X.values.squeeze().tolist()[point]] for point in points]

    return {
        "components": components,
        "phases": phase,
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": [1],
            "sublattice_configuraations": [[components] for _ in range(len(points))],
            "sublattice_occupancies": occupancies,
        },
        "conditions": {
            "P": pressure,
            "T": temperature,
        },
        "output": "HM_MIX",
        "values": hm_mix,
        "weight": 0.1,
    }


def generate_neq_db(
    db: Database,
    phases: list[str],
    components: list[str],
    points: Any,
    pressure: float = 101325,
    temperature: float = 298.15,
) -> PickleableTinyDB:
    """Generate a PickleableTinyDB containing NEQ YAML blocks for multiple phases.

    Args:
        db (Database): The pycalphad Database object containing the thermodynamic data.
        phases (list[str]): A list of phase names for which to generate the YAML blocks.
        components (list[str]): A list of component symbols for the phases.
        points (Any):  A list of points (indices) for which to calculate the properties.
        pressure (float, optional): The pressure at which to calculate the properties, by default 101325 Pa.
        temperature (float, optional): The temperature at which to calculate the properties, by default 298.15 K.

    Returns:
        PickleableTinyDB: A PickleableTinyDB instance containing the NEQ YAML blocks for the specified phases.
    """
    with PickleableTinyDB(storage=MemoryStorage) as tiny_db:
        for phase in phases:
            tiny_db.insert(
                generate_neq_yaml(
                    db=db,
                    phase=phase,
                    components=components,
                    points=points,
                    pressure=pressure,
                    temperature=temperature,
                )
            )
    return tiny_db
