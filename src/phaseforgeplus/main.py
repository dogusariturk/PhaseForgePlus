"""PhaseForgePlus: A Python package for thermodynamic database optimization."""

from pathlib import Path

import numpy as np
from espei.utils import database_symbols_to_fit, unpack_piecewise
from scipy import optimize as scipy_optimize

from phaseforgeplus.io import load_tdb, search_and_load_yaml
from phaseforgeplus.utils import generate_neq_db
from phaseforgeplus.utils.objective import objective_function


class PhaseForgePlus:
    """PhaseForgePlus: A Python package for thermodynamic database optimization.

    This class provides methods to optimize a thermodynamic database using
    a combination of ZPF and non-equilibrium thermochemical data.

    Attributes:
        db (str | Path): The thermodynamic database containing the model parameters.
        zpf_path (str): Path to the ZPF data file.
        points (list): List of points for optimization.
        pressure (float): Pressure in Pascals.
        temperature (float): Temperature in Kelvin.
        components (list): List of components in the database.
        phases (list): List of phases in the database.
        db_zpf (PickleableTinyDB): Database containing ZPF data points.
        db_neq (PickleableTinyDB): Database containing non-equilibrium thermochemical data.
    """

    def __init__(self, db: str | Path, zpf_path: str | Path, points: list, pressure: float, temperature: float) -> None:
        """Initialize the PhaseForgePlus class.

        Args:
            db (str | Path): Path to the TDB file or a pycalphad Database object.
            zpf_path (str | Path): Path to the ZPF data file.
            points (list): List of points for optimization.
            pressure (float): Pressure in Pascals.
            temperature (float): Temperature in Kelvin.
        """
        self.db = load_tdb(db)
        self.zpf_path = zpf_path
        self.points = points
        self.components = list(self.db.elements)
        self.phases = list(self.db.phases.keys())
        self.pressure = pressure
        self.temperature = temperature

        self.db_zpf = search_and_load_yaml(zpf_path)

        self.db_neq = generate_neq_db(
            db=self.db,
            phases=self.phases,
            components=self.components,
            points=self.points,
            pressure=self.pressure,
            temperature=self.temperature,
        )

    def optimize(self) -> scipy_optimize.OptimizeResult:
        """Optimize the thermodynamic database using a combination of ZPF and non-equilibrium data."""
        func = objective_function(self.db, self.db_zpf, self.db_neq)
        optimized_tdb = scipy_optimize.minimize(
            func,
            self.get_initial_values(),
            jac=True,
            method="CG",
            options={"disp": True},
        )
        return optimized_tdb

    def get_initial_values(self) -> np.ndarray:
        """Get initial values for the optimization based on the database symbols to fit."""
        symbols_to_fit = database_symbols_to_fit(self.db)
        initial_values = np.array([unpack_piecewise(self.db.symbols[s]) for s in symbols_to_fit])
        return initial_values
