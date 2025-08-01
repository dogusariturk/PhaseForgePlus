"""Objective function for optimization combining ZPF and non-equilibrium data."""

import numpy as np
from espei.error_functions.non_equilibrium_thermochemical_error import (
    FixedConfigurationPropertyResidual,
)
from espei.error_functions.zpf_error import ZPFResidual
from espei.utils import PickleableTinyDB, database_symbols_to_fit
from pycalphad import Database


def objective_function(db: Database, db_zpf: PickleableTinyDB, db_neq: PickleableTinyDB):
    """Create an objective function for optimization combining ZPF and non-equilibrium data.

    Args:
        db (Database): The thermodynamic database containing the model parameters
        db_zpf (PickleableTinyDB): Database containing ZPF data points
        db_neq (PickleableTinyDB): Database containing non-equilibrium thermochemical data

    Returns:
        callable: A function that takes an array of parameters and returns the negative log likelihood
                 and its gradient as a tuple (neg_likelihood, neg_gradient)
    """
    symbols_to_fit = database_symbols_to_fit(db)

    def func(params):
        """Objective function that combines ZPF and non-equilibrium data."""
        residual_func_zpf = ZPFResidual(db, db_zpf, phase_models=None, symbols_to_fit=symbols_to_fit)
        likelihood_zpf, gradient_zpf = residual_func_zpf.get_likelihood(np.array(params))

        residual_func_neq = FixedConfigurationPropertyResidual(
            db, db_neq, phase_models=None, symbols_to_fit=symbols_to_fit
        )
        likelihood_neq, gradient_neq = residual_func_neq.get_likelihood(np.array(params))

        neg_likelihood = -(likelihood_zpf + likelihood_neq)
        neg_gradient = -1 * (gradient_zpf + gradient_neq)

        return neg_likelihood, neg_gradient

    return func
