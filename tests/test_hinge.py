"""Test cases for the hinge penalty in phaseforgeplus.utils.hinge."""

from pathlib import Path

import numpy as np
import pytest
from espei.utils import database_symbols_to_fit, unpack_piecewise
from pycalphad import Database
from scipy import optimize as scipy_optimize

from phaseforgeplus import PhaseForgePlus, main
from phaseforgeplus.io import load_tdb
from phaseforgeplus.utils import objective
from phaseforgeplus.utils.hinge import HingeOptions, HingePenalty

EXAMPLE = Path(__file__).parent.joinpath("../example/data")
SYNTHETIC = Path(__file__).parent.joinpath("data/hinge_synthetic.tdb")
GAS_CONSTANT = 8.3145
SPURIOUS_CELL = {
    "T": 1000.0,
    "x": 0.5,
    "expected": ["SOLUTION"],
    "penalize": ["SPURIOUS"],
    "sigma": 700.0,
    "confidence": 0.8,
}


def setup_synthetic(**symbols):
    """Return the synthetic database, its fit parameters and their values, with some values replaced."""
    db = load_tdb(SYNTHETIC)
    names = database_symbols_to_fit(db)
    values = {name: float(unpack_piecewise(db.symbols[name])) for name in names}
    return db, names, np.array([symbols.get(name, values[name]) for name in names])


def central_differences(hinge, theta, step=1e-3):
    """Return the central-difference gradient of the penalty."""
    gradient = np.zeros(len(theta))
    for j in range(len(theta)):
        delta = np.zeros(len(theta))
        delta[j] = step * max(1.0, abs(theta[j]))
        gradient[j] = (hinge(theta + delta)[0] - hinge(theta - delta)[0]) / (2 * delta[j])
    return gradient


class TestHingePenalty:
    """Unit tests of the penalty and its gradient."""

    def test_spurious_phase_matches_closed_form(self):
        """Test the penalty and gradient of a stoichiometric phase below a known target line."""
        db, names, theta = setup_synthetic()
        hinge = HingePenalty(db, names, [SPURIOUS_CELL], "W")
        interaction, spurious = theta[names.index("VV0000")], theta[names.index("VV0001")]
        dinteraction = 0.49 * 0.51
        mixing = GAS_CONSTANT * 1000.0 * (0.49 * np.log(0.49) + 0.51 * np.log(0.51))
        driving_force = spurious - (mixing + interaction * dinteraction)
        scaled = (50.0 - driving_force) / 700.0
        penalty, gradient = hinge(theta)
        assert penalty == pytest.approx(0.5 * 0.8 * scaled**2, rel=1e-3)
        assert gradient[names.index("VV0001")] == pytest.approx(-0.8 * scaled / 700.0, rel=1e-3)
        assert gradient[names.index("VV0000")] == pytest.approx(0.8 * scaled / 700.0 * dinteraction, rel=1e-3)
        assert [entry["phase"] for entry in hinge.violations(theta)] == ["SPURIOUS"]

    def test_zero_when_well_above(self):
        """Test that a phase well above the target line gives no penalty and no gradient."""
        db, names, theta = setup_synthetic(VV0001=20000.0)
        hinge = HingePenalty(db, names, [SPURIOUS_CELL], "W")
        penalty, gradient = hinge(theta)
        assert penalty == 0.0
        assert np.all(gradient == 0.0)
        assert hinge.violations(theta) == []

    def test_gradient_matches_finite_differences(self):
        """Test the gradient for one and two expected phases, with vacancies and two mixing sublattices."""
        db, names, theta = setup_synthetic()
        sigma = {"SOLUTION": 500.0, "SPURIOUS": 700.0, "ORDERED": 900.0}
        hinge = HingePenalty(
            db,
            names,
            [
                {"T": 1000.0, "x": 0.3, "expected": ["SOLUTION"], "penalize": ["SPURIOUS", "ORDERED"], "sigma": sigma},
                {
                    "T": 1200.0,
                    "x": 0.5,
                    "expected": ["SOLUTION", "SPURIOUS"],
                    "tie_line": {"SOLUTION": 0.2, "SPURIOUS": 0.5},
                    "penalize": ["ORDERED"],
                    "sigma": sigma,
                },
                {"T": 800.0, "x": 0.5, "expected": ["ORDERED"], "penalize": ["SOLUTION", "SPURIOUS"], "sigma": sigma},
            ],
            "W",
        )
        gradient = hinge(theta)[1]
        assert np.any(gradient != 0.0)
        assert gradient == pytest.approx(central_differences(hinge, theta), rel=1e-4, abs=1e-8)

    def test_minimizing_the_penalty_lifts_the_spurious_phase(self):
        """Test that the gradient removes the spurious phase in a small fit."""
        db, names, theta = setup_synthetic()
        hinge = HingePenalty(db, names, [SPURIOUS_CELL], "W")
        result = scipy_optimize.minimize(hinge, theta, jac=True, method="L-BFGS-B")
        assert hinge(theta)[0] > 1.0
        assert result.fun == pytest.approx(0.0, abs=1e-6)
        assert result.x[names.index("VV0001")] > theta[names.index("VV0001")]

    def test_vacancy_cap_is_an_option(self):
        """Test that the vacancy cap limits the configurations of a sublattice with vacancies."""
        db, names, theta = setup_synthetic()
        cells = [{"T": 800.0, "x": 0.3, "expected": ["SOLUTION"], "penalize": ["ORDERED"], "sigma": 900.0}]
        capped = HingePenalty(db, names, cells, "W")
        no_vacancies = HingePenalty(db, names, cells, "W", HingeOptions(vacancy_cap=0.0))
        assert capped(theta)[0] != no_vacancies(theta)[0]
        rows = no_vacancies._clouds["ORDERED"][0]
        vacancy_columns = [
            i for i, site in enumerate(no_vacancies._phases["ORDERED"].site_fractions) if "VA" in str(site)
        ]
        assert np.all(rows[:, vacancy_columns] == 0.0)

    def test_matches_reference_on_open_databases(self):
        """Test against values frozen from the reference implementation this module was ported from."""
        db = load_tdb(EXAMPLE / "pt-w.tdb")
        names = database_symbols_to_fit(db)
        theta = np.array([float(unpack_piecewise(db.symbols[name])) for name in names])
        sigma = {"FCC_A1": 500.0, "BCC_A2": 700.0, "LIQUID": 900.0}
        hinge = HingePenalty(
            db,
            names,
            [
                {
                    "T": 1000.0,
                    "x": 0.1,
                    "expected": ["FCC_A1"],
                    "penalize": ["BCC_A2", "LIQUID"],
                    "confidence": 0.8,
                    "sigma": sigma,
                },
                {
                    "T": 1500.0,
                    "x": 0.5,
                    "expected": ["BCC_A2", "FCC_A1"],
                    "tie_line": {"FCC_A1": 0.45, "BCC_A2": 0.9},
                    "penalize": ["LIQUID"],
                    "confidence": 0.5,
                    "sigma": sigma,
                },
            ],
            "W",
            HingeOptions(margin=60000.0),
        )
        penalty, gradient = hinge(theta)
        assert penalty == pytest.approx(4928.7924268409415, rel=1e-6)
        assert gradient[:4] == pytest.approx(
            [-29.333382443005146, -36.32049211660082, -81.1950797130943, -80.88454150537893], rel=1e-6
        )
        assert gradient[-2:] == pytest.approx([25.212502698755117, 11.851907230581954], rel=1e-6)

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"T": -1.0}, "temperature"),
            ({"x": 1.5}, "x must be"),
            ({"expected": ["SOLUTION", "SPURIOUS", "ORDERED"], "penalize": []}, "one or two"),
            ({"expected": ["MISSING"]}, "not in the database"),
            ({"penalize": ["SOLUTION"]}, "both expected"),
            ({"sigma": 0.0}, "positive"),
            ({"sigma": {"ORDERED": 1.0}}, "no value"),
            ({"confidence": -1.0}, "non-negative"),
            ({"tie_line": {"SOLUTION": 0.5}}, "only for two"),
            ({"expected": ["SOLUTION", "ORDERED"]}, "need tie_line"),
            ({"expected": ["SOLUTION", "ORDERED"], "tie_line": {"SOLUTION": 0.4, "ORDERED": 0.4}}, "coincide"),
            (
                {
                    "expected": ["SOLUTION", "SPURIOUS"],
                    "tie_line": {"SOLUTION": 0.4, "SPURIOUS": 0.7},
                    "penalize": ["ORDERED"],
                },
                "formula",
            ),
            ({"expected": ["SPURIOUS"], "penalize": ["SOLUTION"]}, "stoichiometric"),
            ({"typo": 1.0}, "unknown keys"),
        ],
    )
    def test_rejects_invalid_cells(self, change, match):
        """Test that invalid cells fail with a clear message."""
        db, names, _ = setup_synthetic()
        with pytest.raises(ValueError, match=match):
            HingePenalty(db, names, [{**SPURIOUS_CELL, **change}], "W")

    def test_rejects_invalid_setup(self):
        """Test that an unknown element, no cells and a wrong parameter count fail clearly."""
        db, names, theta = setup_synthetic()
        with pytest.raises(ValueError, match="not one of"):
            HingePenalty(db, names, [SPURIOUS_CELL], "AL")
        with pytest.raises(ValueError, match="no cells"):
            HingePenalty(db, names, [], "W")
        with pytest.raises(ValueError, match="theta"):
            HingePenalty(db, names, [SPURIOUS_CELL], "W")(theta[:-1])
        with pytest.raises(ValueError, match="not in the database"):
            HingePenalty(db, ["MISSING"], [SPURIOUS_CELL], "W")
        with pytest.raises(ValueError, match="margin"):
            HingeOptions(margin=-1.0)

    def test_rejects_more_than_two_elements(self):
        """Test that a ternary database is rejected."""
        db = Database("ELEMENT AL FCC_A1 1 1 1 !\nELEMENT CR BCC_A2 1 1 1 !\nELEMENT NI FCC_A1 1 1 1 !\n")
        with pytest.raises(ValueError, match="binary"):
            HingePenalty(db, [], [SPURIOUS_CELL], "AL")

    def test_rejects_phase_without_composition_freedom(self):
        """Test that a phase whose sublattices cannot reach a fixed composition is rejected."""
        tdb = "ELEMENT VA VACUUM 0 0 0 !\nELEMENT PT FCC_A1 1 1 1 !\nELEMENT W BCC_A2 1 1 1 !\nPHASE P % 2 1 1 !\nCONSTITUENT P :PT:VA,W: !\nPHASE Q % 1 1 !\nCONSTITUENT Q :PT,W: !\n"
        with pytest.raises(NotImplementedError, match="no sublattice"):
            HingePenalty(Database(tdb), [], [{"T": 1000.0, "x": 0.2, "expected": ["P"], "sigma": 1.0}], "PT")


class TestHook:
    """Tests of the penalty in the objective and in PhaseForgePlus."""

    def test_off_by_default(self, monkeypatch):
        """Test that without cells there is no penalty and optimize passes None to the objective."""
        passed = {}

        def fake_objective(db, db_zpf, db_neq, hinge=None):
            passed["hinge"] = hinge
            return lambda params: (0.0, np.zeros(len(params)))

        monkeypatch.setattr(main, "objective_function", fake_objective)
        for kwargs in ({}, {"hinge_cells": None}, {"hinge_cells": []}):
            pfp = PhaseForgePlus(EXAMPLE / "pt-w.tdb", EXAMPLE, [], 101325, 298.15, **kwargs)
            assert pfp.hinge is None
            pfp.optimize()
            assert passed["hinge"] is None

    def test_cells_are_passed_to_the_objective(self, monkeypatch):
        """Test that PhaseForgePlus builds the penalty from the cells and optimize uses it."""
        passed = {}

        def fake_objective(db, db_zpf, db_neq, hinge=None):
            passed["hinge"] = hinge
            return lambda params: (0.0, np.zeros(len(params)))

        monkeypatch.setattr(main, "objective_function", fake_objective)
        cells = [{"T": 1000.0, "x": 0.1, "expected": ["FCC_A1"], "penalize": ["BCC_A2"], "sigma": 500.0}]
        pfp = PhaseForgePlus(EXAMPLE / "pt-w.tdb", EXAMPLE, [], 101325, 298.15, hinge_cells=cells, hinge_element="W")
        pfp.optimize()
        assert passed["hinge"] is pfp.hinge
        assert pfp.hinge.symbols == database_symbols_to_fit(pfp.db)
        assert pfp.get_initial_values().shape == (len(pfp.hinge.symbols),)

    def test_objective_is_unchanged_and_penalty_adds(self, monkeypatch):
        """Test the objective against its plain formula, without and with the penalty.

        The residuals are replaced, because the ESPEI of uv.lock removes the fit parameters from the database and leaves
        process-wide state that breaks later tests.
        """

        class FakeResidual:
            """A residual with a fixed likelihood and gradient, scaled by the parameters."""

            scale = 1.0

            def __init__(self, db, data, phase_models, symbols_to_fit):
                self.n = len(symbols_to_fit)

            def get_likelihood(self, params):
                return -self.scale * float(params.sum()), -self.scale * np.arange(1.0, self.n + 1)

        class FakeNeqResidual(FakeResidual):
            """The second residual."""

            scale = 0.5

        monkeypatch.setattr(objective, "ZPFResidual", FakeResidual)
        monkeypatch.setattr(objective, "FixedConfigurationPropertyResidual", FakeNeqResidual)
        db, names, theta = setup_synthetic()
        hinge = HingePenalty(db, names, [SPURIOUS_CELL], "W")

        plain_value = 1.5 * float(theta.sum())
        plain_gradient = 1.5 * np.arange(1.0, len(names) + 1)
        for func in (objective.objective_function(db, None, None), objective.objective_function(db, None, None, None)):
            value, gradient = func(theta)
            assert value == plain_value
            assert np.array_equal(gradient, plain_gradient)

        penalty, penalty_gradient = hinge(theta)
        value, gradient = objective.objective_function(db, None, None, hinge)(theta)
        assert penalty > 0.0
        assert value == plain_value + penalty
        assert np.array_equal(gradient, plain_gradient + penalty_gradient)
