"""Regression tests for fixed-composition refinement and public target diagnostics."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pycalphad import Database
from pycalphad.core.errors import EquilibriumError

from phaseforgeplus.utils.hinge import HingeOptions, HingePenalty

ENERGY_TOL = 0.1
COMPOSITION_TOL = 1e-12
DATA = Path(__file__).parent / "data"
CONTROL = {"T": 1468.0, "x": 0.705, "expected": ["ALTI3_D019"], "penalize": ["LIQUID"], "sigma": 100.0}


def control(options=None):
    """Construct the published D0_19 antisite regression control."""
    return HingePenalty(Database(str(DATA / "AlTi-08Wit.tdb")), [], [CONTROL], "TI", options)


def test_antisite_control_matches_pycalphad():
    """Default refinement removes both coarse-grid errors without changing overall composition."""
    refined = control()
    report = refined.check_targets(np.zeros(0))
    assert report["passed"]
    assert not report["splits"]
    assert len(report["endpoints"]) == 2
    for row in report["endpoints"]:
        assert abs(row["difference"]) < ENERGY_TOL
        assert row["refinement"]["improvement"] > 100
        assert row["refinement"]["composition_error"] < COMPOSITION_TOL
        assert row["refinement"]["converged"]
        assert row["tolerance"] == 10
    coarse = control(HingeOptions(refine_targets=False)).check_targets(np.zeros(0))
    assert not coarse["passed"]
    assert len(coarse["failures"]) == 2
    assert all(row["difference"] > 100 for row in coarse["failures"])


def test_unaffected_targets_and_objective_are_byte_identical():
    """Single mixing sublattices and a line compound bypass refinement exactly."""
    db = Database(str(DATA / "hinge_synthetic.tdb"))
    cell = {
        "T": 1000,
        "x": 0.3,
        "expected": ["SOLUTION", "SPURIOUS"],
        "tie_line": {"SOLUTION": 0.2, "SPURIOUS": 0.5},
        "penalize": ["ORDERED"],
        "sigma": 100,
    }
    old = HingePenalty(db, [], [cell], "W", HingeOptions(refine_targets=False))
    new = HingePenalty(db, [], [cell], "W")
    for phase, x in new._targets[0]:
        assert (
            np.float64(old._minimum_at(phase, x, 1000, np.zeros(0))[0]).tobytes()
            == np.float64(new._minimum_at(phase, x, 1000, np.zeros(0))[0]).tobytes()
        )
    assert old(np.zeros(0))[0] == new(np.zeros(0))[0]
    assert not new.refinement_records


def test_refinement_failure_is_retained_and_reported():
    """An iteration limit cannot silently turn a grid result into a converged target."""
    hinge = control(HingeOptions(refinement_maxiter=1))
    with pytest.raises(RuntimeError, match="did not converge"):
        hinge(np.zeros(0))
    assert any(not record["converged"] for record in hinge.refinement_records.values())
    report = hinge.check_targets(np.zeros(0))
    assert not report["passed"]
    assert len(report["failures"]) == 2
    assert report["excluded_search"]["errors"]
    assert all(not row["refinement"]["converged"] for row in report["failures"])


def test_split_is_separate_and_parameter_values_reach_reference():
    """A positive regular-solution interaction produces a convexified reference, not a target failure."""
    db = Database(str(DATA / "hinge_synthetic.tdb"))
    cell = {"T": 800, "x": 0.5, "expected": ["SOLUTION"], "penalize": ["SPURIOUS"], "sigma": {"SPURIOUS": 20}}
    hinge = HingePenalty(db, ["VV0000"], [cell], "W")
    report = hinge.check_targets(np.array([30000.0]))
    assert report["passed"]
    assert not report["failures"]
    assert len(report["splits"]) == 2
    assert all(row["difference"] > 100 for row in report["splits"])
    assert all(row["tolerance"] == 2 for row in report["splits"])
    assert not hinge.check_targets(np.array([-2000.0]))["splits"]


def test_cloud_comparison_is_independent_of_target_refinement():
    """Doubling a coarse excluded-phase cloud detects a driving-force and verdict change."""
    db = Database("""
ELEMENT PT FCC_A1 1 1 1 !
ELEMENT W BCC_A2 1 1 1 !
TYPE_DEFINITION % SEQ * !
PHASE SOLUTION % 1 1 !
CONSTITUENT SOLUTION :PT,W: !
PHASE ORDERED % 2 1 1 !
CONSTITUENT ORDERED :PT,W:PT,W: !
PARAMETER G(ORDERED,PT:PT;0) 298.15 100000; 6000 N !
PARAMETER G(ORDERED,PT:W;0) 298.15 100000; 6000 N !
PARAMETER G(ORDERED,W:PT;0) 298.15 100000; 6000 N !
PARAMETER G(ORDERED,W:W;0) 298.15 100000; 6000 N !
""")
    cell = {"T": 1000, "x": 0.5, "expected": ["SOLUTION"], "penalize": ["ORDERED"], "sigma": 100}
    hinge = HingePenalty(db, [], [cell], "W", HingeOptions(cloud_points_multiple=3))
    report = hinge.check_targets(np.zeros(0))
    comparison = report["excluded_search"]["comparisons"][0]
    assert abs(comparison["change"]) > 1
    margin = (comparison["before"] + comparison["after"]) / 2
    hinge = HingePenalty(db, [], [cell], "W", replace(hinge.options, margin=margin))
    before = hinge(np.zeros(0))[0]
    report = hinge.check_targets(np.zeros(0))
    assert len(report["excluded_search"]["verdict_changes"]) == 1
    assert report["excluded_search"]["cloud_points_multiple"] == [3, 6]
    assert hinge(np.zeros(0))[0] == before
    assert hinge.options.cloud_points_multiple == 3


@pytest.mark.parametrize("bad", [float("nan"), -1000.0, 1000.0])
def test_bad_reference_is_not_silently_accepted(monkeypatch, bad):
    """Both signs of energy disagreement and nonfinite references fail the diagnostic."""
    hinge = control()

    def reference(phase, temperature, x, theta, pdens):
        return {"reference_energy": hinge._minimum_at(phase, x, temperature, theta)[0] + bad, "split": False}

    monkeypatch.setattr(hinge, "_reference_target", reference)
    report = hinge.check_targets(np.zeros(0))
    assert not report["passed"]
    assert len(report["failures"]) == 2


@pytest.mark.parametrize(
    "options",
    [{"refinement_tol": 0}, {"refinement_tol": float("nan")}, {"refinement_starts": 0}, {"refinement_maxiter": 1.5}],
)
def test_invalid_refinement_options(options):
    """Invalid numerical settings fail at construction."""
    with pytest.raises(ValueError):
        HingeOptions(**options)


def test_nearby_compositions_do_not_share_a_refined_target():
    """Refinement retains composition differences below the legacy cache rounding."""
    cells = [CONTROL, {**CONTROL, "x": CONTROL["x"] + 2e-7}]
    hinge = HingePenalty(Database(str(DATA / "AlTi-08Wit.tdb")), [], cells, "TI")
    hinge(np.zeros(0))
    assert len(hinge.refinement_records) == 4
    for record in hinge.refinement_records.values():
        assert record["composition_error"] < COMPOSITION_TOL


def test_reference_solver_error_does_not_skip_endpoints(monkeypatch):
    """A pycalphad solver error is reported for both endpoints instead of aborting the diagnostic."""
    hinge = control()

    def fail_reference(*args):
        raise EquilibriumError("reference solver failed")

    monkeypatch.setattr(hinge, "_reference_target", fail_reference)
    report = hinge.check_targets(np.zeros(0))
    assert len(report["endpoints"]) == len(report["failures"]) == 2
    assert all(row["error"] == "reference solver failed" for row in report["failures"])


@pytest.mark.parametrize("cap", [0.0, 0.4])
def test_vacancies_relax_within_bounds_and_cache_tracks_parameters(cap):
    """Vacancies on the solved sublattice can relax, and a new fit vector recomputes the target."""
    db = Database(str(DATA / "hinge_synthetic.tdb"))
    cell = {"T": 800, "x": 0.5, "expected": ["ORDERED"], "penalize": ["SPURIOUS"], "sigma": 900}
    hinge = HingePenalty(db, ["VV0002"], [cell], "W", HingeOptions(vacancy_cap=cap))
    first = hinge(np.array([-5000.0]))
    original_records = dict(hinge.refinement_records)
    second = hinge(np.array([-4000.0]))
    assert second[0] != first[0]
    assert original_records != hinge.refinement_records
    for record in hinge.refinement_records.values():
        fractions = record["site_fractions"]
        assert -COMPOSITION_TOL <= fractions[1] <= cap + COMPOSITION_TOL
        assert -COMPOSITION_TOL <= fractions[4] <= cap + COMPOSITION_TOL
        if cap:
            assert fractions[1] > 0
            assert fractions[4] > 0
    assert hinge(np.array([-5000.0]))[0] == first[0]
