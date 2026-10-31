<div align="center">

# PhaseForge+

![Python](https://img.shields.io/badge/python-3.11-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)
[![Test](https://github.com/dogusariturk/PhaseForgePlus/actions/workflows/tests.yml/badge.svg)](https://github.com/dogusariturk/PhaseForgePlus/actions/workflows/tests.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17519087.svg)](https://doi.org/10.5281/zenodo.17519087)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11669--025--01222--2-blue)](https://doi.org/10.1007/s11669-025-01222-2)

`PhaseForgePlus` is a Python-based, fully open-source workflow for generating and tuning physically-informed CALPHAD models. It integrates Machine-Learned Interatomic Potentials (MLIPs), the Alloy Theoretic Automated Toolkit (ATAT), and experimental data to efficiently produce accurate phase diagrams.

<p>
  <a href="https://github.com/dogusariturk/PhaseForgePlus/issues/new?labels=bug">Report a Bug</a> |
  <a href="https://github.com/dogusariturk/PhaseForgePlus/issues/new?labels=enhancement">Request a Feature</a>
</p>

</div>

---

## Features

* Automatic construction of CALPHAD models from MLIP-based thermodynamic data
* Integration with ATAT and PyCalphad for Gibbs energy and phase diagram calculations
* Efficient parameter fitting using the Jansson derivative method with gradient-based optimization
* Support for physically-grounded adjustments using experimental phase equilibria
* Compatibility with ESPEI and PyCalphad toolchain for advanced thermodynamic modeling

---

## Relationship to PhaseForge

`PhaseForgePlus` builds upon the foundation of [`PhaseForge`](https://github.com/dogusariturk/PhaseForge), extending its capabilities for advanced CALPHAD model optimization. While `PhaseForge` enables users to automatically generate initial thermodynamic database (TDB) files from MLIPs, `PhaseForgePlus` takes the workflow a step further by providing robust tools for parameter fitting, integration with experimental data, and gradient-based optimization.

You can use `PhaseForge` to create an initial `.tdb` file, which serves as the starting point for further optimization in `PhaseForgePlus`.

By combining `PhaseForge` and `PhaseForgePlus`, you can seamlessly transition from automated TDB generation to advanced model optimization in a single, reproducible workflow.

---

## Installation

You can install `PhaseForgePlus` via pip:

```sh
pip install phaseforgeplus
```

---

## Quick Start

Here's a minimal example of optimizing a CALPHAD model using `PhaseForgePlus`:

```python
from phaseforgeplus import PhaseForgePlus

pfp = PhaseForgePlus(
    db="./data/pt-w.tdb",  # Path to your thermodynamic database
    zpf_path="./data",  # Path to Zero Phase Fraction (ZPF) data
    points=[1801, 1601, 1401, 1201, 1001, 802, 602, 402, 202],  # Points for optimization
    pressure=101325,  # Pressure in Pa
    temperature=298.15,  # Temperature in K
)

pfp.optimize()
```

---

## Optimizer Options

`optimize()` runs conjugate gradient with `scipy.optimize.minimize` by default. Another method, its options and starting points can be given. `options` replaces the default `{"disp": True}`.

```python
result = pfp.optimize(method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})

# One run from each start, in the order of database_symbols_to_fit. The result is the run with the lowest objective,
# with result.start_index and all results in result.runs.
start = pfp.get_initial_values()
result = pfp.optimize(options={"maxiter": 500, "gtol": 1e-3}, starts=[start, 1.1 * start])
```

---

## Hinge Penalty on Spurious Phases

An optional penalty keeps phases out of regions where the evidence says they are not stable. It is off unless cells are given, and then the objective is exactly the one above.

Each cell gives the phases expected at a temperature `T` and mole fraction `x`. Their Gibbs energies define a target line: the chord of one phase over `x ± chord_half_width`, or the line through two phases at the tie-line ends. Every other phase whose lowest energy over that line is below `margin` adds `0.5 * confidence * ((margin - DG) / sigma) ** 2` to the negative log-likelihood, with its exact gradient.

```python
from phaseforgeplus import PhaseForgePlus
from phaseforgeplus.utils.hinge import HingeOptions

cells = [
    # One expected phase. sigma is in J per mole of atoms.
    {"T": 1000.0, "x": 0.1, "expected": ["FCC_A1"], "sigma": 500.0},
    # Two expected phases need the tie-line ends. Optional: confidence (default 1), penalize (default all other phases),
    # sigma per penalized phase.
    {
        "T": 1500.0,
        "x": 0.5,
        "expected": ["FCC_A1", "BCC_A2"],
        "tie_line": {"FCC_A1": 0.45, "BCC_A2": 0.9},
        "confidence": 0.5,
        "penalize": ["LIQUID"],
        "sigma": {"LIQUID": 900.0},
    },
]

pfp = PhaseForgePlus(
    db="./data/pt-w.tdb",
    zpf_path="./data",
    points=[1801, 1601, 1401, 1201, 1001, 802, 602, 402, 202],
    pressure=101325,
    temperature=298.15,
    hinge_cells=cells,
    hinge_element="W",  # x is the mole fraction of this element
    hinge_options=HingeOptions(margin=50.0, chord_half_width=0.01),
)

pfp.optimize()
```

Limits:

* Binary systems only, with pure neutral elements and vacancies as constituents, and at most three constituents per sublattice. Other input raises an error.
* Stoichiometric phases can be penalized and can be a tie-line end, but cannot be the single expected phase of a cell. A cell's `penalize` should leave out other composition sets of an expected phase, which the default does not know about.
* Minima are taken over fixed grids of configurations (see `HingeOptions`), with vacancy fractions capped at 0.4 by default. The gradient is exact for this discretized function, which has kinks where a minimizer changes and where the driving force crosses the margin.

---

## Citation

If you use PhaseForgePlus in your research, please cite the following:

> Courtney Kunselman, Siya Zhu, Doğuhan Sarıtürk, Raymundo Arróyave. *Construction and Tuning of CALPHAD Models Using Machine-Learned Interatomic Potentials and Experimental Data: A Case Study of the Pt-W System*, Journal of Phase Equilibria and Diffusion, 2025. https://doi.org/10.1007/s11669-025-01222-2

> Sarıtürk, D., Kunselman, C., & Zhu, S. (2025). PhaseForgePlus (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.17519087

BibTeX:

```bibtex
@article{kunselman2025construction,
  author       = {Kunselman, Courtney and Zhu, Siya and Sarıtürk, Doğuhan and Arróyave, Raymundo},
  title        = {Construction and Tuning of {CALPHAD} Models Using Machine-Learned Interatomic Potentials and Experimental Data: A Case Study of the {Pt-W} System},
  journal      = {Journal of Phase Equilibria and Diffusion},
  year         = 2025,
  doi          = {10.1007/s11669-025-01222-2},
  url          = {https://doi.org/10.1007/s11669-025-01222-2},
}

@software{sariturk_2025_17519087,
  author       = {Sarıtürk, Doğuhan and Kunselman, Courtney and Zhu, Siya},
  title        = {PhaseForgePlus},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17519087},
  url          = {https://doi.org/10.5281/zenodo.17519087},
}
```
