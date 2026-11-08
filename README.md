<div align="center">

# PhaseForge+

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
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

The PyPI release is out of date until the next pycalphad release, so for now install PhaseForgePlus from GitHub with [uv](https://docs.astral.sh/uv/).

To start a new project:

```sh
uv init --python 3.13 my-project
cd my-project
uv add git+https://github.com/dogusariturk/PhaseForgePlus.git
```

Keep `--python` on `uv init`. Without it, uv may pick Python 3.14, which PhaseForgePlus doesn't support yet.

To install into an environment you already have:

```sh
uv pip install git+https://github.com/dogusariturk/PhaseForgePlus.git
```

To work on PhaseForgePlus itself, clone the repository and run `uv sync`:

```sh
git clone https://github.com/dogusariturk/PhaseForgePlus.git
cd PhaseForgePlus
uv sync
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

By default, `optimize()` runs conjugate gradient (`method="CG"`) through `scipy.optimize.minimize`. You can pass a different method, options for that method, and starting points. Passing `options` replaces the default `{"disp": True}` rather than adding to it. Without `starts`, the run begins at the parameter values in the database.

```python
result = pfp.optimize(method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})

# Each start is a parameter vector in the order of database_symbols_to_fit and gets its own run. The returned result
# is the run with the lowest objective. result.start_index tells you which start it came from, and result.runs holds
# every run in the order of the starts.
start = pfp.get_initial_values()
result = pfp.optimize(options={"maxiter": 500, "gtol": 1e-3}, starts=[start, 1.1 * start])
```

---

## Hinge Penalty on Spurious Phases

The optional hinge penalty discourages parameters that make a phase stable where the evidence says it should not be. It is off unless you pass cells. Without cells, the objective is only the negative log-likelihood of the ZPF and non-equilibrium data.

Each cell lists the phases expected to be stable at a temperature `T` and mole fraction `x`, and their Gibbs energies set a target line. For one expected phase, the line is that phase's chord over `x ± chord_half_width`. For two, it passes through both phases at the tie-line ends. Every penalized phase (by default, every phase not in `expected`) gets a driving force `DG`, the lowest value of its Gibbs energy minus the target line. When `DG` falls below `margin`, that phase adds `0.5 * confidence * ((margin - DG) / sigma) ** 2` to the negative log-likelihood. The penalty also passes its exact gradient to the optimizer.

```python
from phaseforgeplus import PhaseForgePlus
from phaseforgeplus.utils import HingeOptions

cells = [
    # One expected phase. sigma is in J per mole of atoms.
    {"T": 1000.0, "x": 0.1, "expected": ["FCC_A1"], "sigma": 500.0},
    # Two expected phases also need the tie-line ends. confidence defaults to 1 and penalize to every other phase.
    # sigma can also be given per penalized phase.
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

* The penalty handles binary systems only. Constituents must be pure neutral elements or vacancies, with at most three per sublattice. Anything else raises an error.
* Stoichiometric phases can be penalized and can sit at a tie-line end, but they cannot be the only expected phase in a cell.
* The default `penalize` list cannot tell that a phase is another composition set of an expected phase, so it penalizes that set too. Give `penalize` explicitly to leave such phases out.
* Each minimum is found on a fixed grid of configurations (see `HingeOptions`), with vacancy site fractions capped at 0.4 by default. The gradient is exact for this discretized function, which has kinks where the minimizing configuration changes and where `DG` crosses `margin`.

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

---

## License

This project is licensed under the GNU GPLv3 License. See the [LICENSE](https://github.com/dogusariturk/PhaseForgePlus/blob/main/LICENSE) file for details.
