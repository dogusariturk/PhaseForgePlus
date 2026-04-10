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
