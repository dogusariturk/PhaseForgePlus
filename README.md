<div align="center">

# PhaseForge+

![Python](https://img.shields.io/badge/python-3.11-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)
[![Test](https://github.com/dogusariturk/PhaseForgePlus/actions/workflows/tests.yml/badge.svg)](https://github.com/dogusariturk/PhaseForgePlus/actions/workflows/tests.yml)

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

You can install `PhaseForgePlus` via pip (once available on PyPI):

```sh
pip install phaseforgeplus
```

Or clone the repository directly:

```sh
pip install git+https://github.com/dogusariturk/PhaseForgePlus.git
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
