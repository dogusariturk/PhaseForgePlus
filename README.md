<div align="center">

# PhaseForge+

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)

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

### Python Workflow Usage

Here's a minimal example of optimizing a CALPHAD model using `PhaseForgePlus`:

```python
from phaseforgeplus import PhaseForgePlus

pfp = PhaseForgePlus(
    db="./data/pt-w.tdb",  # Path to your thermodynamic database
    zpf_path="./data",  # Path to Zero-Point Free Energy data
    points=[1801, 1601, 1401, 1201, 1001, 802, 602, 402, 202],  # Points for optimization
    pressure=101325,  # Pressure in Pa
    temperature=298.15,  # Temperature in K
)

pfp.optimize()
```
