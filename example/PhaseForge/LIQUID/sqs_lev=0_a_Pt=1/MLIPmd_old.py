import materialsframework
import numpy as np
import pickle
from pymatgen.core import Structure
from ase import Atoms
from materialsframework.calculators import GraceCalculator
structure = Structure.from_file("POSCAR")
calculator = GraceCalculator(
    model="GRACE-2L-OMAT",
    ensemble="nvt_nose_hoover",
    timestep=1.0,
    temperature=2091.4,
    logfile='-',
)
res = calculator.run(structure=structure, steps=2000)
total_energies = res["total_energy"]
with open('total_energy_nvt.txt', 'w') as f:
    for item in total_energies:
        f.write(f"{item}\\n")
average = np.mean(total_energies[-2000:])
with open("energy", "w") as f:
    f.write(f"{average}\\n")
