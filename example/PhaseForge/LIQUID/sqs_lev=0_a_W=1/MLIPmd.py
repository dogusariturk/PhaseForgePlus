import numpy as np
from pymatgen.core import Structure

from materialsframework.calculators import GraceCalculator


structure = Structure.from_file("POSCAR")

calculator = GraceCalculator(
    model="GRACE-2L-OMAT",
    ensemble="nvt_nose_hoover",
    timestep=1.0,
    temperature=3745.00000000000000000000,
    logfile='-',
)

res = calculator.run(structure=structure, steps=2000)

res["final_structure"].to("CONTCAR", fmt="poscar")
np.savetxt('total_energy_nvt.txt', res["total_energy"])
np.savetxt('energy', [np.mean(res["total_energy"])], fmt="%.8f")
