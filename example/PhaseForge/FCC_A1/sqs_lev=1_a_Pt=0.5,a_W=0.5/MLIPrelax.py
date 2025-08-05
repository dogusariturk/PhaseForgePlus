import numpy as np
from pymatgen.core import Structure

from materialsframework.calculators import GraceCalculator


struct = Structure.from_file("POSCAR")

calculator = GraceCalculator(model='GRACE-2L-OMAT', fmax=0.001, verbose=True)
calculate_results = calculator.relax(struct)

calculate_results["final_structure"].to("CONTCAR", fmt="poscar")
np.savetxt('energy', [calculate_results["energy"]])
np.savetxt('stress_temp.out', calculate_results["stress"])
np.savetxt('force.out', calculate_results["forces"])
