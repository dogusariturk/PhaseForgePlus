"""Example of using PhaseForgePlus for a phase diagram optimization."""

from phaseforgeplus import PhaseForgePlus

pfp = PhaseForgePlus(
    db="./data/pt-w.tdb",
    zpf_path="./data",
    points=[1801, 1601, 1401, 1201, 1001, 802, 602, 402, 202],
    pressure=101325,
    temperature=298.15,
)

pfp.optimize()
