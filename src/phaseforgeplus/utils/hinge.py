"""One-sided hinge penalty on phases that should not be stable in a binary system.

Each cell states which phases are expected to be stable at a temperature and composition. The expected phases define
a target line ``H(x)``:

* one expected phase: the chord of its minimum Gibbs energy over ``[x - h, x + h]``;
* two expected phases: the line through the minimum Gibbs energies of both phases at the tie-line ends.

For every other phase ``psi`` the driving force over this line is ``DG = min_y [G_psi(y) - H(x(y))]`` over a fixed set
of configurations ``y``, and the penalty is ``P = 0.5 * sum(conf * (max(0, margin - DG) / sigma) ** 2)``.

Energies are in J per mole of atoms and ``P`` is in negative log-likelihood units. As the configuration sets are
fixed, the gradient with respect to the fit parameters follows from the envelope theorem at the minimizing
configurations. It is the exact derivative of the discretized function, which has kinks where a minimizer changes
and where ``DG`` crosses ``margin``. Those two operations are ``_minimum`` and ``_hinge_term``.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pycalphad import Database, Model, variables
from symengine import Lambdify, Symbol

__all__ = ["HingeCell", "HingeOptions", "HingePenalty"]

_VACANCY = "VA"
_MAX_CONSTITUENTS = 3
_MAX_EXPECTED = 2
_BISECTION_STEPS = 50
_EPS = 1e-9
_X_MIN = 1e-4
_MAX_CHORD_HALF_WIDTH = 0.5
_STOICHIOMETRIC_TOLERANCE = 1e-3


@dataclass(frozen=True)
class HingeOptions:
    """Settings of the hinge penalty.

    Attributes:
        margin (float): Required distance above the target line, in J/mol of atoms.
        chord_half_width (float): Half width ``h`` in mole fraction of the chord for a single expected phase.
        vacancy_cap (float): Largest vacancy site fraction in the configuration sets.
        cloud_points_single (int): Grid points on a two-constituent sublattice when one sublattice mixes.
        cloud_points_multiple (int): Grid points on a two-constituent sublattice when several sublattices mix.
        cloud_step_three (float): Site-fraction step on a three-constituent sublattice.
        fixed_x_step (float): Site-fraction step on the free sublattices at fixed composition.
    """

    margin: float = 50.0
    chord_half_width: float = 0.01
    vacancy_cap: float = 0.4
    cloud_points_single: int = 41
    cloud_points_multiple: int = 31
    cloud_step_three: float = 0.1
    fixed_x_step: float = 0.05

    def __post_init__(self) -> None:
        """Check the settings."""
        if not (math.isfinite(self.margin) and self.margin >= 0):
            raise ValueError(f"margin must be finite and non-negative, got {self.margin}")
        if not 0 < self.chord_half_width < _MAX_CHORD_HALF_WIDTH:
            raise ValueError(f"chord_half_width must be in (0, {_MAX_CHORD_HALF_WIDTH}), got {self.chord_half_width}")
        if not 0 <= self.vacancy_cap <= 1:
            raise ValueError(f"vacancy_cap must be in [0, 1], got {self.vacancy_cap}")
        if self.cloud_points_single < 2 or self.cloud_points_multiple < 2:
            raise ValueError("cloud_points_single and cloud_points_multiple must be at least 2")
        if not (0 < self.cloud_step_three <= 1 and 0 < self.fixed_x_step <= 1):
            raise ValueError("cloud_step_three and fixed_x_step must be in (0, 1]")


@dataclass(frozen=True)
class HingeCell:
    """A temperature and composition with its expected stable phases.

    Attributes:
        temperature (float): Temperature in K.
        x (float): Mole fraction of the element named in ``HingePenalty``.
        expected (tuple[str, ...]): The one or two phases expected to be stable.
        sigma (float | Mapping[str, float]): Scale in J/mol of atoms, single or per penalized phase.
        tie_line (Mapping[str, float] | None): Mole fraction at the tie-line end of each phase, for two expected phases.
        confidence (float): Non-negative weight of the cell.
        penalize (tuple[str, ...] | None): Phases to penalize. All other phases of the database when None.
    """

    temperature: float
    x: float
    expected: tuple[str, ...]
    sigma: float | Mapping[str, float]
    tie_line: Mapping[str, float] | None = None
    confidence: float = 1.0
    penalize: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, cell: Mapping) -> "HingeCell":
        """Build a cell from a mapping with keys T, x, expected, sigma and optionally tie_line, confidence, penalize.

        Args:
            cell (Mapping): The cell description.

        Returns:
            HingeCell: The cell.

        Raises:
            ValueError: If a key is missing or unknown.
        """
        required, optional = {"T", "x", "expected", "sigma"}, {"tie_line", "confidence", "penalize"}
        if missing := required - cell.keys():
            raise ValueError(f"cell {dict(cell)} is missing {sorted(missing)}")
        if unknown := cell.keys() - required - optional:
            raise ValueError(f"cell {dict(cell)} has unknown keys {sorted(unknown)}")
        penalize = cell.get("penalize")
        return cls(
            temperature=float(cell["T"]),
            x=float(cell["x"]),
            expected=tuple(cell["expected"]),
            sigma=cell["sigma"],
            tie_line=cell.get("tie_line"),
            confidence=float(cell.get("confidence", 1.0)),
            penalize=None if penalize is None else tuple(penalize),
        )


class _PhaseEnergy:
    """Gibbs energy per mole of atoms and its parameter gradient for configuration arrays of one phase."""

    def __init__(self, db: Database, components: list[str], phase: str, symbols: list[str], element: str) -> None:
        """Compile the energy of a phase and check that its constituents are supported."""
        self.phase = phase
        self.element = element
        self.constituents = [
            sorted(species.name for species in sublattice) for sublattice in db.phases[phase].constituents
        ]
        self.ratios = [float(ratio) for ratio in db.phases[phase].sublattices]
        if db.phases[phase].model_hints.get("ionic_liquid_2SL", False):
            raise NotImplementedError(f"{phase}: ionic liquid models are not supported")
        for sublattice in db.phases[phase].constituents:
            for species in sublattice:
                if species.name != _VACANCY and (len(species.constituents) != 1 or species.charge != 0):
                    raise NotImplementedError(f"{phase}: species {species.name} is not a pure neutral element")
        if any(len(names) > _MAX_CONSTITUENTS for names in self.constituents):
            raise NotImplementedError(f"{phase}: more than {_MAX_CONSTITUENTS} constituents per sublattice")
        model = Model(db, components, phase, parameters=list(symbols))
        self.site_fractions = list(model.site_fractions)
        if len(self.site_fractions) != sum(len(names) for names in self.constituents):
            raise ValueError(f"{phase}: constituents outside the components {components}")
        parameters = [Symbol(name) for name in symbols]
        self.function = Lambdify(
            [variables.T, *self.site_fractions, *parameters],
            [model.GM] + [model.GM.diff(parameter) for parameter in parameters],
        )

    @property
    def stoichiometric(self) -> bool:
        """Whether the phase has a single configuration."""
        return all(len(names) == 1 for names in self.constituents)

    def composition(self, configurations: np.ndarray) -> np.ndarray:
        """Return the mole fraction of the element for rows of site fractions."""
        offset, atoms_element, atoms = 0, np.zeros(len(configurations)), np.zeros(len(configurations))
        for names, ratio in zip(self.constituents, self.ratios, strict=True):
            for k, name in enumerate(names):
                if name != _VACANCY:
                    atoms += ratio * configurations[:, offset + k]
                    if name == self.element:
                        atoms_element += ratio * configurations[:, offset + k]
            offset += len(names)
        with np.errstate(invalid="ignore", divide="ignore"):
            return atoms_element / atoms

    @staticmethod
    def _simplex(names: list[str], step: float, vacancy_cap: float) -> np.ndarray:
        """Return site-fraction rows of one sublattice with a capped vacancy fraction."""
        if len(names) == 1:
            return np.ones((1, 1))
        n_steps = round(1 / step)
        if len(names) == 2:
            grid = np.linspace(0, 1, n_steps + 1)
            if _VACANCY in names:
                if names.index(_VACANCY) == 1:
                    grid = grid[grid <= vacancy_cap + _EPS]
                else:
                    grid = grid[grid >= 1 - vacancy_cap - _EPS]
            return np.column_stack([1 - grid, grid])
        vacancy = names.index(_VACANCY) if _VACANCY in names else None
        rows = []
        for a in range(n_steps + 1):
            for b in range(n_steps + 1 - a):
                row = np.array([a, b, n_steps - a - b], float) / n_steps
                if vacancy is None or row[vacancy] <= vacancy_cap + _EPS:
                    rows.append(row)
        return np.array(rows)

    def cloud(self, options: HingeOptions) -> tuple[np.ndarray, np.ndarray]:
        """Return the configurations for the minimization over all compositions, and their compositions."""
        n_mixing = sum(len(names) > 1 for names in self.constituents)
        points = options.cloud_points_single if n_mixing == 1 else options.cloud_points_multiple
        grids = [
            self._simplex(
                names, 1.0 / (points - 1) if len(names) == 2 else options.cloud_step_three, options.vacancy_cap
            )
            for names in self.constituents
        ]
        rows = grids[0]
        for grid in grids[1:]:
            rows = np.vstack([np.hstack([a, b]) for a in rows for b in grid])
        compositions = self.composition(rows)
        finite = np.isfinite(compositions)
        return rows[finite], compositions[finite]

    def _configuration(self, solved: int, fraction: float, others: np.ndarray) -> np.ndarray:
        """Return one configuration, with the element fraction set on the solved sublattice."""
        rows, position = [], 0
        for k, names in enumerate(self.constituents):
            if k == solved:
                row = np.zeros(len(names))
                row[names.index(self.element)] = fraction
                row[[i for i, name in enumerate(names) if name not in (self.element, _VACANCY)][0]] = 1 - fraction
            else:
                row = others[position : position + len(names)]
                position += len(names)
            rows.append(row)
        return np.concatenate(rows)

    def at_composition(self, x: float, options: HingeOptions) -> np.ndarray:
        """Return configurations with overall composition exactly x.

        The first mixing sublattice with the element and another element is solved for x by bisection, and the other
        sublattices are gridded.
        """
        if self.stoichiometric:
            return self.cloud(options)[0]
        solved = [
            k
            for k, names in enumerate(self.constituents)
            if self.element in names and len(names) > 1 and any(name not in (self.element, _VACANCY) for name in names)
        ]
        if not solved:
            raise NotImplementedError(f"{self.phase}: no sublattice mixes {self.element} with the other element")
        grids = [
            self._simplex(names, options.fixed_x_step, options.vacancy_cap)
            for k, names in enumerate(self.constituents)
            if k != solved[0]
        ]
        others = [np.zeros(0)]
        for grid in grids:
            others = [np.r_[a, b] for a in others for b in grid]
        rows = []
        for other in others:

            def excess(fraction: float, other: np.ndarray = other) -> float:
                configuration = self._configuration(solved[0], fraction, other)
                return self.composition(configuration[None])[0] - x

            low, high = 0.0, 1.0
            excess_low, excess_high = excess(low), excess(high)
            if not (np.isfinite(excess_low) and np.isfinite(excess_high)) or (excess_low > 0) == (excess_high > 0):
                continue
            for _ in range(_BISECTION_STEPS):
                middle = 0.5 * (low + high)
                if (excess(middle) > 0) == (excess_low > 0):
                    low = middle
                else:
                    high = middle
            rows.append(self._configuration(solved[0], 0.5 * (low + high), other))
        if not rows:
            raise ValueError(f"{self.phase}: no configuration reaches x = {x}")
        return np.array(rows)

    def evaluate(
        self, temperature: float, configurations: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the energies and their parameter gradients for rows of site fractions."""
        n_rows = len(configurations)
        inputs = np.column_stack([np.full(n_rows, temperature), configurations, np.tile(theta, (n_rows, 1))])
        result = np.asarray(self.function(inputs)).reshape(n_rows, -1)
        return result[:, 0], result[:, 1:]


def _minimum(values: np.ndarray) -> int:
    """Return the index of the minimizing configuration."""
    return int(np.argmin(values))


def _hinge_term(driving_force: float, margin: float, sigma: float, confidence: float) -> tuple[float, float]:
    """Return the penalty of one cell and phase and its derivative with respect to the driving force."""
    if driving_force >= margin:
        return 0.0, 0.0
    scaled = (margin - driving_force) / sigma
    return 0.5 * confidence * scaled * scaled, -confidence * scaled / sigma


class HingePenalty:
    """Penalty and exact parameter gradient of phases that should not be stable in the given cells.

    Only binary systems are supported, see the module docstring for the definition.

    Attributes:
        symbols (list[str]): The fit parameters, in the order of ``theta``.
        cells (list[HingeCell]): The validated cells.
        options (HingeOptions): The settings.
    """

    def __init__(
        self,
        db: Database,
        symbols: Sequence[str],
        cells: Sequence[HingeCell | Mapping],
        element: str,
        options: HingeOptions | None = None,
    ) -> None:
        """Validate the cells and prepare the configuration sets.

        Args:
            db (Database): The database with the fit parameters as symbols.
            symbols (Sequence[str]): The fit parameters, in the order of ``theta``.
            cells (Sequence[HingeCell | Mapping]): The cells, see ``HingeCell.from_mapping``.
            element (str): The element whose mole fraction is ``x``.
            options (HingeOptions | None): The settings. Defaults are used when None.

        Raises:
            ValueError: If a cell is invalid, a symbol is missing, or the database is not a binary system with the element.
            NotImplementedError: If a phase uses a model that is not supported.
        """
        elements = sorted(str(name) for name in db.elements if str(name) not in ("/-", _VACANCY))
        if len(elements) != 2:
            raise ValueError(f"the hinge penalty supports binary systems only, the database has elements {elements}")
        if element not in elements:
            raise ValueError(f"element {element} is not one of the database elements {elements}")
        if not cells:
            raise ValueError("no cells given")
        if missing := [name for name in symbols if name not in db.symbols]:
            raise ValueError(
                f"symbols {missing} are not in the database. Older ESPEI residuals remove them, build the penalty first"
            )
        self.symbols = list(symbols)
        self.options = options or HingeOptions()
        self.element = element
        self.cells = [cell if isinstance(cell, HingeCell) else HingeCell.from_mapping(cell) for cell in cells]
        self.penalized = [self._validate(cell, db) for cell in self.cells]
        components = [*elements, _VACANCY]
        names = sorted(
            {name for cell, pen in zip(self.cells, self.penalized, strict=True) for name in (*cell.expected, *pen)}
        )
        self._phases = {name: _PhaseEnergy(db, components, name, self.symbols, element) for name in names}
        self._clouds = {name: phase.cloud(self.options) for name, phase in self._phases.items()}
        self._fixed = {}
        self._targets = [self._prepare_target(cell) for cell in self.cells]

    def _validate(self, cell: HingeCell, db: Database) -> tuple[str, ...]:
        """Check one cell and return the phases to penalize in it."""
        where = f"cell at T={cell.temperature}, x={cell.x}"
        if not (math.isfinite(cell.temperature) and cell.temperature > 0):
            raise ValueError(f"{where}: temperature must be positive")
        if not 0 <= cell.x <= 1:
            raise ValueError(f"{where}: x must be in [0, 1]")
        if not (math.isfinite(cell.confidence) and cell.confidence >= 0):
            raise ValueError(f"{where}: confidence must be non-negative")
        if not 1 <= len(cell.expected) <= _MAX_EXPECTED or len(set(cell.expected)) != len(cell.expected):
            raise ValueError(f"{where}: expected must be one or two different phases, got {cell.expected}")
        penalize = cell.penalize
        if penalize is None:
            penalize = tuple(sorted(name for name in db.phases if name not in cell.expected))
        if unknown := [name for name in (*cell.expected, *penalize) if name not in db.phases]:
            raise ValueError(f"{where}: phases {unknown} are not in the database")
        if overlap := set(penalize) & set(cell.expected):
            raise ValueError(f"{where}: phases {sorted(overlap)} are both expected and penalized")
        self._validate_tie_line(cell, where)
        for name in penalize:
            if isinstance(cell.sigma, Mapping) and name not in cell.sigma:
                raise ValueError(f"{where}: sigma has no value for phase {name}")
            sigma = self._sigma(cell, name)
            if not (math.isfinite(sigma) and sigma > 0):
                raise ValueError(f"{where}: sigma for phase {name} must be positive")
        return penalize

    @staticmethod
    def _validate_tie_line(cell: HingeCell, where: str) -> None:
        """Check that the tie line of a cell matches its expected phases."""
        if len(cell.expected) == 1:
            if cell.tie_line is not None:
                raise ValueError(f"{where}: tie_line is only for two expected phases")
            return
        if cell.tie_line is None or set(cell.tie_line) != set(cell.expected):
            raise ValueError(f"{where}: two expected phases need tie_line with an end for each phase")
        ends = [float(cell.tie_line[name]) for name in cell.expected]
        if not all(0 <= end <= 1 for end in ends):
            raise ValueError(f"{where}: tie-line ends must be in [0, 1]")
        if abs(ends[1] - ends[0]) < _EPS:
            raise ValueError(f"{where}: tie-line ends coincide")

    @staticmethod
    def _sigma(cell: HingeCell, phase: str) -> float:
        """Return the scale of a phase in a cell."""
        return float(cell.sigma[phase] if isinstance(cell.sigma, Mapping) else cell.sigma)

    def _prepare_target(self, cell: HingeCell) -> tuple[tuple[str, float], tuple[str, float]]:
        """Return the two (phase, composition) points of the target line of a cell, building their configurations."""
        if len(cell.expected) == 1:
            phase = cell.expected[0]
            if self._phases[phase].stoichiometric:
                raise ValueError(
                    f"cell at T={cell.temperature}, x={cell.x}: a stoichiometric phase alone gives no chord"
                )
            half_width = self.options.chord_half_width
            points = [(phase, max(cell.x - half_width, _X_MIN)), (phase, min(cell.x + half_width, 1 - _X_MIN))]
        else:
            points = sorted(((name, float(cell.tie_line[name])) for name in cell.expected), key=lambda point: point[1])
            points = [(name, self._stoichiometric_x(cell, name, end)) for name, end in points]
        for name, x in points:
            key = (name, round(x, 6))
            if key not in self._fixed:
                self._fixed[key] = self._phases[name].at_composition(x, self.options)
        return points[0], points[1]

    def _stoichiometric_x(self, cell: HingeCell, phase: str, end: float) -> float:
        """Return the given tie-line end, after checking it against the formula of a stoichiometric phase."""
        energy = self._phases[phase]
        if energy.stoichiometric:
            formula = float(self._clouds[phase][1][0])
            if abs(formula - end) > _STOICHIOMETRIC_TOLERANCE:
                raise ValueError(
                    f"cell at T={cell.temperature}, x={cell.x}: tie-line end of {phase} is {end}, its formula has {formula}"
                )
            return formula
        return end

    def _minimum_at(self, phase: str, x: float, temperature: float, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the minimum energy of a phase at fixed composition and its parameter gradient."""
        energies, gradients = self._phases[phase].evaluate(temperature, self._fixed[(phase, round(x, 6))], theta)
        index = _minimum(energies)
        return float(energies[index]), gradients[index]

    def _driving_force(self, k: int, phase: str, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the driving force of a phase over the target line of cell k and its parameter gradient."""
        temperature = self.cells[k].temperature
        (phase_a, x_a), (phase_b, x_b) = self._targets[k]
        g_a, dg_a = self._minimum_at(phase_a, x_a, temperature, theta)
        g_b, dg_b = self._minimum_at(phase_b, x_b, temperature, theta)
        slope = (g_b - g_a) / (x_b - x_a)
        dslope = (dg_b - dg_a) / (x_b - x_a)
        configurations, compositions = self._clouds[phase]
        energies, gradients = self._phases[phase].evaluate(temperature, configurations, theta)
        differences = energies - (g_a + slope * (compositions - x_a))
        index = _minimum(differences)
        return float(differences[index]), gradients[index] - (dg_a + dslope * (compositions[index] - x_a))

    def _theta(self, theta: np.ndarray) -> np.ndarray:
        """Return the parameters as a float array after checking their number."""
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.symbols),):
            raise ValueError(f"theta has shape {theta.shape}, expected ({len(self.symbols)},)")
        return theta

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the penalty and its gradient with respect to the fit parameters.

        Args:
            theta (np.ndarray): The fit parameters, in the order of ``symbols``.

        Returns:
            tuple[float, np.ndarray]: The penalty and its gradient.
        """
        theta = self._theta(theta)
        penalty, gradient = 0.0, np.zeros(len(theta))
        for k, (cell, penalized) in enumerate(zip(self.cells, self.penalized, strict=True)):
            for phase in penalized:
                driving_force, dforce = self._driving_force(k, phase, theta)
                value, slope = _hinge_term(
                    driving_force, self.options.margin, self._sigma(cell, phase), cell.confidence
                )
                if slope != 0.0:
                    penalty += value
                    gradient += slope * dforce
        return penalty, gradient

    def violations(self, theta: np.ndarray) -> list[dict]:
        """Return the cells and phases whose driving force is below the margin.

        Args:
            theta (np.ndarray): The fit parameters, in the order of ``symbols``.

        Returns:
            list[dict]: One entry with T, x, phase, driving_force and sigma per violation.
        """
        theta = self._theta(theta)
        found = []
        for k, (cell, penalized) in enumerate(zip(self.cells, self.penalized, strict=True)):
            for phase in penalized:
                driving_force = self._driving_force(k, phase, theta)[0]
                if driving_force < self.options.margin:
                    found.append(
                        {
                            "T": cell.temperature,
                            "x": cell.x,
                            "phase": phase,
                            "driving_force": driving_force,
                            "sigma": self._sigma(cell, phase),
                        }
                    )
        return found
