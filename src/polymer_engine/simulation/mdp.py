"""GROMACS ``.mdp`` generation with validated parameters.

Every parameter is derived from :class:`SimulationDefaults` and validated before a
file is written.  The generator fixes several classes of error that are easy to make
and hard to notice afterwards:

* **``ref_t``, not ``target_t``.**  ``target_t`` is not a GROMACS option; grompp warns
  about the unknown keyword and silently uses the default temperature.
* **A real barostat.**  Berendsen does not sample the isothermal-isobaric ensemble
  (its volume fluctuations are wrong), and it was removed outright in GROMACS 2025+.
  Both equilibration and production default to the stochastic C-rescale barostat and
  V-rescale thermostat: they sample the correct ensemble without the Parrinello-Rahman /
  Nose-Hoover oscillation that blows a not-quite-equilibrated melt up a few ns into
  production (LINCS failure).
* **Distinct seeds per replica.**  ``gen_seed``/``ld_seed`` left at a fixed value make
  "independent replicas" byte-identical trajectories, which turns replica agreement
  into a tautology and understates uncertainty.
* **Explicit continuation.**  ``continuation`` and ``gen_vel`` must be consistent with
  where the stage's velocities come from, or the run either loses its equilibration
  or double-constrains its starting structure.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from polymer_engine.core.config import SimulationDefaults
from polymer_engine.core.errors import ParameterValidationError

StageName = Literal["em", "nvt", "anneal", "npt", "prod"]
STAGE_ORDER: tuple[StageName, ...] = ("em", "nvt", "anneal", "npt", "prod")

#: Thermostat/barostat names as GROMACS spells them.
_THERMOSTATS = {"v-rescale": "V-rescale", "nose-hoover": "Nose-Hoover", "berendsen": "Berendsen"}
_BAROSTATS = {
    "c-rescale": "C-rescale",
    "parrinello-rahman": "Parrinello-Rahman",
    "berendsen": "Berendsen",
}

#: GROMACS 2025 removed the Berendsen barostat; using it below that is still a bad
#: idea because it does not produce correct volume fluctuations.
BAROSTAT_REMOVED_IN = 2025


@dataclass(frozen=True, slots=True)
class MdpStage:
    """One generated stage."""

    name: StageName
    text: str
    parameters: dict[str, Any]
    input_structure: str
    output_prefix: str
    seed: int | None
    ns: float | None
    nsteps: int

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.mdp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text, encoding="utf-8")
        return path


def steps_for(ns: float, dt_ps: float) -> int:
    """Convert a duration in ns to an integer step count."""
    if dt_ps <= 0:
        raise ParameterValidationError("Timestep must be positive", timestep_ps=dt_ps)
    if ns < 0:
        raise ParameterValidationError("Simulation length cannot be negative", ns=ns)
    return round(ns * 1000.0 / dt_ps)


def validate_parameters(defaults: SimulationDefaults, *, gromacs_version: str | None = None) -> list[str]:
    """Check a parameter set for physical and software-compatibility problems.

    Returns a list of problems; empty means usable.  Raising is left to the caller so
    a planner can report every problem at once.
    """
    problems: list[str] = []

    if not (0 < defaults.temperature_k <= 2000):
        problems.append(f"temperature {defaults.temperature_k} K is outside (0, 2000]")
    if defaults.pressure_bar <= 0:
        problems.append(f"pressure {defaults.pressure_bar} bar must be positive")
    if not (0 < defaults.timestep_ps <= 0.005):
        problems.append(f"timestep {defaults.timestep_ps} ps is outside (0, 0.005]")
    if defaults.timestep_ps > 0.0025 and defaults.constraints == "none":
        problems.append(
            f"timestep {defaults.timestep_ps} ps needs bond constraints; got constraints=none"
        )
    if defaults.production_ns <= 0:
        problems.append("production length must be positive")
    if not (1 <= defaults.replicas <= 64):
        problems.append(f"replica count {defaults.replicas} is outside [1, 64]")
    if defaults.cutoff_nm <= 0:
        problems.append(f"cutoff {defaults.cutoff_nm} nm must be positive")
    if defaults.tau_t_ps <= 10 * defaults.timestep_ps:
        problems.append(
            f"tau_t {defaults.tau_t_ps} ps is too short for a {defaults.timestep_ps} ps timestep"
        )
    if defaults.tau_p_ps <= defaults.tau_t_ps:
        problems.append(
            f"tau_p ({defaults.tau_p_ps} ps) should exceed tau_t ({defaults.tau_t_ps} ps) "
            "so the barostat does not outrun the thermostat"
        )
    if defaults.base_seed <= 0:
        problems.append("base_seed must be positive; -1 would make runs irreproducible")
    if defaults.trajectory_output_ps < defaults.timestep_ps:
        problems.append("trajectory output interval is shorter than the timestep")

    if defaults.production_barostat == "berendsen":
        problems.append(
            "Berendsen is not a valid production barostat: it does not sample the NPT ensemble"
        )
    if gromacs_version:
        major = _major_version(gromacs_version)
        if major is not None and major >= BAROSTAT_REMOVED_IN:
            for label, value in (("barostat", defaults.barostat), ("production_barostat", defaults.production_barostat)):
                if value == "berendsen":
                    problems.append(
                        f"{label}=berendsen was removed in GROMACS {BAROSTAT_REMOVED_IN}+ "
                        f"(detected {gromacs_version}); use c-rescale"
                    )
    return problems


def _major_version(text: str) -> int | None:
    head = text.split(".")[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def replica_seed(base_seed: int, replica_index: int, stage: str) -> int:
    """A distinct, reproducible seed per (replica, stage).

    Derived rather than random so a campaign can be reproduced exactly, and distinct
    so replicas are genuinely independent samples rather than copies of one another.
    """
    if base_seed <= 0:
        raise ParameterValidationError("base_seed must be positive", base_seed=base_seed)
    if replica_index < 0:
        raise ParameterValidationError("replica_index must be non-negative", replica_index=replica_index)
    # zlib.crc32, not hash(): Python randomises string hashing per process, which
    # would give a different "reproducible" seed on every run.
    stage_key = zlib.crc32(stage.encode("utf-8"))
    mixed = (base_seed * 1_000_003) ^ ((replica_index + 1) * 2_654_435_761) ^ stage_key
    return (mixed % (2**31 - 2)) + 1


def _common_block(defaults: SimulationDefaults) -> list[str]:
    """Neighbour-search, electrostatics, and van der Waals settings shared by all MD stages."""
    return [
        "; --- neighbour searching ---",
        "cutoff-scheme            = Verlet",
        "nstlist                  = 20",
        "pbc                      = xyz",
        "verlet-buffer-tolerance  = 0.005",
        "",
        "; --- electrostatics ---",
        "coulombtype              = PME",
        f"rcoulomb                 = {defaults.cutoff_nm}",
        "pme-order                = 4",
        f"fourierspacing           = {defaults.pme_fourier_spacing_nm}",
        "",
        "; --- van der Waals ---",
        "vdwtype                  = Cut-off",
        "vdw-modifier             = Potential-shift-Verlet",
        f"rvdw                     = {defaults.cutoff_nm}",
        "DispCorr                 = EnerPres",
    ]


def _output_block(defaults: SimulationDefaults, dt_ps: float, *, production: bool) -> list[str]:
    nstxout_compressed = max(1, round(defaults.trajectory_output_ps / dt_ps))
    nstenergy = max(1, round(defaults.energy_output_ps / dt_ps))
    nstlog = max(1, round(defaults.log_output_ps / dt_ps))
    lines = [
        "; --- output control ---",
        "nstxout                  = 0",
        "nstvout                  = 0",
        "nstfout                  = 0",
        f"nstxout-compressed       = {nstxout_compressed}",
        f"nstenergy                = {nstenergy}",
        f"nstlog                   = {nstlog}",
        "compressed-x-precision   = 1000",
    ]
    if production:
        # Velocities are needed for transport properties; write them sparsely.
        lines.insert(2, f"nstvout                  = {nstxout_compressed * 10}")
        lines.pop(3)
    return lines


def _thermostat_block(defaults: SimulationDefaults, *, thermostat: str, seed: int) -> list[str]:
    name = _THERMOSTATS[thermostat]
    lines = [
        "; --- temperature coupling ---",
        f"tcoupl                   = {name}",
        "tc-grps                  = System",
        f"tau-t                    = {defaults.tau_t_ps}",
        # ref_t, NOT target_t: 'target_t' is not a GROMACS keyword and would be ignored.
        f"ref-t                    = {defaults.temperature_k}",
    ]
    if name == "V-rescale":
        # A fixed ld-seed would make every replica's thermostat noise identical.
        lines.append(f"ld-seed                  = {seed}")
    return lines


def _barostat_block(defaults: SimulationDefaults, *, barostat: str) -> list[str]:
    return [
        "; --- pressure coupling ---",
        f"pcoupl                   = {_BAROSTATS[barostat]}",
        "pcoupltype               = isotropic",
        f"tau-p                    = {defaults.tau_p_ps}",
        f"ref-p                    = {defaults.pressure_bar}",
        f"compressibility          = {defaults.compressibility_bar_inv}",
        "refcoord-scaling         = com",
    ]


def _constraints_block(defaults: SimulationDefaults) -> list[str]:
    return [
        "; --- constraints ---",
        f"constraints              = {defaults.constraints}",
        "constraint-algorithm     = LINCS",
        "lincs-order              = 4",
        "lincs-iter               = 1",
    ]


def generate_minimization(defaults: SimulationDefaults) -> MdpStage:
    lines = [
        "; Energy minimisation -- generated by polymer-engine",
        "integrator               = steep",
        f"nsteps                   = {defaults.minimization_steps}",
        "emtol                    = 100.0",
        "emstep                   = 0.01",
        "",
        *_common_block(defaults),
        "",
        "; --- constraints ---",
        # Minimisation runs unconstrained so a bad starting geometry can actually relax.
        "constraints              = none",
        "",
        "; --- output control ---",
        "nstlog                   = 100",
        "nstenergy                = 100",
        "",
    ]
    return MdpStage(
        name="em",
        text="\n".join(lines),
        parameters={
            "integrator": "steep",
            "nsteps": defaults.minimization_steps,
            "emtol_kj_mol_nm": 100.0,
            "constraints": "none",
            "cutoff_nm": defaults.cutoff_nm,
        },
        input_structure="system.gro",
        output_prefix="em",
        seed=None,
        ns=None,
        nsteps=defaults.minimization_steps,
    )


def generate_nvt(defaults: SimulationDefaults, *, seed: int) -> MdpStage:
    nsteps = steps_for(defaults.nvt_ns, defaults.timestep_ps)
    lines = [
        "; NVT equilibration -- generated by polymer-engine",
        "integrator               = md",
        f"dt                       = {defaults.timestep_ps}",
        f"nsteps                   = {nsteps}",
        "tinit                    = 0",
        "",
        *_output_block(defaults, defaults.timestep_ps, production=False),
        "",
        *_common_block(defaults),
        "",
        *_constraints_block(defaults),
        # Coming from minimisation there are no velocities to continue from.
        "continuation             = no",
        "",
        *_thermostat_block(defaults, thermostat=defaults.thermostat, seed=seed),
        "",
        "; --- pressure coupling ---",
        "pcoupl                   = no",
        "",
        "; --- velocity generation ---",
        "gen-vel                  = yes",
        f"gen-temp                 = {defaults.temperature_k}",
        # A per-replica seed is what makes replicas independent samples.
        f"gen-seed                 = {seed}",
        "",
    ]
    return MdpStage(
        name="nvt",
        text="\n".join(lines),
        parameters={
            "integrator": "md",
            "dt_ps": defaults.timestep_ps,
            "nsteps": nsteps,
            "ns": defaults.nvt_ns,
            "temperature_k": defaults.temperature_k,
            "thermostat": defaults.thermostat,
            "constraints": defaults.constraints,
            "gen_vel": True,
            "seed": seed,
        },
        input_structure="em.gro",
        output_prefix="nvt",
        seed=seed,
        ns=defaults.nvt_ns,
        nsteps=nsteps,
    )


def anneal_schedule(defaults: SimulationDefaults) -> tuple[list[float], list[float]]:
    """The (time, temperature) points GROMACS interpolates between.

    Four points: start at the target temperature, ramp to the peak, hold, ramp back.
    Returned separately from the mdp text so the schedule can be asserted on directly --
    a cooling ramp that is accidentally instantaneous is a quench, not an anneal, and
    the difference does not show up anywhere else.
    """
    total_ps = defaults.anneal_ns * 1000.0
    heat = total_ps * defaults.anneal_heat_fraction
    hold = total_ps * defaults.anneal_hold_fraction
    if heat + hold >= total_ps:
        raise ParameterValidationError(
            "the anneal has no time left to cool in",
            problems=[f"heat {defaults.anneal_heat_fraction} + hold "
                      f"{defaults.anneal_hold_fraction} must be below 1.0"],
        )
    times = [0.0, heat, heat + hold, total_ps]
    temps = [defaults.temperature_k, defaults.anneal_temperature_k,
             defaults.anneal_temperature_k, defaults.temperature_k]
    return times, temps


def generate_anneal(defaults: SimulationDefaults, *, seed: int) -> MdpStage:
    """NPT with a heat-hold-cool temperature schedule.

    Run under pressure coupling rather than at fixed volume: the point is to let the box
    find its own density at each temperature, and a fixed-volume anneal would hold the
    system at whatever wrong density the packing produced.
    """
    nsteps = steps_for(defaults.anneal_ns, defaults.timestep_ps)
    times, temps = anneal_schedule(defaults)
    lines = [
        "; Simulated-annealing equilibration -- generated by polymer-engine",
        f"; heat {defaults.temperature_k:g} -> {defaults.anneal_temperature_k:g} K, "
        f"hold, cool back over {defaults.anneal_ns:g} ns",
        "integrator               = md",
        f"dt                       = {defaults.timestep_ps}",
        f"nsteps                   = {nsteps}",
        "tinit                    = 0",
        "",
        *_output_block(defaults, defaults.timestep_ps, production=False),
        "",
        *_common_block(defaults),
        "",
        *_constraints_block(defaults),
        "continuation             = yes",
        "",
        *_thermostat_block(defaults, thermostat=defaults.thermostat, seed=seed),
        "",
        *_barostat_block(defaults, barostat=defaults.barostat),
        "",
        "; --- annealing schedule ---",
        "annealing                = single",
        f"annealing-npoints        = {len(times)}",
        "annealing-time           = " + " ".join(f"{x:g}" for x in times),
        "annealing-temp           = " + " ".join(f"{x:g}" for x in temps),
        "",
        "; --- velocity generation ---",
        "gen-vel                  = no",
        "",
    ]
    return MdpStage(
        name="anneal",
        text="\n".join(lines),
        parameters={
            "integrator": "md", "dt_ps": defaults.timestep_ps, "nsteps": nsteps,
            "ns": defaults.anneal_ns, "temperature_k": defaults.temperature_k,
            "anneal_temperature_k": defaults.anneal_temperature_k,
            "annealing_time_ps": times, "annealing_temp_k": temps,
            "pressure_bar": defaults.pressure_bar, "barostat": defaults.barostat,
            "continuation": True, "seed": seed,
        },
        input_structure="nvt.gro", output_prefix="anneal", seed=seed,
        ns=defaults.anneal_ns, nsteps=nsteps,
    )


def generate_npt(defaults: SimulationDefaults, *, seed: int) -> MdpStage:
    nsteps = steps_for(defaults.npt_ns, defaults.timestep_ps)
    lines = [
        "; NPT equilibration -- generated by polymer-engine",
        "integrator               = md",
        f"dt                       = {defaults.timestep_ps}",
        f"nsteps                   = {nsteps}",
        "tinit                    = 0",
        "",
        *_output_block(defaults, defaults.timestep_ps, production=False),
        "",
        *_common_block(defaults),
        "",
        *_constraints_block(defaults),
        # Velocities carry over from NVT, so do not regenerate them.
        "continuation             = yes",
        "",
        *_thermostat_block(defaults, thermostat=defaults.thermostat, seed=seed),
        "",
        *_barostat_block(defaults, barostat=defaults.barostat),
        "",
        "; --- velocity generation ---",
        "gen-vel                  = no",
        "",
    ]
    return MdpStage(
        name="npt",
        text="\n".join(lines),
        parameters={
            "integrator": "md",
            "dt_ps": defaults.timestep_ps,
            "nsteps": nsteps,
            "ns": defaults.npt_ns,
            "temperature_k": defaults.temperature_k,
            "pressure_bar": defaults.pressure_bar,
            "thermostat": defaults.thermostat,
            "barostat": defaults.barostat,
            "continuation": True,
            "seed": seed,
        },
        input_structure="anneal.gro" if defaults.anneal_ns > 0 else "nvt.gro",
        output_prefix="npt",
        seed=seed,
        ns=defaults.npt_ns,
        nsteps=nsteps,
    )


def generate_production(defaults: SimulationDefaults, *, seed: int, plumed: bool = False) -> MdpStage:
    nsteps = steps_for(defaults.production_ns, defaults.timestep_ps)
    lines = [
        "; Production MD -- generated by polymer-engine",
        "integrator               = md",
        f"dt                       = {defaults.timestep_ps}",
        f"nsteps                   = {nsteps}",
        "tinit                    = 0",
        "",
        *_output_block(defaults, defaults.timestep_ps, production=True),
        "",
        *_common_block(defaults),
        "",
        *_constraints_block(defaults),
        "continuation             = yes",
        "",
        *_thermostat_block(defaults, thermostat=defaults.production_thermostat, seed=seed),
        "",
        *_barostat_block(defaults, barostat=defaults.production_barostat),
        "",
        "; --- velocity generation ---",
        "gen-vel                  = no",
        "",
    ]
    if plumed:
        lines.insert(1, "; PLUMED bias is supplied via 'gmx mdrun -plumed', not from the mdp")
    return MdpStage(
        name="prod",
        text="\n".join(lines),
        parameters={
            "integrator": "md",
            "dt_ps": defaults.timestep_ps,
            "nsteps": nsteps,
            "ns": defaults.production_ns,
            "temperature_k": defaults.temperature_k,
            "pressure_bar": defaults.pressure_bar,
            "thermostat": defaults.production_thermostat,
            "barostat": defaults.production_barostat,
            "continuation": True,
            "seed": seed,
            "plumed": plumed,
        },
        input_structure="npt.gro",
        output_prefix="prod",
        seed=seed,
        ns=defaults.production_ns,
        nsteps=nsteps,
    )


def generate_stages(
    defaults: SimulationDefaults,
    *,
    replica_index: int,
    gromacs_version: str | None = None,
    plumed: bool = False,
) -> list[MdpStage]:
    """Generate EM -> NVT -> [anneal] -> NPT -> production for one replica.

    The anneal is included only when ``anneal_ns`` is positive, so an existing
    configuration reproduces its original four-stage protocol exactly.
    """
    problems = validate_parameters(defaults, gromacs_version=gromacs_version)
    if problems:
        raise ParameterValidationError(
            "Simulation parameters are not usable", problems=problems, replica_index=replica_index
        )
    stages = [
        generate_minimization(defaults),
        generate_nvt(defaults, seed=replica_seed(defaults.base_seed, replica_index, "nvt")),
    ]
    if defaults.anneal_ns > 0:
        stages.append(generate_anneal(
            defaults, seed=replica_seed(defaults.base_seed, replica_index, "anneal")))
    return [
        *stages,
        generate_npt(defaults, seed=replica_seed(defaults.base_seed, replica_index, "npt")),
        generate_production(
            defaults, seed=replica_seed(defaults.base_seed, replica_index, "prod"), plumed=plumed
        ),
    ]


def parse_mdp(text: str) -> dict[str, str]:
    """Parse mdp text into a normalised ``{key: value}`` map.

    GROMACS treats ``-`` and ``_`` in option names as equivalent and is
    case-insensitive; the parser normalises both so tests can assert on one spelling.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip().lower().replace("_", "-")] = value.strip()
    return out


__all__ = [
    "BAROSTAT_REMOVED_IN",
    "STAGE_ORDER",
    "MdpStage",
    "generate_minimization",
    "generate_npt",
    "generate_nvt",
    "generate_production",
    "generate_stages",
    "parse_mdp",
    "replica_seed",
    "steps_for",
    "validate_parameters",
]
