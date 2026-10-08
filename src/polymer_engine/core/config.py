"""Configuration.

Precedence, highest first:

1. explicit keyword arguments to :func:`load_config`
2. environment variables (``POLYMER_*``)
3. a YAML/JSON config file (``--config``, ``$POLYMER_CONFIG``, ``./polymer.yaml``,
   ``./configs/default.yaml``, ``~/.config/polymer-engine/config.yaml``)
4. the defaults declared on the models below

Secrets are never stored in the config file that ships with the repository and are
never rendered by :meth:`EngineConfig.redacted`.  ``repr`` of a :class:`Secret` is
masked so a credential cannot leak through a log line or a traceback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from polymer_engine.core.errors import ConfigError

DEFAULT_CONFIG_FILENAMES = (
    "polymer.yaml",
    "polymer.yml",
    "configs/default.yaml",
)


class Secret:
    """A string that refuses to render itself.

    ``str(secret)`` also returns the mask; call :meth:`reveal` to obtain the value.
    That makes accidental interpolation into a log message or an f-string safe.
    """

    __slots__ = ("_value",)
    MASK = "***REDACTED***"

    def __init__(self, value: str | None) -> None:
        self._value = value or None

    def reveal(self) -> str | None:
        return self._value

    def __bool__(self) -> bool:
        return self._value is not None

    def __repr__(self) -> str:
        return f"Secret({self.MASK})" if self._value else "Secret(None)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and self._value == other._value

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self._value)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
class PathsConfig(BaseModel):
    """Filesystem layout.  No path in the engine is hard-coded outside this model."""

    model_config = {"extra": "forbid"}

    root: Path = Field(default=Path(), description="Project root all relative paths resolve against.")
    data_dir: Path = Path("data")
    workspace_dir: Path = Path("workspaces")
    scratch_dir: Path = Path("scratch")
    state_dir: Path = Path("engine_state")
    cache_dir: Path = Path(".polymer-cache")
    database: Path = Path("engine_state/engine.sqlite")

    def resolved(self, name: str) -> Path:
        value = getattr(self, name)
        if not isinstance(value, Path):  # pragma: no cover - defensive
            raise ConfigError(f"{name} is not a path")
        return value if value.is_absolute() else (self.root / value)

    def ensure(self) -> None:
        for name in ("data_dir", "workspace_dir", "scratch_dir", "state_dir", "cache_dir"):
            self.resolved(name).mkdir(parents=True, exist_ok=True)
        self.resolved("database").parent.mkdir(parents=True, exist_ok=True)


class ToolConfig(BaseModel):
    """A local executable.

    ``path`` overrides ``PATH`` lookup entirely when set; otherwise ``executable``
    is resolved against ``PATH``.

    A bare string in configuration is accepted as shorthand for ``{executable: ...}``,
    so ``gromacs: gmx`` and ``gromacs: {executable: gmx}`` mean the same thing.
    """

    model_config = {"extra": "forbid"}

    executable: str
    path: Path | None = None
    min_version: str | None = None
    max_version: str | None = None
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, value: Any) -> Any:
        return {"executable": value} if isinstance(value, str) else value


class LocalToolsConfig(BaseModel):
    model_config = {"extra": "forbid"}

    gromacs: ToolConfig = ToolConfig(executable="gmx", min_version="2021")
    orca: ToolConfig = ToolConfig(executable="orca")
    plumed: ToolConfig = ToolConfig(executable="plumed", min_version="2.7")
    python: ToolConfig = ToolConfig(executable="python3", min_version="3.11")


class ResourceConfig(BaseModel):
    model_config = {"extra": "forbid"}

    max_concurrent_jobs: int = Field(default=1, ge=1)
    cpus_per_job: int | None = Field(default=None, ge=1)
    gpu_available: bool | None = Field(
        default=None,
        description="None means 'detect'.  True/False force the answer for reproducibility.",
    )
    gpu_device_ids: list[int] = Field(default_factory=list)
    job_timeout_s: float | None = Field(default=None, gt=0)


class HttpConfig(BaseModel):
    model_config = {"extra": "forbid"}

    timeout_s: float = Field(default=30.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_base_s: float = Field(default=0.5, ge=0)
    backoff_max_s: float = Field(default=30.0, ge=0)
    rate_limit_per_s: float = Field(default=3.0, gt=0)
    cache_enabled: bool = True
    cache_ttl_s: float = Field(default=86400.0, ge=0)
    user_agent: str = "polymer-autonomous-engine"
    offline: bool = Field(
        default=False,
        description="When true every outbound request raises instead of touching the network.",
    )


class CredentialsConfig(BaseModel):
    """Credential *locations*, never credential values in a file we ship.

    Values are read from environment variables or a file referenced by
    ``*_file``.  The model holds :class:`Secret` objects so they cannot be printed.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    charmm_gui_email: str | None = None
    charmm_gui_password: Secret = Field(default_factory=lambda: Secret(None))
    charmm_gui_token: Secret = Field(default_factory=lambda: Secret(None))
    charmm_gui_token_file: Path | None = None
    materials_project_api_key: Secret = Field(default_factory=lambda: Secret(None))
    crossref_mailto: str | None = None
    openalex_mailto: str | None = None
    anthropic_api_key: Secret = Field(default_factory=lambda: Secret(None))

    @field_validator(
        "charmm_gui_password",
        "charmm_gui_token",
        "materials_project_api_key",
        "anthropic_api_key",
        mode="before",
    )
    @classmethod
    def _wrap_secret(cls, value: Any) -> Secret:
        return value if isinstance(value, Secret) else Secret(value)


class SimulationDefaults(BaseModel):
    """Scientific assumptions.  Explicit, versioned, and copied into provenance.

    Nothing in the simulation layer may hard-code any of these.
    """

    model_config = {"extra": "forbid"}

    force_field: str = Field(
        default="UNSPECIFIED",
        description="Recorded verbatim in provenance.  'UNSPECIFIED' is honest; a wrong name is not.",
    )
    water_model: str = "UNSPECIFIED"
    temperature_k: float = Field(default=300.0, gt=0, le=2000)
    pressure_bar: float = Field(default=1.0, gt=0)
    timestep_ps: float = Field(default=0.002, gt=0, le=0.005)
    replicas: int = Field(default=3, ge=1, le=64)
    base_seed: int = Field(default=20240101, ge=1, le=2**31 - 1)
    minimization_steps: int = Field(default=50_000, ge=1)
    nvt_ns: float = Field(default=1.0, ge=0)
    #: Simulated-annealing equilibration, run between NVT and NPT. Zero disables it.
    #:
    #: `gmx insert-molecules` can only pack rigid pre-built chains at roughly a third of
    #: the melt density, so NPT has to compress the box threefold. That compression
    #: leaves some replicas jammed in configurations that hours of NPT do not relax:
    #: measured across three polyolefins, effective sample counts came out bimodal --
    #: 4 to 19 for the stuck replicas against 600 to 44000 for the rest, with the stuck
    #: ones sitting 30 to 50 kg/m^3 away from their siblings in *both* directions.
    #: Heating well above the melting point lets the chains move past one another before
    #: the box is squeezed, which is the standard remedy and the one this addresses.
    anneal_ns: float = Field(default=0.0, ge=0)
    #: Peak annealing temperature. Must exceed the melting point of the longest chain
    #: present, or the crystallites the anneal exists to erase simply survive it.
    anneal_temperature_k: float = Field(default=500.0, gt=0, le=2000)
    #: Fractions of the anneal spent heating, holding at the peak, and cooling. The
    #: cooling ramp is the longest because ordering happens on the way down.
    anneal_heat_fraction: float = Field(default=0.15, gt=0, lt=1)
    anneal_hold_fraction: float = Field(default=0.35, gt=0, lt=1)
    npt_ns: float = Field(default=5.0, ge=0)
    production_ns: float = Field(default=50.0, gt=0)
    cutoff_nm: float = Field(default=1.2, gt=0)
    pme_fourier_spacing_nm: float = Field(default=0.12, gt=0)
    constraints: Literal["none", "h-bonds", "all-bonds"] = "h-bonds"
    thermostat: Literal["v-rescale", "nose-hoover", "berendsen"] = "v-rescale"
    # Production uses the stochastic V-rescale / C-rescale couplings, not Nose-Hoover /
    # Parrinello-Rahman. Both stochastic schemes sample the correct NPT ensemble (C-rescale
    # is Bernetti-Bussi 2020, a proper barostat, not Berendsen), and unlike the oscillatory
    # NH/PR pair they do not resonate and blow up when production starts from a not-quite-
    # equilibrated box -- the LINCS failure that killed polypropylene and poly(acrylamide)
    # a few ns into production despite a cleanly condensed melt.
    production_thermostat: Literal["v-rescale", "nose-hoover"] = "v-rescale"
    barostat: Literal["c-rescale", "parrinello-rahman", "berendsen"] = "c-rescale"
    production_barostat: Literal["parrinello-rahman", "c-rescale"] = "c-rescale"
    tau_t_ps: float = Field(default=1.0, gt=0)
    tau_p_ps: float = Field(default=5.0, gt=0)
    compressibility_bar_inv: float = Field(default=4.5e-5, gt=0)
    trajectory_output_ps: float = Field(default=10.0, gt=0)
    energy_output_ps: float = Field(default=1.0, gt=0)
    log_output_ps: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> SimulationDefaults:
        if self.timestep_ps > 0.002 and self.constraints == "none":
            raise ConfigError(
                "A timestep above 2 fs requires bond constraints",
                timestep_ps=self.timestep_ps,
                constraints=self.constraints,
            )
        if self.energy_output_ps < self.timestep_ps:
            raise ConfigError(
                "Energy output interval is shorter than the timestep",
                energy_output_ps=self.energy_output_ps,
                timestep_ps=self.timestep_ps,
            )
        return self


class UmbrellaDefaults(BaseModel):
    model_config = {"extra": "forbid"}

    # 0.08 nm at k=1000 kJ/mol/nm^2 is ~1.6 sigma at 300 K, comfortably inside the
    # ~2 sigma limit beyond which adjacent windows stop overlapping.
    spacing_nm: float = Field(default=0.08, gt=0)
    force_constant_kj_mol_nm2: float = Field(default=1000.0, gt=0)
    window_ns: float = Field(default=10.0, gt=0)
    equilibration_fraction: float = Field(default=0.2, ge=0, lt=1)
    min_pair_overlap: float = Field(default=0.10, gt=0, le=1)
    min_windows: int = Field(default=5, ge=2)
    bootstrap_samples: int = Field(default=200, ge=20)
    max_adaptive_rounds: int = Field(default=3, ge=0)


class AnalysisDefaults(BaseModel):
    model_config = {"extra": "forbid"}

    equilibration_detection: Literal["fraction", "reverse-cumulative"] = "reverse-cumulative"
    discard_fraction: float = Field(default=0.2, ge=0, lt=1)
    stride: int = Field(default=1, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=50)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    min_effective_samples: float = Field(default=20.0, gt=0)
    max_drift_fraction: float = Field(default=0.02, gt=0)
    max_relative_stderr: float = Field(default=0.02, gt=0)
    replica_agreement_alpha: float = Field(default=0.05, gt=0, lt=1)
    random_seed: int = Field(default=20240101, ge=0)


class SafetyConfig(BaseModel):
    model_config = {"extra": "forbid"}

    execution_enabled: bool = Field(
        default=False, description="Local scientific executables never run unless this is true."
    )
    allow_network: bool = True
    require_provenance_for_promotion: bool = True
    require_uncertainty_for_promotion: bool = True
    require_replica_agreement: bool = True
    require_equilibration_before_production: bool = True
    max_archive_bytes: int = Field(default=2 * 1024**3, gt=0)
    max_archive_members: int = Field(default=200_000, gt=0)


class PlanningConfig(BaseModel):
    model_config = {"extra": "forbid"}

    information_weight: float = 1.5
    value_weight: float = 1.0
    cost_weight: float = 0.5
    risk_weight: float = 1.0
    max_cost_per_action: float = Field(default=1e9, gt=0)
    tie_break: Literal["cheapest", "lowest-risk", "deterministic-id"] = "cheapest"


class LoggingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["text", "json"] = "text"
    file: Path | None = None


class EngineConfig(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    paths: PathsConfig = Field(default_factory=PathsConfig)
    local_tools: LocalToolsConfig = Field(default_factory=LocalToolsConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    credentials: CredentialsConfig = Field(default_factory=CredentialsConfig)
    simulation: SimulationDefaults = Field(default_factory=SimulationDefaults)
    umbrella: UmbrellaDefaults = Field(default_factory=UmbrellaDefaults)
    analysis: AnalysisDefaults = Field(default_factory=AnalysisDefaults)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    source_files: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _tolerate_empty_sections(cls, value: Any) -> Any:
        """Treat a fully commented-out YAML section as absent rather than as null.

        Writing ``credentials:`` with every entry commented out is the natural way to
        document a section, and YAML parses it as ``None``.  Rejecting that would make
        a helpful config file an error.
        """
        if not isinstance(value, dict):
            return value
        return {k: ({} if v is None and k in cls.model_fields else v) for k, v in value.items()}

    def redacted(self) -> dict[str, Any]:
        """A JSON-safe dump with every credential masked.

        This is the only representation that may be logged or written to a manifest.
        """
        data = self.model_dump(mode="json", exclude={"credentials"})
        creds = self.credentials
        data["credentials"] = {
            "charmm_gui_email": _mask_email(creds.charmm_gui_email),
            "charmm_gui_password": _present(creds.charmm_gui_password),
            "charmm_gui_token": _present(creds.charmm_gui_token),
            "charmm_gui_token_file": str(creds.charmm_gui_token_file) if creds.charmm_gui_token_file else None,
            "materials_project_api_key": _present(creds.materials_project_api_key),
            "anthropic_api_key": _present(creds.anthropic_api_key),
            "crossref_mailto": _mask_email(creds.crossref_mailto),
            "openalex_mailto": _mask_email(creds.openalex_mailto),
        }
        return data

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"EngineConfig(sources={self.source_files!r})"


def _present(secret: Secret) -> str:
    return "configured" if secret else "not-configured"


def _mask_email(value: str | None) -> str | None:
    if not value or "@" not in value:
        return None if not value else "***"
    local, _, domain = value.partition("@")
    head = local[0] if local else "*"
    return f"{head}***@{domain}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
# environment variable -> dotted config path
ENV_MAP: dict[str, str] = {
    "POLYMER_ROOT": "paths.root",
    "POLYMER_DATA_DIR": "paths.data_dir",
    "POLYMER_WORKSPACE_DIR": "paths.workspace_dir",
    "POLYMER_SCRATCH_DIR": "paths.scratch_dir",
    "POLYMER_STATE_DIR": "paths.state_dir",
    "POLYMER_CACHE_DIR": "paths.cache_dir",
    "POLYMER_DATABASE": "paths.database",
    "POLYMER_GMX": "local_tools.gromacs.path",
    "POLYMER_ORCA": "local_tools.orca.path",
    "POLYMER_PLUMED": "local_tools.plumed.path",
    "POLYMER_PYTHON": "local_tools.python.path",
    "POLYMER_MAX_CONCURRENT_JOBS": "resources.max_concurrent_jobs",
    "POLYMER_CPUS_PER_JOB": "resources.cpus_per_job",
    "POLYMER_GPU_AVAILABLE": "resources.gpu_available",
    "POLYMER_HTTP_TIMEOUT": "http.timeout_s",
    "POLYMER_HTTP_OFFLINE": "http.offline",
    "POLYMER_HTTP_CACHE_ENABLED": "http.cache_enabled",
    "POLYMER_EXECUTION_ENABLED": "safety.execution_enabled",
    "POLYMER_ALLOW_NETWORK": "safety.allow_network",
    "POLYMER_LOG_LEVEL": "logging.level",
    "POLYMER_LOG_FORMAT": "logging.format",
    "POLYMER_LOG_FILE": "logging.file",
    # credentials
    "CHARMM_GUI_EMAIL": "credentials.charmm_gui_email",
    "CHARMM_GUI_PASSWORD": "credentials.charmm_gui_password",
    "CHARMM_GUI_TOKEN": "credentials.charmm_gui_token",
    "CHARMM_GUI_TOKEN_FILE": "credentials.charmm_gui_token_file",
    "MP_API_KEY": "credentials.materials_project_api_key",
    "CROSSREF_MAILTO": "credentials.crossref_mailto",
    "OPENALEX_MAILTO": "credentials.openalex_mailto",
    "ANTHROPIC_API_KEY": "credentials.anthropic_api_key",
}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce_env(raw: str, dotted: str) -> Any:
    """Coerce an environment string using the target field's declared type."""
    model: type[BaseModel] | None = EngineConfig
    parts = dotted.split(".")
    for part in parts[:-1]:
        if model is None:
            return raw
        field = model.model_fields.get(part)
        annotation = field.annotation if field else None
        model = annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None
    if model is None:
        return raw
    field = model.model_fields.get(parts[-1])
    if field is None:
        return raw
    annotation = field.annotation
    lowered = raw.strip().lower()
    if annotation is bool or annotation == (bool | None):
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
        raise ConfigError(f"Cannot interpret {raw!r} as a boolean", setting=dotted)
    if annotation in (int, int | None):
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"Cannot interpret {raw!r} as an integer", setting=dotted) from None
    if annotation in (float, float | None):
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(f"Cannot interpret {raw!r} as a number", setting=dotted) from None
    return raw


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - defensive
            raise ConfigError(f"Cannot set {dotted}: {part} is not a section")
    node[parts[-1]] = value


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON config file into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}", path=str(path))
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".json"}:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}", path=str(path)) from exc
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - PyYAML is a hard dependency
            raise ConfigError("PyYAML is required to read YAML configuration") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}", path=str(path)) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping, got {type(data).__name__}", path=str(path))
    return data


def discover_config_file(root: Path | None = None) -> Path | None:
    """Locate a config file using ``$POLYMER_CONFIG`` then well-known filenames."""
    explicit = os.environ.get("POLYMER_CONFIG")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise ConfigError(f"POLYMER_CONFIG points at a missing file: {path}", path=explicit)
        return path
    root = root or Path.cwd()
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = root / name
        if candidate.exists():
            return candidate
    user = Path.home() / ".config" / "polymer-engine" / "config.yaml"
    return user if user.exists() else None


def load_config(
    config_file: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
    discover: bool = True,
) -> EngineConfig:
    """Build an :class:`EngineConfig` following the documented precedence."""
    environment: Mapping[str, str] = os.environ if env is None else env
    data: dict[str, Any] = {}
    sources: list[str] = []

    path = Path(config_file) if config_file else (discover_config_file() if discover else None)
    if path is not None:
        data = _deep_merge(data, read_config_file(path))
        sources.append(str(path))

    if use_env:
        applied: dict[str, Any] = {}
        for var, dotted in ENV_MAP.items():
            raw = environment.get(var)
            if raw is None or raw == "":
                continue
            _set_dotted(applied, dotted, _coerce_env(raw, dotted))
        if applied:
            data = _deep_merge(data, applied)
            sources.append("environment")

    if overrides:
        data = _deep_merge(data, overrides)
        sources.append("overrides")

    data.pop("source_files", None)
    try:
        config = EngineConfig.model_validate(data)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigError(f"Invalid configuration: {exc}", sources=sources) from exc

    config = config.model_copy(update={"source_files": sources})
    _load_token_file(config)
    return config


def _load_token_file(config: EngineConfig) -> None:
    """Read a CHARMM-GUI token from a file when one is configured and no token is set."""
    creds = config.credentials
    if creds.charmm_gui_token or creds.charmm_gui_token_file is None:
        return
    path = creds.charmm_gui_token_file
    if not path.exists():
        raise ConfigError("charmm_gui_token_file does not exist", path=str(path))
    mode = path.stat().st_mode & 0o077
    if mode:
        raise ConfigError(
            "Refusing to read a credential file that is group- or world-accessible",
            path=str(path),
            hint="chmod 600 the token file",
        )
    object.__setattr__(creds, "charmm_gui_token", Secret(path.read_text(encoding="utf-8").strip()))


__all__ = [
    "AnalysisDefaults",
    "CredentialsConfig",
    "EngineConfig",
    "HttpConfig",
    "LocalToolsConfig",
    "LoggingConfig",
    "PathsConfig",
    "PlanningConfig",
    "ResourceConfig",
    "SafetyConfig",
    "Secret",
    "SimulationDefaults",
    "ToolConfig",
    "UmbrellaDefaults",
    "discover_config_file",
    "load_config",
    "read_config_file",
]
