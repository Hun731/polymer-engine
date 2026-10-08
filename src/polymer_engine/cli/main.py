"""Command-line interface.

Command groups mirror the pipeline: engine, polymer, provider, system, campaign,
umbrella, model, design, evidence.

Two conventions apply throughout:

* **Exit codes carry meaning.**  0 success, 1 an engine/scientific failure, 2 a usage
  error, 3 a validation gate that did not pass.  A gate failure is not a crash and is
  not a success either, so it gets its own code that CI can act on.
* **Errors are actionable.**  Every failure prints what went wrong and, where there is
  one, the next thing to try.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer

from polymer_engine.core.config import EngineConfig, load_config
from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import configure_logging, get_logger

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_GATE_FAILED = 3

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Polymer Autonomous Research Engine",
    context_settings={"help_option_names": ["-h", "--help"]},
)

engine_app = typer.Typer(no_args_is_help=True, help="Engine setup and environment inspection")
polymer_app = typer.Typer(no_args_is_help=True, help="Polymer ingestion, descriptors, clustering")
provider_app = typer.Typer(no_args_is_help=True, help="External data providers")
charmm_app = typer.Typer(no_args_is_help=True, help="CHARMM-GUI job access")
system_app = typer.Typer(no_args_is_help=True, help="Simulation system import and validation")
campaign_app = typer.Typer(no_args_is_help=True, help="Campaign lifecycle")
umbrella_app = typer.Typer(no_args_is_help=True, help="Umbrella sampling")
model_app = typer.Typer(no_args_is_help=True, help="Surrogate models")
design_app = typer.Typer(no_args_is_help=True, help="Candidate generation and ranking")
evidence_app = typer.Typer(no_args_is_help=True, help="Claims and reproducibility")
qm_app = typer.Typer(no_args_is_help=True, help="Quantum-chemistry workflows")
property_app = typer.Typer(no_args_is_help=True, help="Property calculations")
research_app = typer.Typer(no_args_is_help=True, help="Autonomous research loop")
param_app = typer.Typer(no_args_is_help=True, help="Polymer parameterization and force-field routing")
browser_app = typer.Typer(no_args_is_help=True,
                          help="CHARMM-GUI Polymer Builder browser automation")

app.add_typer(engine_app, name="engine")
app.add_typer(polymer_app, name="polymer")
app.add_typer(provider_app, name="provider")
provider_app.add_typer(charmm_app, name="charmm-gui")
app.add_typer(system_app, name="system")
app.add_typer(campaign_app, name="campaign")
app.add_typer(umbrella_app, name="umbrella")
app.add_typer(model_app, name="model")
app.add_typer(design_app, name="design")
app.add_typer(evidence_app, name="evidence")
app.add_typer(qm_app, name="qm")
app.add_typer(property_app, name="property")
app.add_typer(research_app, name="research")
app.add_typer(param_app, name="parameterization")
app.add_typer(browser_app, name="browser")

logger = get_logger("cli")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def get_config(config_path: str | None = None) -> EngineConfig:
    try:
        config = load_config(config_path)
    except PolymerEngineError as exc:
        fail(f"Configuration error: {exc}")
    configure_logging(config)
    return config


def emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def fail(message: str, *, code: int = EXIT_ERROR, hint: str | None = None) -> NoReturn:
    """Print an actionable error and exit.

    Declared ``NoReturn`` so type checkers narrow correctly at every call site:
    ``if x is None: fail(...)`` really does mean ``x`` is not None afterwards.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    if hint:
        typer.secho(f"hint: {hint}", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code)


def get_store(config: EngineConfig):
    from polymer_engine.db.store import Store

    return Store(config.paths.resolved("database"))


ConfigOption = typer.Option(None, "--config", "-c", help="Path to a config file")


# ==========================================================================
# engine
# ==========================================================================
@engine_app.command("init")
def engine_init(
    config_path: str | None = ConfigOption,
    title: str = typer.Option("Autonomous polymer discovery", help="Objective title"),
    description: str = typer.Option("Discover polymers with improved target properties.", help="Objective description"),
) -> None:
    """Create the workspace layout, database, and an initial objective."""
    config = get_config(config_path)
    config.paths.ensure()
    from polymer_engine.core.models import Objective
    from polymer_engine.orchestrator.strategy import StrategyRegistry

    with get_store(config) as store:
        objective = Objective(title=title, description=description)
        store.save_objective(objective)
        StrategyRegistry().save(store)
        emit(
            {
                "initialised": True,
                "root": str(config.paths.root),
                "database": str(config.paths.resolved("database")),
                "objective_id": objective.id,
                "config_sources": config.source_files,
            }
        )


@engine_app.command("status")
def engine_status(config_path: str | None = ConfigOption) -> None:
    """Summarise what is in the engine's database."""
    config = get_config(config_path)
    database = config.paths.resolved("database")
    if not database.exists():
        fail("No engine database found", hint="run 'polymer-engine engine init' first")
    with get_store(config) as store:
        emit(
            {
                "database": str(database),
                "counts": store.summary(),
                "campaigns": [
                    {"id": c["id"], "status": c["status"], "updated_at": c["updated_at"]}
                    for c in store.list_campaigns()
                ],
            }
        )


@engine_app.command("config")
def engine_config(config_path: str | None = ConfigOption) -> None:
    """Print the resolved configuration with credentials redacted."""
    emit(get_config(config_path).redacted())


@engine_app.command("resources")
def engine_resources(
    config_path: str | None = ConfigOption,
    probe: bool = typer.Option(True, help="Run each tool to read its version"),
) -> None:
    """Report CPUs, memory, GPUs, and discovered tools."""
    from polymer_engine.local.resources import inspect_resources

    emit(inspect_resources(get_config(config_path), probe_tools=probe).as_dict())


@engine_app.command("discover-tools")
def engine_discover_tools(config_path: str | None = ConfigOption) -> None:
    """Locate gmx, orca, plumed and python, reporting versions and capabilities."""
    from polymer_engine.local.discovery import discover_all, find_all_installations

    config = get_config(config_path)
    statuses = discover_all(config)
    payload = {name: status.as_dict() for name, status in statuses.items()}
    for name in statuses:
        installations = find_all_installations(getattr(config.local_tools, name).executable)
        payload[name]["all_installations"] = installations
        if len(installations) > 1:
            payload[name]["warning"] = (
                "multiple installations found; pin one with local_tools."
                f"{name}.path for reproducibility"
            )
    emit(payload)
    if not statuses["gromacs"].usable:
        raise typer.Exit(EXIT_GATE_FAILED)


@engine_app.command("providers")
def engine_providers(
    config_path: str | None = ConfigOption,
    check: bool = typer.Option(False, "--check", help="Contact each provider (requires network)"),
) -> None:
    """List providers, their capabilities, and what they explicitly do not support."""
    from polymer_engine.providers.registry import ProviderRegistry

    config = get_config(config_path)
    registry = ProviderRegistry(config)
    payload: dict[str, Any] = {"status": registry.status(), "unsupported": registry.unsupported()}
    if check:
        payload["health"] = {name: result.as_dict() for name, result in registry.health().items()}
    emit(payload)


# ==========================================================================
# polymer
# ==========================================================================
@polymer_app.command("ingest")
def polymer_ingest(
    path: str = typer.Argument(..., help="CSV or JSONL dataset"),
    config_path: str | None = ConfigOption,
    output: str | None = typer.Option(None, "--output", "-o", help="Write records as JSONL"),
) -> None:
    """Ingest a polymer dataset, normalising identities, units and properties."""
    from polymer_engine.polymer.ingestion import ingest_csv, ingest_jsonl, save_records

    get_config(config_path)
    source = Path(path)
    if not source.exists():
        fail(f"Dataset not found: {source}", code=EXIT_USAGE)
    try:
        report = ingest_jsonl(source) if source.suffix == ".jsonl" else ingest_csv(source)
    except PolymerEngineError as exc:
        fail(str(exc))
    if output:
        save_records(report.records, output)
    emit({**report.summary(), "output": output})
    if not report.reconciles():
        raise typer.Exit(EXIT_ERROR)


@polymer_app.command("descriptors")
def polymer_descriptors(
    smiles: str = typer.Argument(..., help="Repeat-unit SMILES, e.g. '*CC*'"),
    config_path: str | None = ConfigOption,
) -> None:
    """Compute repeat-unit descriptors and classify the polymer family."""
    from polymer_engine.polymer.descriptors import compute_descriptors
    from polymer_engine.polymer.identity import make_identity
    from polymer_engine.polymer.taxonomy import classify

    get_config(config_path)
    try:
        identity = make_identity(name=smiles, repeat_unit_smiles=smiles)
        descriptors = compute_descriptors(smiles, polymer_id=identity.polymer_id)
        classification = classify(smiles)
    except PolymerEngineError as exc:
        fail(str(exc), hint="repeat units need attachment points, e.g. '*CC*'")
    emit(
        {
            "identity": identity.as_dict(),
            "family": classification.as_dict(),
            "descriptors": descriptors.as_dict(),
        }
    )


@polymer_app.command("cluster")
def polymer_cluster(
    path: str = typer.Argument(..., help="Dataset to group"),
    config_path: str | None = ConfigOption,
) -> None:
    """Group a dataset by classified polymer family."""
    from polymer_engine.polymer.ingestion import ingest_csv, ingest_jsonl
    from polymer_engine.polymer.taxonomy import group_by_family

    get_config(config_path)
    source = Path(path)
    if not source.exists():
        fail(f"Dataset not found: {source}", code=EXIT_USAGE)
    report = ingest_jsonl(source) if source.suffix == ".jsonl" else ingest_csv(source)
    groups = group_by_family([(r.polymer_id, r.repeat_unit_smiles) for r in report.records])
    emit(
        {
            "n_records": report.n_accepted,
            "families": {family.value: ids for family, ids in sorted(groups.items(), key=lambda kv: kv[0].value)},
        }
    )


# ==========================================================================
# provider / charmm-gui
# ==========================================================================
@charmm_app.command("login")
def charmm_login(config_path: str | None = ConfigOption) -> None:
    """Authenticate against CHARMM-GUI and cache a JWT for this process."""

    config = get_config(config_path)
    provider = _charmm_gui(config)
    result = provider.login()
    emit(result.as_dict())
    if not result.ok:
        raise typer.Exit(EXIT_ERROR)


@charmm_app.command("status")
def charmm_status(
    job_id: str = typer.Argument(..., help="CHARMM-GUI job id"),
    config_path: str | None = ConfigOption,
) -> None:
    """Query a CHARMM-GUI job's status."""

    config = get_config(config_path)
    result = _charmm_gui(config).job_status(job_id)
    emit(result.as_dict())
    if not result.ok:
        raise typer.Exit(EXIT_ERROR)


@charmm_app.command("download")
def charmm_download(
    job_id: str = typer.Argument(..., help="CHARMM-GUI job id"),
    destination: str = typer.Argument(..., help="Where to write the .tgz archive"),
    config_path: str | None = ConfigOption,
) -> None:
    """Download a finished CHARMM-GUI job archive and verify it."""

    config = get_config(config_path)
    result = _charmm_gui(config).download_job(job_id, destination)
    emit(result.as_dict())
    if not result.ok:
        raise typer.Exit(EXIT_ERROR)


@charmm_app.command("spec")
def charmm_spec(
    polymer: str = typer.Argument(..., help="Repeat-unit SMILES, or a name in the dataset"),
    force_field: str = typer.Option(..., "--force-field", help="e.g. 'CHARMM36 + CGenFF'"),
    degree_of_polymerization: int = typer.Option(..., "--dp", help="Repeat units per chain"),
    n_chains: int = typer.Option(..., "--chains", help="Chains in the box"),
    temperature: float = typer.Option(..., "--temperature", help="Kelvin"),
    pressure: float = typer.Option(1.0, "--pressure", help="bar"),
    density: float | None = typer.Option(None, "--density", help="Target density kg/m^3, to size the box"),
    dataset: str | None = typer.Option(None, "--dataset", help="CSV to resolve a polymer name against"),
    output: str | None = typer.Option(None, "--output", "-o", help="Directory for the spec files"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    """Write the specification for a CHARMM-GUI Polymer Builder job.

    The engine cannot submit the job -- CHARMM-GUI publishes no submission endpoint --
    so this produces the exact inputs a person needs, plus a fingerprint that makes the
    manual step reproducible.
    """
    from polymer_engine.polymer.records import build_record
    from polymer_engine.simulation.charmm_gui_spec import spec_from_record, write_spec

    record = None
    if dataset:
        from polymer_engine.polymer.ingestion import ingest_csv

        wanted = polymer.strip().lower()
        for candidate in ingest_csv(dataset).records:
            if wanted in (candidate.name.lower(), candidate.polymer_id.lower()):
                record = candidate
                break
        if record is None:
            fail(f"No polymer matching {polymer!r} in {dataset}", code=EXIT_USAGE)
    else:
        try:
            record = build_record(name=polymer, repeat_unit_smiles=polymer,
                                  properties={}, source="cli")
        except PolymerEngineError as exc:
            fail(f"Could not read {polymer!r} as a repeat unit: {exc}", code=EXIT_USAGE,
                 hint="pass --dataset to resolve a polymer by name instead")

    try:
        spec = spec_from_record(
            record, force_field=force_field,
            degree_of_polymerization=degree_of_polymerization, n_chains=n_chains,
            temperature_k=temperature, pressure_bar=pressure,
            target_density_kg_m3=density, notes=notes,
        )
    except PolymerEngineError as exc:
        fail(str(exc), code=EXIT_USAGE)

    payload = spec.as_dict()
    if output:
        json_path, brief = write_spec(spec, output)
        payload["written"] = {"spec": str(json_path), "instructions": str(brief)}
    emit(payload)


@charmm_app.command("penalties")
def charmm_penalties(
    path: str = typer.Argument(..., help="A CHARMM stream file, or a directory to search"),
    max_penalty: float | None = typer.Option(
        None, "--max-penalty",
        help="Your accepted tolerance. Without it, anything above the CGenFF 'good' "
             "tier is INCONCLUSIVE rather than judged.",
    ),
) -> None:
    """Read CGenFF penalty scores and judge the parameter set.

    Exits 3 when the penalties do not permit promotion, so a pipeline can gate on it.
    """
    from polymer_engine.simulation.cgenff import (
        find_stream_files,
        parse_stream_file,
        penalty_gates,
    )

    target = Path(path)
    if not target.exists():
        fail(f"No such path: {path}", code=EXIT_USAGE)

    candidates = [target] if target.is_file() else find_stream_files(target)
    if not candidates:
        fail(f"No .str/.prm/.rtf files under {path}", code=EXIT_USAGE,
             hint="point at the toppar directory of an extracted CHARMM-GUI job")

    reports, worst_report = [], None
    for candidate in candidates:
        try:
            report = parse_stream_file(candidate)
        except PolymerEngineError:
            continue          # not a CGenFF stream file; other files legitimately are not
        reports.append(report)
        if worst_report is None or report.max_penalty > worst_report.max_penalty:
            worst_report = report
    if worst_report is None:
        fail(f"None of the {len(candidates)} file(s) under {path} is a CGenFF stream file",
             code=EXIT_USAGE)

    gates = penalty_gates(worst_report, max_penalty=max_penalty)
    emit({
        "files_examined": len(candidates),
        "stream_files_parsed": len(reports),
        "worst_file": worst_report.path,
        "penalties": worst_report.as_dict(),
        "gate_status": gates.status.value,
        "promotable": gates.promotable,
        "gates": [g.model_dump(mode="json") for g in gates.gates],
    })
    if not gates.promotable:
        raise typer.Exit(EXIT_GATE_FAILED)


@charmm_app.command("import")
def charmm_import(
    job_id: str | None = typer.Argument(None, help="CHARMM-GUI job id to download"),
    archive: str | None = typer.Option(None, "--archive", help="A .tgz you downloaded yourself"),
    polymer_id: str = typer.Option(..., "--polymer", help="Polymer this system is for"),
    force_field: str = typer.Option(..., "--force-field"),
    destination: str = typer.Option("systems", "--destination", "-d"),
    max_penalty: float | None = typer.Option(None, "--max-penalty",
                                             help="Accepted CGenFF penalty tolerance"),
    config_path: str | None = ConfigOption,
) -> None:
    """Import a finished CHARMM-GUI job, validate it, and read its CGenFF penalties.

    Exits 3 when the system or its parameters do not qualify.
    """
    from polymer_engine.simulation.builder import SystemBuildRequest, default_builder
    from polymer_engine.simulation.cgenff import find_stream_files, parse_stream_file, penalty_gates

    if not job_id and not archive:
        fail("Give a job id or --archive", code=EXIT_USAGE)
    config = get_config(config_path)
    provider = _charmm_gui(config) if job_id else None

    result = default_builder(provider).build(
        SystemBuildRequest(
            polymer_id=polymer_id, force_field=force_field,
            external_job_id=job_id, source_archive=archive,
        ),
        destination,
    )
    payload = result.as_dict()

    # Penalties are read whatever the build said, because "the system validated" and
    # "the parameters are trustworthy" are different questions.
    penalties: dict[str, Any] = {"status": "no CGenFF stream file found"}
    promotable = result.usable
    root = Path(destination)
    if root.is_dir():
        worst = None
        for candidate in find_stream_files(root):
            try:
                report = parse_stream_file(candidate)
            except PolymerEngineError:
                continue
            if worst is None or report.max_penalty > worst.max_penalty:
                worst = report
        if worst is not None:
            gates = penalty_gates(worst, max_penalty=max_penalty)
            penalties = {
                "file": worst.path, "summary": worst.as_dict(),
                "gate_status": gates.status.value, "promotable": gates.promotable,
                "gates": [g.model_dump(mode="json") for g in gates.gates],
            }
            promotable = promotable and gates.promotable
    payload["cgenff_penalties"] = penalties
    payload["promotable"] = promotable
    emit(payload)
    if not promotable:
        raise typer.Exit(EXIT_GATE_FAILED)


@provider_app.command("search")
def provider_search(
    provider: str = typer.Argument(..., help="Provider name, e.g. crossref"),
    query: str = typer.Argument(..., help="Search query"),
    rows: int = typer.Option(10, help="Maximum records"),
    config_path: str | None = ConfigOption,
) -> None:
    """Search a literature or structure provider."""
    from polymer_engine.providers.registry import ProviderRegistry

    config = get_config(config_path)
    registry = ProviderRegistry(config)
    try:
        client = registry.get(provider)
    except PolymerEngineError as exc:
        fail(str(exc), code=EXIT_USAGE, hint=f"known providers: {', '.join(registry.names())}")
    method = getattr(client, "search", None) or getattr(client, "search_text", None)
    if not callable(method):
        fail(f"{provider} does not support searching", code=EXIT_USAGE)
    result = method(query, rows)
    emit(result.as_dict())
    if not result.ok:
        raise typer.Exit(EXIT_ERROR)


# ==========================================================================
# system
# ==========================================================================
@system_app.command("import")
def system_import(
    source: str = typer.Argument(..., help="Archive (.tgz/.zip) or directory"),
    destination: str | None = typer.Option(None, "--destination", "-d", help="Where to extract"),
    config_path: str | None = ConfigOption,
) -> None:
    """Import a simulation system, extracting safely and validating it."""
    from polymer_engine.simulation.system import import_archive, import_directory

    config = get_config(config_path)
    path = Path(source)
    if not path.exists():
        fail(f"Source not found: {path}", code=EXIT_USAGE)

    with get_store(config) as store:
        graph = store.load_provenance_graph()
        try:
            if path.is_dir():
                imported = import_directory(path, graph=graph)
            else:
                target = Path(destination) if destination else config.paths.resolved("data_dir") / path.stem
                imported = import_archive(
                    path, target, graph=graph,
                    max_total_bytes=config.safety.max_archive_bytes,
                    max_members=config.safety.max_archive_members,
                )
        except PolymerEngineError as exc:
            fail(str(exc), hint="the archive may be corrupt or contain unsafe paths")
        store.save_provenance_graph(graph)
    emit(imported.as_dict())
    if not imported.usable:
        raise typer.Exit(EXIT_GATE_FAILED)


@system_app.command("validate")
def system_validate(
    path: str = typer.Argument(..., help="System directory"),
    config_path: str | None = ConfigOption,
    require_box: bool = typer.Option(True, help="Require box vectors in the coordinate file"),
) -> None:
    """Validate a system directory against every structural gate."""
    from polymer_engine.simulation.system import validate_system

    get_config(config_path)
    if not Path(path).is_dir():
        fail(f"Not a directory: {path}", code=EXIT_USAGE)
    report = validate_system(path, require_box=require_box)
    emit(
        {
            "status": report.status.value,
            "promotable": report.promotable,
            "summary": report.summary(),
            "gates": [g.model_dump(mode="json") for g in report.gates],
        }
    )
    if not report.promotable:
        raise typer.Exit(EXIT_GATE_FAILED)


# ==========================================================================
# campaign
# ==========================================================================
@campaign_app.command("create")
def campaign_create(
    campaign_id: str = typer.Argument(..., help="Campaign identifier"),
    polymer_id: str = typer.Option(..., "--polymer", help="Polymer identifier"),
    system: str = typer.Option(..., "--system", help="Validated system directory"),
    question: str = typer.Option("", "--question", help="The scientific question"),
    config_path: str | None = ConfigOption,
) -> None:
    """Create a campaign and attach a validated system to it."""
    from polymer_engine.orchestrator.campaign import CampaignBuilder, CampaignSpec, save_campaign
    from polymer_engine.simulation.system import import_directory

    config = get_config(config_path)
    config.paths.ensure()
    if not Path(system).is_dir():
        fail(f"System directory not found: {system}", code=EXIT_USAGE)

    with get_store(config) as store:
        graph = store.load_provenance_graph()
        builder = CampaignBuilder(config, graph=graph, software=_software_versions(config))
        spec = CampaignSpec.from_config(
            config, campaign_id=campaign_id, polymer_id=polymer_id, question=question,
            system_source=str(system),
        )
        campaign = builder.create(spec)
        builder.attach_system(campaign, import_directory(system, graph=graph))
        store.save_provenance_graph(graph)
        save_campaign(store, campaign)
    emit(
        {
            "campaign_id": campaign_id,
            "status": campaign.status.value,
            "workdir": str(campaign.workdir),
            "system_validation": campaign.system_report.summary() if campaign.system_report else None,
            "blocked_reason": campaign.blocked_reason,
        }
    )
    if campaign.status.value == "blocked":
        raise typer.Exit(EXIT_GATE_FAILED)


@campaign_app.command("plan")
def campaign_plan(
    campaign_id: str = typer.Argument(...),
    config_path: str | None = ConfigOption,
) -> None:
    """Materialise replicas and build the action graph."""
    from polymer_engine.orchestrator.campaign import CampaignBuilder, load_campaign, save_campaign
    from polymer_engine.simulation.system import import_directory

    config = get_config(config_path)
    with get_store(config) as store:
        campaign = load_campaign(store, campaign_id)
        if campaign is None:
            fail(f"Unknown campaign: {campaign_id}", code=EXIT_USAGE, hint="run 'campaign create' first")
        graph = store.load_provenance_graph()
        builder = CampaignBuilder(config, graph=graph, software=_software_versions(config))
        if campaign.system is None and campaign.spec.system_source:
            builder.attach_system(campaign, import_directory(campaign.spec.system_source, graph=graph))
        try:
            builder.plan(campaign, gromacs_version=_gromacs_version(config))
        except PolymerEngineError as exc:
            fail(str(exc))
        store.save_provenance_graph(graph)
        save_campaign(store, campaign)
    emit(
        {
            "campaign_id": campaign_id,
            "status": campaign.status.value,
            "replicas": campaign.replicas.n_replicas if campaign.replicas else 0,
            "seeds_distinct": campaign.replicas.seeds_are_distinct() if campaign.replicas else None,
            "actions": [{"id": a.id, "kind": a.kind, "title": a.title} for a in campaign.actions],
            "manifest": str(campaign.workdir / "campaign_manifest.json"),
        }
    )


@campaign_app.command("run")
def campaign_run(
    campaign_id: str = typer.Argument(...),
    config_path: str | None = ConfigOption,
    execute: bool = typer.Option(False, "--execute", help="Actually run local scientific software"),
    max_actions: int = typer.Option(0, "--max-actions", help="Stop after this many actions (0 = all)"),
) -> None:
    """Run a campaign's actions, choosing each one through the planner."""
    from polymer_engine.orchestrator.runner import CampaignRunner

    config = get_config(config_path)
    if execute:
        config.safety.execution_enabled = True
    with get_store(config) as store:
        runner = CampaignRunner(config, store)
        try:
            summary = runner.run(campaign_id, max_actions=max_actions or None)
        except PolymerEngineError as exc:
            fail(str(exc))
    emit(summary)
    if summary.get("gate_status") == "fail":
        raise typer.Exit(EXIT_GATE_FAILED)


@campaign_app.command("status")
def campaign_status(
    campaign_id: str = typer.Argument(...),
    config_path: str | None = ConfigOption,
) -> None:
    """Show a campaign's state, actions, and decision history."""
    from polymer_engine.orchestrator.campaign import load_campaign

    config = get_config(config_path)
    with get_store(config) as store:
        campaign = load_campaign(store, campaign_id)
        if campaign is None:
            fail(f"Unknown campaign: {campaign_id}", code=EXIT_USAGE)
        emit(
            {
                "campaign_id": campaign_id,
                "status": campaign.status.value,
                "actions": [
                    {"id": a.id, "kind": a.kind, "status": a.status.value} for a in campaign.actions
                ],
                "execution_records": [
                    {"id": r.id, "kind": r.kind, "state": r.state.value, "attempt": r.attempt}
                    for r in campaign.records
                ],
                "decisions": store.list_decisions(campaign_id=campaign_id),
            }
        )


@campaign_app.command("analyze")
def campaign_analyze(
    campaign_id: str = typer.Argument(...),
    config_path: str | None = ConfigOption,
) -> None:
    """Re-run replica convergence and agreement analysis for a campaign."""
    from polymer_engine.core.models import Action
    from polymer_engine.executors.registry import ExecutorRegistry
    from polymer_engine.orchestrator.campaign import load_campaign

    config = get_config(config_path)
    with get_store(config) as store:
        campaign = load_campaign(store, campaign_id)
        if campaign is None:
            fail(f"Unknown campaign: {campaign_id}", code=EXIT_USAGE)
        replica_root = campaign.workdir / "replicas"
        replica_dirs = sorted(str(p) for p in replica_root.glob("replica_*")) if replica_root.is_dir() else []
        if not replica_dirs:
            fail("No replica directories found", hint="run 'campaign plan' and 'campaign run' first")
        action = Action(
            kind="analyze_replicas",
            title="Analyse replicas",
            campaign_id=campaign_id,
            inputs={"workdir": str(campaign.workdir), "replica_dirs": replica_dirs},
        )
        result = ExecutorRegistry(config).get("analyze_replicas").run(action)
        for observation in result.observations:
            store.save_observation(observation)
    emit(
        {
            "campaign_id": campaign_id,
            "status": result.status.value,
            "gate_status": result.report.status.value,
            "summary": result.report.summary(),
            "gates": [g.model_dump(mode="json") for g in result.report.gates],
        }
    )
    if not result.report.promotable:
        raise typer.Exit(EXIT_GATE_FAILED)


def _charmm_gui(config: EngineConfig):
    """Fetch the CHARMM-GUI provider with its concrete type."""
    from polymer_engine.providers.charmm_gui import CHARMMGUIProvider
    from polymer_engine.providers.registry import ProviderRegistry

    provider = ProviderRegistry(config).get("charmm_gui")
    assert isinstance(provider, CHARMMGUIProvider)
    return provider


def _software_versions(config: EngineConfig) -> dict[str, str]:
    from polymer_engine.local.discovery import discover_all

    return {
        name: status.version or "unknown"
        for name, status in discover_all(config, probe=True).items()
        if status.found
    }


def _gromacs_version(config: EngineConfig) -> str | None:
    from polymer_engine.local.discovery import discover_tool

    return discover_tool("gromacs", config.local_tools.gromacs).version


def main() -> None:
    try:
        app()
    except PolymerEngineError as exc:  # pragma: no cover - top-level safety net
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        sys.exit(EXIT_ERROR)




# ==========================================================================
# umbrella
# ==========================================================================
@umbrella_app.command("plan")
def umbrella_plan(
    output: str = typer.Argument(..., help="Directory to write windows into"),
    minimum: float = typer.Option(..., "--min", help="Lower bound of the reaction coordinate (nm)"),
    maximum: float = typer.Option(..., "--max", help="Upper bound of the reaction coordinate (nm)"),
    group_a: str = typer.Option("1-100", "--group-a", help="PLUMED atom selection for group A"),
    group_b: str = typer.Option("101-200", "--group-b", help="PLUMED atom selection for group B"),
    justification: str = typer.Option(..., "--justification", help="Why this coordinate is physically meaningful"),
    config_path: str | None = ConfigOption,
    strict: bool = typer.Option(False, "--strict", help="Refuse a spacing that cannot overlap"),
) -> None:
    """Plan umbrella windows, checking the spacing can actually produce overlap."""
    from polymer_engine.simulation.umbrella import ReactionCoordinate, plan_windows, write_windows

    config = get_config(config_path)
    coordinate = ReactionCoordinate(
        name="d", kind="com_distance", units="nm",
        justification=justification, group_a=group_a, group_b=group_b,
    )
    try:
        plan = plan_windows(
            coordinate, minimum=minimum, maximum=maximum,
            temperature_k=config.simulation.temperature_k,
            defaults=config.umbrella, strict=strict,
        )
    except PolymerEngineError as exc:
        fail(str(exc), hint="reduce the spacing or raise the force constant")
    write_windows(plan, output)
    emit({**plan.as_dict(), "output": output})
    if not plan.expected_to_overlap:
        raise typer.Exit(EXIT_GATE_FAILED)


@umbrella_app.command("analyze")
def umbrella_analyze(
    windows_dir: str = typer.Argument(..., help="Directory holding window_*/COLVAR files"),
    config_path: str | None = ConfigOption,
    output: str | None = typer.Option(None, "--output", "-o", help="Write the PMF as JSON"),
) -> None:
    """Build a PMF from window COLVAR files, with overlap and convergence checks."""

    from polymer_engine.analysis.free_energy import WindowSamples, build_pmf, pmf_gates, trim_equilibration

    config = get_config(config_path)
    root = Path(windows_dir)
    if not root.is_dir():
        fail(f"Not a directory: {root}", code=EXIT_USAGE)

    windows: list[WindowSamples] = []
    for directory in sorted(root.glob("window_*")):
        meta_path = directory / "window.json"
        colvar = directory / "COLVAR"
        if not meta_path.exists() or not colvar.exists():
            continue
        meta = json.loads(meta_path.read_text())
        values = _read_colvar(colvar)
        trimmed, discarded = trim_equilibration(values, config.umbrella.equilibration_fraction)
        windows.append(
            WindowSamples(
                index=int(meta["index"]), center=float(meta["center"]),
                force_constant=float(meta["force_constant"]), values=trimmed,
                units=meta.get("units", "nm"), discarded=discarded,
            )
        )
    if len(windows) < 2:
        fail(
            f"Found {len(windows)} usable window(s)",
            hint="each window needs window.json and a COLVAR file",
        )

    try:
        result = build_pmf(
            windows, temperature_k=config.simulation.temperature_k,
            defaults=config.umbrella, bootstrap_samples=config.umbrella.bootstrap_samples,
            seed=config.analysis.random_seed,
        )
    except PolymerEngineError as exc:
        fail(str(exc))
    report = pmf_gates(result, defaults=config.umbrella)
    payload = {
        "trustworthy": result.trustworthy,
        "determination": result.determination.value,
        "problems": result.problems,
        "barrier": result.barrier().model_dump(mode="json"),
        "well_depth": result.well_depth().model_dump(mode="json"),
        "gate_status": report.status.value,
        "gates": [g.model_dump(mode="json") for g in report.gates],
    }
    if output:
        Path(output).write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        payload["output"] = output
    emit(payload)
    if not result.trustworthy:
        raise typer.Exit(EXIT_GATE_FAILED)


def _read_colvar(path: Path):
    """Read the CV column from a PLUMED COLVAR file."""
    import numpy as np

    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue
    if not values:
        fail(f"No numeric data in {path}")
    return np.asarray(values, dtype=float)


# ==========================================================================
# model
# ==========================================================================
@model_app.command("train")
def model_train(
    dataset: str = typer.Argument(..., help="Polymer dataset (CSV or JSONL)"),
    target: str = typer.Option(..., "--target", help="Property to model, e.g. glass_transition_temperature"),
    config_path: str | None = ConfigOption,
    output: str | None = typer.Option(None, "--output", "-o", help="Write the evaluation report"),
) -> None:
    """Fit a surrogate model and report grouped cross-validated performance."""
    from polymer_engine.discovery.qspr import QsprModel, cross_validate

    config = get_config(config_path)
    dataset_obj, skipped = _build_dataset(dataset, target)
    try:
        evaluation = cross_validate(dataset_obj, seed=config.analysis.random_seed)
        model = QsprModel(random_state=config.analysis.random_seed).fit(dataset_obj)
    except PolymerEngineError as exc:
        fail(str(exc))
    payload = {
        "target": target,
        "n_training_polymers": dataset_obj.n_samples,
        "n_features": dataset_obj.n_features,
        "skipped_records": skipped,
        "cross_validation": evaluation.as_dict(),
        "feature_importance": model.feature_importance(),
    }
    if output:
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output"] = output
    emit(payload)


@model_app.command("evaluate")
def model_evaluate(
    dataset: str = typer.Argument(...),
    target: str = typer.Option(..., "--target"),
    folds: int = typer.Option(5, "--folds"),
    config_path: str | None = ConfigOption,
) -> None:
    """Cross-validate a surrogate without keeping the fitted model."""
    from polymer_engine.discovery.qspr import cross_validate

    config = get_config(config_path)
    dataset_obj, skipped = _build_dataset(dataset, target)
    evaluation = cross_validate(dataset_obj, n_folds=folds, seed=config.analysis.random_seed)
    emit(
        {
            "target": target,
            "skipped_records": skipped,
            "duplicate_polymer_ids": dataset_obj.duplicate_ids(),
            "cross_validation": evaluation.as_dict(),
        }
    )
    if evaluation.r2 is None:
        raise typer.Exit(EXIT_GATE_FAILED)


def _build_dataset(path: str, target: str):
    """Assemble a modelling dataset from an ingested polymer file."""
    import numpy as np

    from polymer_engine.discovery.qspr import Dataset
    from polymer_engine.polymer.descriptors import DESCRIPTOR_UNITS
    from polymer_engine.polymer.ingestion import ingest_csv, ingest_jsonl
    from polymer_engine.polymer.normalization import PROPERTIES, normalize_property_name

    source = Path(path)
    if not source.exists():
        fail(f"Dataset not found: {source}", code=EXIT_USAGE)
    canonical_or_none = normalize_property_name(target)
    if canonical_or_none is None:
        fail(
            f"Unknown property {target!r}", code=EXIT_USAGE,
            hint=f"known properties: {', '.join(sorted(PROPERTIES))}",
        )
    canonical = canonical_or_none
    report = ingest_jsonl(source) if source.suffix == ".jsonl" else ingest_csv(source)

    names = sorted(DESCRIPTOR_UNITS)
    ids, rows, targets, skipped = [], [], [], []
    for record in report.records:
        value = record.property_value(canonical)
        if value is None:
            skipped.append({"polymer_id": record.polymer_id, "reason": f"no {canonical} value"})
            continue
        if not record.usable_for_modelling:
            skipped.append({"polymer_id": record.polymer_id, "reason": "record needs curation review"})
            continue
        # descriptor_values returns a {name: value} mapping; index it by `names` so the
        # column order matches feature_names exactly.
        features = record.descriptor_values(names)
        ids.append(record.polymer_id)
        rows.append([np.nan if features[name] is None else features[name] for name in names])
        targets.append(value)

    if len(ids) < 5:
        fail(
            f"Only {len(ids)} usable record(s) for {canonical}",
            hint="ingest more data, or check that the property column is recognised",
        )
    spec = PROPERTIES[canonical]
    return (
        Dataset(
            polymer_ids=ids, feature_names=names, X=np.asarray(rows, dtype=float),
            y=np.asarray(targets, dtype=float), target_name=canonical, target_units=spec.canonical_unit,
        ),
        skipped,
    )


# ==========================================================================
# design
# ==========================================================================
@design_app.command("generate")
def design_generate(
    parent: str = typer.Argument(..., help="Parent repeat-unit SMILES, e.g. '*CC*'"),
    config_path: str | None = ConfigOption,
    max_candidates: int = typer.Option(30, "--max", help="Maximum candidates to emit"),
    output: str | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Generate rationally mutated candidates and validate each one."""
    from polymer_engine.discovery.candidates import CandidateStatus, generate_candidates

    get_config(config_path)
    try:
        generated = generate_candidates(parent, max_candidates=max_candidates)
    except PolymerEngineError as exc:
        fail(str(exc), hint="the parent must be a repeat unit with two attachment points")
    payload = {
        "parent": parent,
        "n_generated": len(generated),
        "counts": {
            status.value: sum(1 for c in generated if c.status is status) for status in CandidateStatus
        },
        "candidates": [c.as_dict() for c in generated],
    }
    if output:
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output"] = output
    emit(payload)


@design_app.command("rank")
def design_rank(
    dataset: str = typer.Argument(..., help="Training dataset"),
    parent: str = typer.Option(..., "--parent", help="Parent repeat-unit SMILES"),
    target: str = typer.Option(..., "--target", help="Property to optimise"),
    objective: str = typer.Option("maximise", "--objective", help="maximise or minimise"),
    batch_size: int = typer.Option(5, "--batch-size"),
    config_path: str | None = ConfigOption,
) -> None:
    """Generate candidates, score them against a surrogate, and rank a diverse batch."""
    import numpy as np

    from polymer_engine.discovery.active_learning import Candidate, select_next_experiments
    from polymer_engine.discovery.candidates import CandidateStatus, generate_candidates
    from polymer_engine.discovery.qspr import QsprModel
    from polymer_engine.polymer.descriptors import DESCRIPTOR_UNITS

    config = get_config(config_path)
    if objective not in {"maximise", "minimise"}:
        fail("objective must be 'maximise' or 'minimise'", code=EXIT_USAGE)
    dataset_obj, _ = _build_dataset(dataset, target)
    try:
        model = QsprModel(random_state=config.analysis.random_seed).fit(dataset_obj)
    except PolymerEngineError as exc:
        fail(str(exc))

    names = sorted(DESCRIPTOR_UNITS)
    candidates = []
    for generated in generate_candidates(parent, max_candidates=50):
        if generated.status is CandidateStatus.INVALID:
            continue
        features = [generated.descriptors.get(n, np.nan) for n in names]
        candidates.append(
            Candidate(
                polymer_id=generated.polymer_id,
                name=generated.description,
                features=np.asarray(features, dtype=float),
                metadata={"smiles": generated.repeat_unit_smiles, "status": generated.status.value},
            )
        )
    if not candidates:
        fail("No usable candidates were generated")

    report = select_next_experiments(
        candidates, model, dataset_obj.X, batch_size=batch_size,
        objective="maximise" if objective == "maximise" else "minimise",
    )
    emit(
        {
            "parent": parent,
            "target": target,
            "objective": objective,
            **report.as_dict(),
        }
    )


# ==========================================================================
# evidence
# ==========================================================================
@evidence_app.command("report")
def evidence_report(config_path: str | None = ConfigOption) -> None:
    """Report every claim and its evidential standing."""
    from polymer_engine.evidence.claims import ClaimLedger

    config = get_config(config_path)
    with get_store(config) as store:
        emit(ClaimLedger.load(store).report())


@evidence_app.command("strategies")
def evidence_strategies(config_path: str | None = ConfigOption) -> None:
    """Show the strategy registry and each strategy's track record."""
    from polymer_engine.orchestrator.strategy import StrategyRegistry

    config = get_config(config_path)
    with get_store(config) as store:
        registry = StrategyRegistry.load(store)
        emit(
            {
                "strategies": [
                    {**strategy.as_dict(), "score": score} for strategy, score in registry.ranked()
                ]
            }
        )


@evidence_app.command("manifest")
def evidence_manifest(
    campaign_id: str = typer.Argument(...),
    config_path: str | None = ConfigOption,
    output: str | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Emit a campaign's reproducibility manifest."""
    from polymer_engine.orchestrator.campaign import load_campaign

    config = get_config(config_path)
    with get_store(config) as store:
        campaign = load_campaign(store, campaign_id)
        if campaign is None:
            fail(f"Unknown campaign: {campaign_id}", code=EXIT_USAGE)
        stored = store.get_campaign(campaign_id) or {}
    manifest = stored or campaign.manifest()
    if output:
        Path(output).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        emit({"campaign_id": campaign_id, "output": output, "fingerprint": manifest.get("fingerprint")})
    else:
        emit(manifest)


@evidence_app.command("lineage")
def evidence_lineage(
    artifact_id: str = typer.Argument(..., help="Artifact to trace"),
    config_path: str | None = ConfigOption,
) -> None:
    """Trace an artifact back to the inputs it came from."""
    config = get_config(config_path)
    with get_store(config) as store:
        graph = store.load_provenance_graph()
        if artifact_id not in graph:
            fail(f"Unknown artifact: {artifact_id}", code=EXIT_USAGE)
        emit(
            {
                "artifact_id": artifact_id,
                "ancestors": graph.ancestors(artifact_id),
                "roots": graph.roots(artifact_id),
                "descendants": graph.descendants(artifact_id),
                "lineage": [a.model_dump(mode="json") for a in graph.lineage(artifact_id)],
                "verification": graph.verify_all(),
            }
        )


# ==========================================================================
# qm
# ==========================================================================
@qm_app.command("run")
def qm_run(
    xyz: str = typer.Argument(..., help="Structure file in XYZ format"),
    kind: str = typer.Option("single_point", "--kind", help="single_point, optimization, frequency, opt_freq, torsion_scan"),
    method: str = typer.Option(..., "--method", help="e.g. B3LYP, HF, wB97X-D3"),
    basis: str = typer.Option(..., "--basis", help="e.g. def2-SVP, STO-3G"),
    charge: int = typer.Option(0, "--charge"),
    multiplicity: int = typer.Option(1, "--multiplicity"),
    workdir: str = typer.Option("qm", "--workdir", "-w"),
    torsion: str | None = typer.Option(None, "--torsion", help="Four zero-based atom indices, e.g. '2,0,1,6'"),
    scan_range: str = typer.Option("0,360,13", "--scan-range", help="start,stop,n_points for a torsion scan"),
    solvent: str | None = typer.Option(None, "--solvent"),
    solvation_model: str | None = typer.Option(None, "--solvation-model", help="CPCM or SMD"),
    nprocs: int = typer.Option(1, "--nprocs"),
    execute: bool = typer.Option(False, "--execute", help="Actually run ORCA"),
    config_path: str | None = ConfigOption,
) -> None:
    """Run a quantum-chemistry job.

    The method and basis set are required: the engine will not choose a level of theory
    for you, because a wrong functional produces a plausible number.
    """
    from polymer_engine.qm.orca_runner import build_orca_runner
    from polymer_engine.qm.spec import JobKind, QMJobSpec, Structure, TorsionSpec
    from polymer_engine.qm.validation import validate_qm_run

    config = get_config(config_path)
    if execute:
        config.safety.execution_enabled = True

    source = Path(xyz)
    if not source.exists():
        fail(f"Structure file not found: {source}", code=EXIT_USAGE)
    try:
        structure = Structure.from_xyz(source.read_text(), charge=charge, multiplicity=multiplicity)
    except PolymerEngineError as exc:
        fail(str(exc), code=EXIT_USAGE)

    try:
        job_kind = JobKind(kind)
    except ValueError:
        fail(
            f"Unknown job kind {kind!r}", code=EXIT_USAGE,
            hint=f"one of: {', '.join(k.value for k in JobKind)}",
        )

    torsion_spec = None
    if torsion:
        try:
            indices = [int(v) for v in torsion.split(",")]
            if len(indices) != 4:
                fail(
                    f"A torsion needs exactly four atom indices, got {len(indices)}",
                    code=EXIT_USAGE, hint="for example --torsion '2,0,1,6'",
                )
            start, stop, points = (float(v) for v in scan_range.split(","))
            torsion_spec = TorsionSpec(
                atoms=(indices[0], indices[1], indices[2], indices[3]),
                start_deg=start, stop_deg=stop, n_points=int(points),
            )
        except (ValueError, TypeError) as exc:
            fail(f"Could not parse the torsion specification: {exc}", code=EXIT_USAGE)

    try:
        spec = QMJobSpec(
            kind=job_kind, structure=structure, method=method, basis=basis,
            label=source.stem, torsion=torsion_spec, n_procs=nprocs,
            solvent=solvent, solvation_model=solvation_model,  # type: ignore[arg-type]
        )
    except PolymerEngineError as exc:
        fail(str(exc), code=EXIT_USAGE)

    with get_store(config) as store:
        graph = store.load_provenance_graph()
        run = build_orca_runner(config).run_job(spec, workdir, graph=graph)
        store.save_provenance_graph(graph)

    validation = validate_qm_run(spec, run.output)
    emit({**run.as_dict(), "validation": validation.as_dict()})
    if not run.succeeded:
        raise typer.Exit(EXIT_GATE_FAILED if run.executed else EXIT_ERROR)


@qm_app.command("parse")
def qm_parse(
    output: str = typer.Argument(..., help="ORCA output file (.out or .out.gz)"),
    expect_geometry: bool = typer.Option(False, "--expect-geometry"),
    expect_frequencies: bool = typer.Option(False, "--expect-frequencies"),
    config_path: str | None = ConfigOption,
) -> None:
    """Parse an existing ORCA output and classify what actually happened."""
    from polymer_engine.qm.orca_parser import parse_orca_file

    get_config(config_path)
    if not Path(output).exists():
        fail(f"Output file not found: {output}", code=EXIT_USAGE)
    try:
        result = parse_orca_file(
            output, expect_geometry=expect_geometry, expect_frequencies=expect_frequencies
        )
    except PolymerEngineError as exc:
        fail(str(exc))
    emit(result.as_dict())
    if not result.succeeded:
        raise typer.Exit(EXIT_GATE_FAILED)


# ==========================================================================
# property
# ==========================================================================
@property_app.command("list")
def property_list(config_path: str | None = ConfigOption) -> None:
    """List every property the engine can compute, with its declared contract."""
    from polymer_engine.properties import default_registry

    config = get_config(config_path)
    emit(default_registry(config.analysis).definitions())


@property_app.command("compute")
def property_compute(
    name: str = typer.Argument(..., help="Property name, e.g. density"),
    xvg: str = typer.Argument(..., help="GROMACS .xvg file holding the time series"),
    replicas: int = typer.Option(1, "--replicas", help="How many independent replicas this represents"),
    simulation_ns: float | None = typer.Option(None, "--simulation-ns"),
    config_path: str | None = ConfigOption,
) -> None:
    """Compute a time-series property from a GROMACS .xvg, with its sampling gates."""
    from polymer_engine.properties import default_registry
    from polymer_engine.properties.thermodynamic import TimeSeriesProperty
    from polymer_engine.simulation.formats import read_xvg

    config = get_config(config_path)
    registry = default_registry(config.analysis)
    try:
        calculator = registry.get(name)
    except PolymerEngineError as exc:
        fail(str(exc), code=EXIT_USAGE, hint=f"known properties: {', '.join(registry.names())}")
    if not isinstance(calculator, TimeSeriesProperty):
        fail(
            f"{name} is not computed from a single time series", code=EXIT_USAGE,
            hint="use the Python API for this property",
        )
    if not Path(xvg).exists():
        fail(f"Data file not found: {xvg}", code=EXIT_USAGE)

    try:
        data = read_xvg(xvg)
    except PolymerEngineError as exc:
        fail(str(exc))
    result = calculator.compute(
        data.y, times_ps=data.x, n_replicas=replicas, simulation_ns=simulation_ns
    )
    emit(result.as_dict())
    if not result.usable:
        raise typer.Exit(EXIT_GATE_FAILED)


# ==========================================================================
# research
# ==========================================================================
@research_app.command("failures")
def research_failures(config_path: str | None = ConfigOption) -> None:
    """Report recorded failures and the patterns in them."""
    from polymer_engine.orchestrator.failure_learning import FailureLedger

    config = get_config(config_path)
    with get_store(config) as store:
        emit(FailureLedger.load(store).report())


@research_app.command("decisions")
def research_decisions(
    config_path: str | None = ConfigOption,
    campaign: str | None = typer.Option(None, "--campaign"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Show the decision log: what was chosen, why, and at what expected cost."""
    config = get_config(config_path)
    with get_store(config) as store:
        emit({"decisions": store.list_decisions(campaign_id=campaign, limit=limit)})


@research_app.command("state")
def research_state(
    state_file: str = typer.Argument("engine_state/research_loop.json"),
    config_path: str | None = ConfigOption,
) -> None:
    """Show the persisted state of the autonomous loop."""
    from polymer_engine.orchestrator.autonomy import LoopState

    get_config(config_path)
    state = LoopState.load(state_file)
    if state is None:
        fail(f"No loop state at {state_file}", code=EXIT_USAGE, hint="the loop has not run yet")
    emit(state.as_dict())


@research_app.command("correlate")
def research_correlate(
    dataset: str = typer.Argument(..., help="Polymer dataset (CSV or JSONL)"),
    target: str = typer.Option(..., "--target", help="Property to relate descriptors to"),
    config_path: str | None = ConfigOption,
) -> None:
    """Screen descriptors against a property, correcting for multiple comparisons."""
    import numpy as np

    from polymer_engine.science.correlation import screen_relationships

    config = get_config(config_path)
    dataset_obj, _ = _build_dataset(dataset, target)
    predictors = {
        name: dataset_obj.X[:, i] for i, name in enumerate(dataset_obj.feature_names)
    }
    usable = {
        name: values for name, values in predictors.items()
        if np.isfinite(values).sum() >= 8 and np.nanstd(values) > 0
    }
    if not usable:
        fail("No descriptor has enough variation to correlate", code=EXIT_ERROR)

    matrix = screen_relationships(
        usable, {dataset_obj.target_name: dataset_obj.y},
        n_permutations=config.analysis.bootstrap_samples,
        seed=config.analysis.random_seed,
    )
    emit(
        {
            "n_tests": matrix.n_tests,
            "significant_after_correction": [r.as_dict() for r in matrix.significant()],
            "all": matrix.as_dict()["relationships"],
        }
    )


# ==========================================================================
# Parameterization
# ==========================================================================
DatasetOption = typer.Option("examples/benchmark_campaign/candidates.csv", "--dataset",
                             help="Candidate CSV to resolve polymer names against")
PropertyOption = typer.Option("bulk_density", "--property-class",
                              help="What the parameters are for; qualification is per class")


def _resolve_polymer(name: str, dataset: str):
    """A polymer by name or id from the dataset, or built from a SMILES."""
    from polymer_engine.polymer.ingestion import ingest_csv
    from polymer_engine.polymer.records import build_record

    wanted = name.strip().lower()
    if Path(dataset).is_file():
        for record in ingest_csv(dataset).records:
            if wanted in (record.name.lower(), record.polymer_id.lower()):
                return record
    try:
        return build_record(name=name, repeat_unit_smiles=name, properties={},
                            source="cli")
    except PolymerEngineError:
        fail(f"No polymer {name!r} in {dataset}, and it is not a readable repeat unit",
             code=EXIT_USAGE)


def _property_class(value: str):
    from polymer_engine.parameterization.models import PropertyClass

    try:
        return PropertyClass(value)
    except ValueError:
        fail(f"Unknown property class {value!r}", code=EXIT_USAGE,
             hint="one of: " + ", ".join(p.value for p in PropertyClass))


@param_app.command("inventory")
def parameterization_inventory(
    output: str | None = typer.Option(None, "--output", "-o", help="Directory to write to"),
) -> None:
    """Discover which parameterization software is actually installed here."""
    from polymer_engine.parameterization.capability import discover_tools, summarise

    tools = discover_tools()
    payload = summarise(tools)
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "parameterization_capability_inventory.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")
        lines = ["# Parameterization capability inventory", "",
                 f"{payload['n_available']} of {payload['n_tools']} tools available.", "",
                 "| Tool | Backend | Kind | Available | Version | Path |",
                 "|---|---|---|---|---|---|"]
        for tool in payload["tools"]:
            lines.append(
                f"| `{tool['tool']}` | {tool['backend']} | {tool['kind']} | "
                f"{'yes' if tool['available'] else 'no'} | {tool['version'] or '—'} | "
                f"`{tool['path'] or '—'}` |"
            )
        (directory / "parameterization_capability_inventory.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        payload["written"] = str(directory)
    emit(payload)


@param_app.command("backends")
def parameterization_backends() -> None:
    """List every registered backend, available or not."""
    from polymer_engine.parameterization.backend import default_registry

    emit(default_registry().as_dict())


@param_app.command("assess")
def parameterization_assess(
    polymer: str = typer.Argument(..., help="Polymer name, id, or repeat-unit SMILES"),
    dataset: str = DatasetOption,
    property_class: str = PropertyOption,
) -> None:
    """Ask every backend how far it could carry this polymer."""
    from polymer_engine.parameterization import ParameterizationEngine

    record = _resolve_polymer(polymer, dataset)
    engine = ParameterizationEngine()
    assessments = engine.assess(record)
    decision = engine.route(record, property_class=_property_class(property_class))
    emit({
        "polymer": record.name, "polymer_id": record.polymer_id,
        "family": record.family.value,
        "assessments": [a.as_dict() for a in assessments],
        "decision": decision.as_dict(),
    })
    if not decision.decided:
        raise typer.Exit(EXIT_GATE_FAILED)


@param_app.command("compare")
def parameterization_compare(
    polymer: str = typer.Argument(...),
    dataset: str = DatasetOption,
    property_class: str = PropertyOption,
) -> None:
    """Show every route side by side, without collapsing to a winner."""
    from polymer_engine.parameterization import ParameterizationEngine

    record = _resolve_polymer(polymer, dataset)
    comparison = ParameterizationEngine().compare(
        record, property_class=_property_class(property_class))
    emit(comparison.as_dict())


@param_app.command("qualify")
def parameterization_qualify(
    dataset: str = DatasetOption,
    property_class: str = PropertyOption,
    output: str | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Route the whole candidate set and report coverage. Claims nothing it cannot show."""
    from polymer_engine.parameterization import ParameterizationEngine
    from polymer_engine.polymer.ingestion import ingest_csv

    if not Path(dataset).is_file():
        fail(f"Dataset not found: {dataset}", code=EXIT_USAGE)
    engine = ParameterizationEngine()
    wanted = _property_class(property_class)
    rows: list[dict[str, Any]] = []
    by_backend: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for record in ingest_csv(dataset).records:
        decision = engine.route(record, property_class=wanted)
        backend = decision.selected_backend or "NONE"
        by_backend[backend] = by_backend.get(backend, 0) + 1
        family = by_family.setdefault(record.family.value,
                                      {"total": 0, "routed": 0, "automatic": 0})
        family["total"] += 1
        if decision.selected_backend:
            family["routed"] += 1
            if not decision.requires_human_step:
                family["automatic"] += 1
        rows.append({
            "polymer": record.name, "polymer_id": record.polymer_id,
            "family": record.family.value, "route": backend,
            "confidence": decision.confidence,
            "requires_human_step": decision.requires_human_step,
            "reason": decision.reason,
        })
    payload = {
        "dataset": dataset, "property_class": wanted.value,
        "n_candidates": len(rows), "n_families": len(by_family),
        "by_backend": by_backend, "by_family": by_family,
        "candidates": rows,
        "note": ("routing is not qualification: a route says a backend could try, not "
                 "that the parameters were validated"),
    }
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "parameterization_routing.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")
        payload["written"] = str(directory)
    emit(payload)


@param_app.command("validate")
def parameterization_validate(
    topology: str = typer.Argument(..., help="A GROMACS .top to check"),
    property_class: str = PropertyOption,
    expected_charge: float = typer.Option(0.0, "--expected-charge"),
) -> None:
    """Check a topology for parameter completeness and charge consistency.

    Exits 3 when the topology does not qualify. A successful grompp is not this check.
    """
    from polymer_engine.parameterization.charges import analyse_charges, charge_gates
    from polymer_engine.parameterization.completeness import (
        analyse_topology,
        completeness_gates,
    )

    if not Path(topology).is_file():
        fail(f"Topology not found: {topology}", code=EXIT_USAGE)
    completeness = analyse_topology(topology)
    completeness_report = completeness_gates(completeness)
    charges = analyse_charges(topology)
    charge_report = charge_gates(charges, expected_system_charge=expected_charge)
    promotable = completeness_report.promotable and charge_report.promotable
    emit({
        "topology": topology, "property_class": _property_class(property_class).value,
        "completeness": {
            "summary": completeness.as_dict(),
            "status": completeness_report.status.value,
            "gates": [g.model_dump(mode="json") for g in completeness_report.gates],
        },
        "charges": {
            "summary": charges.as_dict(), "status": charge_report.status.value,
            "gates": [g.model_dump(mode="json") for g in charge_report.gates],
        },
        "promotable": promotable,
    })
    if not promotable:
        raise typer.Exit(EXIT_GATE_FAILED)


# ---------------------------------------------------------------------------
# CHARMM-GUI browser automation
# ---------------------------------------------------------------------------

@browser_app.command("check")
def browser_check() -> None:
    """Report whether the isolated browser environment can actually drive a browser."""
    from polymer_engine.browser.credentials import from_environment, live_test_enabled
    from polymer_engine.browser.driver import WorkerDriver, install_hint

    capabilities = WorkerDriver.capabilities()
    credentials = from_environment()
    emit({
        "browser": capabilities,
        "credentials": credentials.as_dict(),
        "live_tests_enabled": live_test_enabled(),
        "install": None if capabilities.get("available") else install_hint(),
        "note": (
            "Credentials are read from CHARMM_GUI_EMAIL and CHARMM_GUI_PASSWORD only. "
            "This command reports whether they are set; it never prints them."
        ),
    })
    if not capabilities.get("available"):
        raise typer.Exit(EXIT_ERROR)


@browser_app.command("discover")
def browser_discover(
    headless: bool = typer.Option(True, help="Run without a visible window"),
    output: Path = typer.Option(Path("data/charmm_gui"), help="Where to write the catalogue"),
) -> None:
    """Log in, reach Polymer Builder, and capture the catalogue and form schema.

    Writes ``monomer_catalog.json``, ``monomer_catalog.md`` and
    ``builder_form_schema.json``, and reports what changed since the last capture.
    """
    from polymer_engine.browser.acquisition import CharmmGuiAcquisition
    from polymer_engine.browser.catalog import Catalog, diff
    from polymer_engine.browser.session import Session

    previous = Catalog.read(output)
    with Session(headless=headless) as session:
        catalog, form, problem = CharmmGuiAcquisition(session=session).discover(session)
        if catalog is None or form is None:
            fail(problem or "discovery failed",
                 hint="run `polymer-engine browser check` first")
        changes = diff(previous, catalog) if previous else None
        emit({
            "catalog_version": catalog.version,
            "n_monomers": len(catalog.monomers),
            "completeness": catalog.completeness,
            "system_modes": catalog.system_modes,
            "tacticity_options": catalog.tacticity_options,
            "form_fields": sorted(form.schema.fields),
            "semantic_hints": form.semantics,
            "ambiguous_semantics": form.ambiguous(),
            "changes": changes.as_dict() if changes else "no previous capture",
            "written": [str(output / "monomer_catalog.json"),
                        str(output / "monomer_catalog.md"),
                        str(output / "builder_form_schema.json")],
        })


@browser_app.command("queue")
def browser_queue(
    path: Path = typer.Option(Path("campaign/charmm_gui/build_queue.json")),
    markdown: bool = typer.Option(False, "--markdown", help="Render as a table"),
) -> None:
    """Show the CHARMM-GUI build queue."""
    from polymer_engine.browser.queue import BuildQueue

    queue = BuildQueue(path)
    if markdown:
        typer.echo(queue.to_markdown())
        return
    emit({"path": str(path), "counts": queue.counts(),
          "next_ready": (entry.as_dict() if (entry := queue.next_ready()) else None),
          "entries": [e.as_dict() for e in queue.entries]})


@browser_app.command("coverage")
def browser_coverage(
    candidates: Path = typer.Argument(..., help="Candidate CSV"),
    output: Path = typer.Option(Path("campaign/parameterization")),
) -> None:
    """Score the candidate set against the discovered catalogue (§58)."""
    from polymer_engine.browser.coverage import write_coverage

    written = write_coverage(candidates, output)
    emit({k: str(v) for k, v in written.items()})


if __name__ == "__main__":
    main()
