"""Statistical analysis, convergence gates, trajectories, and free energy."""

from polymer_engine.analysis.convergence import (
    ReplicaAgreement,
    SeriesAnalysis,
    analyse_series,
    build_convergence_report,
    combine_replicas,
    convergence_gates,
    replica_agreement_gate,
)
from polymer_engine.analysis.free_energy import (
    OverlapDiagnostic,
    PmfResult,
    WindowSamples,
    build_pmf,
    diagnose_overlap,
    pmf_gates,
    wham,
)
from polymer_engine.analysis.md import AnalysisSpec, TrajectoryAnalysis, radius_of_gyration
from polymer_engine.analysis.statistics import (
    bootstrap_ci,
    describe,
    detect_equilibration,
    effective_sample_size,
    pearson,
    spearman,
    statistical_inefficiency,
)

__all__ = [
    "AnalysisSpec",
    "OverlapDiagnostic",
    "PmfResult",
    "ReplicaAgreement",
    "SeriesAnalysis",
    "TrajectoryAnalysis",
    "WindowSamples",
    "analyse_series",
    "bootstrap_ci",
    "build_convergence_report",
    "build_pmf",
    "combine_replicas",
    "convergence_gates",
    "describe",
    "detect_equilibration",
    "diagnose_overlap",
    "effective_sample_size",
    "pearson",
    "pmf_gates",
    "radius_of_gyration",
    "replica_agreement_gate",
    "spearman",
    "statistical_inefficiency",
    "wham",
]
