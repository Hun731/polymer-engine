"""The parameterization subsystem: capability, routing, validation, qualification.

Most of these are refusals. A topology file is trivially easy to produce and proves
nothing, so the interesting behaviour is everywhere the engine declines to treat one as
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.markers import requires_rdkit

from polymer_engine.core.errors import ChemistryError, PolymerEngineError
from polymer_engine.core.models import Determination, GateStatus
from polymer_engine.parameterization.backend import BackendRegistry, default_registry
from polymer_engine.parameterization.capability import (
    CapabilityState,
    at_least,
    discover_tools,
    rank,
    summarise,
)
from polymer_engine.parameterization.charges import (
    GROSS_MISMATCH,
    ChargeAdjustment,
    analyse_charges,
    charge_gates,
)
from polymer_engine.parameterization.completeness import (
    analyse_topology,
    completeness_gates,
    expected_counts,
)
from polymer_engine.parameterization.models import (
    ParameterizationState,
    PropertyClass,
    QMPriority,
)
from polymer_engine.parameterization.quality import (
    ParameterQualityGate,
    detect_sensitive_terms,
    overall_priority,
    standard_for,
)
from polymer_engine.parameterization.registry import (
    MIN_FAMILY_REPRESENTATIVES,
    ParameterizationRecord,
    ParameterizationRegistry,
)
from polymer_engine.parameterization.representatives import (
    Candidate,
    select_representatives,
)
from polymer_engine.parameterization.router import SystemBuildRouter
from polymer_engine.parameterization.state import (
    ParameterizationTrack,
    TrackStore,
    allowed_transitions,
)
from polymer_engine.simulation.cgenff import parse_stream_text

CGENFF_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "charmm_gui"


def polymer(name: str = "polyethylene", smiles: str = "*CC*"):
    from polymer_engine.polymer.records import build_record

    return build_record(name=name, repeat_unit_smiles=smiles, properties={},
                        source="test")


# ==========================================================================
# Capability
# ==========================================================================
class TestCapabilityStates:
    def test_capability_is_a_ladder_not_a_boolean(self) -> None:
        assert rank(CapabilityState.QUALIFIED) > rank(CapabilityState.PARAMETERS_VALIDATED)
        assert rank(CapabilityState.PARAMETERS_VALIDATED) > rank(
            CapabilityState.PARAMETERIZATION_AVAILABLE)
        assert rank(CapabilityState.PARAMETERIZATION_AVAILABLE) > rank(
            CapabilityState.STRUCTURE_SUPPORTED)

    def test_a_verdict_is_never_progress(self) -> None:
        """BLOCKED must not read as 'further along than UNAVAILABLE'."""
        for verdict in (CapabilityState.BLOCKED, CapabilityState.REQUIRES_EXPERT_REVIEW):
            assert rank(verdict) == -1
            assert not at_least(verdict, CapabilityState.STRUCTURE_SUPPORTED)
            assert not at_least(verdict, CapabilityState.UNAVAILABLE)

    def test_structure_support_is_not_parameterization(self) -> None:
        """The distinction CHARMM-GUI depends on."""
        assert not at_least(CapabilityState.STRUCTURE_SUPPORTED,
                            CapabilityState.PARAMETERIZATION_AVAILABLE)

    def test_discovery_measures_rather_than_declares(self) -> None:
        tools = discover_tools()
        assert tools
        for tool in tools:
            assert tool.available == (tool.path is not None)
        summary = summarise(tools)
        assert summary["n_available"] <= summary["n_tools"]


# ==========================================================================
# Backends
# ==========================================================================
class TestBackendRegistry:
    def test_unavailable_backends_stay_registered(self) -> None:
        """'OpenFF is not installed' differs from 'OpenFF was not considered'."""
        registry = default_registry()
        assert {"opls_aa", "charmm_gui", "openff", "gaff"} <= set(registry.names())

    def test_duplicate_registration_is_refused(self) -> None:
        from polymer_engine.parameterization.backends.opls import OplsBackend

        registry = BackendRegistry()
        registry.register(OplsBackend())
        with pytest.raises(PolymerEngineError, match="already registered"):
            registry.register(OplsBackend())

    def test_an_unknown_backend_names_the_known_ones(self) -> None:
        with pytest.raises(PolymerEngineError, match="Unknown parameterization backend"):
            default_registry().get("nonexistent")

    @requires_rdkit
    def test_one_broken_backend_does_not_hide_the_others(self) -> None:
        from polymer_engine.parameterization.backend import ForceFieldBackend

        class Exploding(ForceFieldBackend):
            name = "exploding"

            def capabilities(self):
                return {"available": False}

            def assess(self, polymer):
                raise RuntimeError("boom")

            def parameterize(self, request):
                raise NotImplementedError

            def validate(self, result):
                raise NotImplementedError

        registry = default_registry()
        registry.register(Exploding())
        assessments = registry.assess_all(polymer())
        broken = next(a for a in assessments if a.backend == "exploding")
        assert broken.state is CapabilityState.UNAVAILABLE
        assert "RuntimeError" in broken.reason
        assert len(assessments) == len(registry.names())

    def test_uninstalled_backends_report_how_to_install(self) -> None:
        for name in ("openff", "gaff"):
            capabilities = default_registry().get(name).capabilities()
            if not capabilities["available"]:
                assert capabilities["missing"]
                assert "install" in capabilities["install_hint"]


@requires_rdkit
class TestBackendAssessment:
    def test_opls_reaches_system_build_for_a_hydrocarbon(self) -> None:
        assessment = default_registry().get("opls_aa").assess(polymer())
        if assessment.state is CapabilityState.UNAVAILABLE:
            pytest.skip("OPLS-AA is not installed in this environment")
        assert assessment.state is CapabilityState.SYSTEM_BUILD_AVAILABLE
        assert assessment.usable is True

    def test_opls_blocks_a_chemistry_it_cannot_type(self) -> None:
        assessment = default_registry().get("opls_aa").assess(
            polymer("poly(vinyl chloride)", "*CC(Cl)*"))
        if assessment.state is CapabilityState.UNAVAILABLE:
            pytest.skip("OPLS-AA is not installed in this environment")
        assert assessment.state is CapabilityState.BLOCKED
        assert assessment.usable is False
        assert any("Cl" in item for item in assessment.unsupported)

    def test_charmm_gui_never_claims_more_than_structure_support(self) -> None:
        """Nothing is parameterized until a person has run the job."""
        assessment = default_registry().get("charmm_gui").assess(
            polymer("poly(lactic acid)", "*OC(C)C(=O)*"))
        assert assessment.state is CapabilityState.STRUCTURE_SUPPORTED
        assert assessment.usable is False
        assert assessment.requires_human_step is True

    def test_an_uninstalled_backend_says_so_rather_than_blocking(self) -> None:
        """UNAVAILABLE and BLOCKED mean different things and must not be merged.

        Uses whichever backend is genuinely absent here, so the test asserts the
        distinction rather than which tools happen to be installed.
        """
        registry = default_registry()
        absent = [b for b in registry.all() if not b.capabilities()["available"]]
        if not absent:
            pytest.skip("every backend is installed; nothing to assert absence with")
        assessment = absent[0].assess(polymer())
        assert assessment.state is CapabilityState.UNAVAILABLE
        assert assessment.state is not CapabilityState.BLOCKED
        assert assessment.usable is False


# ==========================================================================
# Routing
# ==========================================================================
@requires_rdkit
class TestRouting:
    def test_a_hydrocarbon_routes_to_the_automatic_backend(self) -> None:
        from polymer_engine.parameterization import ParameterizationEngine

        decision = ParameterizationEngine().route(
            polymer(), property_class=PropertyClass.BULK_DENSITY)
        if decision.selected_backend is None:
            pytest.skip("no backend available in this environment")
        assert decision.selected_backend == "opls_aa"
        assert decision.requires_human_step is False
        assert decision.confidence == "high"

    def test_the_human_route_is_used_only_when_no_automatic_one_covers_it(self) -> None:
        """Synthetic, so the assertion does not depend on what is installed today."""
        from polymer_engine.parameterization.models import ForceFieldAssessment

        assessments = [
            ForceFieldAssessment(backend="local", polymer_id="p", polymer_name="P",
                                 state=CapabilityState.BLOCKED,
                                 reason="no tabulated type for this chemistry"),
            ForceFieldAssessment(backend="charmm_gui", polymer_id="p", polymer_name="P",
                                 state=CapabilityState.STRUCTURE_SUPPORTED,
                                 requires_human_step=True, reason="a person must build it"),
        ]
        decision = SystemBuildRouter().route(
            assessments, property_class=PropertyClass.BULK_DENSITY)
        assert decision.selected_backend == "charmm_gui"
        assert decision.requires_human_step is True

    def test_an_automatic_route_is_preferred_over_a_human_one(self) -> None:
        """Never send a person to do what a local backend can do."""
        from polymer_engine.parameterization.models import ForceFieldAssessment

        assessments = [
            ForceFieldAssessment(backend="auto", polymer_id="p", polymer_name="P",
                                 state=CapabilityState.PARAMETERIZATION_AVAILABLE,
                                 reason="covers it", estimated_cost="seconds"),
            ForceFieldAssessment(backend="charmm_gui", polymer_id="p", polymer_name="P",
                                 state=CapabilityState.STRUCTURE_SUPPORTED,
                                 requires_human_step=True, reason="a person must build it"),
        ]
        decision = SystemBuildRouter().route(
            assessments, property_class=PropertyClass.BULK_DENSITY)
        assert decision.selected_backend == "auto"
        assert decision.requires_human_step is False

    def test_two_equal_routes_are_not_silently_separated(self) -> None:
        """Choosing between equally-supported force fields is not a sort order."""
        from polymer_engine.parameterization.models import ForceFieldAssessment

        equal = [
            ForceFieldAssessment(
                backend=name, polymer_id="p", polymer_name="P",
                state=CapabilityState.PARAMETERIZATION_AVAILABLE,
                estimated_cost="seconds", reason="covers it",
            )
            for name in ("alpha", "beta")
        ]
        decision = SystemBuildRouter().route(equal, property_class=PropertyClass.BULK_DENSITY)
        assert decision.ambiguous is True
        assert decision.selected_backend is None
        assert decision.decided is False
        assert len(decision.alternatives) == 2

    def test_no_route_explains_why_not_just_that_there_is_none(self) -> None:
        from polymer_engine.parameterization.models import ForceFieldAssessment

        none_available = [
            ForceFieldAssessment(backend="a", polymer_id="p", polymer_name="P",
                                 state=CapabilityState.BLOCKED,
                                 reason="no tabulated type for fluorine"),
            ForceFieldAssessment(backend="b", polymer_id="p", polymer_name="P",
                                 state=CapabilityState.UNAVAILABLE, reason="not installed"),
        ]
        decision = SystemBuildRouter().route(none_available,
                                             property_class=PropertyClass.BULK_DENSITY)
        assert decision.selected_backend is None
        assert "fluorine" in decision.reason

    def test_an_explicit_preference_is_honoured_but_still_checked(self) -> None:
        from polymer_engine.parameterization.models import ForceFieldAssessment

        assessments = [ForceFieldAssessment(
            backend="alpha", polymer_id="p", polymer_name="P",
            state=CapabilityState.BLOCKED, reason="cannot type it")]
        decision = SystemBuildRouter().route(
            assessments, property_class=PropertyClass.BULK_DENSITY, prefer="alpha")
        assert decision.selected_backend is None
        assert "BLOCKED" in decision.reason

    def test_routing_is_deterministic(self) -> None:
        from polymer_engine.parameterization import ParameterizationEngine

        engine = ParameterizationEngine()
        first = engine.route(polymer(), property_class=PropertyClass.BULK_DENSITY)
        second = engine.route(polymer(), property_class=PropertyClass.BULK_DENSITY)
        assert first.selected_backend == second.selected_backend
        assert first.reason == second.reason

    def test_a_comparison_keeps_every_route(self) -> None:
        from polymer_engine.parameterization import ParameterizationEngine

        comparison = ParameterizationEngine().compare(
            polymer(), property_class=PropertyClass.BULK_DENSITY)
        assert comparison.as_dict()["n_routes"] >= 4
        assert comparison.decision is not None


# ==========================================================================
# Parameter completeness -- the x2top class of failure
# ==========================================================================
class TestCompleteness:
    @staticmethod
    def topology(tmp_path: Path, *, dihedrals: int, angles: int = 24) -> Path:
        """A butane-shaped topology with a controllable number of bonded terms."""
        bonds = [(1, 2), (2, 3), (3, 4)] + [(1, 5 + i) for i in range(3)] \
            + [(2, 8), (2, 9), (3, 10), (3, 11)] + [(4, 12 + i) for i in range(3)]
        lines = ['#include "oplsaa.ff/forcefield.itp"', "", "[ moleculetype ]", "BUT 3",
                 "", "[ atoms ]"]
        for index in range(1, 15):
            kind = "opls_135" if index <= 4 else "opls_140"
            charge = -0.18 if index <= 4 else 0.06
            lines.append(f"{index} {kind} 1 BUT C{index} {index} {charge:.4f}")
        lines += ["", "[ bonds ]"] + [f"{a} {b} 1" for a, b in bonds]
        lines += ["", "[ angles ]"] + [f"1 2 {3 + i} 1" for i in range(angles)]
        lines += ["", "[ dihedrals ]"] + [f"1 2 3 {4 + i} 3" for i in range(dihedrals)]
        path = tmp_path / "topol.top"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_connectivity_implies_the_expected_counts(self) -> None:
        """Counting from bonds is what makes a short section visible."""
        bonds = [(1, 2), (2, 3), (3, 4)]
        angles, dihedrals = expected_counts(bonds)
        assert angles == 2
        assert dihedrals == 1

    def test_a_complete_topology_passes(self, tmp_path: Path) -> None:
        report = analyse_topology(self.topology(tmp_path, dihedrals=27, angles=24))
        gates = completeness_gates(report)
        assert gates.promotable is True

    def test_the_x2top_failure_is_caught(self, tmp_path: Path) -> None:
        """Regression: x2top emits 3 dihedrals for a molecule that needs 27.

        grompp accepts that topology and mdrun runs it; the torsional energy comes out
        near 300 kJ/mol against a true 21 kJ/mol profile.
        """
        report = analyse_topology(self.topology(tmp_path, dihedrals=3))
        gates = completeness_gates(report)
        assert gates.promotable is False
        gate = next(g for g in gates.gates
                    if g.gate == "completeness:dihedrals_vs_connectivity")
        assert gate.status is GateStatus.FAIL
        assert "missing" in gate.message

    def test_a_missing_dihedral_section_is_a_failure_not_an_omission(
        self, tmp_path: Path
    ) -> None:
        """No dihedrals at all preprocesses fine and simulates free rotation."""
        path = tmp_path / "t.top"
        path.write_text(
            "[ moleculetype ]\nX 3\n\n[ atoms ]\n1 opls_135 1 X C1 1 0.0000\n"
            "2 opls_135 1 X C2 2 0.0000\n3 opls_135 1 X C3 3 0.0000\n"
            "\n[ bonds ]\n1 2 1\n2 3 1\n",
            encoding="utf-8",
        )
        gates = completeness_gates(analyse_topology(path))
        assert gates.promotable is False
        gate = next(g for g in gates.gates if g.gate == "completeness:sections")
        assert gate.status is GateStatus.FAIL
        assert "angles" in gate.message and "dihedrals" in gate.message

    def test_placeholder_parameters_are_detected(self, tmp_path: Path) -> None:
        """The other half of the x2top defect: numbers written inline, never looked up."""
        path = tmp_path / "t.top"
        path.write_text(
            "[ moleculetype ]\nX 3\n\n[ atoms ]\n1 opls_135 1 X C1 1 0.0000\n"
            "2 opls_135 1 X C2 2 0.0000\n\n[ bonds ]\n1 2 1\n"
            "\n[ angles ]\n1 2 1 1\n\n[ dihedrals ]\n1 2 1 2 3 0.0000 0.0000 0.0000\n",
            encoding="utf-8",
        )
        report = analyse_topology(path)
        assert report.inline_parameters > 0
        assert any("placeholder" in problem for problem in report.problems)
        assert completeness_gates(report).promotable is False

    def test_an_unresolved_include_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "t.top"
        path.write_text(
            '#include "missing.itp"\n[ moleculetype ]\nX 3\n\n[ atoms ]\n'
            "1 opls_135 1 X C1 1 0.0000\n\n[ bonds ]\n\n[ angles ]\n\n[ dihedrals ]\n",
            encoding="utf-8",
        )
        report = analyse_topology(path)
        assert "missing.itp" in report.unresolved_includes
        assert completeness_gates(report).promotable is False

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ChemistryError, match="not found"):
            analyse_topology(tmp_path / "absent.top")


# ==========================================================================
# Charges
# ==========================================================================
class TestCharges:
    @staticmethod
    def topology(tmp_path: Path, charges: list[float], count: int = 1) -> Path:
        lines = ["[ moleculetype ]", "MOL 3", "", "[ atoms ]"]
        for index, charge in enumerate(charges, start=1):
            lines.append(f"{index} opls_135 1 MOL C{index} {index} {charge:.6f}")
        lines += ["", "[ system ]", "test", "", "[ molecules ]", f"MOL {count}"]
        path = tmp_path / "topol.top"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_a_neutral_molecule_passes(self, tmp_path: Path) -> None:
        gates = charge_gates(analyse_charges(
            self.topology(tmp_path, [-0.18, 0.06, 0.06, 0.06])))
        assert gates.promotable is True

    def test_a_gross_mismatch_fails(self, tmp_path: Path) -> None:
        """A residual this size is a missing charge group, not accumulated round-off."""
        gates = charge_gates(analyse_charges(self.topology(tmp_path, [0.7, 0.06])))
        assert gates.promotable is False
        gate = next(g for g in gates.gates if g.gate.startswith("charge:molecule"))
        assert gate.status is GateStatus.FAIL
        assert "structural error" in gate.message

    def test_a_small_residual_is_inconclusive_not_repaired(self, tmp_path: Path) -> None:
        """The PEO case: +0.120 e per repeat unit. Never silently renormalised."""
        gates = charge_gates(analyse_charges(self.topology(tmp_path, [0.12, 0.0])))
        assert gates.promotable is False
        gate = next(g for g in gates.gates if g.gate.startswith("charge:molecule"))
        assert gate.status is GateStatus.INCONCLUSIVE
        assert "not repaired" in gate.message
        assert abs(gate.value - 0.12) < 1e-9

    def test_the_system_charge_multiplies_by_molecule_count(self, tmp_path: Path) -> None:
        report = analyse_charges(self.topology(tmp_path, [0.01, 0.0], count=20))
        assert report.system_charge == pytest.approx(0.2)
        assert charge_gates(report).promotable is False

    def test_a_topology_without_charges_is_inconclusive(self, tmp_path: Path) -> None:
        path = tmp_path / "t.top"
        path.write_text("[ moleculetype ]\nX 3\n", encoding="utf-8")
        gates = charge_gates(analyse_charges(path))
        assert gates.promotable is False
        assert gates.gates[0].status is GateStatus.INCONCLUSIVE

    def test_an_adjustment_cannot_be_made_without_provenance(self) -> None:
        """Charges may be changed deliberately; they may never be changed quietly."""
        with pytest.raises(ChemistryError, match="provenance"):
            ChargeAdjustment(molecule="MOL", original_charge=0.12, new_charge=0.0,
                             method="", reason="tidy", software="x", author="y")

    def test_a_recorded_adjustment_keeps_the_original(self) -> None:
        adjustment = ChargeAdjustment(
            molecule="MOL", original_charge=0.12, new_charge=0.0,
            method="uniform redistribution", reason="documented neutralisation",
            software="polymer-engine", author="operator",
        )
        payload = adjustment.as_dict()
        assert payload["original_charge"] == 0.12
        assert payload["delta"] == pytest.approx(-0.12)
        assert payload["method"] and payload["reason"]

    def test_the_gross_threshold_is_above_the_peo_residual(self) -> None:
        """A real chemistry case must land in INCONCLUSIVE, not FAIL."""
        assert 0.12 < GROSS_MISMATCH


# ==========================================================================
# Parameter quality -- penalty is evidence, not a verdict
# ==========================================================================
class TestQualityGate:
    @staticmethod
    def penalties(worst: float):
        return parse_stream_text(
            "RESI X 0.000 ! param penalty=  0.000 ; charge penalty=  0.000\n"
            "BONDS\n"
            f"CG1 CG2 100.0 1.5 ! X , from analogy, penalty= {worst}\n"
        )

    def test_standards_differ_by_property_class(self) -> None:
        """A torsion error that ruins a conformer population barely moves a density."""
        density = standard_for(PropertyClass.BULK_DENSITY)
        conformer = standard_for(PropertyClass.CONFORMATIONAL_FREE_ENERGY)
        assert density.requires_torsion_qm is False
        assert conformer.requires_torsion_qm is True
        assert conformer.max_penalty_without_qm < density.max_penalty_without_qm
        assert conformer.torsion_rmse_kj_mol < 2.0

    def test_every_standard_states_its_justification(self) -> None:
        for property_class in PropertyClass:
            standard = standard_for(property_class)
            assert len(standard.justification) > 40, property_class

    def test_a_high_penalty_is_inconclusive_not_a_failure(self) -> None:
        """The rule: weak analogy means 'check this', not 'this is wrong'."""
        report, determination = ParameterQualityGate(
            PropertyClass.BULK_DENSITY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(64.0),
        )
        gate = next(g for g in report.gates if g.gate == "quality:analogy_penalty")
        assert gate.status is GateStatus.INCONCLUSIVE
        assert gate.status is not GateStatus.FAIL
        assert determination is Determination.REQUIRES_VALIDATION

    def test_a_clean_penalty_passes_for_density_without_qm(self) -> None:
        report, determination = ParameterQualityGate(
            PropertyClass.BULK_DENSITY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(1.0),
        )
        assert report.promotable is True
        assert determination is Determination.KNOWN

    def test_the_same_penalty_demands_qm_for_a_free_energy(self) -> None:
        """Identical parameters, different question, different standard."""
        report, _ = ParameterQualityGate(
            PropertyClass.CONFORMATIONAL_FREE_ENERGY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(1.0),
        )
        assert report.promotable is False
        gate = next(g for g in report.gates if g.gate == "quality:qm_validation")
        assert gate.status is GateStatus.INCONCLUSIVE

    def test_passing_qm_validation_promotes_it(self) -> None:
        report, determination = ParameterQualityGate(
            PropertyClass.CONFORMATIONAL_FREE_ENERGY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(64.0), qm_validated=True, qm_torsion_rmse=0.8,
        )
        assert report.promotable is True
        assert determination is Determination.KNOWN

    def test_failing_qm_validation_is_a_real_failure(self) -> None:
        report, determination = ParameterQualityGate(
            PropertyClass.CONFORMATIONAL_FREE_ENERGY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(64.0), qm_validated=False, qm_torsion_rmse=9.0,
        )
        gate = next(g for g in report.gates if g.gate == "quality:qm_validation")
        assert gate.status is GateStatus.FAIL
        assert determination is Determination.INSUFFICIENT_DATA

    def test_a_missing_force_field_version_warns(self) -> None:
        report, _ = ParameterQualityGate(PropertyClass.BULK_DENSITY).evaluate(
            force_field="CGenFF", force_field_version=None,
            penalties=self.penalties(1.0),
        )
        gate = next(g for g in report.gates if g.gate == "quality:force_field_identified")
        assert gate.status is GateStatus.WARN
        assert "provenance is incomplete" in gate.message

    def test_wildcards_are_refused_where_the_standard_says_so(self) -> None:
        report, _ = ParameterQualityGate(
            PropertyClass.CONFORMATIONAL_FREE_ENERGY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(0.0), qm_validated=True, wildcards=3,
        )
        gate = next(g for g in report.gates if g.gate == "quality:wildcards")
        assert gate.status is GateStatus.FAIL

    def test_a_weak_analogy_cleared_by_qm_no_longer_blocks(self) -> None:
        """Regression: the penalty gate stayed INCONCLUSIVE after QM passed.

        That made the QM run pointless -- nothing could ever clear a weak analogy, which
        contradicts the whole 'penalty is evidence, not a verdict' principle.
        """
        report, _ = ParameterQualityGate(
            PropertyClass.CONFORMATIONAL_FREE_ENERGY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(64.0), qm_validated=True, qm_torsion_rmse=0.8,
        )
        gate = next(g for g in report.gates if g.gate == "quality:analogy_penalty")
        assert gate.status is GateStatus.PASS
        assert "QM validation confirmed" in gate.message

    def test_a_weak_analogy_refuted_by_qm_fails(self) -> None:
        report, _ = ParameterQualityGate(
            PropertyClass.CONFORMATIONAL_FREE_ENERGY).evaluate(
            force_field="CGenFF", force_field_version="4.6",
            penalties=self.penalties(64.0), qm_validated=False, qm_torsion_rmse=9.0,
        )
        gate = next(g for g in report.gates if g.gate == "quality:analogy_penalty")
        assert gate.status is GateStatus.FAIL

class TestSensitiveTerms:
    def test_a_high_penalty_torsion_is_high_priority(self) -> None:
        report = parse_stream_text(
            "DIHEDRALS\nA B C D 0.1 3 0.0 ! X , from analogy, penalty= 64.0\n")
        terms = detect_sensitive_terms(report)
        assert terms and terms[0].priority is QMPriority.HIGH
        assert overall_priority(terms) is QMPriority.HIGH

    def test_a_low_penalty_term_is_not_flagged(self) -> None:
        report = parse_stream_text(
            "BONDS\nA B 100 1.5 ! X , from A B, penalty= 0.5\n")
        assert detect_sensitive_terms(report) == []

    def test_an_ester_linkage_is_flagged_without_any_penalty(self) -> None:
        """A backend with no penalty scheme reports nothing; chemistry still matters."""
        terms = detect_sensitive_terms(None, functional_groups=["ester"])
        assert terms and terms[0].priority is QMPriority.HIGH
        assert "linkage" in terms[0].kind

    def test_halogens_are_flagged_as_sparsely_parameterized(self) -> None:
        terms = detect_sensitive_terms(None, heteroatoms=["F"])
        assert terms and terms[0].identifier == "F"

    def test_nothing_notable_is_low_priority(self) -> None:
        assert overall_priority([]) is QMPriority.LOW


# ==========================================================================
# State machine
# ==========================================================================
class TestStateMachine:
    def test_the_forward_path_is_sequential(self) -> None:
        track = ParameterizationTrack(polymer_id="p", backend="opls_aa")
        for state, reason in (
            (ParameterizationState.BACKEND_SELECTED, "routed"),
            (ParameterizationState.PARAMETERIZED, "topology written"),
            (ParameterizationState.TOPOLOGY_VALIDATED, "consistent"),
            (ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED, "complete"),
            (ParameterizationState.QM_VALIDATION_REQUIRED, "needs qm"),
            (ParameterizationState.QM_VALIDATED, "qm agreed"),
            (ParameterizationState.SYSTEM_VALIDATED, "system ok"),
            (ParameterizationState.QUALIFIED, "qualified for density"),
        ):
            track.advance(state, reason)
        assert track.state is ParameterizationState.QUALIFIED
        assert len(track.history) == 8

    def test_a_skipped_step_is_refused(self) -> None:
        """QUALIFIED must not be assertable straight from DISCOVERED."""
        track = ParameterizationTrack(polymer_id="p", backend="opls_aa")
        with pytest.raises(PolymerEngineError, match="Illegal parameterization transition"):
            track.advance(ParameterizationState.QUALIFIED, "wishful thinking")

    def test_a_verdict_is_reachable_from_anywhere(self) -> None:
        track = ParameterizationTrack(polymer_id="p", backend="opls_aa")
        track.advance(ParameterizationState.BACKEND_SELECTED, "routed")
        track.advance(ParameterizationState.BLOCKED, "chemistry not supported")
        assert track.terminal is True

    def test_qualified_is_an_end_state(self) -> None:
        assert allowed_transitions(ParameterizationState.QUALIFIED) == set()

    def test_a_blocked_track_can_be_revisited(self) -> None:
        """Installing the missing toolchain should not require a new identity."""
        assert ParameterizationState.DISCOVERED in allowed_transitions(
            ParameterizationState.BLOCKED)

    def test_every_transition_records_its_reason(self) -> None:
        track = ParameterizationTrack(polymer_id="p", backend="opls_aa")
        track.advance(ParameterizationState.BACKEND_SELECTED, "routed to opls",
                      evidence={"score": 3})
        entry = track.history[0].as_dict()
        assert entry["reason"] == "routed to opls"
        assert entry["evidence"] == {"score": 3}
        assert entry["at"]

    def test_tracks_survive_a_round_trip(self, tmp_path: Path) -> None:
        store = TrackStore()
        track = store.track("p", "opls_aa")
        track.advance(ParameterizationState.BACKEND_SELECTED, "routed")
        path = store.save(tmp_path / "tracks.json")
        restored = TrackStore.load(path)
        assert restored.track("p", "opls_aa").state is ParameterizationState.BACKEND_SELECTED
        assert len(restored.track("p", "opls_aa").history) == 1


# ==========================================================================
# Registry and family qualification
# ==========================================================================
class TestRegistry:
    @staticmethod
    def record(name: str, family: str, state: ParameterizationState):
        return ParameterizationRecord(
            polymer_id=f"pol_{name}", polymer_name=name, family=family,
            backend="charmm_gui", force_field="CHARMM36 + CGenFF",
            force_field_version="4.6", property_class=PropertyClass.BULK_DENSITY,
            state=state,
        )

    def test_parameterized_validated_and_qualified_are_distinct(self) -> None:
        """The central distinction of the whole subsystem."""
        parameterized = self.record("A", "polyester", ParameterizationState.PARAMETERIZED)
        assert parameterized.parameterized is True
        assert parameterized.qm_validated is False
        assert parameterized.qualified is False

        validated = self.record("B", "polyester", ParameterizationState.QM_VALIDATED)
        assert validated.qm_validated is True
        assert validated.qualified is False

        qualified = self.record("C", "polyester", ParameterizationState.QUALIFIED)
        assert qualified.qualified is True

    def test_the_key_identifies_the_combination_not_the_run(self) -> None:
        first = self.record("A", "polyester", ParameterizationState.PARAMETERIZED)
        second = self.record("A", "polyester", ParameterizationState.QUALIFIED)
        assert first.key == second.key

    def test_one_qualified_polyester_does_not_qualify_polyesters(self) -> None:
        """PLA and PET share a functional group and very little else."""
        registry = ParameterizationRegistry()
        registry.add(self.record("poly(lactic acid)", "polyester",
                                 ParameterizationState.QUALIFIED))
        coverage = registry.family_coverage(
            "polyester", "CHARMM36 + CGenFF", PropertyClass.BULK_DENSITY)
        assert coverage.qualified_for_family is False
        assert coverage.confidence == "single-representative"

    def test_two_representatives_reach_family_confidence(self) -> None:
        registry = ParameterizationRegistry()
        for name in ("poly(lactic acid)", "poly(ethylene terephthalate)"):
            registry.add(self.record(name, "polyester", ParameterizationState.QUALIFIED))
        coverage = registry.family_coverage(
            "polyester", "CHARMM36 + CGenFF", PropertyClass.BULK_DENSITY)
        assert coverage.qualified_for_family is True
        assert len(coverage.qualified) >= MIN_FAMILY_REPRESENTATIVES

    def test_qualification_is_scoped_to_a_property_class(self) -> None:
        """A density qualification says nothing about a free energy."""
        registry = ParameterizationRegistry()
        registry.add(self.record("A", "polyester", ParameterizationState.QUALIFIED))
        other = registry.family_coverage(
            "polyester", "CHARMM36 + CGenFF",
            PropertyClass.CONFORMATIONAL_FREE_ENERGY)
        assert other.qualified == []

    def test_the_registry_round_trips(self, tmp_path: Path) -> None:
        registry = ParameterizationRegistry()
        registry.add(self.record("A", "polyester", ParameterizationState.QUALIFIED))
        path = registry.save(tmp_path / "registry.json")
        restored = ParameterizationRegistry.load(path)
        assert restored.summary()["n_qualified"] == 1
        assert restored.all()[0].polymer_name == "A"

    def test_the_save_is_atomic(self, tmp_path: Path) -> None:
        registry = ParameterizationRegistry()
        registry.add(self.record("A", "polyester", ParameterizationState.QUALIFIED))
        path = registry.save(tmp_path / "registry.json")
        assert not (tmp_path / "registry.json.tmp").exists()
        json.loads(path.read_text())


# ==========================================================================
# Representative selection
# ==========================================================================
class TestRepresentatives:
    def test_the_easiest_example_is_not_the_only_one_chosen(self) -> None:
        """Validating polyethylene and declaring polyolefins covered is the failure."""
        candidates = [
            Candidate("a", "polyethylene", "polyolefin",
                      {"heavy_atom_count": 2, "fraction_csp3": 1.0}),
            Candidate("b", "polystyrene", "polyolefin",
                      {"heavy_atom_count": 8, "aromatic_ring_count": 1,
                       "fraction_csp3": 0.25}),
            Candidate("c", "polypropylene", "polyolefin",
                      {"heavy_atom_count": 3, "fraction_csp3": 1.0}),
        ]
        chosen = [s.name for s in select_representatives(candidates, per_family=2)]
        assert "polystyrene" in chosen, "the most unusual member must be selected"

    def test_a_high_penalty_candidate_is_preferred(self) -> None:
        candidates = [
            Candidate("a", "easy", "polyester", {"heavy_atom_count": 5}, max_penalty=0.0),
            Candidate("b", "hard", "polyester", {"heavy_atom_count": 5}, max_penalty=64.0),
        ]
        chosen = select_representatives(candidates, per_family=1)
        assert chosen[0].name == "hard"
        assert any("penalty" in reason for reason in chosen[0].reasons)

    def test_a_second_pick_is_not_a_near_duplicate(self) -> None:
        candidates = [
            Candidate("a", "one", "f", {"heavy_atom_count": 2}),
            Candidate("b", "twin", "f", {"heavy_atom_count": 2}),
            Candidate("c", "different", "f", {"heavy_atom_count": 40}),
        ]
        chosen = [s.name for s in select_representatives(candidates, per_family=2)]
        assert "different" in chosen

    def test_every_selection_explains_itself(self) -> None:
        chosen = select_representatives(
            [Candidate("a", "x", "f", {"heavy_atom_count": 3})], per_family=1)
        assert chosen[0].reasons

    def test_an_empty_set_selects_nothing(self) -> None:
        assert select_representatives([]) == []


class TestQmStateHonesty:
    """Regression: 'QM not required' was being recorded as 'QM validated'.

    A state that says QM_VALIDATED when no QM ran is a claim nothing tested -- exactly
    the conflation this subsystem exists to prevent.
    """

    def test_skipping_qm_never_records_it_as_validated(self) -> None:
        from polymer_engine.parameterization.state import allowed_transitions

        moves = allowed_transitions(
            ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED)
        assert ParameterizationState.SYSTEM_VALIDATED in moves, (
            "a property class that does not need QM must be able to skip it"
        )
        assert ParameterizationState.QM_VALIDATION_REQUIRED in moves

    def test_qm_validated_is_not_reachable_by_skipping(self) -> None:
        """It must still take running QM to claim QM validation."""
        from polymer_engine.parameterization.state import allowed_transitions

        moves = allowed_transitions(
            ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED)
        assert ParameterizationState.QM_VALIDATED not in moves

    def test_the_shortcut_does_not_let_a_topology_jump_to_qualified(self) -> None:
        from polymer_engine.parameterization.state import ParameterizationTrack

        track = ParameterizationTrack(polymer_id="p", backend="openff")
        track.advance(ParameterizationState.BACKEND_SELECTED, "routed")
        track.advance(ParameterizationState.PARAMETERIZED, "topology written")
        track.advance(ParameterizationState.TOPOLOGY_VALIDATED, "consistent")
        track.advance(ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED, "complete")
        with pytest.raises(PolymerEngineError):
            track.advance(ParameterizationState.QUALIFIED, "skipping the evidence")
