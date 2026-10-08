"""CGenFF penalty parsing, and the refusals built on it.

The penalties are the whole point. A CGenFF topology with a 64-penalty dihedral runs
perfectly well and produces wrong torsional energetics, so a parser that loses them --
which any parser treating `!` as "ignore the rest of the line" does -- turns a warning
into silence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.models import Determination, GateStatus
from polymer_engine.simulation.cgenff import (
    HIGH_PENALTY,
    MODERATE_PENALTY,
    determination_for,
    find_stream_files,
    parse_stream_file,
    parse_stream_text,
    penalty_gates,
    tier_for,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "charmm_gui"
PLA = FIXTURES / "pla_cgenff.str"
CLEAN = FIXTURES / "clean_cgenff.str"


class TestTiers:
    """The tiers are the CGenFF program's published guidance, not local invention."""

    @pytest.mark.parametrize(
        ("penalty", "expected"),
        [(0.0, "good"), (9.99, "good"), (10.0, "moderate"), (49.9, "moderate"),
         (50.0, "poor"), (250.0, "poor")],
    )
    def test_boundaries(self, penalty: float, expected: str) -> None:
        assert tier_for(penalty) == expected

    def test_the_named_thresholds_match_the_tiers(self) -> None:
        assert tier_for(MODERATE_PENALTY - 0.01) == "good"
        assert tier_for(HIGH_PENALTY - 0.01) == "moderate"
        assert tier_for(HIGH_PENALTY) == "poor"


class TestParsing:
    def test_a_real_stream_file_yields_every_penalty(self) -> None:
        report = parse_stream_file(PLA)
        assert report.cgenff_version == "2.5"
        assert report.residues == ["PLA"]
        assert len(report.parameters) == 8
        assert report.max_penalty == pytest.approx(64.0)
        assert report.tier == "poor"

    def test_penalties_are_attributed_to_their_section(self) -> None:
        sections = {p.section for p in parse_stream_file(PLA).parameters}
        assert {"BONDS", "ANGLES", "DIHEDRALS", "IMPROPERS"} <= sections

    def test_the_worst_parameter_is_identified_by_atoms(self) -> None:
        worst = parse_stream_file(PLA).worst(1)[0]
        assert worst.penalty == pytest.approx(64.0)
        assert worst.section == "DIHEDRALS"
        assert worst.atoms == "CG321 CG321 OG302 CG2O2"
        assert worst.tier == "poor"
        assert worst.line_number > 0

    def test_per_atom_charge_penalties_are_kept(self) -> None:
        report = parse_stream_file(PLA)
        assert report.atom_charge_penalties["O1"] == pytest.approx(12.4)
        assert report.atom_charge_penalties["C2"] == pytest.approx(18.673)
        assert report.atom_charge_penalties["C1"] == pytest.approx(0.0)

    def test_the_residue_header_penalties_are_kept(self) -> None:
        report = parse_stream_file(PLA)
        assert report.residue_param_penalty == pytest.approx(22.5)
        assert report.residue_charge_penalty == pytest.approx(18.673)

    def test_counts_above_each_threshold(self) -> None:
        report = parse_stream_file(PLA)
        assert len(report.above(HIGH_PENALTY)) == 1        # the 64.0 dihedral
        assert len(report.above(MODERATE_PENALTY)) == 2    # plus the 32.5 angle

    def test_a_clean_parameter_set_is_good(self) -> None:
        report = parse_stream_file(CLEAN)
        assert report.max_penalty == pytest.approx(0.0)
        assert report.tier == "good"
        assert report.above(MODERATE_PENALTY) == []

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ChemistryError, match="not found"):
            parse_stream_file(tmp_path / "absent.str")

    def test_a_file_that_is_not_a_stream_file_raises(self, tmp_path: Path) -> None:
        """Reporting "no penalties" for an unparsed file reads like a clean set."""
        other = tmp_path / "notes.txt"
        other.write_text("this is not a CHARMM stream file\n", encoding="utf-8")
        with pytest.raises(ChemistryError, match="does not look like"):
            parse_stream_file(other)

    def test_penalties_survive_whitespace_and_case(self) -> None:
        text = (
            "RESI XXX 0.000 ! PARAM PENALTY=  5.000 ; CHARGE PENALTY=  1.000\n"
            "BONDS\n"
            "CG1 CG2  100.0  1.5 ! XXX , from CG1 CG2, Penalty=   7.25\n"
        )
        report = parse_stream_text(text)
        assert report.residue_param_penalty == pytest.approx(5.0)
        assert report.parameters[0].penalty == pytest.approx(7.25)

    def test_a_parameter_line_without_a_penalty_is_not_invented(self) -> None:
        text = "BONDS\nCG1 CG2  100.0  1.5 ! from the published force field\n"
        assert parse_stream_text(text).parameters == []


class TestGates:
    def test_a_clean_set_passes(self) -> None:
        gates = penalty_gates(parse_stream_file(CLEAN))
        assert gates.status is GateStatus.PASS
        assert gates.promotable is True

    def test_a_high_penalty_set_is_inconclusive_without_a_tolerance(self) -> None:
        """Not FAIL: whether 64 is acceptable depends on what is being measured."""
        gates = penalty_gates(parse_stream_file(PLA))
        assert gates.promotable is False
        gate = next(g for g in gates.gates if g.gate == "cgenff:max_parameter_penalty")
        assert gate.status is GateStatus.INCONCLUSIVE
        assert gate.value == pytest.approx(64.0)
        assert "poor" in gate.message

    def test_an_explicit_tolerance_turns_it_into_a_decision(self) -> None:
        report = parse_stream_file(PLA)
        assert penalty_gates(report, max_penalty=100.0).promotable is True
        assert penalty_gates(report, max_penalty=10.0).promotable is False

    def test_a_failing_tolerance_says_what_it_measured_against(self) -> None:
        gates = penalty_gates(parse_stream_file(PLA), max_penalty=10.0)
        gate = next(g for g in gates.gates if g.gate == "cgenff:max_parameter_penalty")
        assert gate.status is GateStatus.FAIL
        assert gate.threshold == pytest.approx(10.0)

    def test_the_worst_offenders_travel_with_the_verdict(self) -> None:
        gates = penalty_gates(parse_stream_file(PLA))
        gate = next(g for g in gates.gates if g.gate == "cgenff:max_parameter_penalty")
        assert gate.evidence["worst"][0]["penalty"] == pytest.approx(64.0)
        assert gate.evidence["n_above_high"] == 1

    def test_a_stripped_parameter_set_is_inconclusive_not_clean(self) -> None:
        """The dangerous case: penalties removed looks identical to zero penalties."""
        text = "RESI XXX 0.000\nBONDS\nCG1 CG2 100.0 1.5\n"
        gates = penalty_gates(parse_stream_text(text))
        assert gates.promotable is False
        gate = next(g for g in gates.gates if g.gate == "cgenff:penalties_present")
        assert gate.status is GateStatus.INCONCLUSIVE
        assert "stripped" in gate.message

    def test_charge_penalties_warn_rather_than_block(self) -> None:
        gates = penalty_gates(parse_stream_file(PLA))
        gate = next(g for g in gates.gates if g.gate == "cgenff:max_charge_penalty")
        assert gate.status is GateStatus.WARN
        assert gate.value == pytest.approx(18.673)


class TestDetermination:
    def test_a_clean_set_is_known(self) -> None:
        assert determination_for(parse_stream_file(CLEAN)) is Determination.KNOWN

    def test_a_high_penalty_set_requires_validation(self) -> None:
        assert determination_for(parse_stream_file(PLA)) is Determination.REQUIRES_VALIDATION

    def test_an_explicit_tolerance_can_accept_it(self) -> None:
        report = parse_stream_file(PLA)
        assert determination_for(report, max_penalty=100.0) is Determination.KNOWN

    def test_no_penalties_at_all_is_unknown(self) -> None:
        assert determination_for(parse_stream_text("BONDS\n")) is Determination.UNKNOWN


class TestDiscovery:
    def test_stream_files_are_found_in_a_directory(self) -> None:
        found = find_stream_files(FIXTURES)
        assert PLA in found and CLEAN in found

    def test_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ChemistryError, match="not found"):
            find_stream_files(tmp_path / "absent")

    def test_unrelated_files_are_not_returned(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("hello", encoding="utf-8")
        (tmp_path / "toppar.str").write_text("RESI X 0.0\n", encoding="utf-8")
        assert [p.name for p in find_stream_files(tmp_path)] == ["toppar.str"]


class TestSpecGeneration:
    """The specification for the manual CHARMM-GUI step."""

    @staticmethod
    def record(name: str = "poly(lactic acid)", smiles: str = "*OC(C)C(=O)*"):
        from polymer_engine.polymer.records import build_record

        return build_record(name=name, repeat_unit_smiles=smiles, properties={},
                            source="test")

    def settings(self) -> dict:
        return {"force_field": "CHARMM36 + CGenFF", "degree_of_polymerization": 30,
                "n_chains": 20, "temperature_k": 300.0}

    def test_a_spec_carries_the_chemistry_and_composition(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        spec = spec_from_record(self.record(), **self.settings())
        assert spec.degree_of_polymerization == 30
        assert spec.n_chains == 20
        assert spec.force_field == "CHARMM36 + CGenFF"
        assert spec.charmm_gui_job_id is None

    def test_the_fingerprint_ignores_the_job_id(self) -> None:
        """Two people building the same spec must agree, which is what makes the
        manual step auditable."""
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        first = spec_from_record(self.record(), **self.settings())
        second = spec_from_record(self.record(), **self.settings())
        assert first.fingerprint() == second.fingerprint()
        second.charmm_gui_job_id = "123456"
        assert second.fingerprint() == first.fingerprint()

    def test_the_fingerprint_changes_with_the_chemistry(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        base = spec_from_record(self.record(), **self.settings())
        other = spec_from_record(self.record(), **{**self.settings(), "n_chains": 21})
        assert base.fingerprint() != other.fingerprint()

    def test_the_box_is_sized_from_the_requested_density(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        spec = spec_from_record(self.record(), **self.settings(),
                                target_density_kg_m3=1250.0)
        assert spec.box_nm is not None and 2.0 < spec.box_nm < 10.0

    def test_no_density_means_no_invented_box(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        assert spec_from_record(self.record(), **self.settings()).box_nm is None

    def test_an_unknown_tacticity_is_omitted_rather_than_asserted(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        spec = spec_from_record(self.record(), **self.settings())
        assert spec.tacticity is None
        assert "Tacticity" not in spec.instructions()

    def test_the_brief_states_that_submission_is_manual(self) -> None:
        """The operator must not be left thinking the engine will submit for them."""
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        text = spec_from_record(self.record(), **self.settings()).instructions()
        assert "cannot submit" in text
        assert "no documented way to create a job" in text

    def test_the_brief_does_not_invent_web_form_field_labels(self) -> None:
        """Transcribing labels would go stale silently, like an invented endpoint."""
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        text = spec_from_record(self.record(), **self.settings()).instructions().lower()
        for invented in ("click the", "press the button", "select the dropdown"):
            assert invented not in text

    @pytest.mark.parametrize(
        ("field", "value"),
        [("force_field", "  "), ("degree_of_polymerization", 1), ("n_chains", 0)],
    )
    def test_incomplete_settings_are_refused(self, field: str, value) -> None:
        from polymer_engine.simulation.charmm_gui_spec import spec_from_record

        with pytest.raises(ChemistryError):
            spec_from_record(self.record(), **{**self.settings(), field: value})

    def test_writing_produces_both_machine_and_human_forms(self, tmp_path: Path) -> None:
        import json

        from polymer_engine.simulation.charmm_gui_spec import spec_from_record, write_spec

        spec = spec_from_record(self.record(), **self.settings())
        payload, brief = write_spec(spec, tmp_path)
        assert json.loads(payload.read_text())["fingerprint"] == spec.fingerprint()
        assert brief.read_text().startswith("# CHARMM-GUI Polymer Builder")

    def test_the_box_helper_inverts_correctly(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import AVOGADRO, box_edge_for

        edge = box_edge_for(chain_mass_amu=2000.0, n_chains=20, density_kg_m3=1250.0)
        recovered = 20 * 2000.0 / AVOGADRO * 1e-3 / ((edge**3) * 1e-27)
        assert recovered == pytest.approx(1250.0, rel=1e-9)

    def test_the_box_helper_refuses_impossible_input(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import box_edge_for

        with pytest.raises(ChemistryError):
            box_edge_for(chain_mass_amu=100.0, n_chains=1, density_kg_m3=0.0)
        with pytest.raises(ChemistryError):
            box_edge_for(chain_mass_amu=100.0, n_chains=0, density_kg_m3=1000.0)


class TestSubmissionStaysUnsupported:
    """The rule that shaped this whole interface."""

    def test_the_provider_still_refuses_to_submit(self) -> None:
        from polymer_engine.core.errors import UnsupportedCapability
        from polymer_engine.providers.charmm_gui.client import CHARMMGUIProvider

        provider = CHARMMGUIProvider.__new__(CHARMMGUIProvider)
        with pytest.raises(UnsupportedCapability, match="does not publish a job-submission endpoint"):
            provider.submit_module()

    def test_only_documented_endpoints_are_advertised(self) -> None:
        from polymer_engine.simulation.charmm_gui_spec import DOCUMENTED_ENDPOINTS

        assert set(DOCUMENTED_ENDPOINTS) == {"/api/login", "/api/check_status", "/api/download"}
        assert not any("submit" in e or "build" in e for e in DOCUMENTED_ENDPOINTS)
