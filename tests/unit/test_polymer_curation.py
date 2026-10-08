"""Descriptors, property normalisation, and dataset ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymer_engine.core.errors import ChemistryError, ParameterValidationError
from polymer_engine.core.models import Determination
from polymer_engine.core.units import convert
from polymer_engine.polymer.descriptors import DESCRIPTOR_UNITS, compute_descriptors, descriptor_matrix
from polymer_engine.polymer.ingestion import ingest_csv, ingest_rows, split_unit_from_header
from polymer_engine.polymer.normalization import (
    normalize_property,
    normalize_property_name,
    normalize_unit,
)
from polymer_engine.polymer.records import build_record
from tests.markers import requires_rdkit

DATASETS = Path(__file__).resolve().parents[1] / "fixtures" / "datasets"


# ==========================================================================
# Descriptors -- checked against hand-verifiable chemistry
# ==========================================================================
@requires_rdkit
class TestDescriptors:
    def test_polyethylene_capped_repeat_unit_is_ethane(self) -> None:
        result = compute_descriptors("*CC*", polymer_id="PE")
        assert result.value("repeat_unit_mass") == pytest.approx(30.07, abs=0.02)
        assert result.value("heavy_atom_count") == 2
        assert result.value("backbone_atom_count") == 2
        assert result.value("side_chain_heavy_atoms") == 0

    def test_polystyrene_capped_repeat_unit_is_ethylbenzene(self) -> None:
        result = compute_descriptors("*CC(*)c1ccccc1", polymer_id="PS")
        assert result.value("repeat_unit_mass") == pytest.approx(106.17, abs=0.02)
        assert result.value("aromatic_ring_count") == 1
        assert result.value("backbone_atom_count") == 2
        assert result.value("side_chain_heavy_atoms") == 6

    def test_pmma_capped_repeat_unit_is_methyl_isobutyrate(self) -> None:
        result = compute_descriptors("*CC(C)(*)C(=O)OC", polymer_id="PMMA")
        assert result.value("repeat_unit_mass") == pytest.approx(102.13, abs=0.02)
        assert result.value("side_chain_heavy_atoms") == 5

    def test_peo_has_one_ether_oxygen_in_the_backbone(self) -> None:
        result = compute_descriptors("*CCO*", polymer_id="PEO")
        assert result.value("backbone_atom_count") == 3
        assert result.value("heteroatom_count") == 1

    def test_every_descriptor_declares_a_known_unit(self) -> None:
        result = compute_descriptors("*CC*")
        for name, measurement in result.measurements.items():
            assert name in DESCRIPTOR_UNITS
            assert measurement.units == DESCRIPTOR_UNITS[name][0]

    def test_branched_repeat_unit_has_unknown_backbone_not_a_guess(self) -> None:
        result = compute_descriptors("*CC(*)C*")
        assert result.measurements["backbone_atom_count"].determination is not Determination.KNOWN
        assert result.value("backbone_atom_count") is None

    def test_invalid_structure_raises(self) -> None:
        with pytest.raises(ChemistryError):
            compute_descriptors("*C(((C*")

    def test_matrix_preserves_missing_values_as_none(self) -> None:
        sets = [compute_descriptors("*CC*", polymer_id="a"), compute_descriptors("*CC(*)C*", polymer_id="b")]
        ids, names, matrix = descriptor_matrix(sets, ["repeat_unit_mass", "backbone_atom_count"])
        assert ids == ["a", "b"]
        assert names == ["repeat_unit_mass", "backbone_atom_count"]
        assert matrix[1][1] is None, "unknown must stay None, never be imputed here"


def test_descriptors_without_rdkit_are_unknown_not_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args, **kwargs):
        if name.startswith("rdkit"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = compute_descriptors("*CC*", polymer_id="PE")
    assert result.backend == "none"
    assert result.known() == {}
    assert all(m.determination is Determination.UNKNOWN for m in result.measurements.values())


# ==========================================================================
# Property normalisation
# ==========================================================================
class TestPropertyNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Tg", "glass_transition_temperature"),
            ("glass transition temperature", "glass_transition_temperature"),
            ("Glass_Transition_Temp", "glass_transition_temperature"),
            ("Young's Modulus", "youngs_modulus"),
            ("density", "density"),
            ("Mn", "number_average_molar_mass"),
        ],
    )
    def test_name_aliases(self, raw: str, expected: str) -> None:
        assert normalize_property_name(raw) == expected

    def test_unknown_name_is_none(self) -> None:
        assert normalize_property_name("vibe score") is None

    def test_celsius_to_kelvin(self) -> None:
        m = normalize_property("Tg", 105, "degC")
        assert m.determination is Determination.KNOWN
        assert m.value == pytest.approx(378.15)
        assert m.units == "K"

    def test_kelvin_passes_through_unchanged(self) -> None:
        assert normalize_property("Tg", 378.15, "K").value == pytest.approx(378.15)

    def test_the_two_temperature_paths_agree(self) -> None:
        """A 273-degree unit slip is the classic Tg bug; both routes must match."""
        assert normalize_property("Tg", 105, "degC").value == pytest.approx(
            normalize_property("Tg", 378.15, "K").value
        )

    def test_density_units(self) -> None:
        assert normalize_property("density", 1.18, "g/cm3").value == pytest.approx(1180.0)
        assert normalize_property("density", 1180, "kg/m3").value == pytest.approx(1180.0)

    def test_gpa_is_scaled_to_mpa(self) -> None:
        assert normalize_property("Young's Modulus", 3.2, "GPa").value == pytest.approx(3200.0)

    def test_percent_becomes_a_fraction(self) -> None:
        assert normalize_property("crystallinity", 35, "%").value == pytest.approx(0.35)

    def test_unknown_unit_refuses_to_assume(self) -> None:
        m = normalize_property("Tg", 105, "furlongs")
        assert m.determination is Determination.REQUIRES_VALIDATION
        assert m.value is None
        assert "refusing to assume" in (m.notes or "")

    def test_implausible_value_is_rejected(self) -> None:
        m = normalize_property("Tg", 1e9, "K")
        assert m.determination is Determination.REQUIRES_VALIDATION
        assert m.value is None

    def test_non_numeric_value_is_unknown(self) -> None:
        assert normalize_property("Tg", "n/a", "K").determination is Determination.UNKNOWN

    def test_range_values_take_the_midpoint(self) -> None:
        assert normalize_property("Tm", "250-260", "degC").value == pytest.approx(528.15)

    def test_uncertainty_is_converted_as_a_width_not_a_point(self) -> None:
        """A +/-5 degC interval is +/-5 K, not +/-278 K."""
        m = normalize_property("Tg", 105, "degC", uncertainty=5.0)
        assert m.uncertainty == pytest.approx(5.0)

    def test_uncertainty_scales_with_multiplicative_conversions(self) -> None:
        m = normalize_property("Young's Modulus", 3.2, "GPa", uncertainty=0.1)
        assert m.uncertainty == pytest.approx(100.0)

    def test_unit_alias_resolution(self) -> None:
        assert normalize_unit("g/cm3") == ("g/cm^3", 1.0)
        assert normalize_unit("GPa") == ("MPa", 1000.0)
        assert normalize_unit("furlongs") == (None, 1.0)

    def test_cross_dimension_conversion_is_refused(self) -> None:
        with pytest.raises(ParameterValidationError):
            convert(1.0, "K", "nm")


# ==========================================================================
# Ingestion
# ==========================================================================
class TestIngestion:
    def test_header_unit_extraction(self) -> None:
        assert split_unit_from_header("Tg (degC)") == ("Tg", "degC")
        assert split_unit_from_header("density [g/cm3]") == ("density", "g/cm3")
        assert split_unit_from_header("Tg") == ("Tg", None)

    @requires_rdkit
    def test_csv_ingestion_reconciles_every_row(self) -> None:
        report = ingest_csv(DATASETS / "polymers.csv")
        assert report.rows_read == 8
        assert report.n_accepted == 5
        assert report.n_rejected == 2
        assert report.n_duplicates == 1
        assert report.reconciles(), "rows read must equal accepted + rejected + duplicates"

    @requires_rdkit
    def test_units_are_converted_during_ingestion(self) -> None:
        report = ingest_csv(DATASETS / "polymers.csv")
        ps = next(r for r in report.records if r.name == "polystyrene")
        assert ps.property_value("glass_transition_temperature") == pytest.approx(373.15)
        assert ps.property_value("density") == pytest.approx(1050.0)
        assert ps.property_value("youngs_modulus") == pytest.approx(3200.0)

    @requires_rdkit
    def test_duplicate_notation_is_detected(self) -> None:
        report = ingest_csv(DATASETS / "polymers.csv")
        assert "polyethylene-duplicate" in next(iter(report.duplicates.values()))

    @requires_rdkit
    def test_bad_rows_are_rejected_with_reasons_not_dropped(self) -> None:
        report = ingest_csv(DATASETS / "polymers.csv")
        reasons = " ".join(r.reason for r in report.rejected)
        assert "not a valid molecule" in reasons
        assert "no repeat-unit SMILES" in reasons

    @requires_rdkit
    def test_source_digest_is_recorded(self) -> None:
        report = ingest_csv(DATASETS / "polymers.csv")
        assert report.source_sha256 and len(report.source_sha256) == 64
        assert all(r.provenance["source_sha256"] == report.source_sha256 for r in report.records)

    @requires_rdkit
    def test_missing_property_stays_missing(self) -> None:
        report = ingest_csv(DATASETS / "polymers.csv")
        peo = next(r for r in report.records if r.name == "poly(ethylene oxide)")
        assert peo.property_value("youngs_modulus") is None

    def test_empty_input_produces_an_empty_report(self) -> None:
        report = ingest_rows([])
        assert report.rows_read == 0
        assert report.reconciles()

    @requires_rdkit
    def test_jsonl_round_trip(self, tmp_path: Path) -> None:
        from polymer_engine.polymer.ingestion import load_records, save_records

        report = ingest_rows(
            [{"name": "PE", "repeat_unit_smiles": "*CC*", "Tg (degC)": -120}], source="test"
        )
        path = save_records(report.records, tmp_path / "records.jsonl")
        restored = load_records(path)
        assert len(restored) == 1
        assert restored[0].polymer_id == report.records[0].polymer_id
        assert restored[0].property_value("glass_transition_temperature") == pytest.approx(153.15)

    @requires_rdkit
    def test_records_needing_review_are_not_usable_for_modelling(self) -> None:
        record = build_record(name="mystery", repeat_unit_smiles="*[Se][Se]*")
        assert record.curation_status is Determination.REQUIRES_VALIDATION
        assert record.usable_for_modelling is False

    @requires_rdkit
    def test_clean_record_is_usable_for_modelling(self) -> None:
        record = build_record(name="PE", repeat_unit_smiles="*CC*")
        assert record.curation_status is Determination.KNOWN
        assert record.usable_for_modelling is True
