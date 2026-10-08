"""Property calculations, validated against analytically known answers.

Every numeric assertion here has a closed-form expected value. The refusal tests
matter as much: a property that reports a number when the data cannot support one is
worse than a property that reports nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polymer_engine.core.errors import ParameterValidationError, ScientificError
from polymer_engine.core.models import Determination, GateStatus, Measurement
from polymer_engine.core.units import convert, dimension_of
from polymer_engine.properties import default_registry
from polymer_engine.properties.base import (
    PropertyClass,
    UncertaintyMethod,
    combine_replica_property,
)
from polymer_engine.properties.mechanical import (
    EXPERIMENTAL_STRAIN_RATE_CEILING,
    BulkModulus,
    MetricKind,
    StressStrainCurve,
    TensileAnalysis,
    find_linear_region,
    interpret_strain_rate,
    poisson_ratio,
)
from polymer_engine.properties.structural import FreeVolume, PersistenceLength
from polymer_engine.properties.thermodynamic import (
    Density,
    Pressure,
    StatePoint,
    ThermalExpansion,
)
from polymer_engine.properties.transport import (
    DiffusionCoefficient,
    DiffusionRegime,
    MeanSquaredDisplacement,
    RelaxationTime,
    classify_regime,
)


# ==========================================================================
# The framework contract
# ==========================================================================
class TestPropertyContract:
    def test_every_property_declares_its_full_contract(self) -> None:
        """Observable, units, estimator, uncertainty method and sampling are all required."""
        for name, definition in default_registry().definitions().items():
            assert definition["units"], name
            assert definition["observable"], name
            assert definition["estimator"], name
            assert definition["uncertainty_method"], name
            assert definition["sampling"]["min_replicas"] >= 1, name

    def test_registry_covers_every_property_class(self) -> None:
        registry = default_registry()
        for property_class in PropertyClass:
            assert registry.by_class(property_class), f"no property for {property_class.value}"

    def test_unknown_property_raises(self) -> None:
        with pytest.raises(ScientificError):
            default_registry().get("nonexistent_property")

    def test_units_are_validated_at_definition_time(self) -> None:
        from polymer_engine.core.errors import ParameterValidationError
        from polymer_engine.properties.base import PropertyDefinition

        with pytest.raises(ParameterValidationError):
            PropertyDefinition(
                name="bogus", property_class=PropertyClass.THERMODYNAMIC, units="furlongs",
                observable="x", estimator="y", uncertainty_method=UncertaintyMethod.NONE,
            )


# ==========================================================================
# Thermodynamic
# ==========================================================================
class TestThermodynamic:
    def test_density_recovers_a_known_mean(self) -> None:
        values = 1000.0 + np.random.default_rng(1).normal(0, 2.0, 5000)
        result = Density().compute(values, n_replicas=3, simulation_ns=50.0)
        assert result.measurement.value == pytest.approx(1000.0, abs=0.5)
        assert result.usable is True

    def test_density_uncertainty_is_correlation_aware(self) -> None:
        """A correlated series must report a larger error than its frame count implies."""
        rng = np.random.default_rng(2)
        x = np.zeros(20000)
        for i in range(1, x.size):
            x[i] = 0.95 * x[i - 1] + rng.normal(0, 1.0)
        result = Density().compute(x + 1000.0, n_replicas=3, simulation_ns=50.0)
        naive = float(np.std(x, ddof=1)) / math.sqrt(x.size)
        assert result.measurement.uncertainty > 3 * naive

    def test_too_little_sampling_is_refused(self) -> None:
        result = Density().compute(
            1000.0 + np.random.default_rng(3).normal(0, 2.0, 40), n_replicas=1, simulation_ns=0.4
        )
        assert result.usable is False
        assert any(g.status is GateStatus.FAIL for g in result.report.gates)

    def test_one_replica_is_inconclusive_not_a_pass(self) -> None:
        result = Density().compute(
            1000.0 + np.random.default_rng(4).normal(0, 1.0, 5000), n_replicas=1, simulation_ns=50.0
        )
        replica_gate = next(g for g in result.report.gates if g.gate == "density:replicas")
        assert replica_gate.status is GateStatus.INCONCLUSIVE
        assert result.usable is False

    def test_pressure_demands_more_sampling_than_density(self) -> None:
        """Pressure is far noisier, and its requirement says so."""
        assert (
            Pressure.definition.sampling.min_effective_samples
            > Density.definition.sampling.min_effective_samples
        )

    def test_empty_series_is_refused(self) -> None:
        assert Density().compute([]).usable is False

    def test_non_finite_series_is_refused(self) -> None:
        result = Density().compute([1000.0, float("nan"), 1001.0])
        assert result.usable is False
        assert result.measurement.determination is Determination.UNKNOWN


class TestThermalExpansion:
    def test_recovers_a_known_coefficient(self) -> None:
        """rho = 1200 - 0.7(T-300) gives alpha = 0.7/rho_mean."""
        points = [
            StatePoint(
                t,
                Measurement(name="density", value=1200.0 - 0.7 * (t - 300), uncertainty=0.5, units="kg/m^3"),
            )
            for t in (280.0, 300.0, 320.0, 340.0, 360.0)
        ]
        result = ThermalExpansion().compute(points)
        assert result.usable is True
        mean_density = float(np.mean([p.density.value for p in points]))
        assert result.measurement.value == pytest.approx(0.7 / mean_density, rel=0.02)

    def test_two_state_points_are_not_enough_for_a_derivative(self) -> None:
        points = [
            StatePoint(t, Measurement(name="density", value=1200.0 - 0.7 * (t - 300), units="kg/m^3"))
            for t in (300.0, 320.0)
        ]
        assert ThermalExpansion().compute(points).usable is False

    def test_an_unresolved_slope_is_refused(self) -> None:
        """A slope consistent with zero is not a thermal expansion coefficient."""
        rng = np.random.default_rng(5)
        points = [
            StatePoint(
                t, Measurement(name="density", value=1200.0 + rng.normal(0, 5.0), uncertainty=5.0, units="kg/m^3")
            )
            for t in (298.0, 300.0, 302.0)
        ]
        result = ThermalExpansion().compute(points)
        assert result.usable is False
        assert result.measurement.determination is Determination.INSUFFICIENT_DATA
        assert any("not distinguishable from zero" in g.message for g in result.report.gates)

    def test_unusable_densities_are_excluded(self) -> None:
        points = [
            StatePoint(280.0, Measurement.unknown("density", units="kg/m^3")),
            StatePoint(300.0, Measurement(name="density", value=1200.0, units="kg/m^3")),
            StatePoint(320.0, Measurement(name="density", value=1186.0, units="kg/m^3")),
        ]
        assert ThermalExpansion().compute(points).usable is False


# ==========================================================================
# Structural
# ==========================================================================
class TestStructural:
    def test_persistence_length_is_exact_for_an_ideal_decay(self) -> None:
        bond, persistence = 0.153, 0.6
        correlations = np.exp(-np.arange(20) * bond / persistence)
        result = PersistenceLength().compute(correlations, bond, n_replicas=3)
        assert result.measurement.value == pytest.approx(persistence, rel=1e-6)

    def test_persistence_length_scales_with_stiffness(self) -> None:
        bond = 0.153
        stiff = PersistenceLength().compute(np.exp(-np.arange(30) * bond / 2.0), bond, n_replicas=3)
        floppy = PersistenceLength().compute(np.exp(-np.arange(30) * bond / 0.3), bond, n_replicas=3)
        assert stiff.measurement.value > floppy.measurement.value

    def test_a_correlation_that_never_decays_is_refused(self) -> None:
        result = PersistenceLength().compute(np.ones(20), 0.153, n_replicas=3)
        assert result.measurement.determination is not Determination.KNOWN

    def test_noise_only_correlation_is_refused(self) -> None:
        """Fitting the tail after the signal is gone gives a confident wrong answer."""
        result = PersistenceLength().compute([1.0, 0.02, -0.01, 0.005, -0.002], 0.153, n_replicas=3)
        assert result.measurement.determination is Determination.INSUFFICIENT_DATA
        assert "decays into the noise" in result.diagnostics[0]

    def test_free_volume_matches_an_analytic_sphere(self) -> None:
        """One sphere of radius 0.15 nm in a 1 nm^3 box leaves 1 - (4/3)pi r^3."""
        result = FreeVolume().compute(
            np.array([[0.5, 0.5, 0.5]]), np.array([0.15]), (1.0, 1.0, 1.0),
            probe_radius_nm=0.0, grid_spacing_nm=0.02, n_replicas=3,
        )
        expected = 1.0 - (4.0 / 3.0) * math.pi * 0.15**3
        assert result.measurement.value == pytest.approx(expected, abs=0.005)

    def test_a_larger_probe_finds_less_free_volume(self) -> None:
        positions = np.array([[0.5, 0.5, 0.5]])
        radii = np.array([0.15])
        small = FreeVolume().compute(
            positions, radii, (1.0, 1.0, 1.0), probe_radius_nm=0.0, grid_spacing_nm=0.03, n_replicas=3
        )
        large = FreeVolume().compute(
            positions, radii, (1.0, 1.0, 1.0), probe_radius_nm=0.15, grid_spacing_nm=0.03, n_replicas=3
        )
        assert large.measurement.value < small.measurement.value

    def test_probe_radius_is_recorded_with_the_result(self) -> None:
        """Two free volumes computed with different probes are not comparable."""
        result = FreeVolume().compute(
            np.array([[0.5, 0.5, 0.5]]), np.array([0.15]), (1.0, 1.0, 1.0),
            probe_radius_nm=0.11, grid_spacing_nm=0.05, n_replicas=3,
        )
        assert result.provenance["probe_radius_nm"] == 0.11
        assert "positron" in (result.measurement.notes or "")

    def test_mismatched_radii_are_refused(self) -> None:
        result = FreeVolume().compute(
            np.array([[0.5, 0.5, 0.5]]), np.array([0.15, 0.2]), (1.0, 1.0, 1.0), n_replicas=3
        )
        assert result.usable is False


# ==========================================================================
# Transport
# ==========================================================================
class TestRegimeClassification:
    @pytest.mark.parametrize(
        "exponent,expected",
        [
            (1.0, DiffusionRegime.DIFFUSIVE),
            (2.0, DiffusionRegime.BALLISTIC),
            (0.5, DiffusionRegime.SUBDIFFUSIVE),
            (0.75, DiffusionRegime.SUBDIFFUSIVE),
            (1.4, DiffusionRegime.SUPERDIFFUSIVE),
        ],
    )
    def test_exponent_determines_the_regime(self, exponent: float, expected: DiffusionRegime) -> None:
        lags = np.arange(1, 400, dtype=float)
        diagnosis = classify_regime(lags, 0.05 * lags**exponent)
        assert diagnosis.regime is expected
        assert diagnosis.alpha == pytest.approx(exponent, abs=0.02)

    def test_too_few_lags_is_undetermined(self) -> None:
        assert classify_regime([1.0, 2.0], [1.0, 2.0]).regime is DiffusionRegime.UNDETERMINED

    def test_mismatched_arrays_are_undetermined(self) -> None:
        assert classify_regime([1.0, 2.0, 3.0], [1.0]).regime is DiffusionRegime.UNDETERMINED


class TestDiffusion:
    def test_recovers_a_known_coefficient(self) -> None:
        """MSD = 6*D*t in 3D, so slope 0.06 nm^2/ps gives D = 0.01."""
        lags = np.arange(1, 400, dtype=float)
        result = DiffusionCoefficient().compute(
            lags, 0.06 * lags, n_replicas=3, simulation_ns=100.0
        )
        assert result.measurement.value == pytest.approx(0.01, rel=1e-6)
        assert result.usable is True

    def test_dimensionality_changes_the_divisor(self) -> None:
        lags = np.arange(1, 400, dtype=float)
        three_d = DiffusionCoefficient().compute(lags, 0.06 * lags, dimensionality=3, n_replicas=3, simulation_ns=100.0)
        one_d = DiffusionCoefficient().compute(lags, 0.06 * lags, dimensionality=1, n_replicas=3, simulation_ns=100.0)
        assert one_d.measurement.value == pytest.approx(3 * three_d.measurement.value)

    def test_ballistic_motion_is_refused(self) -> None:
        """Constant-velocity drift is not diffusion, however well a line fits it."""
        lags = np.arange(1, 400, dtype=float)
        result = DiffusionCoefficient().compute(lags, 0.01 * lags**2, n_replicas=3, simulation_ns=100.0)
        assert result.usable is False
        assert result.extra["regime"] == "ballistic"
        assert "Einstein relation does not apply" in (result.measurement.notes or "")

    def test_subdiffusive_motion_is_refused(self) -> None:
        """A polymer melt in the Rouse regime is subdiffusive; D from it would be wrong."""
        lags = np.arange(1, 400, dtype=float)
        result = DiffusionCoefficient().compute(lags, 0.5 * lags**0.5, n_replicas=3, simulation_ns=100.0)
        assert result.usable is False
        assert result.extra["regime"] == "subdiffusive"

    def test_finite_size_caveat_is_recorded(self) -> None:
        lags = np.arange(1, 400, dtype=float)
        result = DiffusionCoefficient().compute(lags, 0.06 * lags, n_replicas=3, simulation_ns=100.0)
        assert "finite-size" in result.definition.caveats.lower()

    def test_msd_reports_its_regime(self) -> None:
        lags = np.arange(1, 400, dtype=float)
        result = MeanSquaredDisplacement().compute(lags, 0.05 * lags, n_replicas=3)
        assert result.extra["regime"] == "diffusive"


class TestRelaxation:
    def test_recovers_a_known_time_constant(self) -> None:
        times = np.arange(0, 500, 1.0)
        result = RelaxationTime().compute(times, np.exp(-times / 50.0), n_replicas=3)
        assert result.measurement.value == pytest.approx(50.0, rel=0.01)

    def test_an_undecayed_correlation_gives_a_bound_not_a_value(self) -> None:
        times = np.arange(0, 20, 1.0)
        result = RelaxationTime().compute(times, np.exp(-times / 500.0), n_replicas=3)
        assert result.measurement.determination is Determination.INSUFFICIENT_DATA
        assert result.extra["lower_bound_ps"] == pytest.approx(19.0)
        assert "exceeds the" in result.measurement.notes
        assert "simulated window" in result.measurement.notes

    def test_a_short_window_is_warned_about(self) -> None:
        times = np.arange(0, 120, 1.0)
        result = RelaxationTime().compute(times, np.exp(-times / 50.0), n_replicas=3)
        window_gate = next(g for g in result.report.gates if g.gate.endswith("window_length"))
        assert window_gate.status is GateStatus.WARN


# ==========================================================================
# Mechanical
# ==========================================================================
def synthetic_curve(strain_rate: float | None = 1.0e8) -> StressStrainCurve:
    """E = 2000 MPa to 2% strain, yielding near 5%, then softening and hardening."""
    strain = np.linspace(0.0, 0.30, 200)
    stress = np.where(
        strain <= 0.02, 2000 * strain,
        np.where(strain <= 0.05, 40 + (60 - 40) * (strain - 0.02) / 0.03,
                 60 - 15 * (strain - 0.05) / 0.10),
    )
    stress = np.where(strain > 0.15, 45 + 30 * (strain - 0.15), stress)
    return StressStrainCurve(
        strain=strain, stress_mpa=stress, strain_rate_per_s=strain_rate, temperature_k=300.0
    )


class TestMechanical:
    def test_elastic_modulus_is_recovered_exactly(self) -> None:
        analysis = TensileAnalysis().analyse(synthetic_curve(), n_replicas=3)
        assert analysis.elastic_modulus.value == pytest.approx(2000.0, rel=1e-6)

    def test_yield_and_peak_are_identified(self) -> None:
        analysis = TensileAnalysis().analyse(synthetic_curve(), n_replicas=3)
        assert analysis.yield_strain.value == pytest.approx(0.05, abs=0.005)
        assert analysis.yield_stress_proxy.value == pytest.approx(60.0, abs=0.5)
        assert analysis.peak_stress.value == pytest.approx(60.0, abs=0.5)

    def test_md_strain_rates_are_not_experimentally_comparable(self) -> None:
        """This is the claim the module exists to prevent."""
        analysis = TensileAnalysis().analyse(synthetic_curve(strain_rate=1.0e8), n_replicas=3)
        assert analysis.interpretation.comparable_to_experiment is False
        assert "rate dependent" in analysis.interpretation.rationale

    def test_yield_is_labelled_a_proxy_not_a_strength(self) -> None:
        analysis = TensileAnalysis().analyse(synthetic_curve(), n_replicas=3)
        assert analysis.yield_stress_proxy.name == "yield_stress_proxy"
        assert "not an experimental" in (analysis.yield_stress_proxy.notes or "")
        assert "not an experimental" in (analysis.peak_stress.notes or "")

    def test_an_unrecorded_strain_rate_blocks_comparability(self) -> None:
        analysis = TensileAnalysis().analyse(synthetic_curve(strain_rate=None), n_replicas=3)
        assert analysis.interpretation.comparable_to_experiment is False
        assert "not recorded" in analysis.interpretation.rationale

    def test_a_monotonic_curve_has_not_yielded(self) -> None:
        strain = np.linspace(0.0, 0.01, 50)
        curve = StressStrainCurve(strain=strain, stress_mpa=2000 * strain, strain_rate_per_s=1e8)
        analysis = TensileAnalysis().analyse(curve, n_replicas=3)
        assert analysis.yield_stress_proxy.determination is Determination.INSUFFICIENT_DATA
        assert analysis.elastic_modulus.value == pytest.approx(2000.0, rel=1e-6)

    def test_a_curve_with_no_linear_region_is_refused_a_modulus(self) -> None:
        strain = np.linspace(0.0, 0.02, 60)
        rng = np.random.default_rng(7)
        curve = StressStrainCurve(
            strain=strain, stress_mpa=rng.normal(50, 30, strain.size), strain_rate_per_s=1e8
        )
        analysis = TensileAnalysis().analyse(curve, n_replicas=3)
        assert analysis.elastic_modulus.determination is Determination.INSUFFICIENT_DATA

    def test_strain_hardening_sign_is_meaningful(self) -> None:
        analysis = TensileAnalysis().analyse(synthetic_curve(), n_replicas=3)
        assert analysis.strain_hardening_slope.value is not None
        assert "softening" in (analysis.strain_hardening_slope.notes or "")

    def test_mismatched_curve_arrays_are_rejected(self) -> None:
        with pytest.raises(ScientificError):
            StressStrainCurve(strain=np.array([0.0, 0.1]), stress_mpa=np.array([0.0]))

    def test_decreasing_strain_is_rejected(self) -> None:
        with pytest.raises(ScientificError, match="non-decreasing"):
            StressStrainCurve(strain=np.array([0.0, 0.2, 0.1]), stress_mpa=np.array([0.0, 1.0, 2.0]))

    def test_slow_strain_rate_is_treated_differently(self) -> None:
        slow = interpret_strain_rate(1.0e-3, MetricKind.SIMULATION_OBSERVABLE)
        fast = interpret_strain_rate(1.0e9, MetricKind.SIMULATION_OBSERVABLE)
        assert slow.strain_rate_per_s < EXPERIMENTAL_STRAIN_RATE_CEILING
        assert fast.comparable_to_experiment is False

    def test_linear_region_detection_adapts_to_the_curve(self) -> None:
        strain = np.linspace(0, 0.05, 100)
        stress = np.where(strain <= 0.01, 3000 * strain, 30 + 200 * (strain - 0.01))
        region = find_linear_region(strain, stress, max_strain=0.05)
        assert region is not None
        assert strain[region[1] - 1] == pytest.approx(0.01, abs=0.005)


class TestBulkModulus:
    def test_recovers_a_known_modulus(self) -> None:
        """K = kT<V>/var(V); construct volumes with the variance that gives a target K."""
        from polymer_engine.core.units import GAS_CONSTANT_KJ_PER_MOL_K

        temperature = 300.0
        mean_volume = 100.0
        target_mpa = 2000.0
        kt = GAS_CONSTANT_KJ_PER_MOL_K * temperature
        target_kj = target_mpa / 1.6605390666
        variance = kt * mean_volume / target_kj

        rng = np.random.default_rng(11)
        volumes = rng.normal(mean_volume, math.sqrt(variance), 200_000)
        result = BulkModulus().compute(volumes, temperature, n_replicas=3, simulation_ns=100.0)
        assert result.measurement.value == pytest.approx(target_mpa, rel=0.05)

    def test_a_variance_estimator_demands_more_sampling(self) -> None:
        assert BulkModulus.definition.sampling.min_effective_samples >= 200

    def test_a_constant_volume_is_refused(self) -> None:
        """No volume fluctuation means the run was not NPT."""
        result = BulkModulus().compute(np.full(1000, 100.0), 300.0, n_replicas=3)
        assert result.usable is False
        assert "not run in an NPT ensemble" in result.diagnostics[0]

    def test_barostat_caveat_is_recorded(self) -> None:
        assert "Berendsen" in BulkModulus.definition.caveats


class TestPoissonRatio:
    def test_recovers_a_known_ratio(self) -> None:
        axial = np.linspace(0, 0.05, 20)
        assert poisson_ratio(axial, -0.35 * axial).value == pytest.approx(0.35, rel=1e-6)

    def test_a_value_outside_thermodynamic_bounds_is_refused(self) -> None:
        axial = np.linspace(0, 0.05, 20)
        result = poisson_ratio(axial, -0.9 * axial)
        assert result.determination is Determination.REQUIRES_VALIDATION
        assert "thermodynamic bounds" in (result.notes or "")

    def test_constant_axial_strain_is_refused(self) -> None:
        assert poisson_ratio(np.full(10, 0.01), np.linspace(0, -0.01, 10)).value is None


# ==========================================================================
# Replica combination
# ==========================================================================
class TestReplicaCombination:
    def test_combined_uncertainty_uses_replica_spread(self) -> None:
        """Using the frame-level error here would be pseudoreplication."""
        per_replica = [
            Measurement(name="density", value=v, uncertainty=0.01, units="kg/m^3",
                        n_samples=1000, effective_samples=500.0)
            for v in (1000.0, 1010.0, 990.0)
        ]
        result = combine_replica_property(Density.definition, per_replica)
        assert result.measurement.uncertainty > 1.0
        assert result.n_replicas == 3

    def test_disagreeing_replicas_fail_the_gate(self) -> None:
        per_replica = [
            Measurement(name="density", value=v, uncertainty=0.5, units="kg/m^3",
                        n_samples=1000, effective_samples=500.0)
            for v in (1000.0, 1080.0, 950.0)
        ]
        result = combine_replica_property(Density.definition, per_replica)
        assert result.usable is False
        assert any("disagree" in g.message for g in result.report.gates)

    def test_two_replicas_are_inconclusive(self) -> None:
        per_replica = [
            Measurement(name="density", value=v, uncertainty=0.5, units="kg/m^3",
                        n_samples=1000, effective_samples=500.0)
            for v in (1000.0, 1001.0)
        ]
        result = combine_replica_property(Density.definition, per_replica)
        assert result.usable is False


class TestDeclaredUnitsAreReal:
    """Regression: four calculators declared ``units="1"`` while carrying real quantities.

    A volume in nm^3 labelled dimensionless is exactly the hidden-unit failure the
    engine exists to prevent -- it converts without complaint against any other
    "dimensionless" number. The fix was to teach the registry about volume, area,
    diffusivity and inverse temperature, not to keep the true unit in prose.
    """

    def test_every_property_declares_a_unit_the_registry_knows(self) -> None:
        registry = default_registry()
        for name in registry.names():
            units = registry.get(name).definition.units
            assert units, name
            dimension_of(units)  # raises for a unit the registry does not know

    @pytest.mark.parametrize(
        ("name", "expected_dimension"),
        [
            ("volume", "volume"),
            ("thermal_expansion_coefficient", "inverse_temperature"),
            ("mean_squared_displacement", "area"),
            ("diffusion_coefficient", "diffusivity"),
        ],
    )
    def test_the_previously_mislabelled_properties_carry_their_dimension(
        self, name: str, expected_dimension: str
    ) -> None:
        units = default_registry().get(name).definition.units
        assert dimension_of(units) == expected_dimension

    def test_a_diffusion_coefficient_cannot_be_converted_to_a_volume(self) -> None:
        """The point of the fix: wrong-dimension comparisons now raise."""
        with pytest.raises(ParameterValidationError):
            convert(1.0, "nm^2/ps", "nm^3")

    def test_a_measured_diffusion_coefficient_reports_its_unit(self) -> None:
        lags = np.linspace(1.0, 100.0, 200)
        msd = 0.6 * lags  # exactly diffusive: D = 0.6 / 6 = 0.1 nm^2/ps
        result = DiffusionCoefficient().compute(lags, msd, n_replicas=3, simulation_ns=100.0)
        assert result.measurement.units == "nm^2/ps"
        assert result.measurement.value == pytest.approx(0.1, rel=1e-6)
        # And the value converts correctly into the unit an experimentalist would use.
        assert convert(result.measurement.value, "nm^2/ps", "cm^2/s") == pytest.approx(1e-3)

    def test_a_refused_diffusion_coefficient_still_reports_its_unit(self) -> None:
        lags = np.linspace(1.0, 100.0, 200)
        result = DiffusionCoefficient().compute(lags, lags**2, n_replicas=3)  # ballistic
        assert result.measurement.determination is not Determination.KNOWN
        assert result.measurement.units == "nm^2/ps"
