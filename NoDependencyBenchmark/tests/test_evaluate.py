"""Tier 1 tests: the benchmark functions and dispatch, without a gRPC server.

These run on any architecture. That is the point of moving the MOPTA
architecture check out of __init__ -- pestcontrol is pure numpy and has no
business being unreachable on an arm64 laptop.
"""
import numpy as np
import pytest
from bencherscaffold.protoclasses.bencher_pb2 import ConstraintType, EvaluationResult

from nodependencybenchmark.main import (BENCHMARKS, NoDependencyServiceServicer,
                                        _pest_control_score, _pressure_vessel, mopta_executable_basename)


@pytest.fixture
def servicer():
    return NoDependencyServiceServicer()


def test_servicer_constructs_on_any_architecture(servicer):
    """Regression: this used to raise RuntimeError on anything but x86/armv7l."""
    assert servicer is not None


def test_registry_shape_is_declared():
    assert set(BENCHMARKS) == {"mopta08", "pestcontrol", "pressure-vessel"}
    assert BENCHMARKS["pestcontrol"]["dimensions"] == 25


class TestPestControl:
    def test_is_deterministic_for_a_fixed_seed(self, servicer):
        x = np.ones(25)
        assert servicer.evaluate("pestcontrol", x, seed=7) == \
               servicer.evaluate("pestcontrol", x, seed=7)

    def test_different_seeds_give_different_draws(self, servicer):
        x = np.ones(25)
        # Stochastic by construction; two seeds agreeing exactly would mean the
        # seed is being ignored.
        assert servicer.evaluate("pestcontrol", x, seed=1) != \
               servicer.evaluate("pestcontrol", x, seed=2)

    def test_returns_a_single_finite_objective(self, servicer):
        result = servicer.evaluate("pestcontrol", np.ones(25), seed=0)
        assert isinstance(result, EvaluationResult)
        assert [o.name for o in result.objectives] == ["f0"]
        assert np.isfinite(result.objectives[0].value)

    def test_is_unconstrained(self, servicer):
        result = servicer.evaluate("pestcontrol", np.ones(25), seed=0)
        assert len(result.constraints) == 0

    @pytest.mark.parametrize("category", [0, 1, 2, 3, 4])
    def test_accepts_every_valid_category(self, servicer, category):
        """Values are used directly as categorical indices, 0 (no control) to 4."""
        result = servicer.evaluate("pestcontrol", np.full(25, category), seed=3)
        assert np.isfinite(result.objectives[0].value)

    def test_all_zero_means_no_control_is_applied(self):
        """With no pesticide applied, nothing is paid for control."""
        assert np.isfinite(_pest_control_score(np.zeros(25), seed=1).objectives[0].value)


class TestDispatch:
    def test_unknown_benchmark_is_rejected_by_name(self, servicer):
        with pytest.raises(ValueError, match="Invalid benchmark name"):
            servicer.evaluate("not-a-benchmark", np.ones(5))

    def test_mopta_architecture_error_names_the_architecture(self):
        """On a machine with no MOPTA binary the message should say so clearly."""
        import platform
        if (platform.machine().lower(), 64) in {("x86_64", 64), ("amd64", 64)}:
            assert mopta_executable_basename().startswith("mopta08_")
        else:
            with pytest.raises(RuntimeError, match="no executable for architecture"):
                mopta_executable_basename()

    def test_every_declared_benchmark_is_dispatched(self, servicer):
        """A name added to BENCHMARKS without a branch must not fall through silently."""
        for name in BENCHMARKS:
            if name == "mopta08":
                continue  # needs the x86 binary; covered by the e2e tier
            dimensions = BENCHMARKS[name]["dimensions"]
            result = servicer.evaluate(name, np.zeros(dimensions, dtype=int), seed=0)
            assert isinstance(result, EvaluationResult)


class TestPressureVessel:
    def test_published_optimum_is_feasible_with_g1_and_g3_active(self, servicer):
        """Best known design (0.8125, 0.4375, 42.098446, 176.636596), f* ~ 6059.714."""
        x = np.array([12.5 / 99, 6.5 / 99, (42.098446 - 10) / 190, (176.636596 - 10) / 190])
        result = servicer.evaluate("pressure-vessel", x)
        assert result.objectives[0].value == pytest.approx(6059.714, abs=1e-2)
        g1, g2, g3, g4 = (c.value for c in result.constraints)
        # g3 is a volume in cubic inches (~1.3e6 at the bound), hence the looser tolerance
        assert abs(g1) < 1e-3 and abs(g3) < 1.0
        assert g2 < 0 and g4 < 0

    @pytest.mark.parametrize("unit, native", [
        (0.0, (0.0625, 0.0625, 10.0, 10.0)),
        (1.0, (6.1875, 6.1875, 200.0, 200.0)),
    ])
    def test_unit_cube_corners_map_to_the_native_bounds(self, servicer, unit, native):
        result = servicer.evaluate("pressure-vessel", np.full(4, unit))
        expected = _pressure_vessel(np.array(native))
        assert result.objectives[0].value == pytest.approx(expected.objectives[0].value)
        assert [c.value for c in result.constraints] == \
               pytest.approx([c.value for c in expected.constraints])

    @pytest.mark.parametrize("unit, thickness", [
        (0.0, 0.0625),
        (12.5 / 99, 0.8125),
        (0.5, 3.125),
        (1.0, 6.1875),
    ])
    def test_thicknesses_snap_to_multiples_of_0_0625(self, servicer, unit, thickness):
        # with radius at its lower bound (10), g1 = 0.193 - x1 and g2 = 0.0954 - x2
        result = servicer.evaluate("pressure-vessel", np.array([unit, unit, 0.0, 0.0]))
        g1, g2, _, _ = (c.value for c in result.constraints)
        assert 0.0193 * 10 - g1 == pytest.approx(thickness)
        assert 0.00954 * 10 - g2 == pytest.approx(thickness)

    def test_constraints_are_named_inequalities(self, servicer):
        result = servicer.evaluate("pressure-vessel", np.full(4, 0.5))
        assert [c.name for c in result.constraints] == ["g1", "g2", "g3", "g4"]
        assert all(c.type == ConstraintType.INEQUALITY for c in result.constraints)

    def test_wrong_dimension_is_rejected(self, servicer):
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate("pressure-vessel", np.ones(3))
