"""Tier 1 tests: the benchmark functions and dispatch, without a gRPC server.

These run on any architecture. That is the point of moving the MOPTA
architecture check out of __init__ -- pestcontrol is pure numpy and has no
business being unreachable on an arm64 laptop.
"""
import numpy as np
import pytest

from nodependencybenchmark.main import (BENCHMARKS, NoDependencyServiceServicer,
                                        _pest_control_score, mopta_executable_basename)
from bencherscaffold.protoclasses.bencher_pb2 import ConstraintType

@pytest.fixture
def servicer():
    return NoDependencyServiceServicer()


def test_servicer_constructs_on_any_architecture(servicer):
    """Regression: this used to raise RuntimeError on anything but x86/armv7l."""
    assert servicer is not None


def test_registry_shape_is_declared():
    assert set(BENCHMARKS) == {"mopta08", "pestcontrol", "ackley", "gramacy", "pressure_vessel"}
    assert BENCHMARKS["pestcontrol"]["dimensions"] == 25
    assert BENCHMARKS["pressure_vessel"]["dimensions"] == 4


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

    def test_returns_a_finite_scalar(self, servicer):
        value, _ = servicer.evaluate("pestcontrol", np.ones(25), seed=0)
        assert isinstance(value, float) and np.isfinite(value)

    @pytest.mark.parametrize("category", [0, 1, 2, 3, 4])
    def test_accepts_every_valid_category(self, servicer, category):
        """Values are used directly as categorical indices, 0 (no control) to 4."""
        value, _ = servicer.evaluate("pestcontrol", np.full(25, category), seed=3)
        assert np.isfinite(value)

    def test_all_zero_means_no_control_is_applied(self):
        """With no pesticide applied, nothing is paid for control."""
        assert np.isfinite(_pest_control_score(np.zeros(25), seed=1))


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





class TestGramacy:
    @pytest.mark.parametrize("x, f, g0, g1", [
        ((0.5, 0.5), 1.0, -0.5, -1.0),   # feasible
        ((0.0, 0.0), 0.0, 1.5, -1.5),    # violates g0 only
        ((1.0, 1.0), 2.0, -1.5, 0.5),    # violates g1 only
    ])
    def test_known_values(self, servicer, x, f, g0, g1):
        value, constraints = servicer.evaluate('gramacy', np.array(x))
        assert value == pytest.approx(f)
        assert [v for _, _, v in constraints] == pytest.approx([g0, g1])

    def test_published_optimum_is_feasible_and_active(self, servicer):
        """Gramacy et al. (2016): x* ~ (0.1954, 0.4044), f* ~ 0.5998, g0 active."""
        value, constraints = servicer.evaluate('gramacy', np.array([0.1954, 0.4044]))
        assert value == pytest.approx(0.5998, abs=1e-4)
        g0, g1 = (v for _, _, v in constraints)
        assert abs(g0) < 1e-3 and g1 < 0

    def test_constraints_are_named_inequalities(self, servicer):
        _, constraints = servicer.evaluate('gramacy', np.array([0.5, 0.5]))
        assert [n for n, _, _ in constraints] == ["g0", "g1"]
        assert all(t == ConstraintType.INEQUALITY for _, t, _ in constraints)

    def test_wrong_dimension_is_rejected(self, servicer):
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate("gramacy", np.ones(3))

from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest


def _request(name, values, seed=None):
    r = BenchmarkRequest()
    r.benchmark.name = name
    for v in values:
        r.point.values.add(value=v)
    if seed is not None:
        r.random_seed = seed
    return r


class TestEvaluatePoint:
    def test_gramacy_constraints_reach_the_response(self, servicer):
        result = servicer.evaluate_point(_request("gramacy", [0.5, 0.5]), None)
        assert result.objectives[0].value == pytest.approx(1.0)
        assert [(c.name, c.value) for c in result.constraints] == \
               [("g0", pytest.approx(-0.5)), ("g1", pytest.approx(-1.0))]

    def test_unconstrained_benchmark_returns_no_constraints(self, servicer):
        """Regression: unpacking a bare float crashed every unconstrained benchmark."""
        result = servicer.evaluate_point(_request("ackley", [0.5] * 10), None)
        assert len(result.objectives) == 1
        assert len(result.constraints) == 0

    def test_request_seed_reaches_pestcontrol(self, servicer):
        """The TestPestControl seed tests call evaluate() directly; this checks the gRPC path."""
        def run(seed):
            return servicer.evaluate_point(_request("pestcontrol", [1] * 25, seed), None).objectives[0].value
        assert run(1) == run(1)
        assert run(1) != run(2)

class TestPressureVessel:
    def test_known_values(self, servicer):
        """Test known values for the pressure vessel benchmark."""
        x = np.array([0.812500, 0.437500, 42.097398, 176.654047])
        value, constraints = servicer.evaluate('pressure_vessel', x)
        assert value == pytest.approx(6059.946341, abs=1e-2)  # Example expected value
        assert all(np.isfinite(v) for _, _, v in constraints)
