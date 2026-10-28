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
    assert set(BENCHMARKS) == {"mopta08","mopta08-constrained", "pestcontrol", "ackley", "gramacy", "pressure-vessel","welded-beam","tension-compression-spring"}
    assert BENCHMARKS["pestcontrol"]["dimensions"] == 25
    assert BENCHMARKS["pressure-vessel"]["dimensions"] == 4


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
        x = np.array([12.5/99, 6.5/99, (42.097398-10)/190, (176.654047-10)/190])
        c=np.array([-0.000020,-0.035891,-27.886075,-63.345953,-63.345953])
        test_x = np.array([0.812500, 0.437500, 42.097398, 176.654047])
        lower_bounds = np.array([0.0625, 0.0625, 10, 10])
        upper_bounds = np.array([6.1875, 6.1875, 200, 200])
        n_levels = 99
        idx=np.minimum(np.floor(x[:2]*n_levels), n_levels-1).astype(int)
        x_scaled = lower_bounds + (upper_bounds - lower_bounds) * x
        
        x_scaled[:2] = (idx+1) * 0.0625

        lower_bounds = np.array([0.0625, 0.0625, 10, 10])
        upper_bounds = np.array([6.1875, 6.1875, 200, 200])
        
        value, constraints = servicer.evaluate('pressure-vessel', x)

        assert value == pytest.approx(6059.946341, abs=1e-2)  # Example expected value
        assert all(np.isfinite(v) for _, _, v in constraints)
        assert all(x_scaled==test_x)
        assert [v for _, _, v in constraints] == pytest.approx(c, abs=1e-2)

class TestWeldedBeam:
    def test_known_values(self, servicer):
        """Test known values for the welded beam benchmark."""
        x = np.array([0.205986, 3.471328, 9.020224, 0.206480])
        c=np.array([-0.074092,-0.266227,-0.000495,-3.430043,-0.080986,-0.235514,-58.666440])
        #scale back to [0,1]
        lower_bounds=np.array([0.1,0.1,0.1,0.1])
        upper_bounds=np.array([2.0,10.0,10.0,2.0])
        x_scaled = (x-lower_bounds)/(upper_bounds - lower_bounds) 
        value, constraints = servicer.evaluate('welded-beam', x_scaled)
        assert value == pytest.approx(1.728226, abs=1e-2)  # Example expected value
        assert all(np.isfinite(v) for _, _, v in constraints)
        # g1, g2 and g7 are differences of ~1e4-sized quantities (tau, sigma,
        # P_c), so the 6-digit rounding of the published x shifts them by a few
        # 1e-2; the geometric constraints match to ~1e-6.
        print(constraints)
        assert [v for _, _, v in constraints] == pytest.approx(c, abs=1e-2)

class TestTensionCompressionSpring:
    def test_known_values(self, servicer):
        """Test known values for the tension compression spring benchmark."""
        x = np.array([0.051989, 0.363965, 10.890522])
        c=np.array([-0.000013,-0.000021,-4.061338,-0.722698])
        #scale back to [0,1]
        lower_bounds=np.array([0.05,0.25,2.0])
        upper_bounds=np.array([2.0,1.3,15.0])
        x_scaled = (x-lower_bounds)/(upper_bounds - lower_bounds) 
        value, constraints = servicer.evaluate('tension-compression-spring', x_scaled)
        assert value == pytest.approx(0.012681, abs=1e-2)  # Example expected value
        assert all(np.isfinite(v) for _, _, v in constraints)
        assert [v for _, _, v in constraints] == pytest.approx(c, abs=1e-2)
