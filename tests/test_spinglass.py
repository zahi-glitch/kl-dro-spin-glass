import numpy as np

from tapdual.spinglass import PATTERNS, REM, SK, Hopfield, heat_bath


def test_patterns_nearly_orthogonal():
    C = PATTERNS @ PATTERNS.T / PATTERNS.shape[1]
    assert np.allclose(np.diag(C), 1) and np.abs(C[np.triu_indices(3, 1)]).max() < 0.35


def test_hopfield_recalls_from_noise():
    net, rng = Hopfield(), np.random.default_rng(0)
    for mu in range(3):
        s = net.corrupt(mu, 0.25, rng)
        heat_bath(s, net.J, 0.0, rng, 20 * net.N)
        assert net.overlaps(s)[mu] > 0.99


def test_hopfield_forgets_at_high_temperature():
    net, rng = Hopfield(), np.random.default_rng(1)
    s = PATTERNS[0].copy()
    heat_bath(s, net.J, 3.0, rng, 40 * net.N)
    assert abs(net.overlaps(s)[0]) < 0.3


def test_sk_low_temperature_energy_near_parisi():
    sk, rng = SK(seed=3), np.random.default_rng(2)
    s = rng.choice([-1.0, 1.0], sk.N)
    for T in np.linspace(1.5, 0.05, 30):           # slow annealing
        heat_bath(s, sk.J, T, rng, 10 * sk.N)
    assert -0.80 < sk.energy(s) < -0.68            # N = 256: close to -0.7633


def test_rem_freezing():
    rem = REM(n=16, seed=0)
    assert rem.entropy_per_spin(0.0) > 0.69        # uniform: log 2
    assert rem.effective_states(3.0) < 5           # frozen onto a few states
    assert abs(rem.free_energy(0.5) - REM.free_energy_theory(0.5)) < 0.02
