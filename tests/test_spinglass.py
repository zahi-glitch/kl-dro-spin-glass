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


def test_gibbs_variational_principle():
    from tapdual.spinglass import boltzmann, log_Z, terms
    for beta in (0.3, 1.2, 2.5):
        best = terms(boltzmann(beta), beta)[2]
        assert abs(best - log_Z(beta)) < 1e-12
        for t in (0.0, beta / 2, beta + 1):
            q, p = boltzmann(t), boltzmann(beta)
            kl = float((q * np.log(q / p)).sum())
            assert abs(log_Z(beta) - terms(q, beta)[2] - kl) < 1e-12


def test_rbm_marginal_is_hopfield():
    """Block Gibbs on the RBM samples the Hopfield distribution (exact check on 2^6 states).

    Many independent chains, so slow mixing between memory valleys does not bias the check."""
    from itertools import product
    from tapdual.spinglass import RBM
    rng = np.random.default_rng(0)
    xi = rng.choice([-1.0, 1.0], (2, 6))
    rbm = RBM(xi, beta=1.5)
    states = np.array(list(product([-1.0, 1.0], repeat=6)))
    logw = np.array([rbm.hopfield_log_weight(s) for s in states])
    exact = np.exp(logw - logw.max()); exact /= exact.sum()
    S = np.tile(states[0], (20000, 1))
    for _ in range(150):                                   # vectorised block Gibbs, 20000 chains
        Z = rbm.scale * (S @ xi.T) + rng.standard_normal((S.shape[0], 2))
        H = rbm.scale * (Z @ xi)
        S = np.where(rng.random(S.shape) < 1.0 / (1.0 + np.exp(-2.0 * H)), 1.0, -1.0)
    codes = ((S > 0) * (2 ** np.arange(5, -1, -1))).sum(1)
    empirical = np.bincount(codes, minlength=64) / S.shape[0]
    assert 0.5 * np.abs(empirical - exact).sum() < 0.02    # total variation distance


def test_rbm_recalls_memories():
    from tapdual.spinglass import RBM
    rbm, rng = RBM(beta=3.0), np.random.default_rng(1)
    for mu in range(3):
        s = PATTERNS[mu].copy()
        s[128:] = rng.choice([-1.0, 1.0], 128)           # half unknown
        for _ in range(15):
            s = rbm.sample_visible(rbm.sample_hidden(s, rng), rng)
        assert rbm.overlaps(s)[mu] > 0.95
