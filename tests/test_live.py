"""Live tool on a synthetic price file (no internet)."""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")


@pytest.fixture
def prices_csv(tmp_path):
    from tapdual.live import UNIVERSE
    rng = np.random.default_rng(0)
    R = rng.standard_t(5, (700, len(UNIVERSE))) * 0.008 + 0.0002
    df = pd.DataFrame(100 * np.cumprod(1 + R, 0), index=pd.bdate_range("2023-01-02", periods=700), columns=list(UNIVERSE))
    path = tmp_path / "prices.csv"; df.to_csv(path)
    return path


def test_compute_and_log(prices_csv, tmp_path, monkeypatch):
    from tapdual import live
    monkeypatch.setattr(live, "LOG", tmp_path / "log.csv")
    today = live.compute(live.get_prices(list(live.UNIVERSE), prices_csv))
    for w in today.weights.values():
        assert abs(w.sum() - 1) < 1e-9 and (w >= -1e-12).all()
    assert len(today.worst_days) == 5 and today.window == 504
    assert live.trades_since_last_log(today) is None
    live.append_log(today)
    assert live.trades_since_last_log(today) == []          # same weights: nothing to trade
    # a different list of assets starts a fresh log instead of mixing columns
    today.tickers = today.tickers[:-1]
    assert "moved" in live.start_fresh_log_if_universe_changed(today.tickers)
