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


def test_virtual_investment(prices_csv, tmp_path, monkeypatch):
    from tapdual import live
    monkeypatch.setattr(live, "VIRTUAL", tmp_path / "virtual.json")
    prices = live.get_prices(list(live.UNIVERSE), prices_csv)
    start = prices.index[-80].date()
    v = live.simulate(prices, start, 10_000)
    for name, values in v.values.items():
        assert len(values) == 80
        assert abs(values[0] - 9_990) < 1e-6                  # bought at the start, paying 0.1%
        assert abs(v.weights_now[name].sum() - 1) < 1e-9
    # no look-ahead: the path up to a day does not depend on later prices
    v_short = live.simulate(prices.iloc[:-20], start, 10_000)
    for name in v.values:
        assert np.allclose(v_short.values[name], v.values[name][:60])
    live.start_virtual(prices)
    assert live.load_virtual()[0] == prices.index[-1].date()


def test_track_script(prices_csv, tmp_path, monkeypatch):
    """The automatic tracker starts the virtual investment, writes the report, and logs once per month."""
    import importlib.util
    from pathlib import Path
    from tapdual import live
    monkeypatch.setenv("TAPDUAL_PRICES_CSV", str(prices_csv))
    monkeypatch.setattr(live, "VIRTUAL", tmp_path / "virtual_portfolio.json")
    monkeypatch.setattr(live, "LOG", tmp_path / "log.csv")
    spec = importlib.util.spec_from_file_location("track", Path(__file__).resolve().parents[1] / "scripts" / "track.py")
    track = importlib.util.module_from_spec(spec); spec.loader.exec_module(track)
    track.main(); track.main()
    assert (tmp_path / "virtual_report.md").exists() and (tmp_path / "virtual_values.csv").exists()
    lines = (tmp_path / "log.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + len(live.STRATEGIES)        # header + one entry: the second run did not log again
