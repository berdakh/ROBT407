"""Metrics against known answers, and the report card's warning logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade import metrics as M
from algotrade.backtest import BacktestConfig, run_backtest
from algotrade.execution import RETAIL_CRYPTO, ZERO_COST
from algotrade.report import compare, report_card
from algotrade.strategy import BuyAndHold, RandomStrategy, SMACrossover

BPY = 8760


@pytest.fixture(scope="module")
def known_returns():
    """Gaussian returns with an exactly specified annualised volatility."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.10 / BPY, 0.20 / np.sqrt(BPY), BPY * 3))


def test_annualized_volatility_recovers_the_truth(known_returns):
    assert M.annualized_volatility(known_returns, BPY) == pytest.approx(0.20, abs=0.01)


def test_sharpe_recovers_the_truth(known_returns):
    # True Sharpe is 0.10/0.20 = 0.5. Three years is still a noisy estimate,
    # which is the point of also reporting the standard error.
    sr = M.sharpe_ratio(known_returns, BPY)
    se = M.sharpe_standard_error(known_returns, BPY)
    assert abs(sr - 0.5) < 3 * se


def test_zero_volatility_gives_nan_not_infinity():
    flat = pd.Series(np.zeros(1000))
    assert np.isnan(M.sharpe_ratio(flat, BPY))


def test_max_drawdown_on_a_handmade_curve():
    equity = pd.Series([100, 120, 60, 80, 200.0])   # peak 120 -> trough 60
    assert M.max_drawdown(equity) == pytest.approx(-0.5)


def test_drawdown_is_never_positive(gbm_data):
    res = run_backtest(gbm_data, SMACrossover(12, 48))
    assert M.drawdown_series(res.equity).max() <= 1e-12


def test_turnover_counts_every_flip():
    weights = pd.Series(np.tile([1.0, 0.0], 500))    # 999 flips of size 1
    assert M.turnover(weights, bars_per_year=1000) == pytest.approx(999.0, rel=0.01)


def test_cagr_matches_compounding():
    equity = pd.Series(np.full(BPY + 1, 1.0))
    equity.iloc[-1] = 2.0
    equity = pd.Series(np.linspace(1.0, 2.0, BPY + 1))
    assert M.cagr(equity, BPY) == pytest.approx(1.0, rel=0.01)


def test_cvar_is_at_least_as_bad_as_var(known_returns):
    assert M.conditional_value_at_risk(known_returns) <= M.value_at_risk(known_returns)


def test_psr_is_a_probability(known_returns):
    assert 0.0 <= M.probabilistic_sharpe_ratio(known_returns, BPY) <= 1.0


def test_deflated_sharpe_falls_as_trials_rise(known_returns):
    few = M.deflated_sharpe_ratio(known_returns, BPY, n_trials=2)
    many = M.deflated_sharpe_ratio(known_returns, BPY, n_trials=1000)
    assert many < few


def test_metrics_survive_a_wiped_out_account():
    equity = pd.Series([100.0, 50.0, 0.0, 0.0])
    assert np.isfinite(M.max_drawdown(equity))
    assert M.max_drawdown(equity) == pytest.approx(-1.0)


def test_compute_all_has_no_surprises(gbm_data):
    res = run_backtest(gbm_data, SMACrossover(12, 48))
    m = M.compute_all(res.equity, res.weights, bars_per_year=BPY, n_trades=res.n_trades)
    for key in ("sharpe", "max_drawdown", "turnover", "n_trades", "cagr"):
        assert key in m


# -- report card -----------------------------------------------------------
def test_zero_cost_run_is_flagged(gbm_data):
    res = run_backtest(gbm_data, SMACrossover(12, 48), BacktestConfig(cost_model=ZERO_COST))
    card = report_card(res)
    assert any("COSTS ARE ZERO" in w for w in card.warnings)


def test_optimistic_timing_is_flagged(gbm_data):
    cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, fill_timing="this_close")
    card = report_card(run_backtest(gbm_data, SMACrossover(12, 48), cfg))
    assert any("EXECUTION OPTIMISM" in w for w in card.warnings)


def test_in_sample_is_flagged_by_default(gbm_data):
    card = report_card(run_backtest(gbm_data, SMACrossover(12, 48)))
    assert any("IN-SAMPLE" in w for w in card.warnings)


def test_out_of_sample_flag_suppresses_it(gbm_data):
    card = report_card(run_backtest(gbm_data, SMACrossover(12, 48)), is_out_of_sample=True)
    assert not any("IN-SAMPLE" in w for w in card.warnings)


def test_multiple_testing_is_flagged(gbm_data):
    card = report_card(run_backtest(gbm_data, SMACrossover(12, 48)), n_trials=100)
    assert any("MULTIPLE TESTING" in w for w in card.warnings)


def test_few_trades_is_flagged(gbm_data):
    card = report_card(run_backtest(gbm_data, BuyAndHold()))
    assert any("TRADES" in w for w in card.warnings)


def test_underperforming_buy_and_hold_is_flagged(momentum_data):
    cfg = BacktestConfig(cost_model=RETAIL_CRYPTO)
    bad = run_backtest(momentum_data, RandomStrategy(seed=3, every=4), cfg)
    bench = run_backtest(momentum_data, BuyAndHold(), cfg)
    card = report_card(bad, benchmark=bench)
    if card.metrics["sharpe"] < card.benchmark["sharpe"]:
        assert any("UNDERPERFORMS BUY & HOLD" in w for w in card.warnings)


def test_an_honest_run_can_have_no_warnings(momentum_data):
    """The harness must be capable of staying quiet, or it teaches nothing."""
    res = run_backtest(momentum_data, SMACrossover(12, 48), BacktestConfig(cost_model=RETAIL_CRYPTO))
    card = report_card(res, is_out_of_sample=True, n_trials=1)
    structural = [
        w for w in card.warnings
        if any(k in w for k in ("COSTS ARE ZERO", "EXECUTION OPTIMISM", "IN-SAMPLE", "MULTIPLE TESTING"))
    ]
    assert not structural


def test_summary_renders(gbm_data):
    card = report_card(run_backtest(gbm_data, SMACrossover(12, 48)))
    text = card.summary()
    assert "REPORT CARD" in text and "WARNINGS" in text
    assert card.to_frame().shape[1] >= 1


def test_compare_ranks_by_sharpe(gbm_data):
    results = {
        "buy&hold": run_backtest(gbm_data, BuyAndHold()),
        "sma": run_backtest(gbm_data, SMACrossover(12, 48)),
        "random": run_backtest(gbm_data, RandomStrategy(seed=1)),
    }
    table = compare(results)
    assert len(table) == 3
    assert table["sharpe"].is_monotonic_decreasing
