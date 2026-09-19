"""Synthetic generators must deliver the ground truth they advertise.

If ``momentum`` has no momentum, every exercise built on it teaches the wrong
lesson silently. These tests are the answer key's answer key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade import metrics as M
from algotrade.backtest import BacktestConfig, run_backtest
from algotrade.data import simple_returns, validate_ohlcv
from algotrade.execution import ZERO_COST
from algotrade.risk import (
    DrawdownGuard, VolatilityTargeted, fixed_fractional_size, kelly_fraction,
    vol_target_series, volatility_target_weight,
)
from algotrade.strategy import MomentumStrategy, ZScoreReversion
from algotrade.synthetic import DATASETS, generate

BPY = 8760


@pytest.mark.parametrize("kind", sorted(DATASETS))
def test_every_dataset_is_valid_ohlcv(kind):
    df, truth = generate(**DATASETS[kind])
    validate_ohlcv(df)
    assert truth.kind == DATASETS[kind]["kind"]
    assert truth.notes


@pytest.mark.parametrize("kind", sorted(DATASETS))
def test_generation_is_reproducible(kind):
    a, _ = generate(**DATASETS[kind])
    b, _ = generate(**DATASETS[kind])
    pd.testing.assert_frame_equal(a, b)


def test_gbm_volatility_matches_its_parameter():
    df, truth = generate("gbm", n=50_000, seed=1, sigma_annual=0.45)
    realized = simple_returns(df["close"]).std() * np.sqrt(BPY)
    assert realized == pytest.approx(0.45, rel=0.05)
    assert truth.params["sigma_annual"] == 0.45


def test_momentum_dataset_actually_has_momentum():
    df, truth = generate("momentum", n=50_000, seed=2, phi=0.08)
    ac = simple_returns(df["close"]).dropna().autocorr(1)
    assert ac > 0.04, f"advertised phi={truth.params['phi']} but measured autocorr {ac:.3f}"


def test_gbm_has_no_autocorrelation():
    df, _ = generate("gbm", n=50_000, seed=3)
    ac = simple_returns(df["close"]).dropna().autocorr(1)
    assert abs(ac) < 0.02, f"the control dataset has structure in it: autocorr {ac:.3f}"


def test_reversion_strategy_beats_momentum_on_reverting_data():
    """Ground truth check: the right tool must win on the right data."""
    df, _ = generate("mean_reverting", n=20_000, seed=4)
    cfg = BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY)
    rev = run_backtest(df, ZScoreReversion(48, 1.5), cfg)
    mom = run_backtest(df, MomentumStrategy(48), cfg)
    assert M.sharpe_ratio(rev.returns, BPY) > M.sharpe_ratio(mom.returns, BPY)


def test_momentum_strategy_beats_reversion_on_trending_data():
    df, _ = generate("momentum", n=20_000, seed=5, phi=0.08)
    cfg = BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY)
    mom = run_backtest(df, MomentumStrategy(24), cfg)
    rev = run_backtest(df, ZScoreReversion(24, 1.5), cfg)
    assert M.sharpe_ratio(mom.returns, BPY) > M.sharpe_ratio(rev.returns, BPY)


def test_regime_dataset_has_two_volatility_regimes():
    df, truth = generate("regime_switching", n=30_000, seed=6)
    vol = simple_returns(df["close"]).rolling(500).std().dropna()
    assert vol.max() / vol.min() > 2.0
    assert 0.05 < truth.params["pct_turbulent"] < 0.95


# -- risk ------------------------------------------------------------------
def test_vol_target_weight_scales_inversely():
    assert volatility_target_weight(0.40, 0.20, max_leverage=10) == pytest.approx(0.5)
    assert volatility_target_weight(0.10, 0.20, max_leverage=10) == pytest.approx(2.0)


def test_vol_target_weight_respects_the_cap():
    assert volatility_target_weight(0.001, 0.20, max_leverage=1.0) == 1.0


def test_vol_target_weight_handles_zero_vol():
    assert volatility_target_weight(0.0, 0.20) == 0.0
    assert volatility_target_weight(float("nan"), 0.20) == 0.0


def test_vol_target_series_is_causal(gbm_data):
    rets = simple_returns(gbm_data["close"]).fillna(0)
    scale = vol_target_series(rets, target_vol=0.2, lookback=100, bars_per_year=BPY)
    corrupted = rets.copy()
    corrupted.iloc[500:] *= 10
    scale2 = vol_target_series(corrupted, target_vol=0.2, lookback=100, bars_per_year=BPY)
    pd.testing.assert_series_equal(scale.iloc[:500], scale2.iloc[:500])


def test_vol_targeting_moves_realized_vol_towards_the_target():
    df, _ = generate("regime_switching", n=20_000, seed=7)
    cfg = BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY, max_leverage=3.0)
    plain = run_backtest(df, MomentumStrategy(48), cfg)
    targeted = run_backtest(
        df, VolatilityTargeted(MomentumStrategy(48), target_vol=0.20, max_leverage=3.0), cfg
    )
    plain_vol = M.annualized_volatility(plain.returns, BPY)
    target_vol = M.annualized_volatility(targeted.returns, BPY)
    assert abs(target_vol - 0.20) < abs(plain_vol - 0.20)


def test_drawdown_guard_limits_the_drawdown():
    df, _ = generate("trending_with_crash", n=17_520, seed=12)
    cfg = BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY)
    plain = run_backtest(df, MomentumStrategy(48), cfg)
    guarded = run_backtest(df, DrawdownGuard(MomentumStrategy(48), max_drawdown=0.20), cfg)
    assert M.max_drawdown(guarded.equity) > M.max_drawdown(plain.equity)


def test_drawdown_guard_logs_its_halts():
    df, _ = generate("trending_with_crash", n=17_520, seed=12)
    guard = DrawdownGuard(MomentumStrategy(48), max_drawdown=0.15)
    run_backtest(df, guard, BacktestConfig(cost_model=ZERO_COST))
    assert guard.halt_log, "the guard never triggered on a dataset built around crashes"


def test_kelly_is_scaled_down_by_default():
    full = kelly_fraction(0.10, 0.04, safety_factor=1.0)
    quarter = kelly_fraction(0.10, 0.04)
    assert quarter == pytest.approx(full * 0.25)


def test_fixed_fractional_risks_the_stated_fraction():
    units = fixed_fractional_size(equity=10_000, price=100, stop_distance=5, risk_per_trade=0.01)
    assert units * 5 == pytest.approx(100.0)       # 1% of 10,000
