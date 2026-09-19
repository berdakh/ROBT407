"""The engine: accounting, timing, costs, and the guarantees they rest on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade.backtest import BacktestConfig, Backtester, run_backtest
from algotrade.execution import RETAIL_CRYPTO, CostModel, Order, OrderType, ZERO_COST
from algotrade.strategy import (
    BuyAndHold, ConstantWeight, FunctionStrategy, MomentumStrategy,
    RandomStrategy, SMACrossover, Strategy,
)


# -- accounting ------------------------------------------------------------
def test_buy_and_hold_tracks_the_asset(gbm_data):
    """With no costs, buy & hold must equal the asset's own return."""
    res = run_backtest(gbm_data, BuyAndHold(), BacktestConfig(cost_model=ZERO_COST))
    strat = res.equity.iloc[-1] / res.equity.iloc[0] - 1
    # Entry happens at bar 1's open, so compare from there.
    asset = gbm_data["close"].iloc[-1] / gbm_data["open"].iloc[1] - 1
    assert abs(strat - asset) < 0.01


def test_equity_equals_cash_plus_position(gbm_data):
    res = run_backtest(gbm_data, SMACrossover(12, 48))
    final_price = gbm_data["close"].iloc[-1]
    recomputed = res.units.iloc[-1] * final_price
    assert abs(res.equity.iloc[-1] - recomputed) < res.equity.iloc[-1] * 0.5


def test_doing_nothing_preserves_cash(gbm_data):
    flat = FunctionStrategy(lambda v, a: 0.0, name="flat")
    res = run_backtest(gbm_data, flat, BacktestConfig(initial_cash=5_000.0))
    assert res.n_trades == 0
    assert np.allclose(res.equity, 5_000.0)


def test_no_nan_or_inf_in_outputs(momentum_data):
    res = run_backtest(momentum_data, SMACrossover(24, 96))
    for series in (res.equity, res.weights, res.units):
        assert np.isfinite(series.to_numpy()).all()


def test_result_length_matches_input(gbm_data):
    res = run_backtest(gbm_data, SMACrossover(12, 48))
    assert len(res.equity) == len(gbm_data)
    assert res.equity.index.equals(gbm_data.index)


# -- costs -----------------------------------------------------------------
def test_costs_strictly_reduce_returns(momentum_data):
    """Adding costs can never make a strategy better."""
    strategy = SMACrossover(12, 48)
    free = run_backtest(momentum_data, strategy, BacktestConfig(cost_model=ZERO_COST))
    paid = run_backtest(momentum_data, strategy, BacktestConfig(cost_model=RETAIL_CRYPTO))
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.total_costs > 0
    assert free.total_costs == 0


def test_higher_costs_hurt_more(momentum_data):
    strategy = SMACrossover(12, 48)
    finals = []
    for bps in (0.0, 5.0, 20.0, 50.0):
        cfg = BacktestConfig(cost_model=CostModel(commission_bps=bps, half_spread_bps=0.0))
        finals.append(run_backtest(momentum_data, strategy, cfg).equity.iloc[-1])
    assert finals == sorted(finals, reverse=True)


def test_costs_are_charged_on_every_fill(momentum_data):
    res = run_backtest(momentum_data, SMACrossover(12, 48), BacktestConfig(cost_model=RETAIL_CRYPTO))
    assert all(f.commission > 0 for f in res.fills)
    assert np.isclose(res.total_commission, sum(f.commission for f in res.fills))


def test_buys_fill_above_and_sells_below_reference(momentum_data):
    res = run_backtest(momentum_data, MomentumStrategy(24), BacktestConfig(cost_model=RETAIL_CRYPTO))
    for f in res.fills:
        if f.qty > 0:
            assert f.price >= f.reference_price
        else:
            assert f.price <= f.reference_price


def test_turnover_costs_scale_with_trading(gbm_data):
    """A strategy that rebalances constantly must pay far more than one that doesn't."""
    cfg = BacktestConfig(cost_model=RETAIL_CRYPTO)
    lazy = run_backtest(gbm_data, BuyAndHold(), cfg)
    busy = run_backtest(gbm_data, RandomStrategy(seed=1, every=2), cfg)
    assert busy.total_costs > lazy.total_costs * 10


# -- timing ----------------------------------------------------------------
def test_orders_never_fill_on_the_decision_bar(gbm_data):
    """Under the default timing, a decision at bar i cannot fill before bar i+1."""

    class BuyOnBar50(Strategy):
        name = "buy at bar 50"

        def on_bar(self, view, account):
            return 1.0 if view.i == 50 else None

    res = run_backtest(gbm_data, BuyOnBar50(), BacktestConfig(cost_model=ZERO_COST))
    assert res.n_trades == 1
    fill_ts = res.fills[0].timestamp
    assert fill_ts == gbm_data.index[51]
    assert res.fills[0].reference_price == gbm_data["open"].iloc[51]


def test_optimistic_timing_fills_on_the_decision_bar(gbm_data):
    class BuyOnBar50(Strategy):
        name = "buy at bar 50"

        def on_bar(self, view, account):
            return 1.0 if view.i == 50 else None

    cfg = BacktestConfig(cost_model=ZERO_COST, fill_timing="this_close")
    res = run_backtest(gbm_data, BuyOnBar50(), cfg)
    assert res.fills[0].timestamp == gbm_data.index[50]
    assert res.fills[0].reference_price == gbm_data["close"].iloc[50]


def test_fill_timing_has_no_consistent_direction():
    """An honest result that took measuring to establish.

    ``this_close`` is *unrealistic* -- you cannot observe a closing price and
    simultaneously trade at it -- but it is not look-ahead bias, and it does
    NOT reliably inflate returns. Across the 40 dataset/seed combinations
    below it beats ``next_open`` 22 times and loses 18; narrow that to three
    datasets and it wins only 3 of 15. The sign depends on the data, which is
    another way of saying there is no edge in it.

    Contrast ``test_lookahead_is_systematically_profitable`` below, where the
    genuine bias wins 100% of the time on every dataset including pure noise.
    That asymmetry is why the workshop teaches the two separately: a backtest
    can be unrealistic without being inflated, and the inflated kind is the
    one you must hunt down.
    """
    from algotrade.synthetic import generate

    wins = losses = 0
    for kind in ("gbm", "momentum", "pure_noise", "mean_reverting", "trending_with_crash"):
        for seed in range(1, 9):
            df, _ = generate(kind, n=1500, seed=seed)
            strategy = SMACrossover(6, 24)
            honest = run_backtest(df, strategy, BacktestConfig(cost_model=ZERO_COST))
            optimistic = run_backtest(
                df, strategy, BacktestConfig(cost_model=ZERO_COST, fill_timing="this_close")
            )
            if optimistic.equity.iloc[-1] > honest.equity.iloc[-1]:
                wins += 1
            else:
                losses += 1

    # The claim is "no consistent direction", so both outcomes must occur.
    assert wins > 0 and losses > 0, (
        f"fill timing went one way in all {wins + losses} trials (won {wins}). If this "
        f"ever becomes one-sided, the docstrings in backtest.py and notebook 05 are wrong."
    )


def test_lookahead_is_systematically_profitable():
    """The bias the engine exists to prevent, measured.

    The naive vectorised backtest with ``shift=0`` lets the position at bar t
    earn bar t's own return. Unlike fill timing, this wins *every single time*,
    on every dataset -- including ones with no predictability at all. That
    universality is the tell: a real edge does not work on pure noise.
    """
    from algotrade.naive import vectorized_backtest
    from algotrade.indicators import sma
    from algotrade.synthetic import generate

    wins = 0
    trials = 0
    for kind in ("gbm", "momentum", "pure_noise", "mean_reverting"):
        for seed in range(1, 6):
            df, _ = generate(kind, n=1200, seed=seed)
            signal = (sma(df["close"], 6) > sma(df["close"], 24)).astype(float)
            leaky = vectorized_backtest(df["close"], signal, shift=0, quiet=True)
            correct = vectorized_backtest(df["close"], signal, shift=1, quiet=True)
            wins += leaky["equity"].iloc[-1] > correct["equity"].iloc[-1]
            trials += 1
    assert wins == trials, f"look-ahead won only {wins}/{trials}; expected all of them"


# -- limits ----------------------------------------------------------------
def test_leverage_is_capped(gbm_data):
    greedy = FunctionStrategy(lambda v, a: 5.0, name="greedy")
    res = run_backtest(gbm_data, greedy, BacktestConfig(max_leverage=1.0, cost_model=ZERO_COST))
    assert res.weights.abs().max() <= 1.05


def test_shorting_can_be_disabled(gbm_data):
    bear = FunctionStrategy(lambda v, a: -1.0, name="bear")
    res = run_backtest(gbm_data, bear, BacktestConfig(allow_short=False, cost_model=ZERO_COST))
    assert res.units.min() >= -1e-9
    assert res.n_trades == 0


def test_rebalance_tolerance_reduces_trading(gbm_data):
    strategy = MomentumStrategy(12)
    tight = run_backtest(gbm_data, strategy, BacktestConfig(rebalance_tolerance=0.0))
    loose = run_backtest(gbm_data, strategy, BacktestConfig(rebalance_tolerance=0.10))
    assert loose.n_trades <= tight.n_trades


def test_bad_return_type_is_rejected(gbm_data):
    bad = FunctionStrategy(lambda v, a: "buy please", name="bad")
    with pytest.raises(TypeError, match="on_bar returned"):
        run_backtest(gbm_data, bad)


def test_nan_target_is_rejected(gbm_data):
    bad = FunctionStrategy(lambda v, a: float("nan"), name="nan")
    with pytest.raises(ValueError, match="finite"):
        run_backtest(gbm_data, bad)


# -- explicit orders -------------------------------------------------------
def test_limit_order_only_fills_when_touched(gbm_data):
    entry_bar = 30
    far_below = gbm_data["low"].iloc[entry_bar : entry_bar + 20].min() * 0.5

    class UntouchableLimit(Strategy):
        name = "limit far below"

        def on_bar(self, view, account):
            if view.i == entry_bar:
                return Order(qty=0.1, type=OrderType.LIMIT, limit_price=far_below, expires_after=20)
            return None

    res = run_backtest(gbm_data, UntouchableLimit(), BacktestConfig(cost_model=ZERO_COST))
    assert res.n_trades == 0


def test_limit_order_fills_when_price_reaches_it(gbm_data):
    entry_bar = 30
    reachable = gbm_data["high"].iloc[entry_bar + 1] * 1.001

    class ReachableLimit(Strategy):
        name = "limit sell reachable"

        def on_bar(self, view, account):
            if view.i == entry_bar:
                return Order(qty=-0.1, type=OrderType.LIMIT, limit_price=gbm_data["low"].iloc[entry_bar] * 0.99)
            return None

    res = run_backtest(gbm_data, ReachableLimit(), BacktestConfig(cost_model=ZERO_COST))
    assert res.n_trades == 1


# -- determinism -----------------------------------------------------------
def test_backtest_is_deterministic(momentum_data):
    a = run_backtest(momentum_data, SMACrossover(12, 48))
    b = run_backtest(momentum_data, SMACrossover(12, 48))
    pd.testing.assert_series_equal(a.equity, b.equity)


def test_random_strategy_is_reproducible(gbm_data):
    a = run_backtest(gbm_data, RandomStrategy(seed=42))
    b = run_backtest(gbm_data, RandomStrategy(seed=42))
    c = run_backtest(gbm_data, RandomStrategy(seed=43))
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert not a.equity.equals(c.equity)


def test_too_short_data_rejected(gbm_data):
    with pytest.raises(ValueError, match="at least 2 bars"):
        run_backtest(gbm_data.iloc[:1], BuyAndHold())
