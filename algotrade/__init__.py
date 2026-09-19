"""algotrade -- a small, readable library for learning algorithmic trading honestly.

    PAPER TRADING AND EDUCATION ONLY. Nothing in this package connects to a
    live-money account, and nothing in it is investment advice. See the README.

The design is built around one claim: the hard part of algorithmic trading is
not writing a strategy, it is finding out whether your evaluation of that
strategy is lying to you. So the central object is not the strategy, it is
:class:`~algotrade.view.MarketView` -- the read-only, time-truncated window
that makes look-ahead bias structurally impossible rather than merely
discouraged.

Module map
----------
``view``         The look-ahead guard. Read this first.
``data``         Loading, validating and resampling OHLCV.
``synthetic``    Price series with known ground truth.
``indicators``   Strictly causal technical indicators.
``strategy``     The Strategy base class and reference strategies.
``execution``    Orders, fills, and a realistic cost model.
``portfolio``    Cash and position accounting.
``backtest``     The event-driven engine.
``metrics``      Performance statistics, with their error bars.
``report``       The report card that every strategy must face.
``validation``   Walk-forward and purged splits.
``features``     Causal feature engineering.
``labeling``     Forward-looking targets that remember their horizon.
``risk``         Volatility targeting and drawdown control.
``paper``        A simulated broker that cannot reach an exchange.
``naive``        The tutorial backtest, bugs included, for demonstration only.
"""

from __future__ import annotations

__version__ = "1.0.0"

DISCLAIMER = """
EDUCATIONAL USE ONLY -- NOT INVESTMENT ADVICE

This software teaches a methodology for evaluating trading strategies. It does
not recommend any strategy, asset or trade.

* Most retail algorithmic trading strategies lose money after costs.
* A good backtest is weak evidence. It is a statement about the past, filtered
  through choices you made after seeing that past.
* Nothing here connects to a live-money account. Keep it that way until you can
  explain, in writing, why your strategy should work -- and what would have to
  be true for it to fail.
* Past performance, simulated or real, does not predict future results.
"""

from .errors import (  # noqa: E402
    AlgotradeError,
    DataIntegrityError,
    InsufficientHistoryError,
    LookAheadError,
    RiskLimitError,
)
from .view import Clock, MarketView, SeriesView  # noqa: E402
from .data import load_ohlcv, resample_ohlcv, simple_returns, log_returns, BARS_PER_YEAR  # noqa: E402
from .synthetic import generate  # noqa: E402
from .execution import CostModel, Order, OrderType, Fill, ZERO_COST, RETAIL_CRYPTO  # noqa: E402
from .portfolio import Account  # noqa: E402
from .strategy import (  # noqa: E402
    Strategy, BuyAndHold, SMACrossover, MomentumStrategy, ZScoreReversion,
    RandomStrategy, FunctionStrategy, ConstantWeight,
)
from .backtest import BacktestConfig, BacktestResult, Backtester, run_backtest  # noqa: E402
from .report import ReportCard, report_card, compare  # noqa: E402
from .validation import (  # noqa: E402
    Split, walk_forward_splits, expanding_window_splits, purged_kfold_splits,
    assert_no_leakage, describe_splits,
)
from .risk import VolatilityTargeted, DrawdownGuard, vol_target_series, kelly_fraction  # noqa: E402
from .paper import PaperBroker, PaperTradingSession  # noqa: E402

__all__ = [
    "__version__", "DISCLAIMER",
    "AlgotradeError", "LookAheadError", "DataIntegrityError",
    "InsufficientHistoryError", "RiskLimitError",
    "Clock", "MarketView", "SeriesView",
    "load_ohlcv", "resample_ohlcv", "simple_returns", "log_returns", "BARS_PER_YEAR",
    "generate",
    "CostModel", "Order", "OrderType", "Fill", "ZERO_COST", "RETAIL_CRYPTO",
    "Account",
    "Strategy", "BuyAndHold", "SMACrossover", "MomentumStrategy", "ZScoreReversion",
    "RandomStrategy", "FunctionStrategy", "ConstantWeight",
    "BacktestConfig", "BacktestResult", "Backtester", "run_backtest",
    "ReportCard", "report_card", "compare",
    "Split", "walk_forward_splits", "expanding_window_splits", "purged_kfold_splits",
    "assert_no_leakage", "describe_splits",
    "VolatilityTargeted", "DrawdownGuard", "vol_target_series", "kelly_fraction",
    "PaperBroker", "PaperTradingSession",
]
