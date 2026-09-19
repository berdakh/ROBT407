"""Two promises the repository makes, enforced as tests.

1. **Everything runs offline.** No test, and nothing in the required notebook
   path, may open a socket or read a market feed. A workshop that needs the
   internet fails in the room where it is taught.
2. **Nothing can trade real money.** There is no live order path, and the
   paper broker refuses to pretend otherwise.
"""

from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest

import algotrade
from algotrade.paper import PaperBroker, PaperTradingSession
from algotrade.strategy import SMACrossover


# -- offline ---------------------------------------------------------------
@pytest.fixture
def no_network(monkeypatch):
    """Make any socket connection raise, for the duration of a test."""

    def blocked(*args, **kwargs):
        raise AssertionError("this code tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    return True


def test_full_workflow_runs_with_no_network(no_network):
    """Generate, backtest, score -- the whole required path, offline."""
    df, truth = algotrade.generate("momentum", n=1500, seed=5)
    result = algotrade.run_backtest(df, SMACrossover(12, 48))
    card = algotrade.report_card(result)
    assert len(result.equity) == 1500
    assert "REPORT CARD" in card.summary()
    assert truth.params["phi"] > 0


def test_paper_session_runs_with_no_network(no_network):
    df, _ = algotrade.generate("gbm", n=800, seed=6)
    session = PaperTradingSession(SMACrossover(12, 48), df.iloc[:400])
    for ts, bar in df.iloc[400:500].iterrows():
        session.on_new_bar(ts, bar)
    assert len(session.logs) == 100
    assert "NO REAL ORDERS" in session.summary()


def test_package_imports_no_network_clients():
    """The package must not depend on ccxt, requests or any exchange SDK."""
    import sys

    forbidden = {"ccxt", "requests", "urllib3", "aiohttp", "websocket", "websockets", "httpx"}
    # Import the package fresh-ish and see what it pulled in.
    loaded = {name.split(".")[0] for name in sys.modules}
    leaked = forbidden & loaded
    assert not leaked, (
        f"importing algotrade pulled in network libraries: {sorted(leaked)}. "
        f"The required path must stay offline."
    )


# -- safety ----------------------------------------------------------------
def test_paper_broker_cannot_connect():
    with pytest.raises(NotImplementedError, match="cannot connect"):
        PaperBroker().connect()


def test_live_trading_is_declared_unsupported():
    assert PaperBroker.LIVE_TRADING_SUPPORTED is False


def test_package_carries_a_disclaimer():
    text = algotrade.DISCLAIMER
    assert "NOT INVESTMENT ADVICE" in text
    assert "lose money" in text


def test_paper_session_marks_itself_simulated():
    df, _ = algotrade.generate("gbm", n=300, seed=7)
    session = PaperTradingSession(SMACrossover(12, 48), df.iloc[:200])
    for ts, bar in df.iloc[200:250].iterrows():
        session.on_new_bar(ts, bar)
    assert "SIMULATED" in session.summary()


def test_paper_and_backtest_agree_bar_for_bar():
    """The live loop and the backtester must produce identical target weights.

    If these ever diverge, one of them is wrong, and the notebooks' claim that
    the paper session 'is the same logic in a live shape' would be false.
    """
    df, _ = algotrade.generate("momentum", n=1200, seed=8)
    warmup, strategy = 400, SMACrossover(12, 48)

    session = PaperTradingSession(strategy, df.iloc[:warmup])
    for ts, bar in df.iloc[warmup:].iterrows():
        session.on_new_bar(ts, bar)

    backtest = algotrade.run_backtest(df, SMACrossover(12, 48))
    recon = session.reconcile(backtest.targets)
    assert len(recon) > 0
    assert not recon["mismatch"].any(), (
        f"{int(recon['mismatch'].sum())} bars disagree between live and backtest paths"
    )


def test_nothing_writes_outside_tmp(tmp_path):
    """The paper session must only write where it is told to."""
    df, _ = algotrade.generate("gbm", n=300, seed=9)
    log_dir = tmp_path / "logs"
    session = PaperTradingSession(SMACrossover(12, 48), df.iloc[:200], log_dir=log_dir)
    for ts, bar in df.iloc[200:230].iterrows():
        session.on_new_bar(ts, bar)
    assert (log_dir / "session.jsonl").exists()
    assert len(list(log_dir.iterdir())) == 1
