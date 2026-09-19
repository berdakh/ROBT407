"""Offline tests for scripts/fetch_market_data.py.

This is the only script in the repository that touches the network, which made
it the only one with no coverage. These tests exercise its parsing, pagination
and every error path against a **mocked** exchange, so they run with no network
exactly like the rest of the suite.

What they cannot cover: whether the real Binance endpoint still returns the
shape assumed here. That requires a live call, and the fixtures below encode
the documented shape rather than an observed one.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_market_data.py"


@pytest.fixture(scope="module")
def fetch_mod():
    """Import the script as a module. It guards main() behind __main__."""
    spec = importlib.util.spec_from_file_location("fetch_market_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_sleeping(fetch_mod, monkeypatch):
    """The script is polite to free APIs. Tests need not be."""
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda *_: None)


def make_klines(start_ms: int, n: int, step_ms: int, price: float = 20_000.0):
    """Binance kline shape: [open_time, o, h, l, c, v, close_time, ...]."""
    out = []
    for i in range(n):
        base = price + i
        out.append([
            start_ms + i * step_ms,
            f"{base:.2f}", f"{base + 5:.2f}", f"{base - 5:.2f}", f"{base + 1:.2f}",
            f"{10 + i:.4f}",
            start_ms + (i + 1) * step_ms - 1, "0", 0, "0", "0", "0",
        ])
    return out


# -- happy path ------------------------------------------------------------
def test_parses_a_well_formed_batch(fetch_mod):
    rows = fetch_mod._parse_klines(make_klines(1_600_000_000_000, 3, 3_600_000), "BTCUSDT")
    assert len(rows) == 3
    assert rows[0]["open"] == 20_000.0
    assert rows[0]["high"] == 20_005.0
    assert rows[0]["low"] == 19_995.0
    assert rows[0]["close"] == 20_001.0
    assert isinstance(rows[0]["timestamp"], int)


def test_pagination_walks_forward_and_terminates(fetch_mod, monkeypatch):
    step = fetch_mod.INTERVAL_MS["1h"]
    calls = []

    def fake_get(url, timeout=30):
        calls.append(url)
        if len(calls) > 4:
            return []                       # exchange says "no more data"
        start = 1_600_000_000_000 + (len(calls) - 1) * 10 * step
        return make_klines(start, 10, step)

    monkeypatch.setattr(fetch_mod, "_get", fake_get)
    monkeypatch.setattr(fetch_mod.time, "time", lambda: 1_600_000_000 + 40 * 3600)

    rows = fetch_mod.fetch_binance("BTCUSDT", "1h", days=2)
    assert len(rows) == 40
    timestamps = [r["timestamp"] for r in rows]
    assert timestamps == sorted(timestamps), "rows must come back in time order"
    assert len(set(timestamps)) == len(timestamps), "no duplicates from pagination"


def test_empty_first_response_returns_nothing(fetch_mod, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_get", lambda url, timeout=30: [])
    assert fetch_mod.fetch_binance("BTCUSDT", "1h", days=1) == []


# -- the error paths that actually happen ----------------------------------
def test_invalid_symbol_gives_a_useful_message(fetch_mod, monkeypatch):
    """Binance answers an invalid symbol with an error OBJECT, not a list.

    The original code iterated that dict's keys and died with
    'invalid literal for int() with base 10: c', which tells a student nothing.
    """
    monkeypatch.setattr(
        fetch_mod, "_get",
        lambda url, timeout=30: {"code": -1121, "msg": "Invalid symbol."},
    )
    with pytest.raises(fetch_mod.ExchangeError) as excinfo:
        fetch_mod.fetch_binance("BTC-USD", "1h", days=1)

    message = str(excinfo.value)
    assert "Invalid symbol" in message
    assert "BTC-USD" in message
    assert "BTCUSDT" in message, "the message should show the correct format"


def test_malformed_kline_is_reported_not_crashed(fetch_mod):
    with pytest.raises(fetch_mod.ExchangeError, match="malformed kline"):
        fetch_mod._parse_klines([[1, "2", "3"]], "BTCUSDT")


def test_unexpected_payload_type_is_reported(fetch_mod):
    with pytest.raises(fetch_mod.ExchangeError, match="expected a list"):
        fetch_mod._parse_klines("not json at all", "BTCUSDT")


def test_non_advancing_cursor_does_not_loop_forever(fetch_mod, monkeypatch):
    """A stale response must stop the loop, not spin against a free API."""
    step = fetch_mod.INTERVAL_MS["1h"]
    monkeypatch.setattr(fetch_mod.time, "time", lambda: 1_700_000_000)
    # Always return a batch from far in the past: the cursor would go backwards.
    monkeypatch.setattr(
        fetch_mod, "_get",
        lambda url, timeout=30: make_klines(1_500_000_000_000, 5, step),
    )
    with pytest.raises(fetch_mod.ExchangeError, match="did not advance"):
        fetch_mod.fetch_binance("BTCUSDT", "1h", days=30)


# -- main(), end to end, with a mocked exchange ----------------------------
def test_main_writes_a_valid_csv(fetch_mod, monkeypatch, tmp_path):
    """The whole script path: fetch, parse, validate, write."""
    from algotrade.data import load_ohlcv

    step = fetch_mod.INTERVAL_MS["1h"]
    now_ms = 1_700_000_000_000
    monkeypatch.setattr(fetch_mod.time, "time", lambda: now_ms / 1000)

    served = {"n": 0}

    def fake_get(url, timeout=30):
        served["n"] += 1
        if served["n"] > 1:
            return []
        return make_klines(now_ms - 48 * step, 48, step)

    monkeypatch.setattr(fetch_mod, "_get", fake_get)

    out = tmp_path / "BTCUSDT_1h.csv"
    monkeypatch.setattr("sys.argv", ["fetch", "--symbol", "BTCUSDT", "--interval", "1h",
                                     "--days", "2", "--out", str(out)])

    assert fetch_mod.main() == 0
    assert out.exists()

    # The file it produced must satisfy the same validator the notebooks use.
    df = load_ohlcv(out)
    assert len(df) == 48
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing


def test_main_deduplicates_repeated_bars(fetch_mod, monkeypatch, tmp_path):
    """Exchanges do re-send bars across page boundaries."""
    from algotrade.data import load_ohlcv

    step = fetch_mod.INTERVAL_MS["1h"]
    now_ms = 1_700_000_000_000
    monkeypatch.setattr(fetch_mod.time, "time", lambda: now_ms / 1000)

    served = {"n": 0}

    def fake_get(url, timeout=30):
        served["n"] += 1
        if served["n"] == 1:
            return make_klines(now_ms - 20 * step, 10, step)
        if served["n"] == 2:
            # Overlaps the previous page by 3 bars.
            return make_klines(now_ms - 13 * step, 10, step)
        return []

    monkeypatch.setattr(fetch_mod, "_get", fake_get)
    out = tmp_path / "dupes.csv"
    monkeypatch.setattr("sys.argv", ["fetch", "--days", "1", "--out", str(out)])

    assert fetch_mod.main() == 0
    df = load_ohlcv(out)
    assert not df.index.has_duplicates, "overlapping pages must be deduplicated"


def test_main_reports_network_failure_without_crashing(fetch_mod, monkeypatch, tmp_path, capsys):
    def boom(url, timeout=30):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(fetch_mod, "_get", boom)
    monkeypatch.setattr("sys.argv", ["fetch", "--days", "1", "--out", str(tmp_path / "x.csv")])

    assert fetch_mod.main() == 1
    stderr = capsys.readouterr().err
    assert "not fatal" in stderr
    assert "data/synthetic" in stderr, "must point the student at the offline path"


def test_main_explains_a_rate_limit(fetch_mod, monkeypatch, tmp_path, capsys):
    def rate_limited(url, timeout=30):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(fetch_mod, "_get", rate_limited)
    monkeypatch.setattr("sys.argv", ["fetch", "--days", "1", "--out", str(tmp_path / "x.csv")])

    assert fetch_mod.main() == 1
    assert "rate limit" in capsys.readouterr().err.lower()


def test_main_explains_a_geoblock(fetch_mod, monkeypatch, tmp_path, capsys):
    def blocked(url, timeout=30):
        raise urllib.error.HTTPError(url, 451, "Unavailable For Legal Reasons", {}, None)

    monkeypatch.setattr(fetch_mod, "_get", blocked)
    monkeypatch.setattr("sys.argv", ["fetch", "--days", "1", "--out", str(tmp_path / "x.csv")])

    assert fetch_mod.main() == 1
    assert "geo-blocked" in capsys.readouterr().err


def test_main_returns_distinct_code_for_a_bad_symbol(fetch_mod, monkeypatch, tmp_path, capsys):
    """Exit 2 means 'you asked wrongly'; exit 1 means 'the network failed'."""
    monkeypatch.setattr(
        fetch_mod, "_get",
        lambda url, timeout=30: {"code": -1121, "msg": "Invalid symbol."},
    )
    monkeypatch.setattr("sys.argv", ["fetch", "--symbol", "NOPE", "--days", "1",
                                     "--out", str(tmp_path / "x.csv")])

    assert fetch_mod.main() == 2
    assert "Invalid symbol" in capsys.readouterr().err


def test_script_sends_no_credentials(fetch_mod):
    """The safety claim in the docstring, enforced.

    This script must never grow credential handling. It reads public endpoints
    and writes a CSV; that is the whole of it.
    """
    source = SCRIPT.read_text().lower()
    for forbidden in ("api_key", "apikey", "secret", "signature", "hmac",
                      "authorization", "private_key", "passphrase"):
        assert forbidden not in source, f"{forbidden!r} appears in the fetch script"
