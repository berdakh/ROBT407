#!/usr/bin/env python3
"""Download a real crypto OHLCV snapshot. OPTIONAL, and the only script that needs network.

    python scripts/fetch_market_data.py --symbol BTCUSDT --interval 1h --days 365

Nothing in the required workshop path depends on this. Every notebook runs on
the committed synthetic data, and the tests are offline. Run this once if you
want to repeat the exercises on real prices -- which you should, at least once,
because real data is messier than any generator.

NO API KEY IS NEEDED. These are public market-data endpoints. This script
never sends credentials, never places an order, and has no code path that
could: it issues HTTP GETs and writes a CSV.

If your network blocks these hosts (corporate proxies and some cloud
environments do), the workshop still works end to end on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "real"

#: Public, keyless OHLCV endpoints, tried in order.
SOURCES = {
    "binance": {
        "url": "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&startTime={start_ms}",
        "note": "Binance public klines. Geo-blocked in some jurisdictions.",
    },
    "okx": {
        "url": "https://www.okx.com/api/v5/market/history-candles?instId={okx_symbol}&bar={okx_bar}&limit=100",
        "note": "OKX public candles.",
    },
}

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
               "4h": 14_400_000, "1d": 86_400_000}


class ExchangeError(RuntimeError):
    """The exchange answered, but with something other than market data."""


def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "algotrade-workshop/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _parse_klines(payload, symbol: str) -> "list[dict]":
    """Turn one kline batch into rows, or explain clearly why it cannot.

    Binance answers an invalid symbol with an error OBJECT rather than a list.
    Iterating that object yields its dict keys, so the naive version of this
    function failed with ``invalid literal for int() with base 10: 'c'`` --
    which tells a student nothing at all about what they got wrong.
    """
    if isinstance(payload, dict):
        message = payload.get("msg") or payload.get("message") or str(payload)
        raise ExchangeError(
            f"the exchange rejected the request for {symbol!r}: {message}\n"
            f"Check the symbol. Binance pairs have no separator and are upper case, "
            f"e.g. BTCUSDT or ETHUSDT -- not BTC-USD or btc/usdt."
        )
    if not isinstance(payload, list):
        raise ExchangeError(f"expected a list of klines, got {type(payload).__name__}")

    rows = []
    for k in payload:
        if not isinstance(k, (list, tuple)) or len(k) < 6:
            raise ExchangeError(
                f"malformed kline in the response: {k!r}. The endpoint's shape may "
                f"have changed; this script expects [open_time, o, h, l, c, v, ...]."
            )
        rows.append(
            {
                "timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
            }
        )
    return rows


def fetch_binance(symbol: str, interval: str, days: int) -> "list[dict]":
    step = INTERVAL_MS[interval]
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows, cursor = [], start

    while cursor < end:
        url = SOURCES["binance"]["url"].format(
            symbol=symbol, interval=interval, limit=1000, start_ms=cursor
        )
        batch = _get(url)
        if not batch:
            break

        rows.extend(_parse_klines(batch, symbol))

        # Guard against a cursor that does not advance. A stale or malformed
        # response could otherwise spin this loop forever against someone
        # else's free API, which is a rude way to get yourself rate-limited.
        next_cursor = int(batch[-1][0]) + step
        if next_cursor <= cursor:
            raise ExchangeError(
                f"the response did not advance past {cursor}; stopping rather than "
                f"looping. Last kline open time was {batch[-1][0]}."
            )
        cursor = next_cursor

        print(f"    fetched {len(rows)} bars...", end="\r", flush=True)
        time.sleep(0.25)          # be a good citizen of someone else's free API
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h", choices=sorted(INTERVAL_MS))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT / f"{args.symbol}_{args.interval}.csv"

    print(f"Fetching {args.days} days of {args.symbol} {args.interval} bars...")
    try:
        rows = fetch_binance(args.symbol, args.interval, args.days)
    except ExchangeError as exc:
        # The exchange answered; we just cannot use the answer. Usually a typo.
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        hint = ""
        status = getattr(exc, "code", None)
        if status == 429:
            hint = "\nThat is a rate limit. Wait a minute and try again.\n"
        elif status in (401, 403, 451):
            hint = (
                "\nThat status usually means the endpoint is geo-blocked or behind a\n"
                "proxy. Try a different exchange, or skip it -- see below.\n"
            )
        print(
            f"\nCould not reach the exchange: {exc}\n{hint}"
            f"\nThis is not fatal. The workshop is designed to run entirely on the\n"
            f"committed synthetic data in data/synthetic/, which is what every\n"
            f"notebook loads by default. Real data is a bonus, not a dependency.",
            file=sys.stderr,
        )
        return 1

    if not rows:
        print("No data returned.", file=sys.stderr)
        return 1

    import pandas as pd

    from algotrade.data import data_quality_report, validate_ohlcv

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")

    validate_ohlcv(df)
    df.to_csv(out_path)

    report = data_quality_report(df, args.interval.replace("m", "min") if args.interval.endswith("m") else args.interval)
    print(f"\nWrote {len(df):,} bars to {out_path}")
    print(f"  span      : {report['start']} -> {report['end']}")
    print(f"  gaps      : {report.get('gap_count', 'n/a')}  ({report.get('missing_bars', 0)} missing bars)")
    print(f"  max |move|: {report['max_abs_return']:.2%}")
    print("\nLook at those gaps before you trust the file. Real feeds have holes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
