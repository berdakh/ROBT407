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


def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "algotrade-workshop/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


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
        for k in batch:
            rows.append(
                {
                    "timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                }
            )
        cursor = batch[-1][0] + step
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
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(
            f"\nCould not reach the exchange: {exc}\n"
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
