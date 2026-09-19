# Real market data goes here

This directory is empty on purpose, and `.gitignore` keeps CSVs out of the
repository.

Populate it by running, from the repository root:

```bash
python scripts/fetch_market_data.py --symbol BTCUSDT --interval 1h --days 365
```

That script uses a public, keyless exchange endpoint. It sends no credentials,
places no orders, and contains no code path that could.

If your network blocks it — corporate proxies and many cloud sandboxes do — the
entire workshop still runs on `../synthetic/`. Nothing in the required path
depends on this directory having anything in it.
