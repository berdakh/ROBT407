"""Shared fixtures. Nothing here touches the network or reads a market feed."""

from __future__ import annotations

import pandas as pd
import pytest

from algotrade.synthetic import generate


@pytest.fixture(scope="session")
def gbm_data() -> pd.DataFrame:
    """Unpredictable by construction. The control group."""
    df, _ = generate("gbm", n=2000, seed=101)
    return df


@pytest.fixture(scope="session")
def momentum_data() -> pd.DataFrame:
    """Contains a real, known, positive autocorrelation."""
    df, _ = generate("momentum", n=2000, seed=102)
    return df


@pytest.fixture(scope="session")
def reverting_data() -> pd.DataFrame:
    df, _ = generate("mean_reverting", n=2000, seed=103)
    return df


@pytest.fixture(scope="session")
def small_data() -> pd.DataFrame:
    df, _ = generate("gbm", n=300, seed=104)
    return df
