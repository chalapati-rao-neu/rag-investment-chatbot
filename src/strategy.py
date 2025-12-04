# src/strategy.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
import yfinance as yf

DateLike = Union[str, date, datetime]


@dataclass
class StrategyResult:
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    breakout_window: int
    sma_short: int
    sma_long: int
    data: pd.DataFrame
    summary: Dict[str, float]


def _to_timestamp(d: DateLike) -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return d
    return pd.to_datetime(d)


def download_price_data(
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
) -> pd.DataFrame:
    """
    Download daily price data for 'symbol' using yfinance.
    Handles MultiIndex columns and returns flat columns with
    at least: 'Close', 'High', 'Low'.
    """
    start_ts = _to_timestamp(start_date)
    end_ts = _to_timestamp(end_date)

    df = yf.download(
        symbol,
        start=start_ts,
        end=end_ts,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(f"No price data found for {symbol} between {start_ts.date()} and {end_ts.date()}.")

    # Handle MultiIndex columns like ('Close','AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        # Typical yfinance pattern: level 1 is ticker name
        df = df.xs(symbol, axis=1, level=-1)

    required_cols = {"Close", "High", "Low"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Downloaded data for {symbol} missing required columns {required_cols}. "
            f"Got {list(df.columns)}."
        )

    df = df.sort_index()
    return df


def run_livermore_strategy(
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
    breakout_window: int = 20,
    sma_short: int = 50,
    sma_long: int = 200,
) -> StrategyResult:
    """
    Reproduces the notebook-style Livermore breakout strategy:

    - 50-day and 200-day moving averages.
    - 20-day high/low (by default) on Close.
    - Position rules:
        * Position = 1 when:
          Close > previous 20-day High AND
          Close > 50MA AND
          Close > 200MA
        * Position = -1 when:
          Close < previous 20-day Low
        * 0 otherwise.
    - Buy-and-Hold Return = pct_change of Close.
    - Strategy Return = Buy-and-Hold Return * Position.
    - Both plotted / summarized using cumsum(), as in the .ipynb.
    """

    df = download_price_data(symbol, start_date, end_date)

    # Moving averages
    df[f"{sma_short}MA"] = df["Close"].rolling(window=sma_short).mean()
    df[f"{sma_long}MA"] = df["Close"].rolling(window=sma_long).mean()

    # Breakout bands
    df[f"{breakout_window}High"] = df["Close"].rolling(window=breakout_window).max()
    df[f"{breakout_window}Low"] = df["Close"].rolling(window=breakout_window).min()

    # Position logic exactly like the notebook
    df["Position"] = 0

    buy_condition = (
        (df["Close"] > df[f"{breakout_window}High"].shift(1)) &
        (df["Close"] > df[f"{sma_short}MA"]) &
        (df["Close"] > df[f"{sma_long}MA"])
    )

    sell_condition = df["Close"] < df[f"{breakout_window}Low"].shift(1)

    df.loc[buy_condition, "Position"] = 1
    df.loc[sell_condition, "Position"] = -1
    df["Position"] = df["Position"].fillna(0)

    # Returns and "Agg" cumulative sums (not compounded), like ipynb
    df["Buy-and-Hold Return"] = df["Close"].pct_change()
    df["Strategy Return"] = df["Buy-and-Hold Return"] * df["Position"]

    df["BH_CumReturn"] = df["Buy-and-Hold Return"].cumsum()
    df["Strategy_CumReturn"] = df["Strategy Return"].cumsum()

    # Summary
    bh_total = df["BH_CumReturn"].iloc[-1]
    strat_total = df["Strategy_CumReturn"].iloc[-1]

    summary = {
        "total_days": float(len(df)),
        "buy_and_hold_agg_return": float(bh_total),
        "strategy_agg_return": float(strat_total),
    }

    result = StrategyResult(
        symbol=symbol,
        start_date=df.index[0],
        end_date=df.index[-1],
        breakout_window=breakout_window,
        sma_short=sma_short,
        sma_long=sma_long,
        data=df,
        summary=summary,
    )

    return result


if __name__ == "__main__":
    symbol = "AAPL"
    start = "2019-01-01"
    end = "2024-12-31"

    print(f"Running Livermore strategy on {symbol} from {start} to {end}...")
    res = run_livermore_strategy(symbol, start, end)

    print("\nSummary (matching notebook style):")
    for k, v in res.summary.items():
        if "return" in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")

    print("\nLast few rows:")
    print(
        res.data[
            [
                "Close",
                f"{res.sma_short}MA",
                f"{res.sma_long}MA",
                f"{res.breakout_window}High",
                f"{res.breakout_window}Low",
                "Position",
                "Buy-and-Hold Return",
                "Strategy Return",
                "BH_CumReturn",
                "Strategy_CumReturn",
            ]
        ].tail(5)
    )
