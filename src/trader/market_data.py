"""Deterministic market-data + account fetching via alpaca-py.

Bars and account equity are pulled directly (not through the LLM) so indicator
math and the risk hook's position-sizing are reproducible and model-independent.
Claude still uses the MCP order tools for execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import AppConfig, SymbolConfig

# alpaca-py timeframe parsing -------------------------------------------------
_TF_UNITS = {"Min": "Minute", "Hour": "Hour", "Day": "Day"}


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    open_position_symbols: list[str]


def _parse_timeframe(tf: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    # e.g. "5Min" -> (5, Minute), "1Hour" -> (1, Hour), "1Day" -> (1, Day)
    for suffix, _ in _TF_UNITS.items():
        if tf.endswith(suffix):
            amount = int(tf[: -len(suffix)] or "1")
            unit = {
                "Min": TimeFrameUnit.Minute,
                "Hour": TimeFrameUnit.Hour,
                "Day": TimeFrameUnit.Day,
            }[suffix]
            return TimeFrame(amount, unit)
    raise ValueError(f"Unsupported bar timeframe: {tf}")


class MarketData:
    """Thin wrapper over alpaca-py historical + trading clients."""

    def __init__(self, cfg: AppConfig):
        from alpaca.data.historical import (
            CryptoHistoricalDataClient,
            StockHistoricalDataClient,
        )
        from alpaca.trading.client import TradingClient

        s = cfg.settings
        self._stock = StockHistoricalDataClient(s.alpaca_api_key, s.alpaca_secret_key)
        # Crypto data is public but the client accepts keys.
        self._crypto = CryptoHistoricalDataClient(s.alpaca_api_key, s.alpaca_secret_key)
        self._trading = TradingClient(
            s.alpaca_api_key, s.alpaca_secret_key, paper=s.alpaca_paper_trade
        )

    def get_bars(self, sym: SymbolConfig) -> pd.DataFrame:
        """Return an OHLCV DataFrame (oldest->newest) for one symbol."""
        from alpaca.data.requests import (
            CryptoBarsRequest,
            StockBarsRequest,
        )

        timeframe = _parse_timeframe(sym.bar_timeframe)
        # Pull a generous window; we slice to lookback_bars after.
        start = datetime.now(timezone.utc) - timedelta(days=10)

        if sym.asset_class == "crypto":
            req = CryptoBarsRequest(
                symbol_or_symbols=sym.symbol, timeframe=timeframe, start=start
            )
            bars = self._crypto.get_crypto_bars(req)
        else:
            req = StockBarsRequest(
                symbol_or_symbols=sym.symbol, timeframe=timeframe, start=start
            )
            bars = self._stock.get_stock_bars(req)

        df = bars.df  # multiindex (symbol, timestamp)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(sym.symbol, level="symbol")
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return df.tail(sym.lookback_bars)

    def get_history(self, sym: SymbolConfig, days: int, timeframe: str | None = None) -> pd.DataFrame:
        """Full OHLCV history for a symbol over the last `days` (no tail) — for
        backtesting. `timeframe` overrides the symbol's configured bar size
        (e.g. "1Day" for trend backtests)."""
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest

        timeframe = _parse_timeframe(timeframe or sym.bar_timeframe)
        start = datetime.now(timezone.utc) - timedelta(days=days)
        if sym.asset_class == "crypto":
            bars = self._crypto.get_crypto_bars(
                CryptoBarsRequest(symbol_or_symbols=sym.symbol, timeframe=timeframe, start=start)
            )
        else:
            from alpaca.data.enums import Adjustment

            # Split/dividend-adjusted prices — essential for backtests spanning splits
            # (raw data shows a split as a fake price crash and corrupts results).
            bars = self._stock.get_stock_bars(
                StockBarsRequest(symbol_or_symbols=sym.symbol, timeframe=timeframe,
                                 start=start, adjustment=Adjustment.ALL)
            )
        df = bars.df
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(sym.symbol, level="symbol")
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def get_account(self) -> AccountSnapshot:
        acct = self._trading.get_account()
        positions = self._trading.get_all_positions()
        return AccountSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            open_position_symbols=[p.symbol for p in positions],
        )

    def is_market_open(self) -> bool:
        return bool(self._trading.get_clock().is_open)

    def get_positions_detail(self) -> dict[str, dict]:
        """Return symbol -> {qty, avg_entry_price, current_price} for open positions."""
        out: dict[str, dict] = {}
        for p in self._trading.get_all_positions():
            out[p.symbol] = {
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
            }
        return out

    def close_position(self, symbol: str) -> None:
        """Close a single position (used by the software stop manager)."""
        self._trading.close_position(symbol)

    def submit_notional_buy(self, symbol: str, notional: float) -> str:
        """Place a fractional market BUY for a dollar amount (long-only). Returns
        the order id. TIF=day is required for notional/fractional equity orders."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = self._trading.submit_order(
            MarketOrderRequest(symbol=symbol, notional=round(notional, 2),
                               side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        )
        return str(order.id)

    def cancel_stale_orders(self, symbols: list[str], older_than_seconds: int) -> int:
        """Cancel this agent's own open orders (whitelist symbols) older than the
        cutoff. Frees buying power and prevents unfilled orders from stacking when
        a venue is slow to fill (e.g. Alpaca paper crypto). Returns count cancelled.
        """
        from datetime import datetime, timedelta, timezone

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        wl = set(symbols)
        n = 0
        for o in self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            if o.symbol in wl and o.submitted_at and o.submitted_at < cutoff:
                try:
                    self._trading.cancel_order_by_id(o.id)
                    n += 1
                except Exception:  # noqa: BLE001 — best effort
                    pass
        return n

    def close_positions(self, symbols: list[str]) -> None:
        """Liquidate only the given symbols + cancel their open orders.

        Scoped form of close_all_positions used by the per-agent drawdown switch
        so one agent's kill-switch never touches another agent's book.
        """
        open_syms = {p.symbol for p in self._trading.get_all_positions()}
        for sym in symbols:
            if sym in open_syms:
                try:
                    self._trading.close_position(sym)
                except Exception:  # noqa: BLE001 — best-effort liquidation
                    pass

    def close_all_positions(self) -> None:
        """Force-liquidate everything (kept for ad-hoc/manual use)."""
        self._trading.close_all_positions(cancel_orders=True)
