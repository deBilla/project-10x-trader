"""Turn a list of trades into edge metrics: expectancy, win rate, profit factor,
compounded return, max drawdown, and an annualized per-trade Sharpe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import Trade


@dataclass
class Metrics:
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float        # mean net return per trade
    profit_factor: float
    total_return: float      # compounded, at `position_fraction` sizing
    max_drawdown: float
    sharpe: float            # annualized, from per-trade returns
    avg_bars_held: float


def _annualization(trades: list[Trade]) -> float:
    """Trades per year, from the span of the trade sequence."""
    if len(trades) < 2:
        return 1.0
    span_days = (trades[-1].exit_time - trades[0].entry_time).total_seconds() / 86400
    if span_days <= 0:
        return 1.0
    per_year = len(trades) / (span_days / 365.0)
    return max(per_year, 1.0)


def compute_metrics(trades: list[Trade], position_fraction: float = 0.2) -> Metrics:
    if not trades:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    rets = [t.ret_pct for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    # Compounded equity at fixed fractional sizing, in trade order.
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity *= 1 + t.ret_pct * position_fraction
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    mean = sum(rets) / len(rets)
    if len(rets) > 1:
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        sharpe = (mean / std) * math.sqrt(_annualization(trades)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return Metrics(
        n_trades=len(trades),
        win_rate=len(wins) / len(trades),
        avg_win=(gross_win / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        expectancy=mean,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        total_return=equity - 1.0,
        max_drawdown=max_dd,
        sharpe=sharpe,
        avg_bars_held=sum(t.bars_held for t in trades) / len(trades),
    )
