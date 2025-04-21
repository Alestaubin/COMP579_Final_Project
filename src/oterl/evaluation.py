from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np


class Trade(TypedDict):
  """ABIDES-style executed order tuple."""

  qty: int  # signed (+buy, –sell)
  price: float  # executed price
  ts: int  # exchange_ts  (ns)


class EpisodeMetrics(TypedDict, total=False):
  impl_shortfall: float
  vwap_slippage: float
  market_impact: float
  pnl: float
  sharpe: float
  participation: float


def _vwap(prices: Sequence[float], vols: Sequence[int]) -> float:
  v = np.array(vols, dtype=float)
  p = np.array(prices, dtype=float)
  return float((p * v).sum() / v.sum())


def _series_sharpe(returns: np.ndarray, eps: float = 1e-9) -> float:
  mu = returns.mean()
  sd = returns.std()
  return float(mu / (sd + eps))


def compute_episode_metrics(
  trades: Sequence[Trade],
  lob_mid_prices: Sequence[float],
  lob_volumes: Sequence[int],
  arrival_price: float,
) -> EpisodeMetrics:
  """Compute key optimal‑execution metrics for *one* episode."""

  if not trades:
    raise ValueError('Episode contains no executed trades')

  qty = np.array([t['qty'] for t in trades])
  px = np.array([t['price'] for t in trades])
  signed_notional = qty * px
  filled_qty = qty.sum()
  avg_fill_price = float(signed_notional.sum() / filled_qty)

  # Implementation shortfall
  impl_shortfall = float((avg_fill_price - arrival_price))

  # VWAP slippage
  vwap_market = _vwap(prices=lob_mid_prices, vols=lob_volumes)
  vwap_slippage = float((avg_fill_price - vwap_market) / vwap_market)

  # Simple market impact (mid‑price before vs after episode)
  market_impact = float(lob_mid_prices[-1] - lob_mid_prices[0])

  # Realised PnL (long‑only formulation; adjust sign if selling)
  pnl = float(-filled_qty * (avg_fill_price - arrival_price))

  # Per‑step PnL Sharpe
  step_pnl = -qty * (px - arrival_price) # per‑trade, treat like returns
  sharpe = _series_sharpe(step_pnl)

  participation = float(abs(filled_qty) / np.array(lob_volumes).sum())

  return EpisodeMetrics(
    impl_shortfall=impl_shortfall,
    vwap_slippage=vwap_slippage,
    market_impact=market_impact,
    pnl=pnl,
    sharpe=sharpe,
    participation=participation,
  )
