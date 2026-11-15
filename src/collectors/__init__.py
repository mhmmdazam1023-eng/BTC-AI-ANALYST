"""Data collection layer (Phase 2).

Each external API has one fetcher class inheriting from
:class:`~src.collectors.base.BaseDataFetcher`.

Scope so far:
* Part 1 — Binance (crypto prices), FRED (macro indicators).
* Part 2 — Blockchain.com (on-chain metrics), Alternative.me (sentiment).
* Part 3 — Binance Vision (derivatives/liquidity; Deribit/Bybit/OKX are
  ISP-blocked — DeribitFetcher is kept for unrestricted networks),
  Stooq (market indices, with FRED fallback).

Phase 8 adds :class:`BinanceMultiFetcher`: the same Binance transport, but
driven by ``coins_registry`` so the pipeline ingests every active asset
(BTC and ETH first — see the class docstring for the hierarchy rule).
"""

from src.collectors.alternative_me_fetcher import AlternativeMeFetcher
from src.collectors.base import BaseDataFetcher, FetcherError, RetryableHTTPError
from src.collectors.binance_fetcher import BinanceFetcher, BinanceMultiFetcher
from src.collectors.binance_vision_fetcher import BinanceVisionFetcher
from src.collectors.blockchain_fetcher import BlockchainFetcher
from src.collectors.deribit_fetcher import DeribitFetcher
from src.collectors.fred_fetcher import FredFetcher
from src.collectors.live_funding_fetcher import LiveFundingFetcher
from src.collectors.stooq_fetcher import StooqFetcher
from src.collectors.supply_fetcher import SupplyFetcher

__all__ = [
    "BaseDataFetcher",
    "FetcherError",
    "RetryableHTTPError",
    "BinanceFetcher",
    "BinanceMultiFetcher",
    "FredFetcher",
    "BlockchainFetcher",
    "AlternativeMeFetcher",
    "BinanceVisionFetcher",
    "DeribitFetcher",
    "LiveFundingFetcher",
    "StooqFetcher",
    "SupplyFetcher",
]
