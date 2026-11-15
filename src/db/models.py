"""SQLAlchemy 2.0 ORM models for the TimescaleDB data warehouse.

Design notes
------------
* Every table is time-series oriented. TimescaleDB requires the time
  partitioning column to be part of every UNIQUE constraint / primary key,
  therefore all tables use a **composite primary key that includes
  ``timestamp``**. Natural keys double as idempotent upsert targets during
  ingestion (Phase 2).
* Tables are converted into hypertables by ``scripts/init_db.py`` using the
  :data:`HYPERTABLES` registry at the bottom of this module.
* Numeric columns use ``Numeric`` (exact decimal) rather than ``Float`` to
  avoid IEEE-754 rounding artefacts in financial data.

Table -> data source mapping (Phase 2)
--------------------------------------
=====================  =====================================================
Table                  Feeding APIs
=====================  =====================================================
crypto_prices          ccxt: Binance & Indodax OHLCV
macro_indicators       fredapi (FRED series), pandas-datareader (Stooq)
on_chain_metrics       Glassnode / blockchain.info (httpx)
sentiment_data         Alternative.me Fear & Greed, Google Trends, CoinGecko
derivatives_metrics    Deribit funding rate / open interest / DVOL
market_indices         Stooq daily closes (S&P 500, Nasdaq 100)
=====================  =====================================================

Phase 8 adds two **dimension** tables that are deliberately NOT hypertables
(they hold reference data, not time series): ``coins_registry`` (the "Coin
DNA" that drives which symbols are ingested) and ``tokenomics_events``
(scheduled supply shocks per coin).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all warehouse models."""


#: The numéraire of the whole system. Every altcoin's relative-strength
#: feature is measured against this symbol, and every BTC-only query in
#: Phase 4-7 filters on it. Kept as one constant so the assumption is
#: greppable instead of scattered as a string literal.
BASE_SYMBOL = "BTC/USDT"

#: Symbols the ingestion pipeline MUST collect first, in this exact order
#: (Phase 8 hierarchical-ingestion rule). BTC is the denominator of every
#: altcoin feature; ETH is the secondary factor most altcoin betas load on.
HIERARCHY_HEAD: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")

#: The intraday timeframe (Phase 8 Part 8). BTC ONLY — the 20 altcoins stay
#: daily. Stored in ``crypto_prices`` alongside the ``1d`` candles: that
#: table's PK has always included ``timeframe``, so no separate candle table
#: is needed (or wanted — it would duplicate working upsert/retry logic).
INTRADAY_TIMEFRAME = "4h"

#: 4-hour bars in one UTC day. Bars open at 00/04/08/12/16/20 UTC.
#: Used to convert a horizon expressed in BARS into calendar terms; the
#: intraday forecast tables deliberately store ``horizon_bars``, never days,
#: because 4h/12h horizons are not whole days and mixing the two units in
#: one column is how "+1 day" silently becomes "+1 bar".
BARS_PER_DAY = 6


class _IngestedAtMixin:
    """Adds a server-side ingestion timestamp for data lineage/debugging."""

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="When the row was written by the ingestion pipeline.",
    )


class CoinSector(StrEnum):
    """Narrative/sector bucket a coin belongs to ("factor exposure").

    Stored as a NATIVE PostgreSQL enum (``coin_sector``). Crypto narratives
    churn, so extending it is expected — add the member here AND run
    ``ALTER TYPE coin_sector ADD VALUE 'NewSector';`` (PostgreSQL 12+ allows
    this inside a transaction). ``scripts/migrate_phase8.py`` does exactly
    that for any member missing from the live type, so re-running the
    migration after editing this class is enough.
    """

    L1 = "L1"
    L2 = "L2"
    DEFI = "DeFi"
    AI = "AI"
    MEME = "Meme"
    RWA = "RWA"
    INFRASTRUCTURE = "Infrastructure"


class TokenomicsEventType(StrEnum):
    """Kind of supply-side shock recorded in ``tokenomics_events``."""

    UNLOCK = "unlock"
    BURN = "burn"
    HALVING = "halving"
    MAINNET_LAUNCH = "mainnet_launch"


#: SQLAlchemy type for :class:`CoinSector`. ``values_callable`` is REQUIRED:
#: without it SQLAlchemy persists the member *name* ("DEFI") instead of the
#: value ("DeFi"), which is not what the registry seed or any human expects.
_SECTOR_ENUM = SAEnum(
    CoinSector,
    name="coin_sector",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)

#: SQLAlchemy type for :class:`TokenomicsEventType` (same rationale).
_EVENT_TYPE_ENUM = SAEnum(
    TokenomicsEventType,
    name="tokenomics_event_type",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class CoinRegistry(Base, _IngestedAtMixin):
    """The "Coin DNA": static per-asset metadata driving the whole pipeline.

    Deliberately a PLAIN table, not a hypertable — it is a *dimension*
    (a handful of slowly-changing rows), not a time series. It is the single
    source of truth for:

    * **which** symbols the collectors ingest (``is_active``),
    * **what** each coin is (``sector``, ``chain``) — its factor exposure,
    * **against what** its relative strength is judged (``base_asset``).

    ``base_asset`` is NULL for the numéraire itself (BTC/USDT): Bitcoin has
    no benchmark above it in this system.
    """

    __tablename__ = "coins_registry"

    symbol: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="Unified ccxt symbol, e.g. 'SOL/USDT'."
    )
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, doc="Human-readable project name, e.g. 'Solana'."
    )
    sector: Mapped[CoinSector] = mapped_column(
        _SECTOR_ENUM, nullable=False, doc="Narrative bucket — see CoinSector."
    )
    chain: Mapped[str] = mapped_column(
        String(32), nullable=False, doc="Settlement chain, e.g. 'Solana', 'Ethereum'."
    )
    base_asset: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="Benchmark symbol for relative strength ('BTC/USDT'/'ETH/USDT'); "
        "NULL for the numéraire itself.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        doc="Only active symbols are ingested and featurised.",
    )
    coingecko_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="CoinGecko API id (e.g. 'avalanche-2'). NULL = skip supply "
        "collection for this coin; the fetcher degrades quietly rather "
        "than guessing an id and silently fetching the wrong asset.",
    )

    __table_args__ = (Index("ix_coins_registry_active", "is_active"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<CoinRegistry {self.symbol} ({self.name}) sector={self.sector} "
            f"chain={self.chain} active={self.is_active}>"
        )


class TokenomicsEvent(Base, _IngestedAtMixin):
    """Scheduled/realised supply shocks per coin (unlocks, burns, halvings).

    Plain table (tiny volume, needs a surrogate ``id``). The Feature Engine
    will consume this to know a supply shock lands on a given date.

    ``supply_impact_pct`` units — read this before writing rows:

    * ``unlock``          — POSITIVE, % of *circulating supply* newly released
                            (e.g. 2.5 = +2.5% supply hitting the market).
    * ``burn``            — NEGATIVE, % of circulating supply destroyed.
    * ``halving``         — change to the *emission rate*, not to the supply
                            stock (-50.0 for a Bitcoin halving). Sign is
                            negative because issuance falls.
    * ``mainnet_launch``  — % of supply that becomes transferable at launch,
                            or NULL when there is no direct supply effect.

    The unique constraint on (symbol, event_date, event_type) makes seeding
    and re-ingestion idempotent, matching every other table in the warehouse.
    """

    __tablename__ = "tokenomics_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("coins_registry.symbol", ondelete="CASCADE"),
        nullable=False,
        doc="FK to coins_registry.symbol.",
    )
    event_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="When the shock lands (UTC)."
    )
    event_type: Mapped[TokenomicsEventType] = mapped_column(
        _EVENT_TYPE_ENUM, nullable=False
    )
    supply_impact_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 6), nullable=True, doc="See the class docstring for units/sign."
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol", "event_date", "event_type", name="uq_tokenomics_event"
        ),
        Index("ix_tokenomics_events_symbol_date", "symbol", "event_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TokenomicsEvent {self.symbol} {self.event_type} "
            f"@ {self.event_date.date()} impact={self.supply_impact_pct}%>"
        )


class SupplyMetric(Base, _IngestedAtMixin):
    """Circulating / total supply per coin (Phase 8 Part 4, source: CoinGecko).

    Hypertable keyed on ``(timestamp, symbol)``. CoinGecko's free endpoint
    returns a SNAPSHOT, not history, so this table only grows forward from
    the day collection starts — every feature derived from it is NULL for
    historical rows. That is honest and deliberate: back-filling supply by
    interpolation would invent the very quantity the inflation feature is
    supposed to measure.

    ``total_supply`` is nullable because CoinGecko reports it as null for
    some assets, and it is the CURRENT issued supply rather than a max cap
    (for BTC it is ~20.06M, not 21M).
    """

    __tablename__ = "supply_metrics"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Observation day (UTC midnight)."
    )
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)

    circulating_supply: Mapped[Decimal | None] = mapped_column(
        Numeric(40, 8), nullable=True
    )
    total_supply: Mapped[Decimal | None] = mapped_column(Numeric(40, 8), nullable=True)

    __table_args__ = (Index("ix_supply_metrics_symbol_ts", "symbol", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<SupplyMetric {self.symbol} {self.timestamp.date()} "
            f"circ={self.circulating_supply}>"
        )


class VolRegime(StrEnum):
    """Where realized volatility sits versus its own trailing year."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class MarketCapTier(StrEnum):
    """Size bucket from close x circulating supply."""

    LARGE = "LARGE"
    MID = "MID"
    SMALL = "SMALL"


_VOL_REGIME_ENUM = SAEnum(
    VolRegime,
    name="vol_regime",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)

_MARKET_CAP_TIER_ENUM = SAEnum(
    MarketCapTier,
    name="market_cap_tier",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class CryptoPrice(Base, _IngestedAtMixin):
    """OHLCV candles for crypto pairs (source: ccxt — Binance, Indodax)."""

    __tablename__ = "crypto_prices"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Candle open time (UTC)."
    )
    exchange: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="Exchange id, e.g. 'binance', 'indodax'."
    )
    symbol: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="Unified ccxt symbol, e.g. 'BTC/USDT'."
    )
    timeframe: Mapped[str] = mapped_column(
        String(8), primary_key=True, doc="Candle timeframe, e.g. '1h', '4h', '1d'."
    )

    open: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 10), nullable=False)

    __table_args__ = (
        Index("ix_crypto_prices_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<CryptoPrice {self.exchange}:{self.symbol} {self.timeframe} "
            f"@ {self.timestamp} close={self.close}>"
        )


class MacroIndicator(Base, _IngestedAtMixin):
    """Macro & traditional-market series (source: FRED via fredapi, Stooq)."""

    __tablename__ = "macro_indicators"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Observation date (UTC)."
    )
    source: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="Data provider, e.g. 'FRED', 'STOOQ'."
    )
    indicator: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        doc="Series id, e.g. 'DGS10', 'T5YIE', 'CPIAUCSL', '^SPX', 'DX.F'.",
    )

    value: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)

    __table_args__ = (Index("ix_macro_indicator_ts", "indicator", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MacroIndicator {self.source}:{self.indicator} @ {self.timestamp} = {self.value}>"


class OnChainMetric(Base, _IngestedAtMixin):
    """On-chain metrics (source: Glassnode / blockchain.info via httpx)."""

    __tablename__ = "on_chain_metrics"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Metric observation time (UTC)."
    )
    source: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="e.g. 'GLASSNODE', 'BLOCKCHAIN_INFO'."
    )
    asset: Mapped[str] = mapped_column(
        String(16), primary_key=True, default="BTC", doc="Asset ticker, e.g. 'BTC'."
    )
    metric: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        doc="Metric path/name, e.g. 'addresses/active_count', 'hash_rate'.",
    )

    value: Mapped[Decimal] = mapped_column(Numeric(38, 12), nullable=False)

    __table_args__ = (Index("ix_on_chain_metric_ts", "metric", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OnChainMetric {self.source}:{self.asset}:{self.metric} @ {self.timestamp}>"


class SentimentData(Base, _IngestedAtMixin):
    """Sentiment series (source: Alternative.me F&G, Google Trends, CoinGecko)."""

    __tablename__ = "sentiment_data"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Observation time (UTC)."
    )
    source: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        doc="e.g. 'ALTERNATIVE_ME', 'GOOGLE_TRENDS', 'COINGECKO', 'NEWS'.",
    )
    metric: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        doc="e.g. 'fear_greed_index', 'search_interest_bitcoin'.",
    )

    value: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 10), nullable=True, doc="Numeric score, if the metric has one."
    )
    value_text: Mapped[str | None] = mapped_column(
        Text, nullable=True, doc="Categorical label, e.g. 'Extreme Fear'."
    )
    raw_payload: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, doc="Original API payload for audit/debugging."
    )

    __table_args__ = (Index("ix_sentiment_metric_ts", "metric", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SentimentData {self.source}:{self.metric} @ {self.timestamp} = {self.value}>"


class MarketIndex(Base, _IngestedAtMixin):
    """Daily closes of traditional market indices (source: Stooq).

    Wide format by design (one row per day) — separated from
    ``macro_indicators`` so equity benchmarks never mix with FRED series.
    Columns are nullable because the two indices can differ on trading
    holidays; a later fetch fills the gap via upsert.
    """

    __tablename__ = "market_indices"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Trading day (UTC midnight)."
    )
    sp500_close: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 4), nullable=True, doc="S&P 500 daily close (Stooq ^SPX)."
    )
    nasdaq_close: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 4), nullable=True, doc="Nasdaq 100 daily close (Stooq ^NDX)."
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MarketIndex {self.timestamp.date()} "
            f"spx={self.sp500_close} ndx={self.nasdaq_close}>"
        )


class DerivativesMetric(Base, _IngestedAtMixin):
    """Derivatives/liquidity metrics (source: Deribit, Binance via ccxt/httpx).

    Grounded in the research corpus: funding rates and open interest are key
    liquidity-regime signals (Zhivkov et al. 2026; Krause 2026).
    """

    __tablename__ = "derivatives_metrics"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Observation time (UTC)."
    )
    exchange: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="e.g. 'binance', 'deribit'."
    )
    symbol: Mapped[str] = mapped_column(
        String(32), primary_key=True, doc="Instrument, e.g. 'BTC/USDT:USDT', 'BTC-PERPETUAL'."
    )
    metric: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        doc="e.g. 'funding_rate', 'open_interest', 'basis', 'dvol'.",
    )

    value: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)

    __table_args__ = (Index("ix_derivatives_metric_ts", "metric", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DerivativesMetric {self.exchange}:{self.symbol}:{self.metric} @ {self.timestamp}>"


class DailyFeature(Base, _IngestedAtMixin):
    """One row per (day, symbol) with EVERY engineered feature (Phase 3/8).

    Purpose: the Phase 4 AI agents need exactly ONE simple query
    (``SELECT * FROM daily_features WHERE timestamp = :day AND symbol = :s``)
    to obtain the full feature vector. Columns are nullable: indicator
    warm-up windows (e.g. 30-day MAs) and source-history limits leave leading
    rows partially empty; recent rows are complete (enforced by the Phase 3
    test).

    Phase 8 (multi-asset) — composite PK ``(timestamp, symbol)``
    -----------------------------------------------------------
    Before Phase 8 the PK was ``timestamp`` alone and every row was Bitcoin.
    The migration backfills ``symbol = 'BTC/USDT'`` on all historical rows,
    so BTC behaviour is bit-for-bit unchanged. **Every read of this table
    MUST filter on ``symbol``** — an unfiltered ``ORDER BY timestamp DESC
    LIMIT 1`` now returns an arbitrary asset.

    Column semantics across assets (important):

    * ``asset_close`` / ``asset_volume`` — THIS row's own asset. All
      technical indicators (RSI, MACD, Bollinger, ATR, realized vol) are
      computed from this asset's OHLCV.
    * ``btc_close`` / ``btc_volume`` — always **Bitcoin's** close/volume on
      that day, on every row, including altcoin rows. It is the market
      reference and the denominator of ``price_btc_ratio``. On a BTC row it
      equals ``asset_close``/``asset_volume`` by construction, which is why
      pre-Phase-8 BTC rows stay valid untouched.
    * Macro / on-chain / sentiment / derivatives / index columns are
      BTC-and-market-wide context, replicated onto every symbol's row.

    All values are computed deterministically with pandas/numpy/`ta`
    (NEVER an LLM — project rule since 2026-07-08).
    """

    __tablename__ = "daily_features"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Feature day (UTC midnight)."
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        # Quoted via text(): a bare Python string is emitted verbatim into the
        # DDL (DEFAULT BTC/USDT -> syntax error). The default exists so the
        # Phase 8 ALTER on the pre-existing BTC-only table is a no-downtime
        # one-liner and any legacy insert still lands on Bitcoin.
        server_default=text(f"'{BASE_SYMBOL}'"),
        doc="Asset this feature row describes, e.g. 'BTC/USDT', 'SOL/USDT'.",
    )

    # --- Own-asset price context (from crypto_prices, this row's symbol) ---
    asset_close: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    asset_volume: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- Bitcoin reference price context (BTC on EVERY row) ---
    btc_close: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    btc_volume: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- Technical (ta library on BTC OHLCV) ---
    rsi_14: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    macd: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    macd_signal: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    macd_hist: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    bb_upper: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    bb_middle: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    bb_lower: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    atr_14: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    realized_vol_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    realized_vol_30d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    ewma_vol_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 12),
        nullable=True,
        doc="Exponentially weighted annualised volatility, lambda=0.97 "
        "(2026-08-02). Better calibrated than realized_vol_30d at every "
        "horizon measured (34.8k asset-days): 91.0/89.8/87.0% band coverage "
        "at +1/+7/+30d versus 89.2/87.5/84.3%. Consumed by the scenario "
        "engine and Kelly's variance. realized_vol_30d is deliberately NOT "
        "replaced — it defines the Phase 9 verdict band, and redefining that "
        "would invalidate every stored evaluation and the null model with it.",
    )

    # --- Macro (FRED levels + rate-of-change) ---
    dxy_value: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    dxy_roc_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    dxy_roc_30d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    us10y_value: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    us10y_roc_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    us10y_roc_30d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- On-chain (moving averages) ---
    hash_rate_ma_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    hash_rate_ma_30d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    tx_count_ma_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    tx_count_ma_30d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- Sentiment ---
    fear_greed_value: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    fear_greed_roc_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- Liquidity (Binance Vision derivatives) ---
    funding_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    funding_rate_change_1d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    open_interest: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    oi_volume_ratio: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- Market indices (research-grounded extra: Krause 2026 Nasdaq beta) ---
    sp500_close: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    nasdaq_close: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    sp500_roc_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    nasdaq_roc_7d: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)

    # --- Relative strength vs Bitcoin (Phase 8 — the core altcoin quant
    #     feature). Defined for EVERY symbol, Bitcoin included: BTC's ratio
    #     against itself is exactly 1.0 and its relative strength exactly
    #     0.0, which keeps the "recent window is complete" gate honest
    #     instead of carving out NULL exceptions. ---
    # 30 decimal places, not the 12 used elsewhere: this column holds a
    # RATIO, not a price, and ratios of low-priced tokens are TINY.
    #
    # Sizing is empirical, not guessed. PEPE at $0.00000279 against BTC at
    # $63k gives a ratio of 4.4e-11. Scale 12 kept ~2 significant digits;
    # scale 18 still lost 2.2e-8 of relative precision and was caught by the
    # Part 1 reconciliation test the day PEPE entered the registry.
    #
    # The correct target is to sit BELOW float64's own limit (~1.1e-16
    # relative), since the value is computed in pandas as a float — storage
    # must never be the binding constraint. Scale 30 gives ~2e-20 relative
    # precision at PEPE's magnitude, with headroom down to ~1e-14 ratios.
    price_btc_ratio: Mapped[Decimal | None] = mapped_column(
        Numeric(40, 30), nullable=True, doc="asset_close / btc_close (same day)."
    )
    rs_7d: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 12),
        nullable=True,
        doc="7d % change of price_btc_ratio — outperformance vs BTC.",
    )
    rs_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 12), nullable=True, doc="30d % change of price_btc_ratio."
    )

    # --- Coin DNA: supply & tokenomics (Phase 8 Part 4) ---
    # All NULL-tolerant. tokenomics_events and supply_metrics start empty
    # for most coins, and a feature that errors on absent data would take
    # the whole pipeline down for a coin nobody has filed unlocks for yet.
    days_to_next_unlock: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Days until the next SUPPLY-INCREASING event; NULL when none is filed.",
    )
    supply_shock_score_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        doc="Net % of circulating supply unlocked/burned in the next 30 days; "
        "0 when nothing is scheduled (0 is a real answer, not missing data).",
    )
    inflation_rate_7d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="7d % change in circulating supply."
    )
    supply_utilization: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 8), nullable=True, doc="circulating_supply / total_supply."
    )

    # --- Coin DNA: volatility regime ---
    vol_percentile_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(9, 4),
        nullable=True,
        doc="realized_vol_30d percentile (0-100) within its trailing year.",
    )
    vol_regime: Mapped[VolRegime | None] = mapped_column(
        _VOL_REGIME_ENUM, nullable=True
    )

    # --- Coin DNA: liquidity & market structure ---
    liquidity_score: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 10),
        nullable=True,
        doc="Turnover: daily base volume / circulating supply (dimensionless).",
    )
    market_cap_tier: Mapped[MarketCapTier | None] = mapped_column(
        _MARKET_CAP_TIER_ENUM, nullable=True
    )
    btc_beta_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 8),
        nullable=True,
        doc="30d BETA vs BTC: cov(r_asset, r_btc) / var(r_btc). Sensitivity.",
    )
    btc_corr_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 8),
        nullable=True,
        doc="30d CORRELATION vs BTC (-1..1). Co-movement, not sensitivity.",
    )

    # --- Coin DNA: momentum ---
    momentum_score: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        doc="-100..+100 composite of 7d RSI trend and 7d MACD-histogram trend.",
    )
    trend_strength: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        doc="|30d return| / 30d realized vol — signal-to-noise, dimensionless.",
    )

    __table_args__ = (Index("ix_daily_features_symbol_ts", "symbol", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DailyFeature {self.symbol} {self.timestamp.date()} "
            f"close={self.asset_close}>"
        )


#: Every feature column on ``daily_features``.
#: Excludes the primary key parts (``timestamp``, ``symbol``) and the
#: lineage column — ``symbol`` is an IDENTIFIER, not a feature, and must
#: never be fed to the numeric conversion in FeatureEngine.save_features()
#: nor into the agents' feature dicts.
DAILY_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column.name
    for column in DailyFeature.__table__.columns
    if column.name not in ("timestamp", "symbol", "ingested_at")
)

#: CATEGORICAL feature columns (Phase 8 Part 4 introduced the first ones).
#: They hold label strings, not numbers. Anything that converts features to
#: float — ``FeatureEngine.save_features``, ``run_inference`` — must route
#: around these, or a daily run dies on ``float('HIGH')``.
DAILY_FEATURE_CATEGORICAL_COLUMNS: frozenset[str] = frozenset(
    {"vol_regime", "market_cap_tier"}
)

#: Feature columns safe to coerce to float.
DAILY_FEATURE_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    name
    for name in DAILY_FEATURE_COLUMNS
    if name not in DAILY_FEATURE_CATEGORICAL_COLUMNS
)


class MarketRegime(StrEnum):
    """Market state the rule engine classifies BTC into (Phase 8 Part 2).

    ``UNDETERMINED`` is not one of the five real regimes: it is the explicit
    "regime tidak jelas" outcome required so the weighting engine has a
    defined fallback (equal weights) instead of guessing. Emitting it is a
    feature — the detector says "I don't know" rather than forcing a label.
    """

    BULL_TRENDING = "BULL_TRENDING"
    BEAR_TRENDING = "BEAR_TRENDING"
    SIDEWAYS_LOW_VOL = "SIDEWAYS_LOW_VOL"
    SIDEWAYS_HIGH_VOL = "SIDEWAYS_HIGH_VOL"
    CRASH_RECOVERY = "CRASH_RECOVERY"
    UNDETERMINED = "UNDETERMINED"


#: SQLAlchemy type for :class:`MarketRegime` (``values_callable`` for the
#: same reason as the Part 1 enums — persist values, not member names).
_REGIME_ENUM = SAEnum(
    MarketRegime,
    name="market_regime",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class MarketRegimeRow(Base, _IngestedAtMixin):
    """Daily market-regime classification + the evidence behind it.

    Hypertable keyed on ``timestamp`` alone: this is the MARKET-WIDE regime,
    measured from Bitcoin (``symbol`` records which series it was measured
    from, for provenance — it is NOT part of the key, because "the market
    regime" is one thing per day, not one per asset).

    Every input that fed the decision is stored as its own column rather
    than only inside ``regime_scores``. A rule-based classifier earns its
    keep by being auditable: when a regime call looks wrong months later,
    these columns answer "what did it see?" without re-running anything.
    """

    __tablename__ = "market_regimes"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Classified day (UTC midnight)."
    )
    regime: Mapped[MarketRegime] = mapped_column(_REGIME_ENUM, nullable=False)
    confidence: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="0-100: rule strength blended with separation."
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text(f"'{BASE_SYMBOL}'"),
        doc="Series the regime was measured from (provenance, not a key).",
    )

    # --- evidence: trend ---
    close: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    sma_50: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    sma_200: Mapped[Decimal | None] = mapped_column(Numeric(38, 12), nullable=True)
    pct_vs_sma_50: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    pct_vs_sma_200: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    macd_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="MACD as % of close (scale-free)."
    )

    # --- evidence: participation & sentiment ---
    volume_ratio: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Recent volume MA / 30d volume MA."
    )
    fear_greed_value: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True
    )

    # --- evidence: volatility & drawdown ---
    realized_vol_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True
    )
    vol_baseline_1y: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="1-year mean of realized_vol_30d."
    )
    drawdown_from_ath_90d_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Negative = below the 90-day high."
    )
    rsi_14: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    rsi_min_30d: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)

    # --- audit ---
    regime_scores: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, doc="Soft score (0..1) per candidate regime."
    )
    rule_version: Mapped[str] = mapped_column(
        String(16), nullable=False, doc="Detector ruleset version that produced this."
    )

    __table_args__ = (Index("ix_market_regimes_regime_ts", "regime", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MarketRegimeRow {self.timestamp.date()} {self.regime} "
            f"confidence={self.confidence}>"
        )


class SectorFlowStatus(StrEnum):
    """Where capital is moving relative to a sector (Phase 8 Part 3).

    A clean 2x2 over the weekly and monthly BTC-relative strength (user
    decision, 2026-07-28). The quadrants are MUTUALLY EXCLUSIVE, so unlike
    the earlier three-state version there is no subset hazard and the order
    of the checks cannot make a state unreachable:

    ==================  ===========  ============  =========================
    Status              ``rs_7d``    ``rs_30d``    Reading
    ==================  ===========  ============  =========================
    ``IN_FLOW``         > 0          > 0           leading and holding it
    ``ACCUMULATING``    > 0          < 0           weak month, turning up
                                                   — bottoming / reversal
    ``PULLBACK``        < 0          > 0           strong month, correcting
    ``OUT_OF_FLOW``     < 0          < 0           lagging outright
    ``NEUTRAL``         anything else, or either value missing
    ==================  ===========  ============  =========================
    """

    IN_FLOW = "IN_FLOW"
    OUT_OF_FLOW = "OUT_OF_FLOW"
    #: Weak over the month but bidding this week — a bottoming signal.
    ACCUMULATING = "ACCUMULATING"
    #: Strong over the month but correcting this week.
    PULLBACK = "PULLBACK"
    #: Either leg is exactly zero, or there is not enough data to place the
    #: sector in a quadrant.
    NEUTRAL = "NEUTRAL"


#: SQLAlchemy type for :class:`SectorFlowStatus` (``values_callable`` for
#: the same reason as every other enum here — persist values, not names).
_SECTOR_STATUS_ENUM = SAEnum(
    SectorFlowStatus,
    name="sector_flow_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class SectorPerformance(Base, _IngestedAtMixin):
    """Daily per-sector capital-flow reading (Phase 8 Part 3).

    Hypertable keyed on ``(timestamp, sector)``. Values are averages of the
    member coins' BTC-relative strength, so a positive number means the
    sector as a whole outperformed Bitcoin over that window.

    ``coin_count`` and ``coin_count_30d`` are tracked SEPARATELY on purpose.
    Averages ignore NULLs, so a sector's 30-day figure can come from fewer
    coins than its 7-day figure — a freshly listed token has an ``rs_7d``
    long before it has an ``rs_30d``. Without both counts, "AI +18%" reads
    identically whether it came from three coins or one, which is the
    difference between a sector signal and a single token's noise.
    """

    __tablename__ = "sector_performance"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Reading day (UTC midnight)."
    )
    sector: Mapped[CoinSector] = mapped_column(_SECTOR_ENUM, primary_key=True)

    avg_rs_7d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Mean rs_7d across the sector (%)."
    )
    avg_rs_30d: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Mean rs_30d across the sector (%)."
    )
    status: Mapped[SectorFlowStatus] = mapped_column(
        _SECTOR_STATUS_ENUM, nullable=False
    )
    ranking: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="1 = strongest sector by avg_rs_7d that day."
    )
    coin_count: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Coins contributing to avg_rs_7d."
    )
    coin_count_30d: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        doc="Coins contributing to avg_rs_30d (can be fewer — new listings).",
    )

    __table_args__ = (Index("ix_sector_performance_sector_ts", "sector", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<SectorPerformance {self.timestamp.date()} {self.sector} "
            f"#{self.ranking} rs7={self.avg_rs_7d} {self.status}>"
        )


class Decision(Base, _IngestedAtMixin):
    """AI decision history (Phase 5 output). One row per analysis day.

    Re-running inference for the same day UPDATES the row (upsert on
    ``timestamp``) — no duplicates. ``sub_agents_json`` preserves the full
    panel (signal, score, reasoning per pillar) for auditability.
    """

    __tablename__ = "decisions"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        doc="Analysis day (matches the daily_features row used, UTC midnight).",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        server_default=text(f"'{BASE_SYMBOL}'"),
        doc="Asset this decision is about, e.g. 'BTC/USDT', 'SOL/USDT'. "
        "Part of the PK since Phase 8 Part 6 — before it, one row per day "
        "meant a multi-asset run would upsert every coin onto the same row "
        "and keep only whichever finished last. EVERY read of this table "
        "must filter on symbol.",
    )
    action: Mapped[str] = mapped_column(
        String(8), nullable=False, doc="Final decision: BUY / HOLD / SELL."
    )
    confidence_score: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Master Agent confidence, 0-100."
    )
    executive_summary: Mapped[str] = mapped_column(Text, nullable=False)
    risk_warning: Mapped[str] = mapped_column(Text, nullable=False)
    sub_agents_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        doc="Full sub-agent panel: [{agent_name, signal, score, reasoning}, ...].",
    )

    __table_args__ = (Index("ix_decisions_symbol_ts", "symbol", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Decision {self.symbol} {self.timestamp.date()} {self.action} "
            f"confidence={self.confidence_score}>"
        )


class PositionRecommendation(Base, _IngestedAtMixin):
    """Kelly-based position sizing per (day, asset) — Phase 8 Part 7.

    MODEL OUTPUT, NOT ADVICE. Every field here is the arithmetic result of
    a published formula applied to this system's own estimates; it carries
    no knowledge of the reader's finances, and the estimates behind it are
    themselves uncertain. The stored provenance columns exist so that a
    number can always be traced back to what it was actually built on.

    Provenance matters more than the number
    ---------------------------------------
    ``win_rate_source`` and ``win_rate_samples`` are persisted precisely so
    a fraction derived from six observations can never be mistaken for one
    derived from six hundred. ``scenario_source`` likewise records whether
    the distribution came from the Monte Carlo engine or from an analytic
    volatility band.
    """

    __tablename__ = "position_recommendations"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Recommendation day (UTC)."
    )
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)

    action: Mapped[str] = mapped_column(
        String(8), nullable=False, doc="The decision this sizing applies to."
    )
    kelly_fraction: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 8), nullable=True, doc="Raw FULL Kelly f* (never used as-is)."
    )
    adjusted_fraction: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 8),
        nullable=True,
        doc="After fractional-Kelly scaling and every risk overlay.",
    )
    position_size_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(9, 4), nullable=True, doc="adjusted_fraction as a percentage."
    )

    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    win_rate_source: Mapped[str] = mapped_column(
        String(32), nullable=False, doc="'empirical' | 'default_insufficient_data'."
    )
    win_rate_samples: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    payoff_ratio: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    scenario_source: Mapped[str | None] = mapped_column(
        String(32), nullable=True, doc="'monte_carlo' | 'realized_vol_analytic'."
    )

    scenario_probs: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, doc="{prob_bull, prob_base, prob_bear, returns...}."
    )
    risk_factors: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="Every overlay applied, in order, with the multiplier each cost.",
    )

    __table_args__ = (
        Index("ix_position_recommendations_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<PositionRecommendation {self.symbol} {self.timestamp.date()} "
            f"{self.action} size={self.position_size_pct}%>"
        )


class Forecast(Base, _IngestedAtMixin):
    """Monte Carlo block-bootstrap price forecast (Phase 6).

    One row per (forecast_date, horizon) — re-running the engine on the same
    day UPDATES the row (upsert on the unique constraint). Deliberately a
    REGULAR table, not a hypertable: volume is tiny (5 rows/day) and the
    surrogate ``id`` PK is needed as an FK target for evaluations.
    """

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    forecast_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Day the forecast was generated from (last candle used, UTC).",
    )
    target_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="forecast_date + horizon_days (UTC midnight).",
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_price_p5: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    predicted_price_p50: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    predicted_price_p95: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    prob_price_up: Mapped[float] = mapped_column(
        Float, nullable=False, doc="P(final price > base price), 0..1."
    )
    base_price: Mapped[Decimal] = mapped_column(
        Numeric(24, 10),
        nullable=False,
        doc="Close at forecast_date — anchor for direction evaluation.",
    )
    ai_decision_snapshot: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, doc="AI decision at/most recent before forecast_date."
    )

    __table_args__ = (
        UniqueConstraint("forecast_date", "horizon_days", name="uq_forecast_date_horizon"),
        Index("ix_forecasts_target_date", "target_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Forecast {self.forecast_date.date()} +{self.horizon_days}d "
            f"p50={self.predicted_price_p50}>"
        )


class ForecastEvaluation(Base):
    """Realized outcome of an expired forecast (Phase 6 continuous learning)."""

    __tablename__ = "forecast_evaluations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    forecast_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("forecasts.id"), nullable=False
    )
    target_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_price: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    price_error_pct: Mapped[Decimal] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        doc="(actual - p50) / p50 * 100.",
    )
    directional_correct: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Predicted direction (prob_price_up vs 0.5) matched actual vs base_price.",
    )
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("forecast_id", name="uq_evaluation_per_forecast"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ForecastEvaluation forecast_id={self.forecast_id} "
            f"err={self.price_error_pct}% correct={self.directional_correct}>"
        )


class LearningMetric(Base):
    """Continuous-learning calibration metrics per horizon (Phase 7).

    One row per (horizon, run). The ForecastEngine reads the LATEST row per
    horizon to self-correct future forecasts. Regular table (tiny volume).

    Units (documented explicitly to avoid ambiguity):
    * ``rolling_mape``    — mean absolute % error over the trailing window
                            (percent, e.g. 4.2 = 4.2%).
    * ``magnitude_bias``  — mean of (actual - p50) / p50 as a FRACTION
                            (e.g. +0.05 = actual came in 5% above p50, i.e.
                            the model predicted too LOW / too pessimistic;
                            negative = predicted too high). This is the
                            mathematically-consistent Task-2 definition.
    * ``coverage_rate``   — FRACTION of actuals within [p5, p95] (0..1;
                            ideal ~0.90). < 0.85 = overconfident (bands too
                            narrow); > 0.95 = underconfident (too wide).
    * ``brier_score``     — mean((prob_up - outcome)^2), 0..1, lower = better
                            probability calibration.
    * ``sample_size``     — number of evaluations the metrics were computed
                            from (needed to interpret the others).
    """

    __tablename__ = "learning_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rolling_mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    magnitude_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_learning_metrics_horizon_ts", "horizon_days", "evaluated_at"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LearningMetric +{self.horizon_days}d n={self.sample_size} "
            f"bias={self.magnitude_bias} cover={self.coverage_rate}>"
        )


# ===========================================================================
# Phase 8 Part 8 — BTC intraday (4H) layer
#
# Scope discipline: BITCOIN ONLY. Nothing here touches the 20 altcoins, and
# nothing here touches the daily tables above. The daily path must keep
# working byte-for-byte when every table in this section is empty.
#
# Why ``horizon_bars`` and not ``horizon_days``
# --------------------------------------------
# The daily engine's horizons (7/30/60/90/180) are whole days. The intraday
# horizons are 4h / 12h / 24h — fractions of a day. Reusing the existing
# ``horizon_days`` column would force either fractional days or a silent
# unit change in a column five other call sites already read as days.
# A separate table with an explicitly-named unit makes that impossible.
# ===========================================================================
class Btc4hFeature(Base, _IngestedAtMixin):
    """Deterministic technical features on ONE closed 4-hour BTC candle.

    Separate table from ``daily_features`` because that table is one row per
    DAY per symbol; this is one row per 4-HOUR BAR. Same reasoning that keeps
    ``crypto_prices`` timeframe-keyed rather than one table per timeframe.

    **Only CLOSED bars are ever written here** (user decision 2026-07-29,
    restated four times in the spec). A 4h bar opening at ``timestamp`` is
    still forming until ``timestamp + 4h``; its high/low are "so far", not
    final. Writing a forming bar would repeat the bug class already fixed
    twice in this project (evaluator 2026-07-25, reconciliation gate Part 1).
    The guarantee is enforced at two layers — the fetcher drops the forming
    bar before saving, and :mod:`src.features.intraday_engine` re-checks.

    ``close`` is denormalised from ``crypto_prices`` on purpose: every reader
    of this table (prompt builder, SQL dashboards) needs the price next to
    the indicator, and re-joining 6 rows/day to fetch one column is friction
    for zero benefit. It is written in the same transaction as the features,
    from the same source row, so it cannot drift.
    """

    __tablename__ = "btc_4h_features"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        doc="Bar OPEN time (UTC), always on a 00/04/08/12/16/20 boundary.",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        server_default=text(f"'{BASE_SYMBOL}'"),
        doc="Always BTC/USDT today; part of the PK so the table can widen "
        "without a re-key (the lesson from daily_features and decisions).",
    )

    close: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    volume: Mapped[Decimal | None] = mapped_column(Numeric(30, 10), nullable=True)

    rsi_14: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        doc="RSI(14) over 4h bars — spans ~2.3 days, NOT comparable to the "
        "daily RSI(14) in daily_features.",
    )
    macd: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    macd_signal: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    macd_hist: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    volume_ma_30: Mapped[Decimal | None] = mapped_column(
        Numeric(30, 10),
        nullable=True,
        doc="Mean volume over the trailing 30 CLOSED 4h bars (~5 days).",
    )

    __table_args__ = (Index("ix_btc_4h_features_symbol_ts", "symbol", "timestamp"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Btc4hFeature {self.symbol} {self.timestamp:%Y-%m-%d %H:%M} "
            f"close={self.close} rsi={self.rsi_14}>"
        )


class IntradayForecast(Base, _IngestedAtMixin):
    """Monte Carlo forecast over 4-hour bars (Phase 8 Part 8).

    Regular table, not a hypertable — same reasoning as ``forecasts``: tiny
    row volume (3 rows/day) and a surrogate ``id`` is needed as an FK target.

    Anchoring, and why it is bar-aligned
    ------------------------------------
    ``base_at`` is the OPEN time of the last CLOSED 4h bar and ``base_price``
    is that bar's close. ``target_at = base_at + horizon_bars * 4h``, which
    is therefore also exactly on a bar boundary.

    That alignment is what makes evaluation exact: the realized value is one
    ``crypto_prices`` lookup of the ``4h`` candle at ``target_at``, with no
    interpolation and no "closest available" fallback. Anchoring to a live
    mid-price instead would leave ``target_at`` between bars and force
    exactly the kind of approximate comparison that produced the 2026-07-25
    evaluator bug.

    Note ``horizon_bars=1`` targets the bar that is still forming at forecast
    time — i.e. genuinely "the next 4 hours", which is the near-term horizon
    this layer exists to serve.
    """

    __tablename__ = "intraday_forecasts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text(f"'{BASE_SYMBOL}'")
    )
    base_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="OPEN time of the last CLOSED 4h bar the forecast is anchored to.",
    )
    target_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="base_at + horizon_bars * 4h — always on a bar boundary.",
    )
    horizon_bars: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Horizon in 4-HOUR BARS (1=4h, 3=12h, 6=24h). Never days.",
    )

    predicted_price_p5: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    predicted_price_p50: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    predicted_price_p95: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    prob_price_up: Mapped[float] = mapped_column(
        Float, nullable=False, doc="P(final price > base_price), 0..1."
    )
    base_price: Mapped[Decimal] = mapped_column(
        Numeric(24, 10),
        nullable=False,
        doc="Close of the bar at base_at — anchor for direction evaluation.",
    )

    #: Bars actually available in the bootstrap pool. Persisted because a
    #: forecast built on 300 bars is not the same claim as one built on 2190,
    #: and months later the pool size is otherwise unrecoverable.
    pool_bars: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "symbol", "base_at", "horizon_bars", name="uq_intraday_forecast_bar_horizon"
        ),
        Index("ix_intraday_forecasts_target_at", "target_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IntradayForecast {self.symbol} {self.base_at:%m-%d %H:%M} "
            f"+{self.horizon_bars}bar p50={self.predicted_price_p50}>"
        )


class IntradayForecastEvaluation(Base):
    """Realized outcome of a matured 4H forecast (Phase 8 Part 8).

    A target bar opening at ``target_at`` has closed once
    ``target_at + 4h <= now``, i.e. ``target_at < floor_4h(now)`` — the SAME
    predicate the fetcher uses to drop forming bars. One helper, one
    definition of "closed", per the single-source-of-truth lesson from
    ``FINALIZATION_BUFFER_DAYS``.
    """

    __tablename__ = "intraday_forecast_evaluations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    forecast_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("intraday_forecasts.id"), nullable=False
    )
    target_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_price: Mapped[Decimal] = mapped_column(Numeric(24, 10), nullable=False)
    price_error_pct: Mapped[Decimal] = mapped_column(
        Numeric(18, 8), nullable=False, doc="(actual - p50) / p50 * 100."
    )
    directional_correct: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Predicted direction (prob_price_up vs 0.5) matched actual vs base_price.",
    )
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("forecast_id", name="uq_intraday_evaluation_per_forecast"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IntradayForecastEvaluation fid={self.forecast_id} "
            f"err={self.price_error_pct}% correct={self.directional_correct}>"
        )


class IntradayLearningMetric(Base):
    """Calibration metrics per 4H horizon (Phase 8 Part 8).

    Units are identical to :class:`LearningMetric` (see its docstring) so the
    two scoreboards can be read side by side: ``rolling_mape`` in percent,
    ``magnitude_bias`` as a fraction where >0 means the model predicted too
    LOW, ``coverage_rate`` as a fraction with ~0.90 ideal, ``brier_score``
    lower-is-better.

    Kept in its own table rather than adding a nullable ``horizon_bars`` to
    ``learning_metrics``: the daily ForecastEngine reads that table to
    self-correct, and a row whose horizon means bars instead of days would
    silently mis-scale the daily bias correction.
    """

    __tablename__ = "intraday_learning_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    horizon_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rolling_mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    magnitude_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_intraday_learning_horizon_ts", "horizon_bars", "evaluated_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IntradayLearningMetric +{self.horizon_bars}bar "
            f"n={self.sample_size} bias={self.magnitude_bias}>"
        )


# ===========================================================================
# Phase 9 — Decision Accuracy & Cross-Asset Learning
#
# The gap this closes: every layer built up to Phase 8 converges on ONE
# output (BUY/HOLD/SELL), and nothing measured whether that output was right.
# ``forecast_evaluations`` scores a DIFFERENT thing — the Monte Carlo P50
# price — and only for Bitcoin. Twenty assets had no evaluation of any kind.
#
# Unlike forecasting, nothing here is BTC-locked: ``decisions`` has carried a
# ``symbol`` since Phase 8 Part 6 and ``crypto_prices`` always has, so this
# layer covers all 21 assets from day one.
# ===========================================================================
class DecisionVerdict(StrEnum):
    """Outcome of one decision at one horizon.

    ``NEUTRAL`` is not a failure mode — it means the price stayed INSIDE the
    asset's volatility band, so a directional call (BUY/SELL) was neither
    vindicated nor refuted. Counting those as wrong would punish the system
    for markets that simply did not move. They are excluded from hit-rate on
    BOTH sides of the ratio, exactly as ``panel_agreement`` excludes pillars
    scoring zero.

    HOLD is the mirror image: for HOLD, staying inside the band IS correct,
    so a HOLD never produces NEUTRAL.
    """

    CORRECT = "CORRECT"
    WRONG = "WRONG"
    NEUTRAL = "NEUTRAL"


_DECISION_VERDICT_ENUM = SAEnum(
    DecisionVerdict,
    name="decision_verdict",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class DecisionEvaluation(Base):
    """Did this BUY/HOLD/SELL turn out to be right? (Phase 9)

    One row per (decision, horizon). A single decision is scored at THREE
    horizons — 1, 7 and 30 days (user decision 2026-07-31) — because a BUY
    can be wrong tomorrow and right in a month, and which horizon the system
    actually has edge at is precisely the thing worth measuring.

    Correctness is a VOLATILITY-SCALED BAND, not raw direction
    ------------------------------------------------------------
    ``band_pct = realized_vol_30d * sqrt(horizon_days / 365)``.

    ``realized_vol_30d`` is annualised (same convention as ``trend_strength``
    in Part 4), so the square-root-of-time factor puts it on the horizon's
    own scale. Then::

        BUY  -> CORRECT if return >  +band, WRONG if return <  -band
        SELL -> CORRECT if return <  -band, WRONG if return >  +band
        BUY/SELL inside the band          -> NEUTRAL (not counted)
        HOLD -> CORRECT if |return| <= band, else WRONG

    A FIXED band (say 2%) was rejected deliberately: across 21 assets whose
    volatility differs by an order of magnitude, a fixed threshold makes
    high-volatility assets look more "accurate" purely because they move
    more. The band has to belong to the asset.

    Three dates, three different jobs (entry lag, 2026-07-31)
    ---------------------------------------------------------
    ``decision_timestamp`` is when the call was made and is where the BAND is
    read. ``entry_date`` is the candle whose close is ``base_price``, and it
    is always strictly later — the decision is computed from day T's candle
    while that candle is still forming, so ``close(T)`` is a price the
    decision partly saw. ``target_date`` is ``entry_date + horizon_days``.

    Before the fix all three collapsed onto ``decision_timestamp + horizon``,
    and the measured return started from a close that on average had not
    happened yet when the decision was written.

    No foreign key to ``decisions``
    -------------------------------
    ``decisions`` is a hypertable keyed ``(timestamp, symbol)``; TimescaleDB
    does not support being an FK target cleanly. The uniqueness that matters
    — one evaluation per decision per horizon — is enforced by the constraint
    below instead. ``action`` and ``confidence_score`` are copied here on
    purpose so an evaluation row is self-contained and a later edit to the
    decision cannot silently rewrite history.
    """

    __tablename__ = "decision_evaluations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    decision_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="Day the decision was made."
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_days: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="1, 7 or 30 — calendar days, not bars."
    )
    entry_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc=(
            "Day whose close is base_price. decision_timestamp + "
            "ENTRY_LAG_DAYS, never decision_timestamp itself — the decision "
            "is built from day T's still-forming candle, so close(T) is a "
            "price it partly saw. Stored rather than derived so a row stays "
            "self-contained if the lag is ever retuned."
        ),
    )
    target_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="entry_date + horizon."
    )

    #: Copied from the decision so this row can be read on its own.
    action: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence_score: Mapped[int] = mapped_column(Integer, nullable=False)

    base_price: Mapped[Decimal] = mapped_column(Numeric(38, 12), nullable=False)
    actual_price: Mapped[Decimal] = mapped_column(Numeric(38, 12), nullable=False)
    return_pct: Mapped[Decimal] = mapped_column(
        Numeric(18, 8), nullable=False, doc="(actual - base) / base * 100."
    )
    band_pct: Mapped[Decimal] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        doc="Volatility band for THIS asset at THIS horizon, in percent.",
    )
    verdict: Mapped[DecisionVerdict] = mapped_column(
        _DECISION_VERDICT_ENUM, nullable=False
    )
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "decision_timestamp",
            "symbol",
            "horizon_days",
            name="uq_decision_evaluation",
        ),
        Index("ix_decision_evaluations_symbol_horizon", "symbol", "horizon_days"),
        Index("ix_decision_evaluations_target", "target_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DecisionEvaluation {self.symbol} {self.decision_timestamp.date()} "
            f"+{self.horizon_days}d {self.action}={self.verdict}>"
        )


class ModelUsage(Base):
    """What each model call actually cost, in tokens and seconds (C3, 2026-08-09).

    Why this table exists
    ---------------------
    The pillar scores are the only judgement in the pipeline, and the
    ablation is being built to measure whether they carry information. That
    is one axis. This is the other: what they cost. Without both, "is this
    model worth it?" has no answer — and the cost half was being thrown away.

    Before C3, ``usage`` was read in ``BaseAgent`` and passed to
    ``logger.debug`` — invisible at the project's INFO level — and
    ``reasoning_tokens`` was never touched anywhere in ``src/``. The number
    existed for a microsecond and was discarded.

    Reasoning is most of the bill
    -----------------------------
    Measured over the first real 21-asset run (2026-08-08): 24,175 of 40,189
    output tokens — **60.2% of the output was reasoning**, and the old log
    line printed only the visible remainder. Time tracks output length, not
    prompt length: on that same run ``corr(completion_tokens, seconds)`` was
    **+0.582** while ``corr(prompt_tokens, seconds)`` was **-0.276**. Bigger
    prompt, FASTER run — BTC carries the largest prompt (4,603 tokens, the
    only one with the forecast and 4H blocks) and finished fastest at 51.4s.

    (A first 41-token probe suggested 83%. It was one tiny sample and it was
    not representative — the same lesson as the wall-clock estimates.)

    ``reasoning_tokens`` is a SUBSET of ``completion_tokens``, not an
    addition to it — the OpenAI convention. Summing them double-counts.

    Regular table with an ``id``, not a hypertable: ~21-30 rows a day is not
    a volume problem, and two calls can share a timestamp, so there is no
    natural composite key.
    """

    __tablename__ = "model_usage"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="When the call completed."
    )
    agent_name: Mapped[str] = mapped_column(String(48), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="Asset the call was about, when the caller supplied it. NULL for "
        "calls with no asset context (probes, ad-hoc use).",
    )

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True, doc="Total output tokens, reasoning INCLUDED."
    )
    reasoning_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Subset of completion_tokens spent on hidden reasoning. Do NOT "
        "add it to completion_tokens — that double-counts.",
    )
    cached_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True, doc="Prompt tokens served from cache."
    )
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 10),
        nullable=True,
        doc="Provider-reported cost. 0 on the free tier — captured anyway so "
        "the column already works the day the tier changes.",
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True, doc="Wall-clock for this single call."
    )

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_model_usage_timestamp", "timestamp"),
        Index("ix_model_usage_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ModelUsage {self.agent_name} {self.symbol or '-'} "
            f"prompt={self.prompt_tokens} completion={self.completion_tokens} "
            f"(reasoning={self.reasoning_tokens})>"
        )


class LiveDerivative(Base):
    """Near-real-time derivatives readings for the EXECUTION path (2026-08-09).

    Deliberately NOT ``derivatives_metrics``, and the two must never be merged
    ---------------------------------------------------------------------------
    ``derivatives_metrics`` is the archive: Binance USDⓈ-M, official, settled,
    internally consistent across years. It is what ``daily_features`` and every
    trained/evaluated number is built on, and its consistency is worth more
    than its freshness.

    But its freshness is genuinely bad, and for a structural reason no schedule
    can fix: **Binance Vision publishes funding rate in MONTHLY archives only**
    — verified 2026-08-08, ``daily/fundingRate/`` returns 404 while
    ``monthly/fundingRate/`` returns 200. The running month has no archive
    until it ends, so funding can legitimately sit ~31 days stale in early
    August. Measured that day: 9 days.

    That is fine for training and unusable for execution. Hence a SEPARATE
    path, per the rule this project has now applied three times (BTC-proxy
    on-chain columns, the unlock fallback, funding_rate_change_1d): **do not
    let a fresh-looking number stand in for one nobody observed.** Keeping
    them in different tables makes the substitution impossible rather than
    merely discouraged.

    The source is a DIFFERENT exchange, and that is recorded
    -------------------------------------------------------
    Binance's own futures API (``fapi``) is ISP-blocked here, as are Deribit,
    Bybit and OKX. Of the exchanges probed via ccxt on 2026-08-09, **Gate.io
    was the only one reachable** that also implements ``fetch_funding_rate``.
    Cross-exchange funding is highly correlated but NOT identical, so
    ``source`` is part of the key and every reader must show it. A Gate.io
    rate presented as Binance's would be a quieter version of the same lie
    this table exists to prevent.

    Age is not optional
    -------------------
    ``observed_at`` is when WE read it, which is the only basis for deciding
    whether it can still be acted on. ``settles_at`` is the exchange's next
    funding settlement, which is a different clock and does not answer the
    freshness question.

    Regular table with history rather than a one-row snapshot: keeping the
    readings costs almost nothing and gives the intraday funding trend that
    the daily archive structurally cannot.
    """

    __tablename__ = "live_derivatives"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="When THIS process read the value. The basis for data age.",
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Exchange the reading came from. NOT interchangeable with the "
        "archive's Binance series — see the class docstring.",
    )
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)

    settles_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Next funding settlement per the exchange. A DIFFERENT clock from "
        "observed_at; it does not tell you how stale the reading is.",
    )
    funding_interval: Mapped[str | None] = mapped_column(
        String(8), nullable=True, doc="e.g. '8h'. Rates are per interval, not per day."
    )
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric(30, 12), nullable=True)
    index_price: Mapped[Decimal | None] = mapped_column(Numeric(30, 12), nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "observed_at", "symbol", "source", "metric",
            name="uq_live_derivative_reading",
        ),
        Index("ix_live_derivatives_lookup", "symbol", "metric", "observed_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LiveDerivative {self.symbol} {self.metric}={self.value} "
            f"@{self.source} observed {self.observed_at:%Y-%m-%d %H:%M}Z>"
        )


class PillarNoiseMeasurement(Base):
    """Run-to-run noise in one pillar's score, from repeated identical calls.

    Why this is a TABLE and not a constant
    --------------------------------------
    The IC attenuation multiplier is derived from this number, and C1 proved
    the multiplier **is not a constant**: the noise floor is irreducible on
    this provider (MoE expert routing under batched inference), but its size
    moves with the model, the temperature, and the provider's own batching.
    A hardcoded figure would keep correcting last month's IC by last month's
    noise, silently, forever. Stored measurements carry their own date, so a
    stale correction can be *seen* to be stale.

    ``temperature`` and ``model`` are not decoration
    ------------------------------------------------
    Noise measured at temperature 0.2 must NEVER be applied to a series
    produced at temperature 0 — they are different quantities. The reader
    (:mod:`src.evaluation.attenuation`) refuses to correct across a
    mismatch, which is only possible because these two columns are stored
    alongside the number.

    One row per (measurement run, pillar). ``n_runs`` is small by nature —
    each run costs one API call per repeat — so ``noise_sd`` is a crude
    estimate and the column is there to let the reader say so.
    """

    __tablename__ = "pillar_noise_measurements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="When the repeats were run."
    )
    pillar: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(
        String(32), nullable=False, doc="Asset the repeats were run against."
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    temperature: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Sampling temperature in force. Corrections across a mismatch "
        "are refused — see the class docstring.",
    )
    seed: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, doc="NULL when seeding was disabled."
    )
    n_runs: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Identical calls behind noise_sd."
    )
    noise_sd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Sample standard deviation of this pillar's score across the "
        "repeats. 0.0 is a real answer (perfectly repeatable).",
    )
    score_range: Mapped[int | None] = mapped_column(
        Integer, nullable=True, doc="max - min across the repeats."
    )
    scores: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, doc="The raw repeat scores, for re-analysis."
    )

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "measured_at", "symbol", "pillar",
            name="uq_pillar_noise_run",
        ),
        Index("ix_pillar_noise_measured_at", "measured_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<PillarNoiseMeasurement {self.pillar} sd={self.noise_sd:.2f} "
            f"n={self.n_runs} temp={self.temperature} @{self.measured_at:%Y-%m-%d}>"
        )


class MeasurementSnapshot(Base):
    """One day's answer to "is this system actually working?" (2026-07-31).

    Three measurement layers landed in one day — the null model, the baseline
    comparator and the pillar ablation — and all three were report-only
    scripts somebody had to remember to run. A number nobody looks at cannot
    change a decision, so ``run_daily`` now writes this row every day.

    The point is the TREND. Any single day's ``edge_vs_null`` on a handful of
    decisive evaluations is noise; the question that matters is whether it is
    drifting toward zero or away from it over months, and that cannot be
    reconstructed later from a log nobody kept.

    Regular table, not a hypertable: three rows a day is not a volume
    problem, and a composite ``(timestamp, horizon_days)`` primary key makes
    the daily upsert idempotent, which a hypertable would only complicate.
    """

    __tablename__ = "measurement_snapshots"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, doc="Day this was computed for."
    )
    horizon_days: Mapped[int] = mapped_column(
        Integer, primary_key=True, doc="1, 7 or 30 — calendar days."
    )

    #: --- the system ---
    system_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    system_decisive: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    system_effective_n: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Overlap-corrected. system_decisive counts rows; this counts "
        "observations, and the two differ by more than an order of magnitude.",
    )

    #: --- what no skill would score on the same band ---
    null_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_vs_null: Mapped[float | None] = mapped_column(
        Float, nullable=True, doc="system_hit_rate - null_hit_rate."
    )

    #: --- what the best dumb strategy scored on the same grid ---
    baseline_best_name: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="Strongest baseline that day. Usually always_hold: HOLD wins ~70% "
        "from band geometry, which is the bar the pipeline has to clear.",
    )
    baseline_best_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_vs_baseline: Mapped[float | None] = mapped_column(
        Float, nullable=True, doc="system_hit_rate - baseline_best_hit_rate."
    )

    #: --- does the LLM's panel carry information at all ---
    ic_panel_mean: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Cross-sectional Information Coefficient of the mean pillar score. "
        "NULL until a day has enough assets with matured returns.",
    )
    ic_n_days: Mapped[int | None] = mapped_column(
        Integer, nullable=True, doc="Days behind the IC — its real sample size."
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MeasurementSnapshot {self.timestamp.date()} +{self.horizon_days}d "
            f"hit={self.system_hit_rate} edge_null={self.edge_vs_null}>"
        )


class DecisionAccuracyMetric(Base):
    """Rolling accuracy of the decision engine per (asset, horizon) — Phase 9.

    The scoreboard :class:`DecisionEvaluation` rows aggregate into, and the
    source of the per-asset win rate the Kelly sizer consumes.

    ``brier_score`` measures something different here than in
    ``learning_metrics``
    --------------------------------------------------------------------
    The forecast Brier scores ``prob_price_up`` against "did it go up". This
    one scores ``confidence_score / 100`` against "was the decision correct".

    That reframing is deliberate and is the more useful question for a
    decision engine: *when this system says confidence 80, is it right about
    80% of the time?* It also works uniformly for HOLD, which has no
    directional probability to calibrate at all — so no action has to be
    dropped from the calibration measure.

    NEUTRAL verdicts are excluded from every metric here: they are the
    absence of a testable claim, not a wrong one.
    """

    __tablename__ = "decision_accuracy_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)

    sample_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Decisive (non-NEUTRAL) evaluation ROWS behind these numbers.",
    )
    effective_sample_size: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Independent-observation equivalent of sample_size after the "
        "overlap correction (2026-07-31). Daily decisions at a 30-day horizon "
        "share 29/30 of their price path, so 90 rows can be ~3 observations. "
        "This — not sample_size — is what the publication gate and every "
        "significance test use.",
    )
    neutral_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        doc="Evaluations excluded as NEUTRAL. Tracked separately because a "
        "high count means the asset barely moved, not that the system failed.",
    )
    hit_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True, doc="Fraction CORRECT among decisive evaluations."
    )
    brier_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, doc="mean((confidence/100 - was_correct)^2)."
    )
    avg_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Mean confidence over the same rows — compare against hit_rate to "
        "see over/under-confidence at a glance.",
    )
    #: Skill-free baseline for the SAME band, asset and horizon (2026-07-31).
    #: band_pct is exactly one sigma, so a HOLD that knows nothing is right
    #: ~70% of the time. Stored beside hit_rate so the two can never be read
    #: apart: the first Phase 9 scoreboard showed 76.5% and 100% and looked
    #: like evidence, when the baselines for those bands were 73.4% and 70.7%.
    null_hit_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Expected hit_rate under ZERO skill, blended over this row's own "
        "BUY/SELL/HOLD mix. Estimated unconditionally from the asset's full "
        "price history, entry lag included.",
    )
    edge_vs_null: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="hit_rate - null_hit_rate. The only one of these numbers that "
        "answers 'is there anything here'. <= 0 means the band explains the "
        "result entirely.",
    )
    null_decisive_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Historical fraction of days the band did NOT swallow (~0.3). How "
        "much of the sample BUY/SELL calls are structurally unable to test.",
    )
    null_samples: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Asset-days behind the baseline. Provenance: a baseline from 400 "
        "days of a new listing is not one from 2,300 days of BTC.",
    )
    #: Per-action breakdown. A 60% headline hit-rate built entirely from HOLD
    #: is a different system than one built from BUY, and the aggregate alone
    #: cannot tell them apart.
    buy_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    buy_correct: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    sell_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    sell_correct: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    hold_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    hold_correct: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    __table_args__ = (
        Index("ix_decision_accuracy_symbol_horizon", "symbol", "horizon_days", "evaluated_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DecisionAccuracyMetric {self.symbol} +{self.horizon_days}d "
            f"n={self.sample_size} hit={self.hit_rate}>"
        )


# ---------------------------------------------------------------------------
# TimescaleDB hypertable registry: (table_name, time_column).
# Consumed by scripts/init_db.py after Base.metadata.create_all().
# NOTE: forecasts, forecast_evaluations & learning_metrics are intentionally
# NOT hypertables — tiny row volume, and they need surrogate id PKs
# (TimescaleDB requires the time column inside every unique constraint).
# Same for the Phase 8 dimension tables coins_registry & tokenomics_events:
# reference data, not time series.
# ---------------------------------------------------------------------------
HYPERTABLES: tuple[tuple[str, str], ...] = (
    ("crypto_prices", "timestamp"),
    ("macro_indicators", "timestamp"),
    ("on_chain_metrics", "timestamp"),
    ("sentiment_data", "timestamp"),
    ("derivatives_metrics", "timestamp"),
    ("market_indices", "timestamp"),
    ("daily_features", "timestamp"),
    ("decisions", "timestamp"),
    ("market_regimes", "timestamp"),
    ("sector_performance", "timestamp"),
    ("supply_metrics", "timestamp"),
    ("position_recommendations", "timestamp"),
    # Phase 8 Part 8: 6 rows/day, genuine time series -> hypertable.
    # intraday_forecasts / _evaluations / _learning_metrics are deliberately
    # NOT here, for the same reason as their daily counterparts: surrogate id
    # PKs, and TimescaleDB requires the time column inside every unique
    # constraint.
    ("btc_4h_features", "timestamp"),
)
