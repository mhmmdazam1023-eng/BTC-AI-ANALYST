"""Coin DNA -> prompt text (Phase 8 Part 5).

The Part 4 features live in ``daily_features`` as numbers and NULLs. A model
reading ``days_to_next_unlock: None`` will either ignore it or, worse,
narrate "unlock in None days". Every value therefore passes through
:func:`format_dna_value`, which turns absence into a sentence that says what
the absence MEANS.

Why the fallbacks differ per field
----------------------------------
NULL does not mean the same thing everywhere, and flattening it to one
phrase would destroy real information:

* ``days_to_next_unlock`` NULL  -> AMBIGUOUS, and the ambiguity is resolved
  by the ``has_tokenomics_data`` provenance flag (C2b, 2026-08-09). With a
  schedule on file, NULL means nothing is scheduled — *bullish* for supply
  pressure. With NO schedule on file, NULL means nobody loaded one, which is
  not news at all. Before C2b both rendered as "tidak ada unlock terjadwal",
  handing 20 of 21 assets an unearned bullish clause.
* ``inflation_rate_7d`` NULL    -> supply collection has not run long
  enough. That is genuinely unknown, and the agent should discount it.
* ``btc_beta_30d`` NULL         -> fewer than 30 days of history.

Conflating the first two would let an agent treat "no unlocks scheduled"
as a data gap and hedge against a risk that is not there.

Language note: these strings are Indonesian because the sub-agents write
their ``reasoning`` in Indonesian, and mirroring the output language keeps
the model from code-switching mid-sentence.
"""

from __future__ import annotations

import math
from typing import Any

from src.db.models import BASE_SYMBOL

#: What a NULL means, per field. Never a generic "N/A".
DNA_FALLBACKS: dict[str, str] = {
    "days_to_next_unlock": "tidak ada unlock terjadwal",
    "supply_shock_score_30d": "tidak ada supply shock terjadwal",
    "inflation_rate_7d": "data suplai belum tersedia",
    "supply_utilization": "data suplai belum tersedia",
    "liquidity_score": "data likuiditas belum tersedia",
    "market_cap_tier": "kapitalisasi belum diketahui",
    "vol_regime": "rezim volatilitas belum diketahui",
    "vol_percentile_30d": "persentil volatilitas belum tersedia",
    "btc_beta_30d": "beta belum bisa dihitung (riwayat < 30 hari)",
    "btc_corr_30d": "korelasi belum bisa dihitung (riwayat < 30 hari)",
    "momentum_score": "skor momentum belum tersedia",
    "trend_strength": "kekuatan tren belum tersedia",
    "sector": "sektor belum terdaftar",
}

#: Human-readable labels for the categorical features.
_VOL_REGIME_LABELS = {"LOW": "RENDAH", "NORMAL": "NORMAL", "HIGH": "TINGGI"}
_MARKET_CAP_LABELS = {"LARGE": "large-cap", "MID": "mid-cap", "SMALL": "small-cap"}


def format_dna_value(
    value: Any,
    fallback_text: str = "tidak tersedia",
    *,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    """Render one DNA value for a prompt, or ``fallback_text`` when absent.

    Absent means ``None`` or NaN. A non-numeric value (the categorical
    features are label strings) is returned as-is, so this is safe to call
    on every field without knowing its type.
    """
    if value is None:
        return fallback_text
    if isinstance(value, str):
        return value or fallback_text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return fallback_text
    return f"{number:.{decimals}f}{suffix}"


def _label(value: Any, mapping: dict[str, str], fallback: str) -> str:
    if value is None:
        return fallback
    return mapping.get(str(value), str(value))


# ---------------------------------------------------------------------------
# Per-agent context blocks. Each returns "" when it has nothing to say, so
# an agent with no usable DNA keeps its original prompt exactly.
# ---------------------------------------------------------------------------
def technical_dna_context(dna: dict[str, Any]) -> str:
    """Volatility regime, trend quality and BTC sensitivity.

    Composed clause by clause rather than by substituting fallback strings
    into fixed slots: a slot like "rezim volatilitas {x}" filled with
    "rezim volatilitas belum diketahui" produces duplicated, broken prose.
    Each clause is either written with its value or replaced wholesale.
    """
    if not dna:
        return ""
    symbol = dna.get("symbol", "Aset ini")
    clauses: list[str] = []

    regime_raw = dna.get("vol_regime")
    percentile_raw = dna.get("vol_percentile_30d")
    if regime_raw is None:
        clauses.append("Rezim volatilitasnya belum bisa ditentukan")
    else:
        regime = _label(regime_raw, _VOL_REGIME_LABELS, "TIDAK DIKETAHUI")
        clause = f"Koin ini berada dalam rezim volatilitas {regime}"
        if percentile_raw is not None:
            clause += (
                f" (persentil {format_dna_value(percentile_raw, '', decimals=0)}"
                f" terhadap satu tahun terakhir)"
            )
        clauses.append(clause)

    beta_raw = dna.get("btc_beta_30d")
    corr_raw = dna.get("btc_corr_30d")
    if beta_raw is None and corr_raw is None:
        clauses.append(
            "Beta dan korelasi terhadap BTC belum bisa dihitung "
            "(riwayat kurang dari 30 hari)"
        )
    else:
        beta = format_dna_value(beta_raw, "belum tersedia", decimals=2)
        corr = format_dna_value(corr_raw, "belum tersedia", decimals=2)
        clauses.append(
            f"Beta terhadap BTC sebesar {beta} dengan korelasi harga {corr}"
        )

    trend_raw = dna.get("trend_strength")
    if trend_raw is not None:
        clauses.append(
            f"Rasio signal-to-noise tren 30 hari "
            f"{format_dna_value(trend_raw, '', decimals=2)}"
        )

    text = f"KARAKTERISTIK KOIN ({symbol}): " + ". ".join(clauses) + "."
    if _is_base(dna):
        text += (
            " CATATAN: aset ini ADALAH acuan pasar, sehingga Beta 1,00 dan "
            "korelasi 1,00 adalah identitas matematis, bukan temuan analitis."
        )
    else:
        text += (
            " Beta di atas 1 berarti pergerakan koin ini memperbesar "
            "pergerakan BTC; korelasi rendah berarti ia punya penggerak "
            "sendiri di luar BTC. Sesuaikan tingkat keyakinan sinyal teknikal "
            "Anda dengan konteks ini."
        )
    return text


def liquidity_dna_context(dna: dict[str, Any]) -> str:
    """Market-cap tier, turnover and scheduled supply pressure."""
    if not dna:
        return ""
    symbol = dna.get("symbol", "Aset ini")
    clauses: list[str] = []

    tier_raw = dna.get("market_cap_tier")
    liquidity_raw = dna.get("liquidity_score")
    if tier_raw is None:
        clauses.append("Kapitalisasi pasarnya belum diketahui")
    else:
        tier = _label(tier_raw, _MARKET_CAP_LABELS, "tidak diketahui")
        clauses.append(f"Koin ini berkapitalisasi {tier}")
    if liquidity_raw is None:
        clauses.append("Skor likuiditas belum tersedia (data suplai belum ada)")
    else:
        clauses.append(
            f"Skor likuiditas (turnover harian dibagi suplai beredar) "
            f"{format_dna_value(liquidity_raw, '', decimals=5)}"
        )

    shock_raw = dna.get("supply_shock_score_30d")
    unlock_days = dna.get("days_to_next_unlock")
    if unlock_days is None:
        # C2b (2026-08-09). A NULL here has TWO meanings and only one of
        # them is good news. `has_tokenomics_data` is the provenance flag
        # that separates them; when it is absent (older callers, tests) we
        # fall back to the historical wording rather than inventing doubt.
        #
        # Why this mattered: 20 of 21 assets have no tokenomics rows at all,
        # yet every one of them was being told "no unlock scheduled" — an
        # unearned bullish nudge in the liquidity pillar, running since
        # 2026-07-28. Absence narrated as information, the same defect the
        # BTC-proxy on-chain columns had.
        if dna.get("has_tokenomics_data") is False:
            clauses.append(
                "Jadwal unlock BELUM DIMUAT untuk koin ini — tidak diketahui, "
                "BUKAN berarti tidak ada unlock. Jangan perlakukan sebagai "
                "kabar baik untuk tekanan suplai"
            )
        else:
            clauses.append("Tidak ada unlock terjadwal dalam 30 hari ke depan")
    else:
        days = format_dna_value(unlock_days, "", decimals=0)
        shock = format_dna_value(shock_raw, "dampak belum diketahui", decimals=1, suffix="%")
        clauses.append(
            f"Unlock berikutnya dalam {days} hari, dengan potensi supply shock "
            f"30 hari ke depan {shock}"
        )

    return (
        f"KARAKTERISTIK KOIN ({symbol}): "
        + ". ".join(clauses)
        + ". Turnover rendah pada aset small-cap memperbesar dampak harga dari "
        "arus dana yang sama besarnya; supply shock terjadwal menambah tekanan "
        "jual terlepas dari posisi derivatif saat ini."
    )


def onchain_dna_context(dna: dict[str, Any]) -> str:
    """Supply utilisation and realised issuance."""
    if not dna:
        return ""
    symbol = dna.get("symbol", "Aset ini")
    utilization_raw = dna.get("supply_utilization")
    inflation_raw = dna.get("inflation_rate_7d")
    clauses: list[str] = []

    if utilization_raw is None:
        clauses.append("Data utilisasi suplai belum tersedia")
    else:
        clauses.append(
            f"Utilisasi suplai koin ini "
            f"{format_dna_value(float(utilization_raw) * 100, '', decimals=1, suffix='%')}"
            f" (porsi suplai total yang sudah beredar)"
        )
    if inflation_raw is None:
        clauses.append("Tingkat inflasi suplai 7 hari belum tersedia")
    else:
        clauses.append(
            f"Tingkat inflasi suplai 7 hari "
            f"{format_dna_value(inflation_raw, '', decimals=3, suffix='%')}"
        )

    return (
        f"KARAKTERISTIK KOIN ({symbol}): "
        + ". ".join(clauses)
        + ". Utilisasi rendah berarti masih banyak suplai yang akan masuk ke "
        "pasar; inflasi 7 hari yang tinggi berarti suplai sedang bertambah "
        "saat ini."
    )


def market_tier_context(dna: dict[str, Any]) -> str:
    """Minimal context for Macro & Sentiment: how big is this asset?"""
    if not dna:
        return ""
    symbol = dna.get("symbol", "Aset ini")
    tier_raw = dna.get("market_cap_tier")
    sector_raw = dna.get("sector")

    if tier_raw is None and sector_raw is None:
        opening = "Kapitalisasi dan sektor aset ini belum terdaftar"
    elif tier_raw is None:
        opening = f"Aset di sektor {sector_raw}, kapitalisasi belum diketahui"
    elif sector_raw is None:
        tier = _label(tier_raw, _MARKET_CAP_LABELS, "tidak diketahui")
        opening = f"Aset {tier}, sektor belum terdaftar"
    else:
        tier = _label(tier_raw, _MARKET_CAP_LABELS, "tidak diketahui")
        opening = f"Aset {tier} di sektor {sector_raw}"

    return (
        f"KARAKTERISTIK KOIN ({symbol}): {opening}. Aset small-cap dan sektor "
        f"spekulatif biasanya bereaksi lebih keras terhadap perubahan kondisi "
        f"makro dan sentimen dibanding aset large-cap; pertimbangkan itu saat "
        f"menilai besarnya dampak."
    )


# ---------------------------------------------------------------------------
def _is_base(dna: dict[str, Any]) -> bool:
    return str(dna.get("symbol", "")) == BASE_SYMBOL


def format_dna_summary(symbol: str, dna: dict[str, Any] | None) -> str:
    """One- to two-sentence narrative for the Master Agent.

    Example::

        TAO/USDT (sektor AI, mid-cap): rezim volatilitas TINGGI, Beta 1,20
        terhadap BTC (korelasi 0,76). Tidak ada unlock terjadwal dalam 30
        hari ke depan. Utilisasi suplai 45,7%.
    """
    if not dna:
        return f"{symbol}: profil Coin DNA belum tersedia."

    sector = dna.get("sector") or DNA_FALLBACKS["sector"]
    tier = _label(
        dna.get("market_cap_tier"), _MARKET_CAP_LABELS, DNA_FALLBACKS["market_cap_tier"]
    )
    regime = _label(
        dna.get("vol_regime"), _VOL_REGIME_LABELS, DNA_FALLBACKS["vol_regime"]
    )
    beta = format_dna_value(
        dna.get("btc_beta_30d"), DNA_FALLBACKS["btc_beta_30d"], decimals=2
    )
    corr = format_dna_value(
        dna.get("btc_corr_30d"), DNA_FALLBACKS["btc_corr_30d"], decimals=2
    )

    parts = [
        f"{symbol} (sektor {sector}, {tier}): rezim volatilitas {regime}, "
        f"Beta {beta} terhadap BTC (korelasi {corr})."
    ]

    unlock_days = dna.get("days_to_next_unlock")
    shock = dna.get("supply_shock_score_30d")
    if unlock_days is None:
        parts.append("Tidak ada unlock terjadwal dalam 30 hari ke depan.")
    else:
        days = format_dna_value(unlock_days, "", decimals=0)
        shock_text = format_dna_value(shock, "dampak tidak diketahui", decimals=1, suffix="%")
        parts.append(
            f"Unlock berikutnya dalam {days} hari; total supply shock 30 hari "
            f"ke depan {shock_text}."
        )

    utilization = dna.get("supply_utilization")
    if utilization is not None:
        parts.append(
            f"Utilisasi suplai "
            f"{format_dna_value(float(utilization) * 100, '', decimals=1, suffix='%')}."
        )
    else:
        parts.append("Data suplai belum tersedia.")

    if _is_base(dna):
        parts.append(
            "(Aset ini adalah acuan pasar: Beta dan korelasi 1,00 bersifat "
            "identitas.)"
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Phase 8 Part 8 — BTC intraday (4H) context
# ---------------------------------------------------------------------------
def format_intraday_context(bars: list[Any], *, live_close: float | None = None) -> str:
    """Prompt block summarising the last CLOSED 4-hour Bitcoin bars.

    ``bars`` are :class:`~src.features.intraday_engine.IntradayBar` instances,
    oldest first. Returns ``""`` when there are none, so the caller can append
    unconditionally and the prompt stays byte-identical without 4H data.

    Three things this block states explicitly, each for a reason:

    * **"CLOSED"** and the up-to-4h lag — otherwise the model reads the newest
      bar as "now" and treats a 4-hour-old close as the current price.
    * **A full date on every bar label.** A rolling window of the last N closed
      bars spans two calendar days for most of the day, so ``04:00-08:00``
      alone would make two different bars indistinguishable.
    * **That 4H is subordinate to the daily read.** The intraday tape is
      noisier; without this the model is invited to flip a daily thesis on one
      red 4-hour candle. The deterministic composer enforces this too — see
      ``decision_composer.apply_intraday_overlay`` — but the prompt should not
      contradict the arithmetic.
    """
    if not bars:
        return ""

    lines = [
        f"BITCOIN 4-HOUR TIMEFRAME — {len(bars)} last CLOSED bar(s), oldest "
        "first. These are FINAL bars: the bar forming right now is deliberately "
        "excluded, so this view lags real time by up to 4 hours.",
    ]
    for bar in bars:
        rsi = format_dna_value(bar.rsi_14, "n/a", decimals=2)
        close = format_dna_value(bar.close, "n/a", decimals=2)
        lines.append(
            f"- [{bar.label}] close={close}  RSI={rsi}  "
            f"MACD={bar.macd_state}  Volume={bar.volume_state}"
        )

    if live_close is not None:
        lines.append(
            f"Latest daily close for reference: "
            f"{format_dna_value(live_close, 'n/a', decimals=2)} — compare it "
            f"against the last 4H close above to judge what has happened since "
            f"that bar closed."
        )

    lines.append(
        "HOW TO USE THIS: the 4-hour tape sets TIMING, not direction. The "
        "daily pillars own the directional thesis; treat 4H agreement as "
        "confirmation and 4H disagreement as a reason for caution or patience, "
        "NEVER as a reason to invert a daily read. A single 4-hour bar is noise."
    )
    return "\n".join(lines)
