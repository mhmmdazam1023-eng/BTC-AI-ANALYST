"""How much do the pillar scores move between identical runs?

Calls the SAME asset twice with the SAME inputs and diffs the five pillar
scores.

**`decisions` is never written** — that is the table the ablation reads, and
re-running inference through the normal path would upsert over it. Be aware
that assembling the prompt context is NOT side-effect free: ``resolve_regime``
and ``resolve_sectors`` upsert the day's regime and sector rows as a
by-product of computing them. Those are idempotent writes of the same day's
values, so nothing is corrupted, but "nothing is persisted" would be a false
claim and this note exists so nobody repeats it.

Why this exists
---------------
The pillar scores are the only non-deterministic input in the whole pipeline;
everything downstream is arithmetic over them. ``temperature=0.2`` means two
identical runs can disagree, and ``CLAUDE.md`` notes run-to-run variance "when
the panel is split" — but nobody ever measured HOW MUCH.

That number decides a real question. The Information Coefficient the ablation
reports is computed over stored pillar scores; if those scores carry sampling
noise, the IC is attenuated toward zero and a null result becomes ambiguous —
"no signal" and "signal destroyed by model noise" look identical. Sampling
noise can only shrink an IC, never manufacture one, so the asymmetry matters:
a positive IC survives this, a zero IC does not interpret.

Reading the result — the verdict depends on the ACTIVE temperature
-------------------------------------------------------------------
* **drift 0** — deterministic. The ablation sample can be trusted as-is.
* **drift > 0 while temperature > 0** — sampling noise, and it is fixable:
  set ``temperature=0`` (+ ``seed``) and re-run before concluding anything.
* **drift > 0 while temperature == 0** — the important case, and the one
  this project actually landed in. At temperature 0 the sampler is argmax,
  deterministic BY DEFINITION, so the residual cannot be sampling. It comes
  from above the sampler: MoE expert routing under batched inference, plus
  floating-point non-associativity in batched GEMMs. No client-side
  parameter reaches it. Measured 2026-08-09 with ``seed=20260808`` and
  ``temperature=0.0``: still 0/5 pillars identical, max drift 15.

So this script's job changed. It began as "does temperature matter?" and is
now "what is the irreducible noise floor, and how much is it attenuating the
IC?" — a number to re-measure periodically, because it is not a constant.

Usage::

    python scripts/test_determinism.py
    python scripts/test_determinism.py --symbol SOL/USDT --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402

from run_inference import (  # noqa: E402
    INTRADAY_CONTEXT_BARS,
    _as_float,
    load_coin_dna,
    load_forecast_context,
    load_intraday_forecast_context,
    load_latest_features,
    resolve_regime,
    resolve_sectors,
)

from src.agents.consolidated_analyst import ConsolidatedAnalyst  # noqa: E402
from src.agents.utils import format_intraday_context  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import BASE_SYMBOL  # noqa: E402
from src.evaluation.attenuation import (  # noqa: E402
    NoiseMeasurement,
    attenuation_from_spread,
    save_noise_measurements,
    within_day_spread_from_decisions,
)
from src.evaluation.pillar_ablation import MIN_ASSETS_PER_DAY  # noqa: E402
from src.features.intraday_engine import IntradayFeatureEngine  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_determinism")

PILLARS = ("technical", "macro", "onchain", "sentiment", "liquidity_risk")


def _scores(analysis) -> dict[str, int]:
    """The five pillar scores as a plain dict, via the canonical names."""
    return {
        "technical": analysis.technical_score,
        "macro": analysis.macro_score,
        "onchain": analysis.onchain_score,
        "sentiment": analysis.sentiment_score,
        "liquidity_risk": analysis.liquidity_risk_score,
    }


async def _report_attenuation(noise_std: dict[str, float], runs: int) -> None:
    """How much is the measured IC being shrunk by this noise?

    Classical attenuation: if an observed score is the true score plus
    independent noise, then

        corr(observed, outcome) = corr(true, outcome) * sqrt(reliability)
        reliability = var_true / (var_true + var_noise)

    ``var_noise`` comes from the repeated calls above. The denominator is the
    WITHIN-DAY spread of that pillar's scores ACROSS ASSETS.

    **The denominator choice is the whole calculation, and the obvious one is
    wrong.** The first version of this function pooled every stored score and
    got a reassuring 0.949 average. That number was meaningless: the pooled
    spread is dominated by stable cross-asset differences (BTC and PEPE simply
    score differently, every day), which the cross-sectional IC never sees.
    The IC ranks assets WITHIN one day, so the variance it competes against is
    the within-day one — measurably tighter, and the honest average dropped to
    0.690. Reliability must be computed against the same population the
    correlation is measured over.

    The output is the number that matters: **the factor your measured IC is
    being multiplied by**. An IC of 0.03 under an attenuation of 0.5 is really
    0.06 — and one measured at 0.00 stays 0.00 no matter what, which is why a
    null result cannot be interpreted while this factor is well below 1.

    A pillar whose noise EXCEEDS its within-day spread has a reliability of
    zero: on any given day two coins differ less than one coin differs from
    itself. Its cross-sectional ranking is not weak information, it is none.
    """
    async with session_scope() as session:
        spreads = await within_day_spread_from_decisions(
            session, min_assets=MIN_ASSETS_PER_DAY
        )

    settings = get_settings()
    measured_at = datetime.now(tz=UTC)

    print()
    print("=" * 72)
    print("  ATENUASI IC — seberapa besar noise ini mengecilkan IC terukur")
    print("=" * 72)
    print(
        f"  {'pilar':<16}{'sd noise':>10}{'sd dalam-hari':>15}"
        f"{'reliabilitas':>14}{'pengali IC':>12}"
    )
    print("  " + "-" * 63)
    factors, total_days = [], 0
    for pillar in PILLARS:
        observed_sd, n_days = spreads.get(pillar, (None, 0))
        total_days += n_days
        att = attenuation_from_spread(
            signal=pillar,
            observed_sd=observed_sd,
            noise=NoiseMeasurement(
                pillar=pillar,
                noise_sd=noise_std.get(pillar, 0.0),
                n_runs=runs,
                temperature=settings.qwen_temperature,
                model=settings.qwen_model,
                measured_at=measured_at,
                symbol=BASE_SYMBOL,
            ),
            # Same run, same settings — a mismatch is impossible here by
            # construction, and saying so beats leaving it implicit.
            series_temperature=settings.qwen_temperature,
            n_days=n_days,
        )
        if att.multiplier is None:
            print(
                f"  {pillar:<16}{att.noise_sd or 0.0:>10.2f}"
                f"{'n/a':>13}{'n/a':>14}{'n/a':>12}   {att.reason}"
            )
            continue
        factors.append(att.multiplier)
        mark = ("  <-- NOL: ranking = noise murni" if att.multiplier == 0.0
                else "  <-- parah" if att.severity == "severe"
                else "  <- perhatian" if att.severity == "notable" else "")
        print(
            f"  {pillar:<16}{att.noise_sd:>10.2f}{att.observed_sd:>13.2f}"
            f"{att.reliability:>14.3f}{att.multiplier:>12.3f}{mark}"
        )
    print("  " + "-" * 63)
    if factors:
        worst, mean = min(factors), sum(factors) / len(factors)
        print(
            f"  Pengali IC rata-rata {mean:.3f} (terburuk {worst:.3f}) atas "
            f"{total_days} hari-pilar dengan >={MIN_ASSETS_PER_DAY} aset."
        )
        if mean > 0:
            print(
                f"  Artinya: IC 0,03 yang Anda ukur sebenarnya ~"
                f"{0.03 / mean:.3f} kalau noise ini dihilangkan."
            )
    print()
    print("  BATASAN: var_noise diestimasi dari SATU aset di SATU hari dengan")
    print(f"  {runs} panggilan; noise bisa berbeda per aset dan per hari. Ini")
    print("  indikasi besaran, bukan konstanta kalibrasi.")
    print("  PENYEBUTNYA di sini adalah SELURUH decisions, bukan frame satu")
    print("  horizon — run_pillar_ablation.py memakai frame horizonnya sendiri,")
    print("  yang lebih tepat sasaran. Aritmetikanya sama persis (satu modul).")


async def _persist(
    results: list[dict[str, int]],
    measured_at,
    symbol: str,
    *,
    quiet: bool,
) -> None:
    """Store what has been measured so far. NEVER raises.

    Called after every completed run, not just at the end: a measurement that
    only becomes durable once everything succeeds is a measurement you lose
    to the first interruption — and each one here costs an API call off a
    50/day budget.
    """
    settings = get_settings()
    try:
        async with session_scope() as session:
            saved = await save_noise_measurements(
                session,
                measured_at=measured_at,
                symbol=symbol,
                model=settings.qwen_model,
                temperature=settings.qwen_temperature,
                seed=settings.qwen_seed,
                per_pillar_scores={
                    pillar: [r[pillar] for r in results] for pillar in PILLARS
                },
            )
    except Exception as exc:  # noqa: BLE001 - measuring beats storing
        logger.warning(
            "Could not persist the noise measurement (%s: %s) — the printed "
            "numbers are still valid", type(exc).__name__, exc,
        )
        return
    if quiet:
        logger.info("  (persisted %d pillar row(s) from %d run(s) so far)",
                    saved, len(results))
        return
    print()
    print(
        f"  Tersimpan: {saved} baris ke 'pillar_noise_measurements' "
        f"(n_runs={len(results)}, temp={settings.qwen_temperature}, "
        f"{measured_at:%Y-%m-%d %H:%M} UTC)."
    )
    print(
        "  run_pillar_ablation.py akan otomatis memakainya untuk melaporkan "
        "IC mentah DAN terkoreksi."
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=BASE_SYMBOL)
    parser.add_argument(
        "--runs", type=int, default=2,
        help="identical calls to make (default 2; each costs 1 request)",
    )
    args = parser.parse_args()
    symbol = args.symbol
    # Stamped ONCE, before any call: it is the identity of this measurement
    # run, and the unique key the persisted rows share.
    started_at = datetime.now(tz=UTC)

    try:
        # ---- assemble EXACTLY the context run_inference would build -------
        # Anything different here and the test measures prompt drift rather
        # than sampling noise.
        analysis_date, features = await load_latest_features(symbol)
        coin_dna = await load_coin_dna(symbol, analysis_date)
        regime = await resolve_regime()
        sectors = await resolve_sectors()

        is_btc = symbol == BASE_SYMBOL
        forecast_context = await load_forecast_context(analysis_date) if is_btc else None
        intraday_forecast = await load_intraday_forecast_context() if is_btc else None
        intraday_context = None
        if is_btc:
            try:
                bars = await IntradayFeatureEngine().recent_closed_bars(
                    count=INTRADAY_CONTEXT_BARS
                )
                if bars:
                    intraday_context = format_intraday_context(
                        bars, live_close=_as_float(features.get("asset_close"))
                    )
            except Exception as exc:  # noqa: BLE001 - context is optional
                logger.warning("4H context unavailable (%s) — continuing", exc)

        logger.info(
            "Calling %s %s time(s) with IDENTICAL inputs (analysis date %s). "
            "No DECISION is written; the noise measurement itself is stored "
            "to 'pillar_noise_measurements' after each run.",
            symbol, args.runs, analysis_date,
        )

        analyst = ConsolidatedAnalyst()
        results: list[dict[str, int]] = []
        # PERSIST AS WE GO (2026-08-09). The first attempt at this saved only
        # after the whole loop, and the run died during call 5 of 5: four
        # perfectly good measurements — four API calls off a 50/day budget —
        # were left in stdout and never reached the table. Same defect C3 had
        # fixed one layer up, reintroduced here. Each completed run is now
        # durable the moment it exists, and `n_runs` records truthfully how
        # many stand behind the sd.
        for attempt in range(1, args.runs + 1):
            analysis = await analyst.analyze(
                features,
                coin_dna,
                symbol=symbol,
                regime_context=(
                    f"{regime.regime.value} (confidence {regime.confidence}/100)"
                    if regime else None
                ),
                sector_context=sectors.to_prompt_context() if sectors else None,
                forecast_context=forecast_context,
                intraday_context=intraday_context,
                intraday_forecast_context=intraday_forecast,
            )
            scores = _scores(analysis)
            results.append(scores)
            logger.info(
                "  run %d: %s", attempt,
                "  ".join(f"{p}={scores[p]:+d}" for p in PILLARS),
            )
            if len(results) >= 2:
                # Upserts on (measured_at, symbol, pillar), so each pass
                # overwrites the previous partial with a wider sample.
                await _persist(results, started_at, symbol, quiet=True)

        # ---- report ------------------------------------------------------
        print()
        print("=" * 72)
        print(f"  DETERMINISME — {symbol}, {args.runs} panggilan identik")
        print("=" * 72)
        header = "  " + f"{'pilar':<16}" + "".join(f"{'run ' + str(i + 1):>9}" for i in range(args.runs))
        print(header + f"{'rentang':>10}")
        print("  " + "-" * (len(header) + 8))

        drifts = []
        noise_std: dict[str, float] = {}
        for pillar in PILLARS:
            values = [r[pillar] for r in results]
            spread = max(values) - min(values)
            drifts.append(spread)
            noise_std[pillar] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            flag = "" if spread == 0 else ("  <-- bergeser" if spread > 5 else "  <- kecil")
            print(
                "  " + f"{pillar:<16}"
                + "".join(f"{v:>+9d}" for v in values)
                + f"{spread:>10d}{flag}"
            )

        worst = max(drifts)
        identical = sum(1 for d in drifts if d == 0)
        # The verdict depends on whether the fix has ALREADY been applied.
        # Telling someone to "set temperature=0" when it is already 0 is
        # advice that cannot be acted on, and it hides the real finding:
        # that the remaining drift is not sampling at all.
        settings = get_settings()
        greedy = settings.qwen_temperature == 0.0
        print("  " + "-" * (len(header) + 8))
        print(f"  {identical}/{len(PILLARS)} pilar identik · pergeseran maksimum {worst}")
        print(
            f"  Setelan aktif: temperature={settings.qwen_temperature}, "
            f"seed={settings.qwen_seed if settings.qwen_seed is not None else 'MATI'}"
        )
        print()
        if worst == 0:
            print("  VONIS: deterministik. Sampel ablation bisa dipercaya apa adanya.")
        elif not greedy:
            print("  VONIS: skor membawa noise SAMPLING, dan itu bisa diperbaiki.")
            print(f"         temperature masih {settings.qwen_temperature} — set ke 0.0")
            print("         (+ seed) lalu jalankan ulang tes ini sebelum menyimpulkan.")
        else:
            # temperature=0 => the sampler is argmax, deterministic by
            # definition. Whatever still moves is NOT sampling: it is the
            # compute path above it (MoE expert routing under batching,
            # float non-associativity in batched GEMMs). No client-side
            # parameter reaches that.
            print("  VONIS: LANTAI IRREDUCIBLE — bukan noise sampling.")
            print("         temperature=0 berarti sampler sudah argmax (deterministik")
            print("         menurut definisi), jadi sisa pergeseran ini datang dari")
            print("         LAPISAN DI ATAS sampler: routing expert MoE pada inferensi")
            print("         ter-batch + non-asosiativitas float. Tidak ada parameter")
            print("         sisi klien yang memperbaikinya.")
            print("         Konsekuensi: IC akan SELALU teratenuasi di provider ini.")
            print("         Laporkan IC mentah DAN terkoreksi, dan ukur ulang pengali")
            print("         ini berkala — ia bukan konstanta.")
        print()
        print("  Catatan: noise sampling hanya bisa MENGECILKAN IC, tidak pernah")
        print("  mengarang. IC positif selamat dari ini; IC nol tidak bisa")
        print("  dibedakan dari 'sinyal terhapus noise'.")

        # Final pass: same helper, now loud, with the complete sample.
        await _persist(results, started_at, symbol, quiet=False)

        if args.runs >= 3:
            await _report_attenuation(noise_std, args.runs)
        return 0
    except Exception:
        logger.exception("Determinism test FAILED")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))
