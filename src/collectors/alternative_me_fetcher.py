"""Alternative.me Crypto Fear & Greed Index collector (free, no API key).

Fetches the full daily history of the Fear & Greed Index
(``https://api.alternative.me/fng/?limit=0&format=json``, available since
February 2018 — aligning with the macro history window) and maps it to the
``sentiment_data`` hypertable:

* ``value``      — index score 0..100 (0 = Extreme Fear, 100 = Extreme Greed)
* ``value_text`` — the API's classification label (e.g. "Fear", "Greed")

Research grounding: the corpus (Putra & Pimada 2026) shows sentiment indices
carry short-horizon signal for BTC, which is why the full history is stored
rather than only the latest reading.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

import httpx

from src.collectors.base import BaseDataFetcher, FetcherError, RetryableHTTPError
from src.config import Settings
from src.db.models import SentimentData


class AlternativeMeFetcher(BaseDataFetcher):
    """Collects the Crypto Fear & Greed Index from Alternative.me."""

    source_name: ClassVar[str] = "alternative_me"

    API_URL: ClassVar[str] = "https://api.alternative.me/fng/"
    METRIC: ClassVar[str] = "fear_greed_index"

    _FETCH_FAILURES: ClassVar[tuple[type[BaseException], ...]] = (
        RetryableHTTPError,
        httpx.HTTPError,
    )

    def __init__(
        self,
        *,
        limit: int = 0,  # 0 = full history per API convention
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        self.limit = limit

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[SentimentData]:
        """Fetch the index history and return validated ORM rows."""
        self.log.info(
            "[alternative_me] fetching Fear & Greed history (limit=%s)",
            self.limit or "all",
        )
        try:
            payload = await self._request_json(
                self.API_URL,
                params={"limit": str(self.limit), "format": "json"},
            )
        except self._FETCH_FAILURES as exc:
            raise FetcherError(
                f"[alternative_me] request failed after {self.max_attempts} "
                f"attempts: {exc}"
            ) from exc

        return self._map_payload(payload)

    # ------------------------------------------------------------------
    def _map_payload(self, payload: Any) -> list[SentimentData]:
        """Validate the API envelope and map items into ORM rows."""
        if not isinstance(payload, dict):
            raise FetcherError(
                f"[alternative_me] unexpected payload type: {type(payload).__name__}"
            )

        metadata = payload.get("metadata") or {}
        api_error = metadata.get("error") if isinstance(metadata, dict) else None
        if api_error:
            raise FetcherError(f"[alternative_me] API reported an error: {api_error}")

        data = payload.get("data")
        if not isinstance(data, list):
            raise FetcherError(
                f"[alternative_me] missing/invalid 'data' array: {str(payload)[:200]}"
            )
        if not data:
            self.log.warning("[alternative_me] API returned an empty history")
            return []

        rows: list[SentimentData] = []
        skipped = 0
        for item in data:
            try:
                if not isinstance(item, dict):
                    raise ValueError("item is not an object")
                score = Decimal(str(item["value"]))
                if not (0 <= score <= 100):
                    raise ValueError(f"score {score} outside 0..100")
                classification = str(item.get("value_classification") or "").strip()
                if not classification:
                    raise ValueError("missing value_classification")
                timestamp = datetime.fromtimestamp(int(item["timestamp"]), tz=UTC)
                rows.append(
                    SentimentData(
                        timestamp=timestamp,
                        source="ALTERNATIVE_ME",
                        metric=self.METRIC,
                        value=score,
                        value_text=classification,
                        raw_payload=None,  # value + label capture the item fully
                    )
                )
            except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError, OSError) as exc:
                skipped += 1
                self.log.warning(
                    "[alternative_me] skipping malformed item %r: %s", item, exc
                )

        if skipped:
            self.log.warning("[alternative_me] skipped %d malformed item(s)", skipped)
        if not rows:
            self.log.warning("[alternative_me] no usable observations after validation")
            return []

        # The API returns newest-first and may repeat timestamps; a single
        # INSERT may not touch one primary key twice.
        unique: dict[tuple, SentimentData] = {}
        for row in rows:
            unique[(row.timestamp, row.source, row.metric)] = row
        dropped = len(rows) - len(unique)
        if dropped:
            self.log.warning("[alternative_me] dropped %d duplicate item(s)", dropped)

        ordered = sorted(unique.values(), key=lambda r: r.timestamp)
        self.log.info(
            "[alternative_me] fetched %d daily readings (%s -> %s, latest=%s '%s')",
            len(ordered),
            ordered[0].timestamp.date(), ordered[-1].timestamp.date(),
            ordered[-1].value, ordered[-1].value_text,
        )
        return ordered

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``sentiment_data`` (score + label refreshed)."""
        return await self._upsert(
            SentimentData, rows, update_columns=("value", "value_text")
        )
