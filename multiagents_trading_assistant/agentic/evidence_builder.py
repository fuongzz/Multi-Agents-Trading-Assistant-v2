"""Phase 2 EvidenceBuilder.

Assembles a broker-grade EvidencePacket from a screener TradeCandidate plus
optional StrategySignal. Single entry point for both live/deep mode and
backtest replay. Each source is wrapped in fail-safe try/except so a missing
upstream does not kill the pipeline.

Anti-lookahead policy:
    * Parquet sources (smart_money_trace, money_cycle) filter `date <= as_of`.
    * Live-only fetchers (`get_fundamentals`, `get_foreign_flow`,
      `get_global_macro`, `get_stock_news`, SQLite positions) cache by run
      date, so they are skipped entirely in `backtest_mode=True` and the
      packet records which fields were skipped.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.agentic.adapters import _clean_mapping, _clean_obj, _safe_float
from multiagents_trading_assistant.agentic.schemas import EvidencePacket, StrategySignal


_DEFAULT_CACHE_ROOT = Path("multiagents_trading_assistant/cache/agentic_packets")
_SMT_PATH = Path("data/research/smart_money_trace/smart_money_by_symbol.parquet")
_CHDM_PATH = Path("data/research/money_cycle/chdm_by_symbol.parquet")
_DS_PATH = Path("data/research/money_cycle/ds_by_symbol.parquet")
_MARKET_MC_PATH = Path("data/research/money_cycle/money_cycle_market.parquet")

_LIVE_ONLY_FIELDS = ["global_macro", "foreign_flow", "fundamental", "news", "portfolio"]


class EvidenceBuilder:
    """Build a replayable broker-grade EvidencePacket."""

    def __init__(
        self,
        as_of_date: str,
        *,
        backtest_mode: bool = False,
        root: Path | str = ".",
        cache_root: Path | str | None = None,
        write_cache: bool = True,
    ) -> None:
        self.as_of_date = as_of_date
        self.as_of_ts = pd.Timestamp(as_of_date).normalize()
        self.backtest_mode = backtest_mode
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root else (self.root / _DEFAULT_CACHE_ROOT)
        self.write_cache = write_cache

    def build(
        self,
        candidate: Any,
        *,
        signal: StrategySignal | None = None,
    ) -> EvidencePacket:
        symbol = str(getattr(candidate, "symbol", "")).upper()
        missing: list[str] = []
        skipped: list[str] = list(_LIVE_ONLY_FIELDS) if self.backtest_mode else []

        builders = {
            "market": lambda: self._build_market(candidate),
            "sector": lambda: self._build_sector(symbol),
            "technical": lambda: self._build_technical(candidate, symbol),
            "money_flow": lambda: self._build_money_flow(candidate, symbol),
            "edge": lambda: self._build_edge(candidate),
            "fundamental": lambda: self._build_fundamental(symbol),
            "news": lambda: self._build_news(symbol),
            "portfolio": lambda: self._build_portfolio(symbol),
        }

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(fn): name for name, fn in builders.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result() or {}
                except Exception as exc:  # noqa: BLE001
                    results[name] = {}
                    missing.append(f"{name}:{type(exc).__name__}")

        data_quality = self._build_data_quality(
            candidate, signal, missing=missing, skipped=skipped
        )

        packet = EvidencePacket(
            symbol=symbol,
            as_of_date=self.as_of_date,
            market=_clean_mapping(results.get("market", {})),
            sector=_clean_mapping(results.get("sector", {})),
            technical=_clean_mapping(results.get("technical", {})),
            money_flow=_clean_mapping(results.get("money_flow", {})),
            edge=_clean_mapping(results.get("edge", {})),
            fundamental=_clean_mapping(results.get("fundamental", {})),
            news=_clean_mapping(results.get("news", {})),
            portfolio=_clean_mapping(results.get("portfolio", {})),
            data_quality=_clean_mapping(data_quality),
        )

        if self.write_cache:
            self._save_cache(packet)
        return packet

    # ------------------------------------------------------------------ market

    def _build_market(self, candidate: Any) -> dict[str, Any]:
        market_context = _clean_obj(getattr(candidate, "market_context", {})) or {}
        out: dict[str, Any] = {"context": market_context if isinstance(market_context, dict) else {}}

        market_mc = self._latest_market_row_le(_MARKET_MC_PATH)
        if market_mc:
            out["money_cycle_market"] = {
                k: v for k, v in market_mc.items() if k != "date"
            }
            out["money_cycle_market"]["date"] = self._iso_date(market_mc.get("date"))

        if not self.backtest_mode:
            try:
                from multiagents_trading_assistant import fetcher

                macro = fetcher.get_global_macro() or {}
                if macro:
                    out["global_macro"] = macro
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ sector

    def _build_sector(self, symbol: str) -> dict[str, Any]:
        row = self._latest_symbol_row_le(_SMT_PATH, symbol)
        if not row:
            return {}
        keep_cols = [
            "smart_money_state",
            "rs_percentile_20",
            "sector_leadership_score",
            "rs_score",
        ]
        out = {k: row.get(k) for k in keep_cols if k in row}
        out["as_of_row_date"] = self._iso_date(row.get("date"))
        return out

    # --------------------------------------------------------------- technical

    def _build_technical(self, candidate: Any, symbol: str) -> dict[str, Any]:
        indicators = _clean_mapping(getattr(candidate, "indicators", {}) or {})
        smt_row = self._latest_symbol_row_le(_SMT_PATH, symbol)
        chdm_row = self._latest_symbol_row_le(_CHDM_PATH, symbol)
        ds_row = self._latest_symbol_row_le(_DS_PATH, symbol)

        smart_money: dict[str, Any] = {}
        if smt_row:
            for k in [
                "smart_money_score",
                "smart_money_state",
                "clv_score",
                "value_flow_quality_score",
                "accumulation_distribution_score",
                "pullback_quality_score",
                "distribution_days_10",
                "accumulation_days_10",
                "cmf20",
                "mfi14",
            ]:
                if k in smt_row:
                    smart_money[k] = smt_row.get(k)
            smart_money["as_of_row_date"] = self._iso_date(smt_row.get("date"))

        money_cycle: dict[str, Any] = {}
        if chdm_row:
            for k in ("CHDM03", "CHDM20", "CHDM50"):
                if k in chdm_row:
                    money_cycle[k] = chdm_row.get(k)
            money_cycle["chdm_row_date"] = self._iso_date(chdm_row.get("date"))
        if ds_row:
            for k in ("DS20", "DS50"):
                if k in ds_row:
                    money_cycle[k] = ds_row.get(k)
            money_cycle["ds_row_date"] = self._iso_date(ds_row.get("date"))

        return {
            "indicators": indicators,
            "smart_money": smart_money,
            "money_cycle": money_cycle,
        }

    # -------------------------------------------------------------- money_flow

    def _build_money_flow(self, candidate: Any, symbol: str) -> dict[str, Any]:
        indicators = getattr(candidate, "indicators", {}) or {}
        mf = _clean_mapping(indicators.get("money_flow_analysis", {}) or {})
        pv = _clean_mapping(indicators.get("price_volume", {}) or {})
        out: dict[str, Any] = {"classification": mf}
        if pv:
            out["price_volume"] = pv

        if not self.backtest_mode:
            try:
                from multiagents_trading_assistant import fetcher

                ff = fetcher.get_foreign_flow(symbol) or {}
                if ff:
                    out["foreign_flow"] = ff
            except Exception:
                pass
        return out

    # -------------------------------------------------------------------- edge

    def _build_edge(self, candidate: Any) -> dict[str, Any]:
        indicators = getattr(candidate, "indicators", {}) or {}
        return _clean_mapping(indicators.get("edge_strategy_analysis", {}) or {})

    # ------------------------------------------------------------- fundamental

    def _build_fundamental(self, symbol: str) -> dict[str, Any]:
        if self.backtest_mode:
            return {}
        try:
            from multiagents_trading_assistant import fetcher

            return fetcher.get_fundamentals(symbol) or {}
        except Exception:
            return {}

    # -------------------------------------------------------------------- news

    def _build_news(self, symbol: str) -> dict[str, Any]:
        if self.backtest_mode:
            return {}
        try:
            from multiagents_trading_assistant import news_fetcher

            items = news_fetcher.get_stock_news(symbol, days=3, max_articles=20) or []
        except Exception:
            return {}
        if not items:
            return {"news_count": 0, "items": []}

        headlines = []
        positives = []
        negatives = []
        for it in items[:10]:
            headline = str(it.get("headline") or it.get("title") or "")
            sentiment = str(it.get("sentiment") or "").upper()
            headlines.append({
                "headline": headline[:200],
                "source": str(it.get("source") or ""),
                "published_date": str(it.get("published_date") or it.get("date") or ""),
                "sentiment": sentiment,
            })
            if sentiment == "POSITIVE":
                positives.append(headline[:160])
            elif sentiment == "NEGATIVE":
                negatives.append(headline[:160])

        return {
            "news_count": len(items),
            "items": headlines,
            "key_positive": positives[:5],
            "key_negative": negatives[:5],
        }

    # --------------------------------------------------------------- portfolio

    def _build_portfolio(self, symbol: str) -> dict[str, Any]:
        if self.backtest_mode:
            return {}
        try:
            from multiagents_trading_assistant import database

            pos = database.get_position(symbol)
        except Exception:
            return {}
        if not pos:
            return {"has_position": False}
        return {
            "has_position": True,
            "entry_price": _safe_float(pos.get("entry_price")),
            "quantity": pos.get("quantity"),
            "entry_date": pos.get("entry_date"),
            "sl": _safe_float(pos.get("sl")),
            "tp": _safe_float(pos.get("tp")),
            "nav_pct": _safe_float(pos.get("nav_pct")),
            "strategy": pos.get("strategy"),
            "peak_price": _safe_float(pos.get("peak_price")),
        }

    # ------------------------------------------------------------ data_quality

    def _build_data_quality(
        self,
        candidate: Any,
        signal: StrategySignal | None,
        *,
        missing: list[str],
        skipped: list[str],
    ) -> dict[str, Any]:
        indicators = getattr(candidate, "indicators", {}) or {}
        last_ohlcv_date = (
            indicators.get("last_ohlcv_date")
            or indicators.get("feature_date")
            or (signal.feature_date if signal else None)
        )
        out: dict[str, Any] = {
            "setup_type": str(getattr(candidate, "setup_type", "") or ""),
            "priority_score": _safe_float(getattr(candidate, "priority_score", None)),
            "last_ohlcv_date": str(last_ohlcv_date) if last_ohlcv_date else None,
            "missing_fields": missing,
            "skipped_in_backtest": skipped,
            "backtest_mode": self.backtest_mode,
        }
        if signal is not None:
            out["signal_strategy"] = signal.strategy_name
            out["signal_source"] = signal.source
        return out

    # ----------------------------------------------------------------- parquet

    def _latest_symbol_row_le(self, path: Path, symbol: str) -> dict[str, Any]:
        full = self.root / path
        if not full.exists():
            return {}
        try:
            df = pd.read_parquet(full)
        except Exception:
            return {}
        if df.empty or "date" not in df.columns or "symbol" not in df.columns:
            return {}
        sym_up = symbol.upper()
        dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        symbols = df["symbol"].astype(str).str.upper()
        mask = (symbols == sym_up) & (dates <= self.as_of_ts)
        sub = df[mask]
        if sub.empty:
            return {}
        idx = dates[mask].idxmax()
        row = sub.loc[idx].to_dict()
        return row

    def _latest_market_row_le(self, path: Path) -> dict[str, Any]:
        full = self.root / path
        if not full.exists():
            return {}
        try:
            df = pd.read_parquet(full)
        except Exception:
            return {}
        if df.empty or "date" not in df.columns:
            return {}
        dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        mask = dates <= self.as_of_ts
        sub = df[mask]
        if sub.empty:
            return {}
        idx = dates[mask].idxmax()
        return sub.loc[idx].to_dict()

    # ------------------------------------------------------------------- cache

    def _save_cache(self, packet: EvidencePacket) -> None:
        try:
            out_dir = self.cache_root / packet.as_of_date
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{packet.symbol}.json").write_text(
                packet.model_dump_json(by_alias=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _iso_date(value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat() if isinstance(value, datetime) else str(value)[:10]
        return str(value)[:10] if value else None
