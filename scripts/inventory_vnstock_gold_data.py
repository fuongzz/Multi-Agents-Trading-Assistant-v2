"""Inventory and sample vnstock_data Gold datasets.

This script is intentionally conservative: it calls a broad set of Unified UI
endpoints with small/default samples, records schema/shape/errors, and stores
successful samples under data/research/vnstock_gold_inventory/.

Run:
    $env:PYTHONPATH=(Get-Location).Path
    $env:PYTHONIOENCODING='utf-8'
    C:\\Users\\NC\\.venv\\Scripts\\python.exe scripts\\inventory_vnstock_gold_data.py
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "research" / "vnstock_gold_inventory"


@dataclass
class EndpointResult:
    layer: str
    endpoint: str
    status: str
    rows: int = 0
    cols: int = 0
    columns: str = ""
    sample_path: str = ""
    error: str = ""


def _to_frame(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    if isinstance(obj, dict):
        return pd.json_normalize(obj)
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    return pd.DataFrame({"value": [str(obj)]})


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text).strip("_").lower()


def _save_sample(frame: pd.DataFrame, out_dir: Path, endpoint: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_name(endpoint)
    parquet_path = out_dir / f"{name}.parquet"
    csv_path = out_dir / f"{name}.csv"
    try:
        frame.to_parquet(parquet_path, index=False)
        return str(parquet_path.relative_to(ROOT))
    except Exception:
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return str(csv_path.relative_to(ROOT))


def _call_endpoint(layer: str, endpoint: str, fn: Callable[[], Any], out_dir: Path) -> EndpointResult:
    try:
        data = fn()
        frame = _to_frame(data)
        sample_path = _save_sample(frame, out_dir, endpoint) if not frame.empty else ""
        return EndpointResult(
            layer=layer,
            endpoint=endpoint,
            status="ok",
            rows=int(len(frame)),
            cols=int(len(frame.columns)),
            columns="|".join(map(str, frame.columns[:120])),
            sample_path=sample_path,
        )
    except Exception as exc:
        return EndpointResult(
            layer=layer,
            endpoint=endpoint,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def build_endpoint_plan(args) -> list[tuple[str, str, Callable[[], Any]]]:
    from vnstock_data import Analytics, Fundamental, Insights, Macro, Market, Reference

    ref = Reference()
    market = Market()
    macro = Macro()
    insights = Insights()
    fund = Fundamental()
    analytics = Analytics()

    symbol = args.symbol
    etf_symbol = args.etf_symbol
    fund_symbol = args.fund_symbol
    index_symbol = args.index_symbol

    equity = market.equity(symbol)
    index = market.index(index_symbol)
    etf = market.etf(etf_symbol)
    fund_market = market.fund(fund_symbol)
    fundamental = fund.equity(symbol)
    valuation = analytics.valuation(index_symbol)
    econ = macro.economy()
    commodity = macro.commodity()
    currency = macro.currency()
    ranking = insights.ranking()
    screener = insights.screener()

    return [
        ("reference", "reference.equity.list", lambda: ref.equity.list()),
        ("reference", "reference.equity.list_by_exchange", lambda: ref.equity.list_by_exchange()),
        ("reference", "reference.equity.list_by_group_VN30", lambda: ref.equity.list_by_group("VN30")),
        ("reference", "reference.equity.list_by_group_HOSE", lambda: ref.equity.list_by_group("HOSE")),
        ("reference", "reference.equity.list_by_industry", lambda: ref.equity.list_by_industry()),
        ("reference", "reference.index.groups", lambda: ref.index.groups()),
        ("reference", "reference.index.list", lambda: ref.index.list()),
        ("reference", "reference.index.members_VN30", lambda: ref.index.members("VN30")),
        ("reference", "reference.industry.list", lambda: ref.industry.list()),
        ("reference", "reference.industry.sectors", lambda: ref.industry.sectors()),
        ("reference", "reference.events.calendar", lambda: ref.events.calendar(start=args.start, end=args.end, limit=5000)),
        ("reference", "reference.events.market", lambda: ref.events.market(start=args.start, end=args.end)),
        ("reference", "reference.etf.list", lambda: ref.etf.list()),
        ("reference", "reference.fund.list", lambda: ref.fund.list()),
        ("reference", "reference.bond.list", lambda: ref.bond.list()),
        ("reference", "reference.market.status", lambda: ref.market.status()),
        ("reference", "reference.search.symbol_USD", lambda: ref.search.symbol("USD", limit=10)),
        ("reference", "reference.search.info_USD", lambda: ref.search.info("USD", limit=10)),
        ("market_equity", f"market.equity.{symbol}.ohlcv", lambda: equity.ohlcv(start=args.start, end=args.end)),
        ("market_equity", f"market.equity.{symbol}.quote", lambda: equity.quote()),
        ("market_equity", f"market.equity.{symbol}.summary", lambda: equity.summary()),
        ("market_equity", f"market.equity.{symbol}.trade_history", lambda: equity.trade_history()),
        ("market_equity", f"market.equity.{symbol}.session_stats", lambda: equity.session_stats()),
        ("market_equity", f"market.equity.{symbol}.foreign_flow", lambda: equity.foreign_flow()),
        ("market_equity", f"market.equity.{symbol}.proprietary_flow", lambda: equity.proprietary_flow()),
        ("market_equity", f"market.equity.{symbol}.block_trades", lambda: equity.block_trades(limit=200)),
        ("market_equity", f"market.equity.{symbol}.odd_lot", lambda: equity.odd_lot(limit=200)),
        ("market_equity", f"market.equity.{symbol}.order_book", lambda: equity.order_book()),
        ("market_equity", f"market.equity.{symbol}.trades", lambda: equity.trades(limit=200)),
        ("market_equity", f"market.equity.{symbol}.volume_profile", lambda: equity.volume_profile()),
        ("market_index", f"market.index.{index_symbol}.ohlcv", lambda: index.ohlcv(start=args.start, end=args.end)),
        ("market_index", f"market.index.{index_symbol}.quote", lambda: index.quote()),
        ("market_index", f"market.index.{index_symbol}.summary", lambda: index.summary()),
        ("market_index", f"market.index.{index_symbol}.trade_history", lambda: index.trade_history()),
        ("market_index", f"market.index.{index_symbol}.session_stats", lambda: index.session_stats()),
        ("market_etf", f"market.etf.{etf_symbol}.ohlcv", lambda: etf.ohlcv(start=args.start, end=args.end)),
        ("market_etf", f"market.etf.{etf_symbol}.quote", lambda: etf.quote()),
        ("market_etf", f"market.etf.{etf_symbol}.summary", lambda: etf.summary()),
        ("market_etf", f"market.etf.{etf_symbol}.order_book", lambda: etf.order_book()),
        ("market_fund", f"market.fund.{fund_symbol}.history", lambda: fund_market.history()),
        ("market_fund", f"market.fund.{fund_symbol}.asset_holding", lambda: fund_market.asset_holding()),
        ("market_fund", f"market.fund.{fund_symbol}.industry_holding", lambda: fund_market.industry_holding()),
        ("market_fund", f"market.fund.{fund_symbol}.top_holding", lambda: fund_market.top_holding()),
        ("fundamental", f"fundamental.equity.{symbol}.ratio", lambda: fundamental.ratio()),
        ("fundamental", f"fundamental.equity.{symbol}.income_statement", lambda: fundamental.income_statement()),
        ("fundamental", f"fundamental.equity.{symbol}.balance_sheet", lambda: fundamental.balance_sheet()),
        ("fundamental", f"fundamental.equity.{symbol}.cash_flow", lambda: fundamental.cash_flow()),
        ("fundamental", f"fundamental.equity.{symbol}.note", lambda: fundamental.note()),
        ("fundamental", f"fundamental.equity.{symbol}.financial_health", lambda: fundamental.financial_health()),
        ("analytics", f"analytics.valuation.{index_symbol}.pe", lambda: valuation.pe(duration=args.duration)),
        ("analytics", f"analytics.valuation.{index_symbol}.pb", lambda: valuation.pb(duration=args.duration)),
        ("analytics", f"analytics.valuation.{index_symbol}.evaluation", lambda: valuation.evaluation(duration=args.duration)),
        ("macro_economy", "macro.economy.gdp", lambda: econ.gdp()),
        ("macro_economy", "macro.economy.cpi", lambda: econ.cpi()),
        ("macro_economy", "macro.economy.industry_prod", lambda: econ.industry_prod()),
        ("macro_economy", "macro.economy.import_export", lambda: econ.import_export()),
        ("macro_economy", "macro.economy.retail", lambda: econ.retail()),
        ("macro_economy", "macro.economy.fdi", lambda: econ.fdi()),
        ("macro_economy", "macro.economy.money_supply", lambda: econ.money_supply()),
        ("macro_economy", "macro.economy.population_labor", lambda: econ.population_labor()),
        ("macro_currency", "macro.currency.exchange_rate", lambda: currency.exchange_rate()),
        ("macro_currency", "macro.currency.interest_rate", lambda: currency.interest_rate()),
        ("macro_commodity", "macro.commodity.gold_VN", lambda: commodity.gold(market="VN")),
        ("macro_commodity", "macro.commodity.gold_GLOBAL", lambda: commodity.gold(market="GLOBAL")),
        ("macro_commodity", "macro.commodity.gas_VN", lambda: commodity.gas(market="VN")),
        ("macro_commodity", "macro.commodity.gas_GLOBAL", lambda: commodity.gas(market="GLOBAL")),
        ("macro_commodity", "macro.commodity.oil_crude", lambda: commodity.oil_crude()),
        ("macro_commodity", "macro.commodity.coke", lambda: commodity.coke()),
        ("macro_commodity", "macro.commodity.steel_GLOBAL", lambda: commodity.steel(market="GLOBAL")),
        ("macro_commodity", "macro.commodity.steel_VN", lambda: commodity.steel(market="VN")),
        ("macro_commodity", "macro.commodity.iron_ore", lambda: commodity.iron_ore()),
        ("macro_commodity", "macro.commodity.fertilizer_ure", lambda: commodity.fertilizer_ure()),
        ("macro_commodity", "macro.commodity.soybean", lambda: commodity.soybean()),
        ("macro_commodity", "macro.commodity.corn", lambda: commodity.corn()),
        ("macro_commodity", "macro.commodity.sugar", lambda: commodity.sugar()),
        ("macro_commodity", "macro.commodity.pork_VN", lambda: commodity.pork(market="VN")),
        ("macro_commodity", "macro.commodity.pork_CHINA", lambda: commodity.pork(market="CHINA")),
        ("insights", "insights.ranking.gainer", lambda: ranking.gainer(index=index_symbol, limit=50)),
        ("insights", "insights.ranking.loser", lambda: ranking.loser(index=index_symbol, limit=50)),
        ("insights", "insights.ranking.value", lambda: ranking.value(index=index_symbol, limit=50)),
        ("insights", "insights.ranking.volume", lambda: ranking.volume(index=index_symbol, limit=50)),
        ("insights", "insights.ranking.deal", lambda: ranking.deal(index=index_symbol, limit=50)),
        ("insights", "insights.ranking.foreign_buy", lambda: ranking.foreign_buy(limit=50)),
        ("insights", "insights.ranking.foreign_sell", lambda: ranking.foreign_sell(limit=50)),
        ("insights", "insights.screener.criteria", lambda: screener.criteria()),
        ("insights", "insights.screener.filter", lambda: screener.filter(limit=args.screener_limit)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="VCB")
    parser.add_argument("--index-symbol", default="VNINDEX")
    parser.add_argument("--etf-symbol", default="E1VFVN30")
    parser.add_argument("--fund-symbol", default="SSISCA")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--duration", default="5Y")
    parser.add_argument("--screener-limit", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.symbol}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_endpoint_plan(args)
    if args.limit > 0:
        plan = plan[: args.limit]

    results: list[EndpointResult] = []
    for idx, (layer, endpoint, fn) in enumerate(plan, start=1):
        print(f"[{idx:03d}/{len(plan):03d}] {endpoint}")
        result = _call_endpoint(layer, endpoint, fn, out_dir)
        print(f"  -> {result.status} rows={result.rows} cols={result.cols} {result.error}")
        results.append(result)

    catalog = pd.DataFrame([asdict(item) for item in results])
    catalog.to_csv(out_dir / "catalog.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "catalog.json").open("w", encoding="utf-8") as fh:
        json.dump([asdict(item) for item in results], fh, ensure_ascii=False, indent=2)
    print(f"[done] saved {out_dir / 'catalog.csv'}")
    print(catalog.groupby(["layer", "status"]).size().to_string())


if __name__ == "__main__":
    main()
