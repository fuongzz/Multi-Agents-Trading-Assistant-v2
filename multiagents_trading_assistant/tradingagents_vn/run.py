"""CLI runner for TradingAgents-VN.

Usage:
    python -m multiagents_trading_assistant.tradingagents_vn.run --symbol VCB
    python -m multiagents_trading_assistant.tradingagents_vn.run --symbol HPG --date 2026-05-01
    python -m multiagents_trading_assistant.tradingagents_vn.run --symbol FPT --cheap
    python -m multiagents_trading_assistant.tradingagents_vn.run --symbol VCB --output reports/
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


def _separator(title: str = "", width: int = 72) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return "\n" + "=" * pad + f" {title} " + "=" * pad
    return "\n" + "=" * width


def _build_report(state: dict, quiet: bool) -> str:
    lines = []
    if not quiet:
        lines.append(_separator("BÁO CÁO KỸ THUẬT"))
        lines.append(state.get("market_report", ""))

        lines.append(_separator("BÁO CÁO CƠ BẢN"))
        lines.append(state.get("fundamentals_report", ""))

        lines.append(_separator("BÁO CÁO TIN TỨC / SENTIMENT"))
        lines.append(state.get("sentiment_report", ""))

        lines.append(_separator("BÁO CÁO DÒNG TIỀN NĐTNN"))
        lines.append(state.get("flow_report", ""))

        lines.append(_separator("TỔNG HỢP NGHIÊN CỨU"))
        lines.append(state.get("investment_plan", ""))

        lines.append(_separator("KẾ HOẠCH GIAO DỊCH"))
        lines.append(state.get("trader_investment_plan", ""))

    lines.append(_separator("QUYẾT ĐỊNH CUỐI CÙNG (Giám đốc Danh mục)"))
    lines.append(state.get("final_trade_decision", "Chưa có quyết định"))
    lines.append(_separator())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingAgents-VN: multi-agent analysis")
    parser.add_argument("--symbol", required=True, help="Stock ticker e.g. VCB")
    parser.add_argument("--date", default=None, help="Trade date yyyy-mm-dd (signal_date, default: today)")
    parser.add_argument("--as-of-date", dest="as_of_date", default=None,
                        help="Data cutoff yyyy-mm-dd (default: same as --date). "
                             "Set khi backtest để tránh leak future data.")
    parser.add_argument("--invest-rounds", type=int, default=2, help="Bull/Bear debate rounds")
    parser.add_argument("--risk-rounds", type=int, default=1, help="Risk debate rounds per analyst")
    parser.add_argument("--cheap", action="store_true", help="Use Haiku for all agents (~$0.03-0.05/mã)")
    parser.add_argument("--quiet", action="store_true", help="Only print final decision")
    parser.add_argument("--output", default=None, metavar="DIR",
                        help="Lưu output vào thư mục này (tạo file .md tự động)")
    args = parser.parse_args()

    trade_date = args.date or date.today().strftime("%Y-%m-%d")
    symbol = args.symbol.upper()

    print(f"TradingAgents-VN | {symbol} | {trade_date}")
    print(f"Debate rounds — invest: {args.invest_rounds}, risk: {args.risk_rounds}")
    if args.cheap:
        print("Mode: CHEAP (full Haiku) — ~$0.03-0.05/mã")
    print(_separator())

    from multiagents_trading_assistant.tradingagents_vn.graph import TradingAgentsVN

    agent = TradingAgentsVN(
        max_invest_rounds=args.invest_rounds,
        max_risk_rounds=args.risk_rounds,
        cheap=args.cheap,
    )

    if args.as_of_date:
        print(f"as_of_date: {args.as_of_date} (backtest mode — data capped)")
    print("Running analysis... (this takes 1-3 minutes)")
    state = agent.analyze(symbol, trade_date, as_of_date=args.as_of_date)

    report = _build_report(state, args.quiet)
    print(report)

    # Lưu file nếu có --output
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"{symbol}_{trade_date}.md"
        header = (
            f"# TradingAgents-VN — {symbol} | {trade_date}\n\n"
            f"- Invest rounds: {args.invest_rounds}\n"
            f"- Risk rounds: {args.risk_rounds}\n"
            f"- Mode: {'Cheap (Haiku)' if args.cheap else 'Normal (Haiku+Sonnet)'}\n\n"
        )
        filename.write_text(header + report, encoding="utf-8")
        print(f"\n📄 Đã lưu: {filename.resolve()}")


if __name__ == "__main__":
    main()
