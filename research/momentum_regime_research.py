"""
Nghiên cứu backtest: Relative Momentum + Regime Filter cho NĐT cá nhân VN
==========================================================================

File nghiên cứu ĐỘC LẬP, nhỏ gọn — KHÔNG phụ thuộc engine/backtest hiện có
của dự án. Chỉ đọc dữ liệu giá đã có sẵn trong project:

    multiagents_trading_assistant/data/ohlcv_master.parquet   (VN100, 2020→nay)
    multiagents_trading_assistant/data/index_master.parquet   (VNINDEX, 2012→nay)

Mục đích (theo triết lý dự án):
    - Backtest để LOẠI ý tưởng tồi, không phải để dự đoán tương lai.
    - Mốc phải vượt: nắm giữ thụ động VN-Index SAU KHI trừ phí/thuế/trượt giá.
    - Ít tham số, có lý do kinh tế rõ ràng (chống overfit).
    - Mô hình chi phí BI QUAN. Xử lý bẫy giá trần đặc thù VN.

Chiến lược (đúng "Hướng chiến lược ưu tiên" trong bối cảnh dự án):
    1. Vũ trụ : lọc thanh khoản POINT-IN-TIME — chỉ giữ mã có giá trị giao
                dịch TB 60 phiên ≥ ngưỡng tỷ đồng (mặc định 20 tỷ, hợp vốn
                500tr–5 tỷ). Đây là tuyến phòng thủ số 1 chống ảo tưởng
                backtest: loại trước những mã bạn KHÔNG thật sự mua bán được.
    2. Tín hiệu: xếp hạng momentum tương đối = lợi suất ~6 tháng (bỏ 5 phiên
                 gần nhất để tránh đảo chiều ngắn hạn). Giữ TOP N dẫn đầu.
    3. Bộ lọc chế độ: chỉ giải ngân khi VNINDEX > MA200; nếu không → tiền mặt.
    4. Tái cân bằng hàng THÁNG. Equal-weight, cap tỷ trọng mỗi mã.
    5. Không bán khống. Chỉ long hoặc đứng ngoài.

Quy tắc CHỐNG look-ahead:
    - Tín hiệu, chế độ & lọc thanh khoản tính bằng dữ liệu ĐẾN HẾT phiên r.
    - Thực thi tại GIÁ MỞ CỬA phiên kế tiếp (r+1). Không dùng giá đóng tín hiệu.

Bẫy giá trần (VN-specific):
    - Khi cần MUA nhưng giá mở cửa phiên thực thi ≥ +6.9% so với close hôm
      trước (chạm trần HOSE) → coi như KHÔNG khớp được (trắng bên bán),
      bỏ qua mã đó kỳ này. Bỏ qua bẫy này = "lợi nhuận ma".
    - Khi cần BÁN nhưng giá mở cửa ≤ -6.9% (chạm sàn) → không thoát được,
      giữ lại tới kỳ sau.

⚠️ GIỚI HẠN DỮ LIỆU (đọc kỹ trước khi tin số liệu):
    - SURVIVORSHIP BIAS: file chỉ chứa VN100 *hiện tại* → đã loại sẵn mã hủy
      niêm yết/rớt rổ. Kết quả vì thế LẠC QUAN hơn thực tế. Lọc thanh khoản
      point-in-time giảm bớt phần nào nhưng KHÔNG khử được bias này. Muốn khử
      cần dữ liệu universe lịch sử + mã đã hủy niêm yết (chưa có trong project).
    - MỘT CHU KỲ RƯỠI: dữ liệu từ 2020 — có xuyên qua sập 2022 (tốt) nhưng
      thiếu các chu kỳ trước. Đọc MaxDD và cột OOS, đừng chỉ đọc Return.

Chạy:
    python -m research.momentum_regime_research
    python -m research.momentum_regime_research --top 8 --min-liq 30 --oos 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Console Windows mặc định cp1252 — ép UTF-8 để in được tiếng Việt
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Cấu hình mặc định
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[1]
_OHLCV = _ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
_INDEX = _ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"

# Mô hình chi phí BI QUAN (mỗi chiều, theo giá trị giao dịch)
FEE_RATE = 0.0015        # phí môi giới 0.15%/chiều
SLIP_RATE = 0.0015       # trượt giá 0.15%/chiều (vốn 500tr–5 tỷ, lệnh vừa)
SELL_TAX = 0.0010        # thuế bán 0.10% giá trị bán (bất kể lãi/lỗ)

CEILING = 0.069          # ngưỡng coi như chạm trần/sàn HOSE (~±7%)
TRADING_DAYS = 252
LIQ_WINDOW = 60          # cửa sổ tính thanh khoản TB (phiên)


# --------------------------------------------------------------------------- #
# Nạp dữ liệu
# --------------------------------------------------------------------------- #
def load_panels(start: str | None = None):
    """Trả về (close_w, open_w, liq_w, vni) — pivot ngày × mã.

    liq_w = giá trị giao dịch TB 60 phiên, đơn vị TỶ ĐỒNG (value/1e6).
    (value trong store = giá_nghìn × khối_lượng → VND = value×1000.)
    """
    px = pd.read_parquet(_OHLCV, columns=["date", "symbol", "open", "close", "value"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date")
    if start:
        # giữ thêm buffer trước 'start' để có sẵn lookback + MA + liquidity
        px = px[px["date"] >= pd.Timestamp(start)]

    close_w = px.pivot(index="date", columns="symbol", values="close").sort_index()
    open_w = px.pivot(index="date", columns="symbol", values="open").sort_index()
    val_w = px.pivot(index="date", columns="symbol", values="value").sort_index()
    liq_w = (val_w / 1e6).rolling(LIQ_WINDOW, min_periods=20).mean()  # tỷ đồng

    idx = pd.read_parquet(_INDEX, columns=["date", "symbol", "close"])
    idx = idx[idx["symbol"] == "VNINDEX"].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    vni = idx.set_index("date")["close"].sort_index().reindex(close_w.index).ffill()
    return close_w, open_w, liq_w, vni


# --------------------------------------------------------------------------- #
# Lõi backtest
# --------------------------------------------------------------------------- #
def run_backtest(
    close_w, open_w, liq_w, vni,
    top_n: int = 10,
    lookback: int = 120,
    skip: int = 5,
    ma_regime: int = 200,
    max_weight: float = 0.15,
    min_liq: float = 20.0,
) -> dict:
    """Momentum + regime + lọc thanh khoản, tái cân bằng cuối mỗi tháng.

    Equity mark-to-market hàng ngày theo close. Theo dõi chi phí & turnover.
    """
    dates = close_w.index
    vni_ma = vni.rolling(ma_regime).mean()

    month_key = dates.to_period("M")
    is_month_end = month_key != np.roll(month_key, -1)
    rebal_dates = [d for d, m in zip(dates, is_month_end) if m]
    min_start = dates[max(lookback + skip, ma_regime) + 1]
    rebal_dates = [d for d in rebal_dates if d >= min_start]
    rebal_set = set(rebal_dates)

    cash = 1.0
    shares: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    total_cost = 0.0          # tổng phí+thuế+slip đã trả (đơn vị NAV)
    total_buy_value = 0.0     # để tính turnover

    pos_first = dates.get_loc(rebal_dates[0])

    def pv(day):
        prices = close_w.loc[day]
        return cash + sum(sh * prices.get(s, np.nan) for s, sh in shares.items()
                          if not np.isnan(prices.get(s, np.nan)))

    for i in range(pos_first, len(dates)):
        day = dates[i]

        if day in rebal_set and i + 1 < len(dates):
            exec_day = dates[i + 1]
            regime_on = (not np.isnan(vni_ma.loc[day])) and bool(vni.loc[day] > vni_ma.loc[day])

            # momentum
            if i - lookback - skip < 0:
                equity_curve.append((day, pv(day)))
                continue
            mom = (close_w.iloc[i - skip] / close_w.iloc[i - lookback - skip] - 1.0).dropna()

            # lọc thanh khoản point-in-time + có giá mở cửa ngày thực thi
            liq_today = liq_w.loc[day]
            liquid = liq_today[liq_today >= min_liq].index
            valid = open_w.loc[exec_day].dropna().index
            mom = mom[mom.index.isin(liquid) & mom.index.isin(valid)]

            target = (list(mom.sort_values(ascending=False).head(top_n).index)
                      if regime_on and len(mom) else [])

            cur_val = pv(day)
            op = open_w.loc[exec_day]
            pc = close_w.loc[day]

            # BÁN mã rớt khỏi target
            for sym in list(shares):
                if sym in target:
                    continue
                o, c = op.get(sym, np.nan), pc.get(sym, np.nan)
                if np.isnan(o):
                    continue
                if not np.isnan(c) and o <= c * (1 - CEILING):   # bẫy sàn → kẹt
                    continue
                sh = shares.pop(sym)
                gross = sh * o
                cost = gross * (FEE_RATE + SLIP_RATE + SELL_TAX)
                cash += gross - cost
                total_cost += cost
                basis = cost_basis.pop(sym, o)
                trades.append({"symbol": sym, "ret": o / basis - 1.0})

            # MUA / cân bằng equal-weight
            if target:
                w = min(1.0 / len(target), max_weight)
                tgt_val = cur_val * w
                for sym in target:
                    o, c = op.get(sym, np.nan), pc.get(sym, np.nan)
                    if np.isnan(o):
                        continue
                    if sym not in shares and not np.isnan(c) and o >= c * (1 + CEILING):  # bẫy trần
                        continue
                    cur_sh = shares.get(sym, 0.0)
                    delta = tgt_val - cur_sh * o
                    if delta > 1e-9:
                        spend = min(delta, cash)
                        if spend <= 1e-9:
                            continue
                        cost = spend * (FEE_RATE + SLIP_RATE) / (1 + FEE_RATE + SLIP_RATE)
                        eff_price = o * (1 + FEE_RATE + SLIP_RATE)
                        add_sh = (spend - cost) / o
                        old_b = cost_basis.get(sym, eff_price)
                        tot = cur_sh + add_sh
                        cost_basis[sym] = (cur_sh * old_b + add_sh * eff_price) / tot if tot else eff_price
                        shares[sym] = tot
                        cash -= spend
                        total_cost += cost
                        total_buy_value += spend
                    elif delta < -1e-9 and cur_sh > 0:
                        red_sh = min(cur_sh, -delta / o)
                        gross = red_sh * o
                        cost = gross * (FEE_RATE + SLIP_RATE + SELL_TAX)
                        shares[sym] = cur_sh - red_sh
                        cash += gross - cost
                        total_cost += cost

        equity_curve.append((day, pv(day)))

    eq = pd.Series(dict(equity_curve)).sort_index()
    avg_equity = eq.mean()
    return {
        "equity": eq,
        "trades": trades,
        "rebal_count": len(rebal_dates),
        "total_cost_pct": round(total_cost / avg_equity * 100, 1),  # % NAV TB đã trả cho chi phí
        "turnover_x": round(total_buy_value / avg_equity, 1),       # tổng giá trị mua / NAV TB
    }


# --------------------------------------------------------------------------- #
# Thống kê hiệu suất
# --------------------------------------------------------------------------- #
def perf_stats(equity: pd.Series, label: str) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"label": label, "total_return_pct": 0, "cagr_pct": 0,
                "sharpe": 0, "max_dd_pct": 0, "vol_pct": 0}
    ret = eq.pct_change(fill_method=None).dropna()
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1.0).min()
    return {
        "label": label,
        "total_return_pct": round(total * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(mdd * 100, 1),
        "vol_pct": round(vol * 100, 1),
    }


def trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0, "avg_ret_pct": 0.0}
    rets = np.array([t["ret"] for t in trades])
    wins, losses = rets[rets > 0], rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    return {
        "n_trades": len(rets),
        "win_rate_pct": round((rets > 0).mean() * 100, 1),
        "profit_factor": round(pf, 2),
        "avg_ret_pct": round(rets.mean() * 100, 2),
    }


def benchmark_buyhold(series: pd.Series, eq_index, label: str) -> dict:
    bh = series.reindex(eq_index).ffill().dropna()
    return perf_stats(bh / bh.iloc[0], label)


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'Chiến lược':<42}{'Return%':>10}{'CAGR%':>9}{'Sharpe':>9}{'MaxDD%':>9}{'Vol%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        print(f"{s['label']:<42}{s['total_return_pct']:>10}{s['cagr_pct']:>9}"
              f"{s['sharpe']:>9}{s['max_dd_pct']:>9}{s['vol_pct']:>8}")


# --------------------------------------------------------------------------- #
# CLI / báo cáo
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Momentum + Regime backtest (research)")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--top", type=int, default=10, help="số mã giữ")
    ap.add_argument("--lookback", type=int, default=120, help="cửa sổ momentum (phiên)")
    ap.add_argument("--skip", type=int, default=5, help="bỏ N phiên gần nhất")
    ap.add_argument("--ma", type=int, default=200, help="MA regime trên VNINDEX")
    ap.add_argument("--maxw", type=float, default=0.15, help="cap tỷ trọng mỗi mã")
    ap.add_argument("--min-liq", type=float, default=20.0, help="thanh khoản TB tối thiểu (tỷ/phiên)")
    ap.add_argument("--oos", default="2024-01-01", help="mốc tách out-of-sample")
    args = ap.parse_args()

    close_w, open_w, liq_w, vni = load_panels(start=args.start)
    n_liquid = int((liq_w.iloc[-1] >= args.min_liq).sum())
    print(f"Dữ liệu : {close_w.shape[1]} mã | {close_w.index[0].date()} → {close_w.index[-1].date()} "
          f"| {len(close_w)} phiên")
    print(f"Vũ trụ  : lọc thanh khoản ≥ {args.min_liq:.0f} tỷ/phiên (TB {LIQ_WINDOW} phiên) "
          f"→ ~{n_liquid} mã đạt ở phiên cuối")
    print(f"Chi phí  : phí {FEE_RATE:.2%}/chiều + trượt {SLIP_RATE:.2%}/chiều + thuế bán {SELL_TAX:.2%}")

    res = run_backtest(
        close_w, open_w, liq_w, vni,
        top_n=args.top, lookback=args.lookback, skip=args.skip,
        ma_regime=args.ma, max_weight=args.maxw, min_liq=args.min_liq,
    )
    eq = res["equity"]

    print(f"\nThông số : top={args.top} lookback={args.lookback} skip={args.skip} "
          f"MA{args.ma} maxw={args.maxw:.0%} | {res['rebal_count']} lần tái cân bằng")
    print(f"Chi phí thực: đã trả ~{res['total_cost_pct']}% NAV-TB cho phí/thuế/slip "
          f"| turnover ~{res['turnover_x']}× NAV\n")

    # --- Bảng FULL PERIOD ---
    print("══ TOÀN KỲ ══")
    strat = perf_stats(eq, "Momentum+Regime (net)")
    ew = close_w.pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
    _print_table([
        strat,
        benchmark_buyhold(vni, eq.index, "VNINDEX buy & hold"),
        perf_stats(ew.reindex(eq.index).ffill().dropna(), "Equal-weight buy & hold"),
    ])
    tr = trade_stats(res["trades"])
    print(f"Lệnh: {tr['n_trades']} round-trip | WR {tr['win_rate_pct']}% "
          f"| PF {tr['profit_factor']} | avg/lệnh {tr['avg_ret_pct']}%")

    # --- Tách IN-SAMPLE / OUT-OF-SAMPLE ---
    oos = pd.Timestamp(args.oos)
    is_eq, oos_eq = eq[eq.index < oos], eq[eq.index >= oos]
    if len(is_eq) > 30 and len(oos_eq) > 30:
        print(f"\n══ IN-SAMPLE (< {oos.date()}) vs OUT-OF-SAMPLE (≥ {oos.date()}) ══")
        _print_table([
            perf_stats(is_eq, f"  IS  strat"),
            benchmark_buyhold(vni, is_eq.index, "  IS  VNINDEX"),
            perf_stats(oos_eq, f"  OOS strat"),
            benchmark_buyhold(vni, oos_eq.index, "  OOS VNINDEX"),
        ])
        print("  → OOS yếu hơn IS nhiều = dấu hiệu overfit / một-chu-kỳ.")

    # --- Kết luận ---
    bench = benchmark_buyhold(vni, eq.index, "VNINDEX")
    edge = strat["total_return_pct"] - bench["total_return_pct"]
    dd = strat["max_dd_pct"] - bench["max_dd_pct"]
    print(f"\nKết luận : alpha vs VNINDEX = {edge:+.1f}pp return | "
          f"MDD {'tốt hơn' if dd > 0 else 'tệ hơn'} {abs(dd):.1f}pp | "
          f"Sharpe {strat['sharpe']} vs {bench['sharpe']}")
    print("⚠️  Số liệu ĐÃ trừ chi phí bi quan + xử lý bẫy giá trần, NHƯNG còn dính")
    print("    SURVIVORSHIP BIAS (universe = VN100 hiện tại). Coi đây là CẬN TRÊN")
    print("    lạc quan. Bắt buộc: OOS-lock + paper trade vài tháng trước tiền thật.")


if __name__ == "__main__":
    main()
