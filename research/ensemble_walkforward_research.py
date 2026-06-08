"""
Nghiên cứu backtest: Ensemble luôn-chạy + Walk-forward cho NĐT cá nhân VN
==========================================================================

File nghiên cứu ĐỘC LẬP. Đọc dữ liệu giá sẵn có trong project:
    multiagents_trading_assistant/data/ohlcv_master.parquet   (VN100, 2020→nay)
    multiagents_trading_assistant/data/index_master.parquet   (VNINDEX)

Ý tưởng (kiến trúc B — đã chốt với người dùng):
    KHÔNG dự báo chế độ thị trường. Chạy 3 nhánh chiến lược TỰ-GẠN cùng lúc;
    mỗi nhánh tự về tiền mặt khi điều kiện của nó không thoả. "Thích nghi chế
    độ" TỰ PHÁT SINH từ logic từng nhánh, không từ một bộ phân loại trung tâm
    (vốn dính hindsight bias). Đúng nhiều → lãi nhiều, sai nhiều → lỗ ít, nhờ:
        - mỗi nhánh tự cắt khi rớt điều kiện (độ lồi/convexity)
        - đa dạng hoá theo NGUỒN return, không chỉ theo mã.

Ba nhánh (lý do kinh tế PHÂN BIỆT, ít tham số, không tối ưu):
    XMOM   — Cross-sectional momentum: top-N return 6 tháng, gate VNINDEX>MA200.
    TREND  — Time-series trend: giữ mã trên MA100 của CHÍNH NÓ + return 60d>0.
             Tự về tiền mặt khi thị trường rộng giảm (ít mã đạt).
    BRKOUT — Breakout-after-accumulation: mã lập đỉnh 60 phiên SAU giai đoạn nén
             biến động (vol20 ở đáy phân phối). Tự gạn theo số breakout.

Tổ hợp (WALK-FORWARD thật — chỉ dùng quá khứ):
    Trọng số INVERSE-VOL ước lượng trên 120 phiên TRƯỚC ĐÓ của từng nhánh,
    chuẩn hoá, áp cho phiên kế. Không có tham số nào fit vào tương lai.
    (Đây là yếu tố walk-forward; bản thân các nhánh CỐ Ý không tối ưu tham số
     để tránh overfit — walk-forward ở đây là stress-test xuyên chế độ + tái
     ước lượng trọng số, không phải dò tham số đẹp.)

Trung thực phương pháp:
    - Chống look-ahead: tín hiệu đến hết phiên r, khớp giá MỞ CỬA r+1.
    - Chi phí BI QUAN: phí 0.15% + trượt 0.15%/chiều + thuế bán 0.10%.
    - Bẫy giá trần/sàn HOSE (~±7%): chạm trần→không mua được; sàn→không bán được.
    - HOLDOUT KHOÁ CỨNG (mặc định 2025-06-01→): báo cáo RIÊNG, "tiêu một lần".
    - ⚠️ SURVIVORSHIP BIAS vẫn còn (universe = VN100 hiện tại) → mọi số là CẬN
      TRÊN lạc quan. Đọc breakdown theo NĂM + holdout, đừng đọc mỗi headline.

Chạy:
    python -m research.ensemble_walkforward_research
    python -m research.ensemble_walkforward_research --top 8 --min-liq 30 --holdout 2025-06-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[1]
_OHLCV = _ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
_INDEX = _ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"

FEE_RATE = 0.0015
SLIP_RATE = 0.0015
SELL_TAX = 0.0010
CEILING = 0.069
TRADING_DAYS = 252
LIQ_WINDOW = 60
VOL_WEIGHT_WINDOW = 120   # cửa sổ trượt ước lượng inverse-vol cho tổ hợp


# --------------------------------------------------------------------------- #
# Nạp + tiền xử lý (precompute tín hiệu wide một lần)
# --------------------------------------------------------------------------- #
class Ctx:
    """Gói dữ liệu + tín hiệu precompute, truy cập theo vị trí dòng i."""

    def __init__(self, start=None, lookback=120, skip=5, ma_stock=100,
                 hi_win=60, vol_win=20, ma_regime=200, data_path=None):
        src = Path(data_path) if data_path else _OHLCV
        px = pd.read_parquet(src, columns=["date", "symbol", "open", "close", "value"])
        px["date"] = pd.to_datetime(px["date"])
        px = px.sort_values("date")
        if start:
            px = px[px["date"] >= pd.Timestamp(start)]

        self.close = px.pivot(index="date", columns="symbol", values="close").sort_index()
        self.open = px.pivot(index="date", columns="symbol", values="open").sort_index()
        val = px.pivot(index="date", columns="symbol", values="value").sort_index()
        self.liq = (val / 1e6).rolling(LIQ_WINDOW, min_periods=20).mean()  # tỷ đồng
        self.dates = self.close.index

        c = self.close
        # XMOM: return lookback (bỏ skip phiên gần nhất)
        self.mom = c.shift(skip) / c.shift(lookback + skip) - 1.0
        # TREND: MA của chính mã + return 60d
        self.ma_stock = c.rolling(ma_stock).mean()
        self.ret60 = c / c.shift(60) - 1.0
        # BRKOUT: đỉnh hi_win phiên (gồm hôm nay = lập đỉnh) + nén biến động
        self.hi = c.rolling(hi_win).max()
        dret = c.pct_change(fill_method=None)
        self.vol20 = dret.rolling(vol_win).std()
        # ngưỡng nén: vol20 ở đáy 40% phân phối 1 năm TRƯỚC ĐÓ (shift chống self-ref)
        self.vol_thr = self.vol20.shift(1).rolling(TRADING_DAYS, min_periods=60).quantile(0.40)

        idx = pd.read_parquet(_INDEX, columns=["date", "symbol", "close"])
        idx = idx[idx["symbol"] == "VNINDEX"].copy()
        idx["date"] = pd.to_datetime(idx["date"])
        vni = idx.set_index("date")["close"].sort_index().reindex(self.dates).ffill()
        self.vni = vni
        self.vni_ma = vni.rolling(ma_regime).mean()
        self.warmup = max(lookback + skip, ma_regime, ma_stock, hi_win) + 1


# --------------------------------------------------------------------------- #
# Bộ chọn mã cho từng nhánh (trả list ĐÃ xếp hạng; [] = về tiền mặt)
# --------------------------------------------------------------------------- #
def sel_xmom(ctx: Ctx, i: int, top_n: int) -> list[str]:
    day = ctx.dates[i]
    if np.isnan(ctx.vni_ma.loc[day]) or ctx.vni.loc[day] <= ctx.vni_ma.loc[day]:
        return []                                   # gate chế độ → tiền mặt
    s = ctx.mom.iloc[i].dropna().sort_values(ascending=False)
    return list(s.head(top_n * 3).index)            # dư để lọc thanh khoản sau


def sel_trend(ctx: Ctx, i: int, top_n: int) -> list[str]:
    c = ctx.close.iloc[i]
    ma = ctx.ma_stock.iloc[i]
    r60 = ctx.ret60.iloc[i]
    cond = (c > ma) & (r60 > 0)                      # trên MA chính nó + đang lên
    cand = ctx.ret60.iloc[i][cond].dropna().sort_values(ascending=False)
    return list(cand.head(top_n * 3).index)          # ít mã đạt khi downtrend → tự gạn


def sel_brkout(ctx: Ctx, i: int, top_n: int) -> list[str]:
    c = ctx.close.iloc[i]
    is_high = c >= ctx.hi.iloc[i]                     # lập đỉnh 60 phiên hôm nay
    squeezed = ctx.vol20.iloc[i] <= ctx.vol_thr.iloc[i]  # sau giai đoạn nén
    cond = is_high & squeezed
    # xếp theo độ "khô" biến động (nén sâu nhất ưu tiên)
    cand = ctx.vol20.iloc[i][cond].dropna().sort_values()
    return list(cand.head(top_n * 3).index)


STRATEGIES = {"XMOM": sel_xmom, "TREND": sel_trend, "BRKOUT": sel_brkout}


# --------------------------------------------------------------------------- #
# Mô phỏng MỘT nhánh (full capital, equity chuẩn hoá về 1.0)
# --------------------------------------------------------------------------- #
def run_sleeve(ctx: Ctx, select_fn, top_n=10, max_weight=0.15, min_liq=20.0) -> dict:
    dates = ctx.dates
    month_key = dates.to_period("M")
    is_me = month_key != np.roll(month_key, -1)
    rebal = [d for d, m in zip(dates, is_me) if m and dates.get_loc(d) >= ctx.warmup]
    rebal_set = set(rebal)
    if not rebal:
        return {"equity": pd.Series(dtype=float), "trades": [], "total_cost": 0.0, "buy_value": 0.0}

    cash, shares, basis = 1.0, {}, {}
    curve, trades = [], []
    total_cost = buy_value = 0.0
    start_i = dates.get_loc(rebal[0])

    def pv(day):
        p = ctx.close.loc[day]
        return cash + sum(sh * p.get(s, np.nan) for s, sh in shares.items()
                          if not np.isnan(p.get(s, np.nan)))

    for i in range(start_i, len(dates)):
        day = dates[i]
        if day in rebal_set and i + 1 < len(dates):
            exec_day = dates[i + 1]
            cand = select_fn(ctx, i, top_n)
            liq_ok = ctx.liq.loc[day]
            liquid = set(liq_ok[liq_ok >= min_liq].index)
            valid = set(ctx.open.loc[exec_day].dropna().index)
            target = [s for s in cand if s in liquid and s in valid][:top_n]

            cur_val = pv(day)
            op, pc = ctx.open.loc[exec_day], ctx.close.loc[day]

            for sym in list(shares):                 # BÁN mã rớt target
                if sym in target:
                    continue
                o, c = op.get(sym, np.nan), pc.get(sym, np.nan)
                if np.isnan(o) or (not np.isnan(c) and o <= c * (1 - CEILING)):
                    continue                         # bẫy sàn → kẹt
                sh = shares.pop(sym)
                gross = sh * o
                cost = gross * (FEE_RATE + SLIP_RATE + SELL_TAX)
                cash += gross - cost
                total_cost += cost
                trades.append({"symbol": sym, "ret": o / basis.pop(sym, o) - 1.0})

            if target:                               # MUA / cân bằng equal-weight
                w = min(1.0 / len(target), max_weight)
                tgt = cur_val * w
                for sym in target:
                    o, c = op.get(sym, np.nan), pc.get(sym, np.nan)
                    if np.isnan(o):
                        continue
                    if sym not in shares and not np.isnan(c) and o >= c * (1 + CEILING):
                        continue                     # bẫy trần → không mua được
                    cur_sh = shares.get(sym, 0.0)
                    delta = tgt - cur_sh * o
                    if delta > 1e-9:
                        spend = min(delta, cash)
                        if spend <= 1e-9:
                            continue
                        cost = spend * (FEE_RATE + SLIP_RATE) / (1 + FEE_RATE + SLIP_RATE)
                        eff = o * (1 + FEE_RATE + SLIP_RATE)
                        add = (spend - cost) / o
                        ob = basis.get(sym, eff)
                        tot = cur_sh + add
                        basis[sym] = (cur_sh * ob + add * eff) / tot if tot else eff
                        shares[sym] = tot
                        cash -= spend
                        total_cost += cost
                        buy_value += spend
                    elif delta < -1e-9 and cur_sh > 0:
                        red = min(cur_sh, -delta / o)
                        gross = red * o
                        cost = gross * (FEE_RATE + SLIP_RATE + SELL_TAX)
                        shares[sym] = cur_sh - red
                        cash += gross - cost
                        total_cost += cost
        curve.append((day, pv(day)))

    eq = pd.Series(dict(curve)).sort_index()
    return {"equity": eq, "trades": trades, "total_cost": total_cost,
            "buy_value": buy_value, "avg_equity": eq.mean()}


# --------------------------------------------------------------------------- #
# Tổ hợp inverse-vol WALK-FORWARD (chỉ dùng quá khứ)
# --------------------------------------------------------------------------- #
def combine_inverse_vol(sleeve_eq: dict[str, pd.Series], window=VOL_WEIGHT_WINDOW):
    """Trả (ensemble_equity, weights_df). Trọng số tỷ lệ nghịch vol trượt
    ước lượng trên 'window' phiên TRƯỚC ĐÓ → áp cho return phiên kế (no lookahead)."""
    rets = pd.DataFrame({k: v.pct_change(fill_method=None) for k, v in sleeve_eq.items()}).dropna(how="all")
    rets = rets.fillna(0.0)
    # vol trượt, shift(1) để chỉ dùng dữ liệu tới hôm trước
    vol = rets.rolling(window, min_periods=20).std().shift(1)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    w = w.fillna(1.0 / rets.shape[1])               # giai đoạn đầu: equal weight
    ens_ret = (w * rets).sum(axis=1)
    ens_eq = (1.0 + ens_ret).cumprod()
    return ens_eq, w, rets


# --------------------------------------------------------------------------- #
# Thống kê
# --------------------------------------------------------------------------- #
def perf(eq: pd.Series, label: str) -> dict:
    eq = eq.dropna()
    if len(eq) < 5:
        return {"label": label, "ret": 0, "cagr": 0, "sharpe": 0, "mdd": 0, "vol": 0}
    r = eq.pct_change(fill_method=None).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {
        "label": label,
        "ret": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
        "cagr": round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 1) if yrs > 0 else 0,
        "sharpe": round((r.mean() * TRADING_DAYS) / vol, 2) if vol > 0 else 0,
        "mdd": round((eq / eq.cummax() - 1).min() * 100, 1),
        "vol": round(vol * 100, 1),
    }


def trade_stat(trades) -> str:
    if not trades:
        return "0 lệnh"
    r = np.array([t["ret"] for t in trades])
    w, l = r[r > 0], r[r < 0]
    pf = w.sum() / abs(l.sum()) if l.sum() else float("inf")
    return f"{len(r)} lệnh | WR {round((r>0).mean()*100,1)}% | PF {round(pf,2)} | avg {round(r.mean()*100,2)}%"


def bh(series, idx, label):
    s = series.reindex(idx).ffill().dropna()
    return perf(s / s.iloc[0], label)


def ptable(rows, title=None):
    if title:
        print(f"\n══ {title} ══")
    h = f"{'':<24}{'Return%':>10}{'CAGR%':>9}{'Sharpe':>9}{'MaxDD%':>9}{'Vol%':>8}"
    print(h)
    print("-" * len(h))
    for s in rows:
        print(f"{s['label']:<24}{s['ret']:>10}{s['cagr']:>9}{s['sharpe']:>9}{s['mdd']:>9}{s['vol']:>8}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Ensemble walk-forward backtest (research)")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--maxw", type=float, default=0.15)
    ap.add_argument("--min-liq", type=float, default=20.0, help="thanh khoản TB tối thiểu (tỷ/phiên)")
    ap.add_argument("--holdout", default="2025-06-01", help="mốc holdout khoá cứng")
    ap.add_argument("--data", default=None,
                    help="parquet OHLCV thay thế (vd research/data/ohlcv_hose_full.parquet)")
    args = ap.parse_args()

    ctx = Ctx(start=args.start, data_path=args.data)
    print(f"Dữ liệu : {ctx.close.shape[1]} mã | {ctx.dates[0].date()} → {ctx.dates[-1].date()} "
          f"| {len(ctx.dates)} phiên")
    print(f"Vũ trụ  : thanh khoản ≥ {args.min_liq:.0f} tỷ/phiên | chi phí phí {FEE_RATE:.2%}"
          f"+trượt {SLIP_RATE:.2%}/chiều+thuế bán {SELL_TAX:.2%}")

    sleeves = {name: run_sleeve(ctx, fn, top_n=args.top, max_weight=args.maxw, min_liq=args.min_liq)
               for name, fn in STRATEGIES.items()}
    sleeve_eq = {k: v["equity"] for k, v in sleeves.items()}
    ens_eq, weights, rets = combine_inverse_vol(sleeve_eq)

    # ---- Bảng TOÀN KỲ: từng nhánh + ensemble + benchmark ----
    rows = [perf(sleeve_eq[k], k) for k in STRATEGIES]
    rows.append(perf(ens_eq, "ENSEMBLE"))
    rows.append(bh(ctx.vni, ens_eq.index, "VNINDEX b&h"))
    ew = ctx.close.pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
    rows.append(perf(ew.reindex(ens_eq.index).ffill().dropna(), "Equal-weight b&h"))
    ptable(rows, "TOÀN KỲ (gồm cả holdout)")
    for k in STRATEGIES:
        print(f"  {k:<7}: {trade_stat(sleeves[k]['trades'])} "
              f"| chi phí ~{round(sleeves[k]['total_cost']/sleeves[k]['avg_equity']*100,1)}% NAV-TB")
    print(f"  Trọng số inverse-vol TB: " +
          " ".join(f"{k}={weights[k].mean():.0%}" for k in STRATEGIES))

    # ---- Breakdown theo NĂM (stress-test xuyên chế độ) ----
    print("\n══ THEO NĂM — Return% (ensemble vs VNINDEX) ══")
    print(f"{'Năm':<6}{'ENS%':>9}{'VNI%':>9}{'ENS MDD%':>11}  nhánh nào dẫn dắt")
    print("-" * 60)
    for yr, grp in ens_eq.groupby(ens_eq.index.year):
        if len(grp) < 5:
            continue
        ens_r = round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 1)
        ens_mdd = round((grp / grp.cummax() - 1).min() * 100, 1)
        vni_y = ctx.vni.reindex(grp.index).ffill()
        vni_r = round((vni_y.iloc[-1] / vni_y.iloc[0] - 1) * 100, 1)
        # nhánh đóng góp nhiều nhất năm đó
        contrib = {k: (sleeve_eq[k].reindex(grp.index).ffill().iloc[-1] /
                       sleeve_eq[k].reindex(grp.index).ffill().iloc[0] - 1)
                   for k in STRATEGIES}
        lead = max(contrib, key=contrib.get)
        print(f"{yr:<6}{ens_r:>9}{vni_r:>9}{ens_mdd:>11}  {lead} (+{contrib[lead]*100:.0f}%)")

    # ---- Tương quan: ngày thường vs ngày STRESS (kiểm tra "lỗ ít" có vỡ không) ----
    print("\n══ TƯƠNG QUAN GIỮA CÁC NHÁNH (kiểm tra diversification có vỡ khi sập) ══")
    vni_ret = ctx.vni.pct_change(fill_method=None).reindex(rets.index).fillna(0)
    stress = vni_ret < -0.015            # phiên VNINDEX giảm > 1.5%
    keys = list(STRATEGIES)
    print(f"{'cặp':<16}{'mọi phiên':>12}{'phiên STRESS':>15}")
    print("-" * 43)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ca = rets[keys[a]].corr(rets[keys[b]])
            cs = rets.loc[stress, keys[a]].corr(rets.loc[stress, keys[b]])
            print(f"{keys[a]+'–'+keys[b]:<16}{ca:>12.2f}{cs:>15.2f}")
    print("  → đọc 2 cột: nếu STRESS >> mọi-phiên thì diversification vỡ lúc sập;")
    print("    nếu STRESS ≈ hoặc < (do regime-gate đẩy về tiền mặt) thì 'lỗ ít' giữ được.")

    # ---- HOLDOUT KHOÁ CỨNG (báo cáo riêng, tiêu một lần) ----
    ho = pd.Timestamp(args.holdout)
    design_eq, hold_eq = ens_eq[ens_eq.index < ho], ens_eq[ens_eq.index >= ho]
    if len(hold_eq) > 20:
        ptable([
            perf(design_eq, "ENS thiết kế"),
            bh(ctx.vni, design_eq.index, "VNI thiết kế"),
            perf(hold_eq, "ENS HOLDOUT"),
            bh(ctx.vni, hold_eq.index, "VNI HOLDOUT"),
        ], f"HOLDOUT KHOÁ CỨNG: thiết kế (<{ho.date()}) vs holdout (≥{ho.date()})")
        print("  ⚠️ Holdout chỉ trung thực ở LẦN CHẠY ĐẦU. Đã nhìn rồi thì nó 'cháy'.")

    if args.data:
        print("\n⚠️ Universe mở rộng (HOSE đang niêm yết) → đã khử phần lớn MEMBERSHIP")
        print("   LOOK-AHEAD, NHƯNG vẫn THIẾU mã đã hủy niêm yết → delisting bias còn lại.")
        print("   Vẫn là cận trên (nhẹ hơn). Cổng cuối cùng: paper trade vài tháng.")
    else:
        print("\n⚠️ Universe = VN100 hiện tại → còn SURVIVORSHIP BIAS NẶNG (membership")
        print("   look-ahead + delisting). Mọi số là cận trên lạc quan. Chạy --data để so.")


if __name__ == "__main__":
    main()
