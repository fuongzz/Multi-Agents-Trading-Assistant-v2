"""Automate market data, research parquet, and demo artifact updates.

This runner is meant for the demo/prototype environment:

* During VN morning and afternoon sessions, refresh open holdings with live
  prices so the layered demo keeps marking open positions to market.
* After market close, update local OHLCV/index parquet files, rebuild the
  research parquet inputs, validate data quality, and regenerate production
  demo reports.

It deliberately does not send broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as day_time, timedelta
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagents_trading_assistant.fetcher import (  # noqa: E402
    get_live_price,
    get_vn100_symbols,
    get_vnindex,
)
from scripts.update_demo_open_holdings_live import (  # noqa: E402
    BASELINE_BACKTEST_DIR,
    DEFAULT_DEMO_DIR,
    _market_session_state,
    _refresh_once,
)


RUNTIME_DIR = REPO_ROOT / "data" / "runtime"
LOG_DIR = REPO_ROOT / "logs"
STATE_PATH = RUNTIME_DIR / "auto_update_data_state.json"
LOCK_PATH = RUNTIME_DIR / "auto_update_data.lock"
DEFAULT_DAILY_TIME = "15:35"
OHLCV_MASTER = REPO_ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
INDEX_MASTER = REPO_ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
LIVE_MARKET_JSON = RUNTIME_DIR / "live_market_snapshot.json"
LIVE_MARKET_HTML = DEFAULT_DEMO_DIR / "00_live_market_snapshot.html"


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    required: bool = True


def _now() -> datetime:
    return datetime.now()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _append_log(path: Path, line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _parse_hhmm(value: str) -> day_time:
    hour, minute = value.split(":", 1)
    return day_time(int(hour), int(minute))


def _today_is_trading_day() -> bool:
    return _now().weekday() < 5


def _latest_parquet_date(path: Path) -> date | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["date"])
        if frame.empty:
            return None
        return pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize().max().date()
    except Exception:
        return None


def _verify_today_close_available() -> dict[str, Any]:
    target = _now().date()
    ohlcv_date = _latest_parquet_date(OHLCV_MASTER)
    index_date = _latest_parquet_date(INDEX_MASTER)
    ok = True
    missing: list[str] = []
    if _today_is_trading_day():
        if ohlcv_date != target:
            ok = False
            missing.append(f"ohlcv_master latest={ohlcv_date}")
        if index_date != target:
            ok = False
            missing.append(f"index_master latest={index_date}")
    return {
        "ok": ok,
        "target_date": target.isoformat(),
        "ohlcv_latest_date": ohlcv_date.isoformat() if ohlcv_date else None,
        "index_latest_date": index_date.isoformat() if index_date else None,
        "missing": missing,
    }


def _resolve_demo_dir(raw: str) -> Path:
    if raw and raw.lower() != "latest":
        return Path(raw)
    candidates = sorted(
        (REPO_ROOT / "reports").glob("layered_pipeline_demo_*"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else DEFAULT_DEMO_DIR


def _acquire_lock(max_age_hours: int = 12) -> bool:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(payload.get("created_at")))
            age_hours = (_now() - created_at).total_seconds() / 3600.0
        except Exception:
            age_hours = max_age_hours + 1
        if age_hours <= max_age_hours:
            return False
    LOCK_PATH.write_text(
        json.dumps({"pid": os.getpid(), "created_at": _now().isoformat(timespec="seconds")}, indent=2),
        encoding="utf-8",
    )
    return True


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _run_stage(stage: Stage, state: dict[str, Any]) -> dict[str, Any]:
    started = _now()
    log_path = LOG_DIR / f"auto_update_data.{started:%Y%m%d}.log"
    display_command = " ".join(stage.command)
    print(f"[auto_update] start stage={stage.name} command={display_command}", flush=True)
    _append_log(log_path, f"\n[{started.isoformat(timespec='seconds')}] START {stage.name}: {display_command}")

    proc = subprocess.run(
        stage.command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    finished = _now()
    duration = round((finished - started).total_seconds(), 2)
    if proc.stdout:
        _append_log(log_path, proc.stdout)
    if proc.stderr:
        _append_log(log_path, "[stderr]\n" + proc.stderr)
    _append_log(
        log_path,
        f"[{finished.isoformat(timespec='seconds')}] END {stage.name}: returncode={proc.returncode} duration={duration}s",
    )

    result = {
        "name": stage.name,
        "command": stage.command,
        "returncode": proc.returncode,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_seconds": duration,
        "log": str(log_path),
    }
    state.setdefault("stage_history", []).append(result)
    state["last_stage"] = result
    _save_state(state)

    if proc.returncode != 0 and stage.required:
        raise RuntimeError(f"Stage failed: {stage.name}. See {log_path}")
    return result


def _build_daily_stages(args: argparse.Namespace, run_date: date) -> list[Stage]:
    py = sys.executable
    ohlcv = "multiagents_trading_assistant/data/ohlcv_master.parquet"
    production_dir = args.production_dir or str(REPO_ROOT / "reports" / f"production_demo_{run_date:%Y-%m-%d}")
    layered_dir = args.layered_demo_dir or str(REPO_ROOT / "reports" / f"layered_pipeline_demo_{run_date:%Y-%m-%d}")
    quality_report = str(REPO_ROOT / "backtest_results" / "data_quality" / f"data_quality_report_{run_date:%Y%m%d}.json")

    stages = [
        Stage("update_ohlcv_master", [py, "scripts/update_ohlcv_daily.py"]),
        Stage("build_index_master", [py, "scripts/build_index_master.py"]),
    ]
    if not args.skip_research:
        stages.extend(
            [
                Stage(
                    "build_money_cycle_parquet",
                    [
                        py,
                        "-m",
                        "multiagents_trading_assistant.research.money_cycle.cli",
                        "--input",
                        ohlcv,
                        "--output",
                        "data/research/money_cycle",
                    ],
                ),
                Stage(
                    "build_smart_money_parquet",
                    [
                        py,
                        "-m",
                        "multiagents_trading_assistant.research.smart_money_trace.cli",
                        "--input",
                        ohlcv,
                        "--output",
                        "data/research/smart_money_trace",
                    ],
                ),
                Stage(
                    "build_sector_rotation_parquet",
                    [py, "-m", "multiagents_trading_assistant.research.sector_rotation.compute_sector_rotation"],
                ),
                Stage(
                    "build_market_regime_parquet",
                    [py, "-m", "multiagents_trading_assistant.research.sector_rotation.compute_market_regime"],
                ),
                Stage(
                    "build_daily_market_state_report",
                    [py, "scripts/generate_daily_market_state_report.py", "--with-live-context"],
                ),
            ]
        )
    stages.append(
        Stage(
            "validate_data_quality",
            [py, "scripts/validate_data_quality.py", "--report", quality_report],
            required=not args.allow_quality_warn,
        )
    )
    if not args.skip_demo:
        stages.extend(
            [
                Stage(
                    "build_production_demo",
                    [
                        py,
                        "-m",
                        "scripts.run_production_dry_run",
                        "--out-dir",
                        production_dir,
                        "--lookback-days",
                        str(args.lookback_days),
                    ],
                ),
                Stage(
                    "build_layered_demo",
                    [
                        py,
                        "-m",
                        "scripts.export_pipeline_layer_demo",
                        "--production-dir",
                        production_dir,
                        "--out-dir",
                        layered_dir,
                        "--lookback-days",
                        str(args.lookback_days),
                    ],
                ),
            ]
        )
    return stages


def run_daily_pipeline(args: argparse.Namespace, *, reason: str) -> None:
    if not _acquire_lock():
        print("[auto_update] daily pipeline skipped because another run is locked", flush=True)
        return

    state = _load_state()
    run_date = _now().date()
    run_id = f"{run_date:%Y%m%d}_{_now():%H%M%S}"
    state["current_run"] = {
        "run_id": run_id,
        "reason": reason,
        "started_at": _now().isoformat(timespec="seconds"),
    }
    _save_state(state)

    try:
        for stage in _build_daily_stages(args, run_date):
            _run_stage(stage, state)
        close_check = _verify_today_close_available()
        state["last_close_data_check"] = {
            "at": _now().isoformat(timespec="seconds"),
            **close_check,
        }
        _save_state(state)
        if args.require_today_close and not close_check["ok"]:
            raise RuntimeError(
                "Today close data is not available yet: "
                + "; ".join(close_check.get("missing") or [])
            )
        if args.live_holdings:
            try:
                refresh_live_holdings(args)
            except Exception as exc:
                print(f"[auto_update] post-close live holdings refresh failed: {type(exc).__name__}: {exc}", flush=True)
        if args.live_market_snapshot:
            try:
                refresh_live_market_snapshot(args)
            except Exception as exc:
                print(f"[auto_update] post-close live market snapshot failed: {type(exc).__name__}: {exc}", flush=True)
        refresh_layered_market_context(args)
        state = _load_state()
        state["last_successful_daily_date"] = run_date.isoformat()
        state["last_successful_daily_run_id"] = run_id
        state["last_successful_daily_at"] = _now().isoformat(timespec="seconds")
        state["current_run"] = None
        _save_state(state)
        print(f"[auto_update] daily pipeline completed run_id={run_id}", flush=True)
    except Exception as exc:
        state["last_failed_daily_date"] = run_date.isoformat()
        state["last_failed_daily_run_id"] = run_id
        state["last_failed_daily_at"] = _now().isoformat(timespec="seconds")
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        state["current_run"] = None
        _save_state(state)
        print(f"[auto_update] daily pipeline failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        _release_lock()


def refresh_live_holdings(args: argparse.Namespace) -> None:
    demo_dir = _resolve_demo_dir(args.demo_dir)
    if not demo_dir.exists():
        print(f"[auto_update] live refresh skipped; demo dir missing: {demo_dir}", flush=True)
        return
    result = _refresh_once(
        demo_dir,
        Path(args.backtest_dir),
        html_refresh_seconds=max(args.live_interval_seconds, 0),
    )
    state = _load_state()
    state["last_live_refresh"] = {
        "at": _now().isoformat(timespec="seconds"),
        "demo_dir": result["demo_dir"],
        "updated": result["updated"],
        "symbols": result["symbols"],
        "priced_at": result["priced_at"],
    }
    _save_state(state)
    print(
        f"[auto_update] live holdings refreshed updated={result['updated']}/{len(result['symbols'])} "
        f"demo_dir={result['demo_dir']}",
        flush=True,
    )


def _format_price(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return ""


def _latest_master_close_prices(symbols: list[str]) -> tuple[dict[str, float], str]:
    if not OHLCV_MASTER.exists():
        return {}, ""
    df = pd.read_parquet(OHLCV_MASTER, columns=["date", "symbol", "close"])
    if df.empty:
        return {}, ""
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    latest_date = df["date"].max()
    wanted = {str(symbol).upper() for symbol in symbols}
    latest = df[(df["date"] == latest_date) & (df["symbol"].astype(str).str.upper().isin(wanted))]
    prices = {
        str(row["symbol"]).upper(): float(row["close"])
        for _, row in latest.iterrows()
        if pd.notna(row.get("close"))
    }
    return prices, latest_date.date().isoformat()


def _latest_index_rows() -> list[dict[str, Any]]:
    if INDEX_MASTER.exists():
        df = pd.read_parquet(INDEX_MASTER)
        if not df.empty:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            cols = [col for col in ["date", "open", "high", "low", "close", "volume"] if col in df.columns]
            return df.tail(5)[cols].to_dict("records")
    vnindex = get_vnindex(n_days=5)
    if vnindex is not None and not vnindex.empty:
        return vnindex.tail(5).to_dict("records")
    return []


def refresh_live_market_snapshot(args: argparse.Namespace) -> None:
    symbols = get_vn100_symbols() if args.live_universe.lower() == "vn100" else [
        item.strip().upper() for item in args.live_symbols.split(",") if item.strip()
    ]
    in_session, session_label, _ = _market_session_state()
    close_date = ""
    if in_session:
        prices = get_live_price(symbols)
        price_mode = "realtime"
    else:
        prices, close_date = _latest_master_close_prices(symbols)
        price_mode = "latest_close"
    vnindex_rows = []
    try:
        vnindex_rows = _latest_index_rows()
    except Exception as exc:
        vnindex_rows = [{"error": f"{type(exc).__name__}: {exc}"}]

    now_iso = _now().isoformat(timespec="seconds")
    rows = [
        {"symbol": symbol, "live_price": prices.get(symbol)}
        for symbol in symbols
        if prices.get(symbol) is not None
    ]
    rows.sort(key=lambda item: str(item["symbol"]))
    payload = {
        "updated_at": now_iso,
        "universe": args.live_universe,
        "price_mode": price_mode,
        "session_label": session_label,
        "close_date": close_date,
        "requested_symbols": len(symbols),
        "priced_symbols": len(rows),
        "prices": rows,
        "vnindex_tail": vnindex_rows,
    }
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_MARKET_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    demo_dir = _resolve_demo_dir(args.demo_dir)
    html_path = demo_dir / "00_live_market_snapshot.html"
    price_rows = "\n".join(
        f"<tr><td>{escape(str(row['symbol']))}</td><td>{_format_price(row['live_price'])}</td></tr>"
        for row in rows
    )
    vni_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in ["date", "open", "high", "low", "close", "volume"])
        + "</tr>"
        for row in vnindex_rows
        if isinstance(row, dict)
    )
    html_path.write_text(
        f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{max(args.live_market_interval_seconds, 15)}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>00 Live Market Snapshot</title>
  <style>
    :root {{ --ink:#14213d; --muted:#5f6b7a; --line:#d9dee7; --bg:#f5f1e8; }}
    body {{ margin:0; padding:28px; font-family:Georgia,'Times New Roman',serif; background:var(--bg); color:var(--ink); }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin:24px 0 10px; font-size:20px; }}
    .note {{ color:var(--muted); font-family:Segoe UI,Arial,sans-serif; }}
    table {{ border-collapse:collapse; width:100%; background:white; font:13px Segoe UI,Arial,sans-serif; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #edf0f5; text-align:left; }}
    th {{ background:var(--ink); color:white; position:sticky; top:0; }}
    .wrap {{ max-height:70vh; overflow:auto; border:1px solid var(--line); }}
    a {{ color:var(--ink); }}
  </style>
</head>
<body>
  <h1>00 Live Market Snapshot</h1>
  <p class="note"><a href="index.html">Index</a> | Updated {escape(now_iso)} | Mode {escape(price_mode)} {escape(close_date)} | Priced {len(rows)}/{len(symbols)} symbols | Paper/data only.</p>
  <h2>VNINDEX Latest Daily Bars</h2>
  <div class="wrap"><table><thead><tr><th>date</th><th>open</th><th>high</th><th>low</th><th>close</th><th>volume</th></tr></thead><tbody>{vni_rows}</tbody></table></div>
  <h2>{escape(args.live_universe.upper())} Live Prices</h2>
  <div class="wrap"><table><thead><tr><th>symbol</th><th>live_price</th></tr></thead><tbody>{price_rows}</tbody></table></div>
  <script src="sortable_tables.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    state = _load_state()
    state["last_live_market_snapshot"] = {
        "at": now_iso,
        "html": str(html_path),
        "json": str(LIVE_MARKET_JSON),
        "requested_symbols": len(symbols),
        "priced_symbols": len(rows),
        "price_mode": price_mode,
        "close_date": close_date,
    }
    _save_state(state)
    print(f"[auto_update] live market snapshot {len(rows)}/{len(symbols)} -> {html_path}", flush=True)


def refresh_layered_market_context(args: argparse.Namespace) -> None:
    demo_dir = _resolve_demo_dir(args.demo_dir)
    if not demo_dir.exists():
        print(f"[auto_update] layered market context skipped; demo dir missing: {demo_dir}", flush=True)
        return
    try:
        from multiagents_trading_assistant.realtime_shadow_dual_pipeline import (
            DATA_DIR,
            _enrich_vnindex_context,
            _safe_read_recent_parquet,
            _write_layer_table,
        )

        vnindex = _safe_read_recent_parquet(DATA_DIR / "index_master.parquet", None, days=260)
        if "symbol" in vnindex.columns:
            vnindex = vnindex[vnindex["symbol"].astype(str).str.upper().eq("VNINDEX")]
        vnindex = _enrich_vnindex_context(vnindex, days=30)
        _write_layer_table(
            demo_dir,
            "01_vnindex",
            "01 VNINDEX Market Context",
            "VNINDEX OHLCV enriched with returns, MA distance, volume ratio, money-cycle breadth and regime state.",
            vnindex,
        )
    except Exception as exc:
        print(f"[auto_update] layered VNINDEX context refresh failed: {type(exc).__name__}: {exc}", flush=True)

    try:
        from scripts.build_feature_engine_visuals import build as build_feature_visuals

        build_feature_visuals()
    except Exception as exc:
        print(f"[auto_update] feature visuals refresh failed: {type(exc).__name__}: {exc}", flush=True)

    try:
        from scripts.build_vnindex_visuals import build as build_vnindex_visuals

        build_vnindex_visuals(demo_dir)
    except Exception as exc:
        print(f"[auto_update] vnindex visuals refresh failed: {type(exc).__name__}: {exc}", flush=True)


def _should_run_daily(args: argparse.Namespace, state: dict[str, Any], daily_time: day_time) -> bool:
    if args.force_daily:
        return True
    now = _now()
    if now.weekday() >= 5:
        return False
    if now.time() < daily_time:
        return False
    if state.get("last_successful_daily_date") == now.date().isoformat():
        return False
    if state.get("last_failed_daily_date") == now.date().isoformat():
        if not args.retry_failed_daily:
            return False
        retry_until = _parse_hhmm(args.eod_retry_until)
        if now.time() > retry_until:
            return False
        try:
            last_failed = datetime.fromisoformat(str(state.get("last_failed_daily_at")))
            if (now - last_failed) < timedelta(minutes=max(1, args.daily_retry_minutes)):
                return False
        except Exception:
            pass
    return True


def loop(args: argparse.Namespace) -> None:
    daily_time = _parse_hhmm(args.daily_time)
    print(
        f"[auto_update] daemon started daily_time={args.daily_time} "
        f"live_interval={args.live_interval_seconds}s poll={args.poll_seconds}s",
        flush=True,
    )
    last_live_attempt = 0.0
    last_market_attempt = 0.0

    while True:
        state = _load_state()
        if _should_run_daily(args, state, daily_time):
            try:
                run_daily_pipeline(args, reason="scheduled_after_close")
            except Exception as exc:
                print(f"[auto_update] daily attempt will retry later if allowed: {type(exc).__name__}: {exc}", flush=True)

        if args.live_holdings:
            in_session, session_label, _ = _market_session_state()
            due = time.monotonic() - last_live_attempt >= args.live_interval_seconds
            if in_session and due:
                last_live_attempt = time.monotonic()
                try:
                    refresh_live_holdings(args)
                except Exception as exc:
                    state = _load_state()
                    state["last_live_error"] = {
                        "at": _now().isoformat(timespec="seconds"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _save_state(state)
                    print(f"[auto_update] live refresh failed: {type(exc).__name__}: {exc}", flush=True)
            elif due:
                last_live_attempt = time.monotonic()
                print(f"[auto_update] live holdings paused={session_label}", flush=True)

        if args.live_market_snapshot:
            in_session, session_label, _ = _market_session_state()
            due = time.monotonic() - last_market_attempt >= args.live_market_interval_seconds
            if in_session and due:
                last_market_attempt = time.monotonic()
                try:
                    refresh_live_market_snapshot(args)
                except Exception as exc:
                    state = _load_state()
                    state["last_live_market_error"] = {
                        "at": _now().isoformat(timespec="seconds"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _save_state(state)
                    print(f"[auto_update] live market snapshot failed: {type(exc).__name__}: {exc}", flush=True)
            elif due:
                last_market_attempt = time.monotonic()
                print(f"[auto_update] live market snapshot paused={session_label}", flush=True)

        time.sleep(max(5, args.poll_seconds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automate live demo prices and project parquet updates.")
    parser.add_argument("--once", action="store_true", help="Run one daily pipeline now and exit.")
    parser.add_argument("--refresh-live-once", action="store_true", help="Refresh demo open holdings once and exit.")
    parser.add_argument("--refresh-live-market-once", action="store_true", help="Refresh live market snapshot once and exit.")
    parser.add_argument("--force-daily", action="store_true", help="Run the daily pipeline even if it already succeeded today.")
    parser.add_argument("--retry-failed-daily", action="store_true", default=True, help="Retry after a failed daily run on the same date.")
    parser.add_argument("--no-retry-failed-daily", dest="retry_failed_daily", action="store_false")
    parser.add_argument("--require-today-close", action="store_true", default=True, help="After close, keep retrying until OHLCV and VNINDEX include today.")
    parser.add_argument("--no-require-today-close", dest="require_today_close", action="store_false")
    parser.add_argument("--eod-retry-until", default="18:30", help="Latest local time HH:MM to retry waiting for today's close data.")
    parser.add_argument("--daily-retry-minutes", type=int, default=10)
    parser.add_argument("--daily-time", default=DEFAULT_DAILY_TIME, help="Local VN time HH:MM for daily parquet/demo update.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--live-interval-seconds", type=int, default=15)
    parser.add_argument("--live-market-interval-seconds", type=int, default=60)
    parser.add_argument("--live-universe", default="vn100", choices=["vn100", "custom"])
    parser.add_argument("--live-symbols", default="", help="Comma-separated symbols when --live-universe custom.")
    parser.add_argument("--demo-dir", default="latest", help="Layered demo dir for live holdings, or 'latest'.")
    parser.add_argument("--backtest-dir", default=str(BASELINE_BACKTEST_DIR))
    parser.add_argument("--production-dir", default="", help="Override production demo output dir.")
    parser.add_argument("--layered-demo-dir", default="", help="Override layered demo output dir.")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--skip-research", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--no-live-holdings", dest="live_holdings", action="store_false")
    parser.add_argument("--no-live-market-snapshot", dest="live_market_snapshot", action="store_false")
    parser.add_argument("--allow-quality-warn", action="store_true", help="Do not fail the pipeline when data quality returns non-zero.")
    parser.set_defaults(live_holdings=True, live_market_snapshot=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.refresh_live_once:
        refresh_live_holdings(args)
        return
    if args.refresh_live_market_once:
        refresh_live_market_snapshot(args)
        return
    if args.once:
        run_daily_pipeline(args, reason="manual_once")
        return
    loop(args)


if __name__ == "__main__":
    main()
