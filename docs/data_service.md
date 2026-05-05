# Data Service

Agents, screeners, backtests, and orchestrators must not import vendor data
SDKs directly. They should call `multiagents_trading_assistant.services.data_service`,
which re-exports the stable public functions from `fetcher.py`.

## Current Design

`fetcher.py` owns cache, schema normalization, and fallback behavior. Vendor
calls are delegated to the internal provider layer:

```text
multiagents_trading_assistant/data/
  providers/base.py             # vendor-neutral contract
  providers/vnstock_provider.py # current vnstock_data Golden implementation
  repository.py                 # provider factory
```

The default provider is `vnstock_data` Golden:

```text
MATA_DATA_PROVIDER=vnstock
```

This preserves the product path: when we build an in-house data library, it can
implement the same `DataProvider` contract without changing agents, screeners,
or backtest modules.

## Public API

```python
get_ohlcv(symbol: str, n_days: int = 200) -> pd.DataFrame
get_ohlcv_history(symbol: str, start: str, end: str | None = None, resolution: str = "1D") -> pd.DataFrame
get_ohlcv_batch(symbols: list[str], n_days: int = 200) -> dict[str, pd.DataFrame]
get_vnindex(n_days: int = 200) -> pd.DataFrame
get_vnmidcap(n_days: int = 200) -> pd.DataFrame
get_vn30_symbols() -> list[str]
get_vn100_symbols() -> list[str]
get_all_symbols() -> list[str]
get_liquid_symbols(min_avg_vol: int = 500_000, n_days: int = 20) -> list[str]
get_price_board(symbols: list[str]) -> pd.DataFrame
get_live_price(symbols: list[str]) -> dict[str, float]
get_fundamentals(symbol: str) -> dict
get_foreign_flow(symbol: str, n_days: int = 20) -> dict
get_global_macro() -> dict
get_vn_macro() -> dict
```

## Stable Schemas

OHLCV always returns:

```text
date, open, high, low, close, volume
```

Fundamentals always returns:

```text
pe, pb, roe, eps, revenue_growth, profit_growth, industry
```

Foreign flow always returns:

```text
room_usage_pct, net_flow_5d, net_flow_20d, flow_history
```

Price board always returns:

```text
symbol, price, listed_share, current_room, total_room,
foreign_buy_vol, foreign_sell_vol, foreign_buy_val, foreign_sell_val
```

## Source Priority

| Data | Primary | Fallback |
| --- | --- | --- |
| Vietnam OHLCV | `vnstock_data.Market` | DNSE OHLCV where available |
| VN30/VN100/HOSE symbols | `vnstock_data.Reference` | static VN100 list |
| Price board | `vnstock_data.Market` | empty DataFrame |
| Fundamentals | `vnstock_data.Fundamental` | FiinQuantX, static VN30 |
| Foreign flow | `vnstock_data.Market.foreign_flow` | FiinQuantX, price board snapshot |
| Live broker price | DNSE WebSocket/REST | `vnstock_data` price board |
| Global macro | yfinance | cached/stale output |
| News | existing crawlers | migrate to `vnstock_news` where coverage is enough |

## Usage Rule

Use the stable project API:

```python
from multiagents_trading_assistant.services.data_service import get_ohlcv

df = get_ohlcv("VCB", n_days=200)
```

Do not use this in agents or screeners:

```python
from vnstock import ...
from vnstock_data import ...
```
