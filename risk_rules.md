# Risk Layer Specification

This document defines the risk layer for the AI Trading Assistant.

The risk layer is rule-based and must not use LLM judgement. LLM agents may
produce analysis, explanations, or candidate actions, but the risk layer is the
final deterministic gate before any order is accepted by backtest, paper
trading, or live execution.

Primary sources:

- FPTS HOSE Trading Regulations:
  https://www.fpts.com.vn/customer-service/securities-trading/stock-trading-guide/trading-regulations/hose-trading-regulations/
- FPTS Securities Trading Settlement:
  https://www.fpts.com.vn/customer-service/securities-trading/securities-trading-settlement/

Scope:

- Mandatory market rules in this version are HOSE-focused because the source
  above is the HOSE regulation page.
- HNX/UPCOM rules must be added from their own source pages before the system
  treats them as first-class execution venues.

---

## 1. Risk Layer Contract

Risk receives a proposed action from the strategy/trader layer:

```python
{
    "symbol": "VCB",
    "exchange": "HOSE",
    "side": "BUY" | "SELL" | "HOLD",
    "order_type": "LO" | "ATO" | "ATC" | "MTL",
    "quantity": int,
    "limit_price": float | None,
    "signal_time": "Asia/Ho_Chi_Minh timestamp",
    "reference_price": float,
    "ceiling_price": float | None,
    "floor_price": float | None,
    "foreign_room_qty": int | None,
    "position_qty": int,
    "settled_sellable_qty": int,
    "portfolio_state": dict,
}
```

Risk returns:

```python
{
    "final_action": "BUY" | "SELL" | "HOLD",
    "allowed": bool,
    "override_reason": str | None,
    "warnings": list[str],
    "sizing_modifier": float,
    "source_rule_ids": list[str],
}
```

If any mandatory rule fails, `allowed=False` and `final_action="HOLD"`.

---

## 2. Exchange-Mandated Rules From FPTS

These are not strategy preferences. They represent market structure constraints
that must be enforced or simulated.

### MR-HOSE-001: Trading Session Gate

Source: FPTS HOSE Trading Regulations, section 1.1.

Regular board-lot/odd-lot order matching sessions:

| Session | Time | Allowed orders |
|---|---:|---|
| ATO periodic call auction | 09:00-09:15 | ATO, LO |
| Continuous morning | 09:15-11:30 | LO, MTL |
| Break | 11:30-13:00 | No normal order matching |
| Continuous afternoon | 13:00-14:30 | LO, MTL |
| ATC periodic call auction | 14:30-14:45 | ATC, LO |

Risk behavior:

- Reject BUY/SELL outside valid order-entry sessions.
- Reject MTL outside continuous sessions.
- Reject ATO outside 09:00-09:15.
- Reject ATC outside 14:30-14:45.
- For EOD strategy, prefer generating orders after close and simulate fill at
  next eligible session.

Implementation status:

- Partially represented in `risk_rules.md` old Rule 5.
- Backtest should not assume a fill during break or after market close.

### MR-HOSE-002: Odd-Lot LO-Only Rule

Source: FPTS HOSE Trading Regulations, section 1.1 and 2.b.

Odd-lot orders only accept LO.

Risk behavior:

- If quantity is 1-99 shares, reject ATO/ATC/MTL.
- Either convert odd-lot orders to LO or hold the signal.
- Prefer board-lot sizing in the strategy layer to avoid odd-lot execution.

Implementation status:

- Missing.

### MR-HOSE-003: Restricted Securities Session Rule

Source: FPTS HOSE Trading Regulations, section 1.3.

Restricted securities trade through ATO, multiple 15-minute periodic call
auction sessions, and ATC. Continuous trading assumptions do not apply.

Risk behavior:

- If security status is restricted, disable normal continuous-session fill
  assumptions.
- Only allow order types and timestamps valid for restricted securities.
- If the system cannot determine restricted status, mark the order as
  `status_unknown` and use conservative sizing or hold.

Implementation status:

- Missing. Requires security status data.

### MR-HOSE-004: Order Type Validity

Source: FPTS HOSE Trading Regulations, section 2.

Order type constraints:

- LO is allowed during continuous trading and periodic call auctions.
- ATO is only for the opening call auction and expires after ATO.
- ATC is only for the closing call auction and expires after ATC.
- MTL is only valid during continuous trading.

Risk behavior:

- Validate order type against session.
- In paper/live trading, expire unfilled ATO/ATC intent after the session.
- In backtest, do not carry ATO/ATC orders into later bars.

Implementation status:

- Partially missing. Current backtest uses next-open fill but does not model
  order-type expiry explicitly.

### MR-HOSE-005: Price-Time Priority

Source: FPTS HOSE Trading Regulations, section 3.

Matching priority:

- Higher buy prices have priority.
- Lower sell prices have priority.
- At the same price, earlier orders have priority.

Risk behavior:

- Backtest should not assume guaranteed fills at crowded ceiling/floor prices.
- Paper/live should treat late LO orders at the same price as lower priority.
- For conservative backtest, apply additional slippage or no-fill probability
  when buying near ceiling or selling near floor.

Implementation status:

- Only approximated through slippage and price-band checks.

### MR-HOSE-006: Continuous Trading Price Rule

Source: FPTS HOSE Trading Regulations, section 4.2.

Continuous trades execute at the available counterparty order price.

Risk behavior:

- A marketable buy should fill near ask, not at last price.
- A marketable sell should fill near bid, not at last price.
- If bid/ask is unavailable, backtest must use conservative slippage.

Implementation status:

- Approximated by `ExecutionConfig.slippage_bps`.

### MR-HOSE-007: Order Modification/Cancellation Lockout

Source: FPTS HOSE Trading Regulations, section 6.

Important constraints:

- During continuous trading, an LO can be modified or canceled, but price and
  quantity cannot both be modified at the same time.
- Increasing quantity or changing price resets priority.
- During ATO/ATC, orders cannot be canceled or modified.
- During periodic call auction sessions, orders cannot be canceled or modified
  in the last 5 minutes.

Risk behavior:

- Live order manager must not issue cancel/modify commands during locked
  sessions.
- If strategy changes its mind during ATO/ATC, risk must hold the change until
  the next modifiable window.
- Backtest should not assume a stop can be modified during a locked auction.

Implementation status:

- Missing.

### MR-HOSE-008: Board-Lot, Odd-Lot, And Max Order Quantity

Source: FPTS HOSE Trading Regulations, section 7.1.

Rules:

- Board-lot quantity must be a multiple of 100 shares.
- Maximum board-lot order size is 500,000 shares.
- Odd-lot quantity is 1-99 shares.

Risk behavior:

- Round board-lot orders down to a multiple of 100.
- Reject or split orders above 500,000 shares.
- Detect odd-lot orders and enforce LO-only behavior.

Implementation status:

- Missing.

### MR-HOSE-009: Tick Size

Source: FPTS HOSE Trading Regulations, section 7.2.

Order-matching tick sizes:

| Price range | Tick size |
|---:|---:|
| <= 10,000 VND | 10 VND |
| 10,000-49,950 VND | 50 VND |
| >= 50,000 VND | 100 VND |

ETF certificates and warrants use a VND10 tick size across all price ranges.

Risk behavior:

- Normalize every limit price to a valid tick.
- Reject prices that are not tick-aligned if normalization would materially
  change the trade.
- Backtest fill prices should be rounded to valid ticks.

Implementation status:

- Missing. Current code rounds to 2 decimals or whole VND in different places,
  which is not the same as HOSE tick-size compliance.

### MR-HOSE-010: Trading Band

Source: FPTS HOSE Trading Regulations, section 8.

Rules:

- Normal HOSE trading band is +/-7% around reference price.
- +/-20% applies for special cases such as first trading date, resumption after
  suspension of at least 25 consecutive trading days, specified ex-rights
  cases, cash dividend greater than or equal to prior close, and first trading
  day of a split listed company.

Risk behavior:

- Reject or no-fill any order price outside floor/ceiling.
- If special +/-20% status is unknown, default to conservative normal band
  unless exchange-provided ceiling/floor is available.
- Use actual ceiling/floor from market data when available; do not recompute if
  official values are provided.

Implementation status:

- Partially implemented in `backtest/execution.py` as `price_band_pct=7.0`.
- Missing special +/-20% handling.

### MR-HOSE-011: Ceiling/Floor Formula

Source: FPTS HOSE Trading Regulations, section 9.

For stocks, closed-end fund certificates, and ETF fund certificates:

- Ceiling = reference price + reference price * trading band.
- Floor = reference price - reference price * trading band.
- Edge cases are adjusted by tick size when ceiling/floor equals reference
  price.

Risk behavior:

- Prefer official ceiling/floor fields from market data.
- If recomputing, apply tick-size adjustment rules.
- Ensure all order prices stay within official floor/ceiling.

Implementation status:

- Partially implemented as a simple percentage band check.
- Missing tick-size adjusted floor/ceiling computation.

### MR-HOSE-012: Reference Price

Source: FPTS HOSE Trading Regulations, section 10.

Normal reference price is the prior trading day's closing price. Special rules
exist for new listings, ex-rights dates, and securities suspended for more than
25 sessions.

Risk behavior:

- Use exchange-provided reference price where possible.
- Do not infer reference price from raw close if corporate actions or suspension
  status are present and official values are available.
- Backtest must use adjusted OHLCV consistently; otherwise ceiling/floor checks
  may be wrong around ex-rights dates.

Implementation status:

- Partially approximated by using previous close.
- Missing corporate-action/suspension awareness.

### MR-HOSE-013: Foreign Investor Room

Source: FPTS HOSE Trading Regulations, section 11.

Rules:

- Foreign buy volume reduces current foreign room immediately when the buy order
  enters the system.
- Foreign sell volume increases current room after settlement.
- A foreign buy order is rejected if current foreign room is less than order
  quantity.
- Buy order cancellation or quantity reduction releases room immediately.

Risk behavior:

- For foreign-investor mode, reject BUY if `foreign_room_qty < quantity`.
- For domestic-investor mode, keep foreign room as an information/warning
  factor, not a hard reject.
- Existing project rule using room percentage should be split into two modes:
  regulatory hard reject for foreign accounts, liquidity/sentiment warning for
  domestic accounts.

Implementation status:

- Partially represented by old Rule 2.
- Needs account-type awareness.

### MR-HOSE-014: No Simultaneous Buy And Sell In Same Periodic Auction

Source: FPTS HOSE Trading Regulations, section 12.

Investors are prohibited from placing simultaneous buy and sell orders for the
same security within a single periodic call auction session, except certain
valid carried-over orders.

Risk behavior:

- Reject a BUY if there is an active SELL for the same symbol in the same
  periodic auction session.
- Reject a SELL if there is an active BUY for the same symbol in the same
  periodic auction session.
- Order manager must track active auction-session orders by symbol.

Implementation status:

- Missing.

### MR-SETTLE-001: Settlement And Sellable Quantity

Source: FPTS Securities Trading Settlement.

For shares, fund certificates, and covered warrants, settlement is before 13:00
on T+2. The actual settlement time depends on VSDC confirmation.

Risk behavior:

- Shares bought on T are not sellable until settlement on T+2.
- In live/paper trading, use broker/VSDC sellable quantity when available.
- In backtest, model per-lot sellability by buy date and block sells before
  settlement.
- After settlement time uncertainty, conservative mode should allow selling only
  from the next full session unless reliable sellable quantity is available.

Implementation status:

- Implemented in `backtest/execution.py` as `settlement_days=2`.
- Old `risk_rules.md` T+3 wording was outdated and is replaced by T+2.

---

## 3. Project Internal Risk Rules

These are not directly mandated by FPTS. They are project-level survival rules.

### IR-001: VN-Index Circuit Breaker

If VN-Index drops more than 3% intraday, block new BUY orders.

Reason:

- Internal market-regime safety rule.
- Not a HOSE regulation in the FPTS source.

Default behavior:

- BUY -> HOLD
- SELL/REDUCE remains allowed unless blocked by market/session/settlement rules.

### IR-002: Portfolio Exposure Cap

Defaults from `backtest/portfolio.py`:

- Max open positions: 5
- Max total exposure: 50% NAV
- Max total risk if all stops hit: 6% NAV
- Max single position: 10% NAV
- Max sector exposure: 20% NAV

Risk behavior:

- Reject new entries or adds that exceed these caps.
- Never let LLM override these caps.

### IR-003: Stop-Loss Required Before Entry

Every BUY must have a valid stop-loss or invalidation level.

Risk behavior:

- Reject BUY if stop-loss is missing, zero, or above/equal to entry price.
- Position size must be computed from distance to stop and portfolio caps.

### IR-004: No Short Selling In Cash Equity Mode

The system trades cash equities only.

Risk behavior:

- Reject SELL quantity above settled sellable quantity.
- Reject net-short state.

Note:

- This is both a project constraint and a practical cash-equity constraint. The
  FPTS page used here does not define a short-selling framework for normal HOSE
  cash equity trading.

### IR-005: Data Integrity Gate

Risk must hold all orders if required market fields are missing:

- reference price
- floor/ceiling or enough data to compute them
- timestamp/session
- OHLCV bar used by the signal
- sellable quantity for sell orders

Risk behavior:

- Missing mandatory data -> HOLD.
- Missing optional data -> warning and conservative sizing.

### IR-006: LLM Isolation

LLM output cannot bypass risk.

Risk behavior:

- LLM can propose action, entry zone, thesis, and warnings.
- Risk decides whether the proposed action is executable.
- Every override must be logged with rule id and source.

---

## 4. Enforcement Priority

Apply rules in this order:

1. Data integrity gate.
2. Account and position legality: sellable quantity, no net short.
3. Trading session and order-type validity.
4. Tick size, board-lot, max order quantity.
5. Price band, floor/ceiling, reference price.
6. Foreign room, if account is foreign-investor mode.
7. Auction/order conflict rules.
8. Portfolio exposure/risk caps.
9. Internal market-regime rules.
10. Strategy-specific stop/trailing/invalidation rules.

Mandatory market rules override internal sizing preferences.

---

## 5. Backtest Requirements

Backtest must simulate these minimum constraints:

- Signal at bar `i` cannot fill before bar `i+1`.
- Fill price must be inside official floor/ceiling.
- Fill price must be tick-aligned.
- Buy quantity must be board-lot aligned unless odd-lot mode is explicitly
  enabled.
- Sell quantity cannot exceed settled sellable quantity.
- T+2 settlement applies per buy lot.
- Slippage is required when bid/ask is unavailable.
- ATO/ATC orders expire after their auction session.
- No guaranteed fill at ceiling/floor; use conservative no-fill or extra
  slippage assumptions.

---

## 6. Paper/Live Requirements

Paper/live mode must log:

- proposed action from strategy/LLM
- final risk action
- rule ids evaluated
- failed rule ids
- source URL for exchange-mandated rules
- original order fields
- normalized order fields after tick/lot adjustment
- broker-reported sellable quantity
- broker-reported foreign room if relevant

No order should be sent without a risk log record.

---

## 7. Implementation Gap List

Already present:

- HOSE +/-7% simple band check.
- T+2 sellability in lifecycle backtest.
- Slippage approximation.
- Portfolio exposure and total-risk caps.
- No sell before settlement in lifecycle engine.

Needs implementation:

- Tick-size normalization.
- Board-lot rounding and max 500,000 shares/order.
- Odd-lot LO-only rule.
- Session/order-type validator.
- ATO/ATC expiry and cancel/modify lockout.
- Special +/-20% trading band handling.
- Official floor/ceiling/reference-price support.
- Foreign room hard reject only for foreign-investor account mode.
- Same-symbol buy/sell conflict tracking in periodic auctions.
- Restricted-security trading session handling.

