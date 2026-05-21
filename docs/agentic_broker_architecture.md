# Agentic Broker-Grade Architecture

## Vision

The system should first match the baseline capabilities users expect from a
brokerage, securities company, or financial data platform. After that baseline
is reliable, GenAI and agentic AI should add an intelligence layer that is hard
for a normal brokerage workflow to replicate: synthesis, personalized review,
strategy memory, automated research, and continuous improvement.

This means the architecture is not "LLM replaces strategy". It is:

```text
Broker-grade data and workflows
  -> deterministic alpha and risk engines
  -> GenAI research and explanation
  -> agentic review, monitoring, and improvement
  -> rule-based final control
```

## Product Baseline: Match The Broker

The system should cover the practical surfaces that serious retail and
semi-professional investors already get from brokers and financial platforms.

### Market Data

- Daily and intraday OHLCV.
- Live price board.
- Market breadth and index contribution.
- Sector and industry movement.
- Liquidity, value traded, foreign flow, proprietary flow when available.
- Corporate actions and adjusted price consistency.
- Security status: restricted, warning, supervision, delisting risk.

### Research Data

- Financial statements, ratios, growth, margins, valuation.
- Peer and sector comparison.
- Ownership, foreign room, insider transactions if available.
- Company events, earnings, dividends, shareholder meetings.
- News by symbol, sector, and market theme.
- Macro data: rates, FX, commodities, global indices.

### Trading Workflow

- Watchlist and alerts.
- Signal generation.
- Entry zone, stop loss, take profit, sizing.
- Portfolio exposure and position tracking.
- T+ settlement awareness.
- Order simulation with HOSE/HNX/UPCOM constraints.
- Trade journal and post-trade review.

### Reporting

- Morning brief.
- Intraday watchlist update.
- End-of-day market review.
- Weekly strategy review.
- Portfolio risk report.
- Signal audit: why accepted, rejected, or delayed.

## AI Advantage: Exceed The Broker

The agentic layer should do things that static dashboards and ordinary broker
reports usually do not do well.

### Cross-Channel Synthesis

Agents combine:

- Price and volume behavior.
- Money flow and sector rotation.
- Fundamental quality.
- News and events.
- Macro regime.
- Historical behavior of similar setups.
- Current portfolio constraints.

The output is not just a recommendation. It is a structured decision record:

```python
{
    "symbol": "FPT",
    "strategy_signal": {...},
    "evidence": {...},
    "agent_review": {...},
    "risk_decision": {...},
    "execution_plan": {...},
    "audit": {...},
}
```

### Personalization

The same market signal should not produce the same action for every user. The
system should account for:

- Portfolio holdings.
- Available cash.
- Maximum acceptable drawdown.
- Preferred holding period.
- Sector concentration.
- Existing exposure to correlated names.
- User's trading style: conservative, balanced, aggressive.

### Continuous Learning

The system should remember decisions and outcomes:

- Which core3 signals worked or failed.
- Which agent objections were useful.
- Which news or flow conditions mattered.
- Which sectors produced better expectancy in each regime.
- Which execution model created better fills.

Learning must update filters, rankings, and research hypotheses. It must not
silently rewrite live strategy rules without backtest or shadow validation.

### Agentic Research

Agents can run controlled research loops:

- Generate hypotheses.
- Backtest them.
- Compare against current core strategies.
- Promote only to shadow.
- Require acceptance criteria before live use.

This is where the system can outperform a typical brokerage report: it can
convert new observations into testable strategy variants.

## Layered Architecture

### Layer 0: Data Foundation

Purpose: broker-grade data reliability.

Responsibilities:

- Normalize all providers behind internal schemas.
- Track data freshness and source.
- Detect missing, stale, duplicated, or unadjusted data.
- Store historical OHLCV and feature tables locally.
- Prevent lookahead in backtest.

Key rule: agents must never call vendor libraries directly. They use service
contracts.

### Layer 1: Market Intelligence

Purpose: reproduce the analytical context brokers provide.

Modules:

- Market regime.
- Sector rotation.
- Breadth.
- Liquidity.
- Foreign flow.
- Macro overlay.
- News/event map.

Output should be structured and reusable by both deterministic engines and
agents.

### Layer 2: Deterministic Alpha Engines

Purpose: produce backtestable candidate signals.

Current core3 belongs here:

- `breakout_after_accumulation_v3`
- `compression_breakout_smt_v1`
- `mean_reversion_uptrend_ma50_v1`

Rules:

- Strategy signals are immutable once generated.
- Each signal records strategy name, feature date, filters, rank, score, risk
  parameters, and data quality.
- LLM agents can reject or adjust execution, but cannot claim a non-core signal
  is a core strategy signal.

### Layer 3: Agentic Review

Purpose: turn raw signals into decision-quality analysis.

Recommended agents:

- Market analyst: checks index, breadth, sector, regime.
- Technical/money-flow analyst: reviews setup quality and failure modes.
- Fundamental analyst: checks valuation, quality, earnings risk.
- News analyst: detects event risk and narrative changes.
- Flow analyst: foreign/proprietary/liquidity confirmation.
- Bull researcher: best case.
- Bear researcher: failure case.
- Execution planner: entry zone and order style.
- Risk reviewer: portfolio and market constraints.

Important: these agents provide evidence and proposals, not final authority.

### Layer 4: Deterministic Risk And Execution Control

Purpose: prevent the AI layer from bypassing market rules or portfolio limits.

Responsibilities:

- Exchange rules: session, lot size, tick size, price band, settlement.
- Portfolio rules: max positions, max sector exposure, max risk per trade.
- Strategy rules: allowed setup, max hold, required R:R, stop depth.
- Data rules: stale data, corporate action uncertainty, missing liquidity.
- Final action: allow, reject, resize, or hold.

This layer is the final gate in live, paper, and backtest.

### Layer 5: User Experience

Purpose: make the system usable like a broker terminal plus analyst desk.

Interfaces:

- CLI for research and backtests.
- Dashboard for watchlists, signals, portfolio, reports.
- Discord/Telegram alerts.
- Markdown reports.
- Trade journal.
- Audit viewer.

## Decision Authority

The system must define who is allowed to change what.

| Layer | Can Create Buy Candidate | Can Reject | Can Change Entry/SL/TP | Can Execute |
|---|---:|---:|---:|---:|
| Core strategy | Yes | No | Provides defaults | No |
| Agentic review | No | Proposes reject | Proposes bounded changes | No |
| Risk engine | No | Yes | Clips/blocks invalid values | Yes, as final gate |
| User | Yes, manual override | Yes | Yes | Yes |

For automated mode, the rule is stricter:

```text
No deterministic core signal -> no automated buy candidate.
No risk approval -> no order or live alert marked actionable.
```

## Target Flow For Core3

```text
Daily data update
  -> feature table build
  -> core3 signal family
  -> StrategySignal envelope
  -> agentic review packet
  -> execution proposal
  -> deterministic validation
  -> risk decision
  -> alert/order journal
  -> outcome tracking
  -> weekly research review
```

The multi-agent system improves the original strategy by filtering and
executing it better. It does not replace the original strategy unless a new
strategy passes research, shadow, and promotion gates.

## Promotion Gates For New Agent-Discovered Strategies

New strategies from Bob/research agents must pass:

1. Written hypothesis with economic rationale.
2. No-lookahead feature implementation.
3. Walk-forward backtest.
4. Regime split.
5. Universe split: VN30, VN100, liquid non-VN30.
6. Transaction cost and slippage assumptions.
7. Shadow run for a fixed period.
8. Explicit promotion to live config.

Until then, they remain research-only.

## Implementation Phases

### Phase 1: Freeze Core3 As The Production Alpha Source

- Create `StrategySignal` schema.
- Emit core3 signals as structured envelopes.
- Make deep mode consume only these envelopes by default.
- Add an audit reason whenever a non-core candidate is considered.

### Phase 2: Broker-Grade Evidence Packet

- Add market, sector, money-flow, fundamental, news, and portfolio context into
  one `EvidencePacket`.
- Store packet per symbol/date for audit and replay.
- Make agents read packets instead of refetching ad hoc data.

### Phase 3: Agentic Review Without Strategy Mutation

- Modify TradingAgents-VN to review a provided signal.
- Agents output `approve`, `reject`, or `adjust_execution`.
- Require structured JSON for every agent summary.
- Keep final decision in deterministic risk.

### Phase 4: Broker Terminal Experience

- Morning brief.
- Signal dashboard.
- Watchlist and position monitor.
- End-of-day and weekly review.
- Audit/history viewer.

### Phase 5: Research And Promotion Loop

- Keep Bob/research agents in shadow mode.
- Track strategy candidates separately from live core3.
- Promote only after acceptance criteria are met.

## Non-Negotiable Guardrails

- No LLM numeric scoring for alpha.
- No LLM final risk approval.
- No lookahead data in backtest or memory.
- No live strategy mutation without explicit promotion.
- Every actionable alert must include source strategy, data date, risk rules,
  and rejection/approval rationale.

