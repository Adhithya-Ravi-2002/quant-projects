# Product Requirement Document (PRD)

## Prediction Market Market-Making & Microstructure Research Project

**Document ID:** PRD-QUANT-2026-004
**Project Name:** Kalshi / Polymarket Market-Making & Microstructure Engine
**Author:** [Friend's name] — independent research/engineering project
**Status:** Active development
**Intended audience:** Quant research / quant systems hiring teams (e.g. prop trading, market making desks)

---

## 1. Document Overview & Scope

### 1.1 Project Vision

Build a well-tested, well-documented Python system that:

1. Ingests and reconstructs L2 order book state for prediction markets (Kalshi and/or
   Polymarket), from historical/replayed data.
2. Computes a market-making signal stack: Avellaneda-Stoikov-style reservation pricing,
   multi-level Order Flow Imbalance (OFI), and a micro-price estimator.
3. Simulates market-making execution against replayed order book data with realistic
   queue-position and fill assumptions.
4. Backtests the strategy with rigorous overfitting controls (Deflated Sharpe Ratio) and
   reports results honestly, including failure modes and known limitations.

The deliverable is a GitHub repository with clean module structure and tests (primary
artifact), supplemented by notebooks for exploratory signal research and plots.

The project is framed explicitly as **research + engineering**, not as a live trading
system. No claim of live deployment, live PnL, or real capital is made anywhere in the
writeup or interview narrative unless it becomes literally true. Recruiters and quant
interviewers are not impressed by numbers that look manufactured; they are impressed by
**correct methodology, honest evaluation, and an engineer who can say precisely why a
result is trustworthy or isn't.**

---

## 2. System Architecture & Component Specifications

### 2.1 Market Data Ingestion Subsystem (MDIS)

- **Protocols:** REST/WebSocket historical data pulls from Kalshi and/or Polymarket public
  APIs; local storage of raw ticks.
- **Reconstruction:** Rebuild L2 depth (a fixed number of levels — start with 10, expand if
  useful) from raw tick/diff messages.
- **Persistence:** Store as Parquet (DuckDB optional, not required) for replay.
- **Latency:** Measured and reported (e.g. "book reconstruction takes X ms per update on
  average on this hardware"), not held to an arbitrary hard threshold. If a threshold is
  useful pedagogically, set one only after seeing real numbers from the pipeline.

### 2.2 Alpha & Microstructure Engine (AME)

- **Reservation Price Generator:** Implement the Avellaneda-Stoikov reservation price
  r(s, q, t) = s − qγσ²(T − t), with volatility estimated from a rolling window (simple
  realized vol first; more sophisticated estimators as a stretch goal, only after the
  simple version is validated).
- **Order Flow Imbalance (OFI) Signal:** Multi-level OFI over a configurable sliding window
  (500ms as a starting point, tuned empirically rather than fixed by fiat).
- **Micro-Price Estimator:** Volume-weighted micro-price to reduce quote lag versus naive
  mid-price.
- Each signal component should have: a written explanation of the underlying math, a unit
  test against a hand-computed example, and a notebook showing it on real data.

### 2.3 Execution & Order Management (Simulated)

- **Quote Manager:** Cancel/replace logic when mid-price or micro-price moves beyond a
  configurable threshold (tunable, not fixed at 0.5 ticks by default).
- **Position & Inventory Controls:** Hard position limits, configurable per experiment
  (start small; there is no real capital at risk, so the limit exists to make the strategy
  behave sensibly in backtests, not to satisfy a compliance requirement).
- **Sizing:** Start with fixed or simple volatility-scaled sizing. Half-Kelly is a
  reasonable stretch goal once the simpler version is working and understood — do not
  implement Kelly sizing as a black box; be able to derive and explain it.

### 2.4 Tick-Level Backtesting Engine (TEB)

- **L2 Book Replay:** Simulate queue priority and fills against historical order book data
  as faithfully as the data allows. Document explicitly where the simulation is
  approximate (e.g. no ability to see true queue position without proprietary data) —
  stating this limitation clearly is a strength in an interview, not a weakness.
- **Overfitting Controls:** Deflated Sharpe Ratio (Bailey & López de Prado) computed
  across the actual number of hyperparameter trials run — not an assumed N. Report the
  real trial count and the real DSR, whatever it is.
- **Fee modeling:** Include maker/taker fees as published by the platform(s) used.

---

## 3. Functional Requirements Table

| Module | Requirement ID | Specification Detail |
|---|---|---|
| Data Engine | SPEC-DAT-01 | Reconstruct L2 order book from historical tick/diff data; store as Parquet. |
| Data Engine | SPEC-DAT-02 | Report book reconstruction latency empirically; no fixed hard threshold. |
| Quant Model | SPEC-ALG-01 | Implement Avellaneda-Stoikov reservation price with rolling volatility estimate. |
| Quant Model | SPEC-ALG-02 | Implement multi-level OFI over a tunable sliding window. |
| Quant Model | SPEC-ALG-03 | Implement micro-price estimator. |
| Execution (sim) | SPEC-EXE-01 | Cancel/replace simulated quotes on configurable price-move threshold. |
| Execution (sim) | SPEC-EXE-02 | Enforce configurable position limits in backtest. |
| Backtester | SPEC-BKT-01 | Replay L2 tick logs with documented, explicit fill/queue assumptions. |
| Backtester | SPEC-BKT-02 | Report Deflated Sharpe Ratio (with real trial count), Brier Score, Max Drawdown, and win/loss breakdown. |
| Evaluation | SPEC-EVL-01 | Written limitations section: what the backtest cannot capture (adverse selection, real queue position, latency in live conditions, etc.). |

---

## 4. Key Performance Indicators (KPIs)

The purpose of these KPIs is to be **defensible under interview questioning**, not to look
impressive on a slide.

- **Sharpe Ratio:** Report the actual in-sample and out-of-sample Sharpe from the backtest.
  No target is set in advance; a modest, well-validated positive Sharpe (even 0.5–1.5) with
  a clean methodology is a better outcome than a fabricated-looking 3.5+.
- **Deflated Sharpe Ratio (DSR):** Computed honestly against the real number of trials run
  during hyperparameter search. The goal is a DSR that is *statistically meaningful given the
  real trial count*, not a specific numeric target.
- **Maximum Drawdown:** Reported as observed, contextualized against strategy volatility.
- **Passive/Maker Fill Rate:** Reported as observed from the simulated queue model, with an
  explicit caveat about the accuracy limits of simulated queue position versus a real
  matching engine.
- **Process metric:** Number of documented failure modes / limitations identified and
  written up. This is treated as a first-class deliverable — a thorough limitations section
  is direct evidence of quant maturity.

---

## 5. Explicit Non-Goals

- This is **not** a live trading system and will not be presented as one.
- This is **not** a low-latency/HFT infrastructure project; latency is measured, not chased.
- C++ is explicitly out of scope for this phase.
- No claim will be made of returns, Sharpe, or performance that has not been independently
  reproduced from the code in this repository.

---

## 6. Tooling

- **Language:** Python throughout.
- **Primary development environment:** VS Code, structured as an installable package with
  `pytest` test coverage.
- **Exploratory work:** Jupyter/Colab notebooks for signal research, EDA, and plots — not
  used for the engine modules themselves (reservation price, OFI, backtester), which live
  as tested modules in the repo.
- **Storage:** Parquet for tick data; DuckDB optional if convenient for querying.
