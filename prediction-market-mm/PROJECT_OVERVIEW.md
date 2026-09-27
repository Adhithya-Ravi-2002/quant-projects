# Prediction Market Market-Making & Microstructure Engine

Research + engineering project studying market-making and microstructure signals on
prediction markets (Kalshi / Polymarket), built as a portfolio project for quant
research/systems roles.

**Status:** Active development. No live trading, no live capital. All results reported
here are only ever claims that are reproducible from the code in this folder — see
`PRD.md` for the explicit non-goals and evaluation methodology.

Full scope, architecture, and KPI methodology: [`PRD.md`](./PRD.md).

## Structure

```
prediction-market-mm/
├── PRD.md              # Full product requirement doc: scope, architecture, KPIs
├── requirements.txt     # Python dependencies
├── src/
│   └── prediction_market_mm/   # Installable package: engine modules live here
├── tests/               # pytest test suite
└── notebooks/           # Exploratory signal research (Jupyter/Colab)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/
```

## Current phase

Phase 1: L2 order book reconstruction (in progress).
