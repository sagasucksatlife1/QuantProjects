# GD Nexus — A Terminal for Quantitative Research

A local-first research system that fuses three instruments into one pipeline: a **council of language-model personas** that propose trading strategies, a **backtesting forge** that hand-tests them against the historical record, and a **portfolio treasury** that composes surviving strategies into an allocated, optimized book.

> Built end-to-end — data layer, sandboxed execution engine, LLM orchestration, portfolio math, and interface — as a single coherent research tool rather than three disconnected scripts.

---

## Why this exists

Most "AI trading bot" projects are a single prompt asking an LLM to output buy/sell signals. That's not research — it's a guess with extra steps.

GD Nexus is built on a different premise: **strategy generation should be adversarial and disposable.** Ten independent model personas — each with a distinct mathematical worldview (statistical mechanics, differential geometry, signal processing, behavioral finance, regime-switching macro) — each *independently* propose a strategy. Almost all of them fail. That's expected, and the failures are not hidden: every idea, every formula, every stack trace is logged and shown, because the failure log is as informative as the survivors.

What survives contact with real price data, transaction costs, and a lookahead-bias checker gets promoted into a vault, backtested properly, and can be deployed against an actual multi-asset portfolio.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          I. THE COUNCIL                          │
│   10 LLM research personas × 2 runs each = 20 attempts/session   │
│   Local inference (Ollama / Qwen3-8B) — nothing leaves the machine│
│                                                                    │
│   persona → IDEA → FORMULA → CODE → validate → backtest          │
│                                          │                        │
│                              ┌───────────┴──────────┐             │
│                          survivors               graveyard        │
│                       (promoted to vault)     (logged, not hidden)│
└─────────────────────────────────┬─────────────────────────────────┘
                                   │
┌──────────────────────────────────▼────────────────────────────────┐
│                           II. THE FORGE                            │
│   Sandboxed strategy execution + vectorized backtesting engine     │
│   yfinance data · lookahead detection · costs · leverage · SL/TP   │
│   → equity curve, drawdown, trade log, full metric suite           │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
┌──────────────────────────────────▼────────────────────────────────┐
│                          III. THE TREASURY                         │
│   Portfolio analytics · Mean-Variance & Hierarchical Risk Parity   │
│   THE FUSION: run a vaulted strategy on the portfolio's own        │
│   combined equity curve — trade the estate, not just one asset     │
└─────────────────────────────────────────────────────────────────────┘
```

**Stack:** FastAPI · SQLite · yfinance · SciPy/NumPy/Pandas · Ollama (Qwen3-8B) · React

---

## I. The Council

<img src="assets/02-the-council.png" width="850">

Ten research personas, each carrying a distinct doctrine rather than a generic "trading bot" prompt:

| # | Persona | Research Style |
|---|---------|-----------------|
| 1 | `physicist_stat_mech` | Statistical mechanics / information theory — rare-event detection, rolling Shannon entropy regime shifts |
| 2 | `physicist_wave_dynamics` | Signal processing — Hilbert transform instantaneous phase & frequency, cycle-state detection |
| 3 | `physicist_disordered_systems` | Regime-change detection — CUSUM change-point accumulation, continuous arctan scaling |
| 4 | `mathematician_stochastic` | Ornstein-Uhlenbeck parameter estimation via rolling OLS — reversion speed, half-life sizing |
| 5 | `mathematician_topology_geometry` | Differential geometry — Savitzky-Golay derivatives, true path curvature κ = y″/(1+y′²)^1.5 |
| 6 | `quant_stat_arb` | Statistical arbitrage — synthetic fair-value spread, half-life-gated mean reversion |
| 7 | `quant_vol_surface` | Garman-Klass OHLC volatility, regime-gated momentum/reversion switching |
| 8 | `quant_ml_engineer` | Orthogonal feature ensemble — z-scored momentum + mean-reversion + volume-confidence |
| 9 | `quant_behavioral` | Behavioral finance — disposition-effect proxy via distance-from-extremes |
| 10 | `quant_macro_regime` | 2×2 Hurst-exponent × volatility-percentile quadrant allocator, including a "go flat" regime |

**The pipeline, per persona, per run (2 runs × 10 personas = 20 attempts per session):**

1. The persona is prompted for an **IDEA** (the market inefficiency being targeted), a **FORMULA** (the governing math, in plain notation), and **CODE** (a `signals(df, params=None)` function returning positions in `[-1, 1]`).
2. Code is validated against a hard contract: no imports, no file/network access, no lookahead (`shift(-1)` is explicitly banned and checked for).
3. Validated code is executed against real historical OHLCV data in a restricted sandbox (whitelisted builtins only, hard timeout).
4. Successful strategies are backtested immediately and their metrics attached to the result.

**Nothing is hidden.** Every attempt — including the ~60-70% that fail — streams live with its stated idea, its formula, its exact code, and its exact error. The idea and formula are stamped as comments directly into the code artifact, so a failed run is still legible on its own:

```python
# IDEA: Treat the log-price path as a curve; sharp curvature marks a turning point.
# FORMULA: kappa = y'' / (1 + y'^2)^1.5, via Savitzky-Golay smoothed derivatives
def signals(df, params=None):
    ...
```

Typical failure modes surfaced by the graveyard (real examples from development): `pandas.Rolling` has no `.zscore()` method; `scipy.signal.hilbert` returns a raw `ndarray`, not a `Series`, so `.diff()` fails; `np.diff()` silently shortens arrays by one element causing length mismatches; models occasionally attempt `shift(-1)`, which is caught and rejected as a lookahead violation before it ever reaches a backtest.

---

## II. The Forge

<img src="assets/03-the-forge.png" width="850">

The backtesting laboratory. Every strategy — whether hand-written or council-generated — is held to the same contract:

```python
def signals(df, params=None) -> pd.Series
# returns positions in {-1, 0, +1} aligned to df.index
# +1 = long, -1 = short, 0 = flat
# df columns available: Open, High, Low, Close, Volume
```

**Execution safety:**
- Sandboxed exec environment — restricted builtins, banned tokens (`import`, `open(`, `exec`, `subprocess`, `socket`, `os.`, etc.), hard timeout
- Strategy code is compiled and validated *before* it ever touches live data
- Positions are shifted forward one bar before being applied to returns — no accidental lookahead in the backtest math itself

**What gets computed:**
- Total return, CAGR, Sharpe ratio, max drawdown, volatility, win rate, trade count
- Full equity curve vs. buy-and-hold benchmark
- Drawdown series
- Complete trade log (entry/exit date, price, side, P&L %)
- Configurable transaction costs (bps), leverage (1–10×), stop-loss, and take-profit

Strategies can be run at 1mo–max lookback windows across 5m/15m/1h/1d bars, on any Yahoo Finance-listed instrument.

---

## III. The Treasury

<img src="assets/04-the-treasury.png" width="850">

Portfolio construction and the system's actual thesis: **strategies shouldn't just trade one instrument — they should be tested against a real, allocated portfolio.**

**Three operations:**

1. **Analyze** — given a set of holdings and weights, compute portfolio-level return, volatility, Sharpe, drawdown, and a full pairwise correlation matrix.
2. **Optimize** — rebalance the same holdings using either:
   - **Mean-Variance Optimization (MVO)** — maximizes Sharpe ratio via constrained SLSQP over the covariance matrix
   - **Hierarchical Risk Parity (HRP)** — clusters assets by correlation distance (single-linkage) and allocates inverse-variance weight recursively down the dendrogram, avoiding the instability MVO exhibits under estimation error
3. **Fusion** — the connective operation. A strategy pulled from the Council's vault is run not against a single stock, but against the **portfolio's own weighted combined equity curve**, treated as a synthetic instrument. This answers a different question than a single-asset backtest: *does this strategy add value on top of my actual diversified book, net of costs?*

---

## Design Philosophy

<img src="assets/01-introduction.png" width="850">

> "The apparatus is technical, but the discipline is old. Nothing that happens here has not happened before — the markets, like galleries, are principally a museum of what has already been done."

The interface deliberately avoids dashboard-style UI in favor of a single-column, editorial document — the system is read like a research notebook, chapter by chapter (Council → Forge → Treasury), rather than clicked through like a SaaS product. Every session is a document: idea, formula, code, result, in that order, for every attempt, success or failure.

---

## Engineering notes

- **Fully local.** LLM inference runs on-device via Ollama (Qwen3-8B) — no API costs, no data leaves the machine, no rate limits on strategy generation.
- **Failure is a first-class citizen.** The graveyard isn't a debug log bolted on after the fact — the event schema (`stage`, `persona_start`, `code`, `validation_failed`, `runtime_error`, `backtested`, `done`) treats success and failure as equally structured, equally displayed outcomes.
- **No lookahead, enforced twice.** Once at the persona-prompt level (explicit instruction + static check for `shift(-1)`), and again structurally in the backtest engine itself (signals are shifted forward before being applied to returns), so a strategy can't accidentally cheat even if a check is bypassed.
- **Sandboxing over trust.** Generated code runs with a whitelisted builtin set and a banned-token filter rather than relying on the model to behave — the code being untrusted output is treated as the default assumption, not an edge case.

---

## Status

Personal research tool, actively developed. Paper trading only — not investment advice, not a production trading system.
