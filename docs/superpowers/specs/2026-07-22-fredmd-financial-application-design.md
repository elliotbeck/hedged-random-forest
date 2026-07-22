# FRED-MD Financial-Econometric Application — Design Spec

## 1. Motivation

The JFEC Associate Editor's sole substantive comment on revision is that the empirical
section needs "a closer match with the readership and standards of Journal of Financial
Econometrics" — specifically, one genuine financial-econometric out-of-sample forecasting
application (return prediction, volatility forecasting, credit-risk prediction, or
macro-financial forecasting), plus diagnostic evidence on when the HRF weighting scheme
helps.

`hrf.tex` currently only gestures at this via a one-paragraph citation
(`\cite{beck:wolf:hrf-inf}`, `hrf.tex:1169`) to the companion paper *Forecasting Inflation
with the Hedged Random Forest* (Beck & Wolf, SNB Working Paper 2025-07). That paper adapts
HRF to time-series forecasting via custom EWMA+shrinkage estimators of the tree-residual
mean/covariance, and demonstrates RMSE ratios (HRF/RF) consistently below 1 for six
inflation series. This project brings an adapted, condensed version of that same validated
methodology into `hrf.tex` itself, applied to a new set of genuinely financial-econometric
FRED-MD targets — not inflation, which the companion paper already owns.

## 2. Data

- Source: `data/hrf-ts/2026-06-MD.csv` (FRED-MD vintage, provided by user). Contains raw,
  untransformed monthly series, Jan 1959 – May 2026, plus a `tcode` row (McCracken & Ng's
  recommended stationarity-inducing transform per series, codes 1–7).
- No external network/API dependency — the target series' raw levels come from this same
  file (no need for `fredr`/FRED API, unlike the companion paper's implementation).

## 3. Target universe (41 series)

Two groups from the FRED-MD classification (McCracken & Ng, 2016), chosen because they are
squarely "financial econometrics" (interest rates, credit spreads, FX) or directly adjacent
to it (price indices), and because using an entire class of series avoids cherry-picking a
single target while still avoiding a "broad horse race across methods" (the AE's specific
objection) — the horse race here is across *targets*, not across competing ML algorithms.

- **Prices** (20 of 21 group members; `NAPMPRI` absent from this vintage): CPI/PPI
  subcomponents, PCE deflator components, oil price. All `tcode=6`.
- **Interest and Exchange Rates** (21 of 22 group members; `TWEXMMTH` absent from this
  vintage): `tcode=2` — FEDFUNDS, CP3Mx, TB3MS, TB6MS, GS1, GS5, GS10, AAA, BAA (9 rate
  levels); `tcode=1` — COMPAPFFx, TB3SMFFM, TB6SMFFM, T1YFFM, T5YFFM, T10YFFM, AAAFFM,
  BAAFFM (8 credit/term spreads); `tcode=5` — EXSZUSx, EXJPUSx, EXUSUKx, EXCAUSx (4 FX
  rates).

Predictors: the full FRED-MD panel (both groups' targets included), restricted to series
with complete history over the estimation window — same convention as the existing
(currently unused) `src/utils/get_fredmd_data.R`.

## 4. Methodology — unified fit/aggregate/evaluate recipe

Every target is handled by the same recipe, branching only on its own `tcode`. This
generalizes the companion paper's validated finding that "path-average" (compound
one-period building-block forecasts) beats "one-shot" (model the multi-step change
directly) — applied consistently across all 41 targets rather than being special-cased to
price series only.

- **Building block** `b_t`, defined per target by its `tcode`:
  - `tcode ∈ {5,6}` (price/FX-type): `b_t := Δlog(x_t)` (one-period log-change — one order
    lower than McCracken-Ng's own `tcode=6` for prices, exactly matching how the companion
    paper treats CPI: MoM `%` change, not "inflation's acceleration").
  - `tcode = 2` (rate-level-type): `b_t := Δx_t` (one-period first difference).
  - `tcode = 1` (already-stationary, e.g. spreads): no building block needed; `b_t := x_t`.
- **Direct forecasting**: for forecast lead `j`, train a separate model
  `ĝ_j(x_t) ≈ b_{t+j}` (ranger RF, then HRF-reweighted) on a 360-month rolling window
  ending at `t`, following Section 4.1 of the companion paper (`R_j := 360 - j - 4` effective
  training observations, `l=4` lags).
- **Aggregation to raw level**, to obtain the horizon-`h` forecast `x̂_{t+h}`:
  - `tcode ∈ {5,6}`: `x̂_{t+h} = x_t · Π_{j=1}^h (1 + b̂_{t+j|t})` (cumulative product —
    companion paper's eq. 4.6, generalized).
  - `tcode = 2`: `x̂_{t+h} = x_t + Σ_{j=1}^h b̂_{t+j|t}` (cumulative sum, additive analogue).
  - `tcode = 1`: `x̂_{t+h} = ĝ_h(x_t)` directly (no aggregation; `h=1` and general-`h` are
    the same single direct model).
- **Evaluation**: RMSE/MAE always computed on the raw level `x_{t+h}` vs. `x̂_{t+h}` — never
  in transformed/differenced space. This keeps every target's error in its own natural,
  economically interpretable units (index points, percentage points, spread bp, FX rate),
  and since we only ever report the *ratio* RMSE^HRF/RMSE^RF, cross-target comparability of
  absolute error units is not required.

### Backtest hyperparameters (mirrors the companion paper exactly)

| Parameter | Value |
|---|---|
| Rolling window | 360 months |
| OOS period | Jan 1990 → May 2026 (~437 months) |
| RF | `ranger`, `mtry = round(d/3)`, `num.trees = 500`, other defaults |
| Feature set | target's 4 AR lags (of its building block) + first 4 PCs of the complete-history panel (recomputed per rolling window) + 4 lags of every panel series |
| HRF inputs (μ̂, Σ̂) | EWMA (λ = 0.15) + linear shrinkage to constant-variance-covariance target, bandwidth `H = 6` (companion paper Appendix D — implemented fresh in this repo; the public `hrf-ts` GitHub code diverges from the published paper in several details, e.g. 3 lags vs. the paper's 4, so it is not treated as authoritative) |
| κ (leverage constraint) | 2 (fixed, matching the companion paper's single default — no κ-sweep) |
| Winsorizing | Building-block training labels winsorized at 1st/99th percentile within each rolling window; evaluation always uses actual (non-winsorized) values |
| Metric | RMSE ratio := RMSE^HRF / RMSE^RF per (target, horizon); MAE ratio analogously. Ratio < 1 ⟹ HRF outperforms RF |

## 5. Execution plan (two phases)

- **Phase 1 — scan**: horizon `h = 1` only, all 41 targets. ~20–30 min wall-clock on 30
  cores. Purpose: identify which targets show a clean, robust ratio < 1, to select the
  subset that goes into the paper (avoiding the AE's "broad horse race" concern by curating
  rather than dumping all 41 into the manuscript).
- **Phase 2 — full grid**: horizons `h = 1..12`, run only on the curated subset selected
  from Phase 1, matching the companion paper's full horizon range exactly.

## 6. Paper integration (`hrf.tex`)

New empirical subsection (placed alongside/replacing the current placeholder paragraph at
`hrf.tex:1160-1169`):

1. Brief recap of the HRF-EWMA time-series estimator (2–3 sentences, citing the companion
   paper for the full derivation — avoids duplicating derivations already published there,
   addressing the AE's separate similarity-score concern).
2. **Diagnostic table**: RMSE ratios at `h=1` across all (or most) of the 41 scanned
   targets, grouped by Prices / Rate levels / Spreads / FX — this doubles as the AE's
   requested "diagnostic evidence on when the weighting scheme works" without being a
   horse race across competing algorithms.
3. **Headline result**: full horizon-by-horizon (h=1,3,6,9,12,mean) table and/or a rolling-
   RMSE plot for 1–2 exemplar targets with the strongest, most robust results — framed as
   the paper's "one substantive financial-econometric application."

Exact target(s) for the headline result are chosen after Phase 1 results are in — not
pre-committed in this spec.

## 7. Implementation architecture

New directory `src/timeseries/` (parallel to existing `src/simulations/`,
`src/cov_estimators/`):

- `get_fredmd_panel.R` — loads `data/hrf-ts/2026-06-MD.csv`, applies the 7 McCracken-Ng
  tcode transforms (implemented directly from the published formulas — no new package
  dependency; `BVAR`, referenced by the existing but currently-unused
  `src/utils/get_fredmd_data.R`, is not installed in this project's `renv.lock` and will not
  be added), drops columns with any `NA` in the estimation window.
- `get_cov_ewma_shrink.R` — EWMA mean/covariance estimator with linear shrinkage
  (companion paper Appendix D), added to `src/cov_estimators/`.
- `get_forecast_horizon_ts.R` — builds features (lags + PCs) for one rolling-window origin
  and one lead `j`, trains RF, extracts tree-level residuals, computes HRF weights via the
  existing `src/utils/get_weights.R` (κ constraint, already implemented).
  `src/utils/get_weights.R` per the earlier repo audit implements the shared
  `(w'μ)² + w'Σw` objective with `‖w‖₁ ≤ κ` and `w'1=1` constraints already — this file is
  reused as-is, not reimplemented.
- `get_forecast_vintage_ts.R` — loops over leads `1..h` for one target/vintage, aggregates
  to the raw-level forecast per the `tcode`-branching rule in Section 4.
- `run_fredmd_scan.R` — orchestrates the two-phase run (Phase 1 / Phase 2) across all 41
  targets, parallelized via `mclapply`, writing per-target RMSE/MAE ratio results to
  `results/fredmd/`.
- `main_fredmd_scan.R` (top level, alongside existing `main.R`, `main_owrf.R`) — thin
  entry-point script setting parameters and invoking `run_fredmd_scan.R`.

## 8. Open items deferred to implementation / results stage

- Exact headline target(s) for the paper's main table/figure: chosen from Phase 1 results.
- Whether the diagnostic table in the paper shows all 41 targets or a representative
  subset per group: decided once Phase 1 output is visible (favor completeness unless the
  table becomes unwieldy).
- `hrf.tex` prose (framing, discussion of results, tie-back to the AE's "incremental value
  vs. modern tree-based ML" question): drafted after Phase 1/2 results exist, not before.
