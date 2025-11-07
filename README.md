# Gas Consumption Forecasting with Feed-Forward Neural Networks (FFNN)

**TL;DR:** Trains FFNNs on hourly temperature and residential gas consumption for Luxembourg (2020–2025). With simple lag features and Adam optimization, the best run achieves **MAPE ≈ 7.2%** and **R² ≈ 0.984** on a 1-year hold-out, supporting temperature-driven ML for operational forecasting and Take-or-Pay analytics.

---

## 1) Project goal
Forecast day-ahead or hour-ahead residential gas consumption using temperature signals and compact FFNNs. The aim is practical forecasting for front-office and risk tasks, not just academic fit.

## 2) Data
- **Temperature (hourly):** Meteostat, Luxembourg Airport (ID 06590), 2020-01-01 → 2025-05-25  
- **Gas consumption (hourly, residential):** ENTSOG Transparency Platform, Luxembourg distribution exit point

> Luxembourg is small enough that a single-station temperature series is representative, reducing spatial averaging noise.

## 3) Feature engineering
Two families of inputs:
- **Non-cumulative lags (k = 1…3 days):** current temperature plus $T_{j-k}$ and $P_{j-k}$
- **Cumulative lags (k = 1…3 days):** current temperature plus all lags up to k for either both series, temperature-only, or consumption-only

\(P\) = consumption, \(T\) = temperature. Day lags are 24-hour offsets. No holiday or calendar features yet.

## 4) Model
- **Architecture:** Dense(32) → Dense(16) → Dense(1); activations tested: ReLU, tanh, sigmoid
- **Loss / optimizer:** MSE / Adam
- **Scaling:** standardization via caret preprocessing
- **Split:** last 365 days (8,760 hours) as hold-out; remainder for training
- **Implementation:** R (keras + tidyverse + caret); notebook-style workflow

## 5) Results (selected)
**Non-cumulative lags, temp + consumption**
- k = 3, ReLU: **R² ≈ 0.983**, **MAPE ≈ 7.23%**
- k = 3, tanh: R² ≈ 0.984, MAPE ≈ 7.35%
- k = 2, tanh: R² ≈ 0.983, MAPE ≈ 7.07%

**Cumulative, temperature-only vs consumption-only**
- Temp-only (k up to 3): R² typically **0.97–0.98**, MAPE **~7.5–12%**
- Cons-only shows similar ranges but tends to degrade at larger k

**Takeaways**
- Pairing **temperature + consumption** lags outperforms single-series inputs
- Simple FFNNs with **1–3 day lags** are competitive and stable across activations
- Accuracy is credible for **ToP contract analytics** and **ops scheduling**, pending geography-specific calibration

## 6) Limitations
- Single geography; broader EU roll-out needs regionalization
- No explicit holiday, weekday or seasonality features yet
- Point forecasts only; no calibrated intervals

## 7) Packages
**R:** `keras`, `tensorflow` (backend), `tidyverse`, `caret`, `ggplot2`, `lubridate`, `scales`  
**Optional R (data access):** `meteostat` (if pulling weather directly), `httr`/`jsonlite` (for API calls)  
**System:** TensorFlow CPU is sufficient for this project

## 8) Suggested extensions
- Add calendar features (weekday/holiday), heating-degree hours, Fourier seasonality
- Baselines: seasonal naïve, ridge/elastic net, GLM with splines; compare to RNN/LSTM
- Probabilistic forecasting (quantile loss) for risk metrics
- Multi-region experiments where data completeness allows

## 9) Business context
Primary use-case is **Take-or-Pay** support: forecast consumption vs contracted volumes, quantify expected penalties or shortfalls, and test sensitivity to temperature scenarios.

## 10) Reference
Ravnik, J., Jovanovac, J., Trupej, A., Vištica, N., & Hriberšek, M. (2021). *A sigmoid regression and artificial neural network models for day-ahead natural gas usage forecasting*. **Cleaner and Responsible Consumption**, 3, 100040.

---

**Author:** Preslav Vachev  
**Period covered:** Jan 2020 – May 2025  
**Location:** Luxembourg (LU)
