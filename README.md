# Carbon-Aware Data Center Scheduling — Complete Project Guide

> **Project:** Carbon-Aware Data Center Scheduling using Big Data Analytics & Time Series Analysis  
> **Dataset:** Alibaba Cluster Trace v2018 — 95 million rows, 4,023 machines, 8 days of telemetry  
> **Tech Stack:** Apache Spark · PySpark · Pandas · Statsmodels · PyTorch · MAPIE · Streamlit · Plotly  
> **Dashboard:** 12-page interactive analytics dashboard with premium dark UI

---

## Table of Contents

1. [What Is This Project About? (Simple Explanation)](#1-what-is-this-project-about)
2. [The Big Picture — End-to-End Pipeline](#2-the-big-picture--end-to-end-pipeline)
3. [Dataset — Alibaba Cluster Trace v2018](#3-dataset--alibaba-cluster-trace-v2018)
4. [Notebook 1 — Spark ETL (Big Data Ingestion)](#4-notebook-1--spark-etl-big-data-ingestion)
5. [Notebook 2 — Time Series Reconstruction](#5-notebook-2--time-series-reconstruction)
6. [Notebook 3 — Statistical Diagnostics (TSA)](#6-notebook-3--statistical-diagnostics-tsa)
7. [Notebook 4 — Forecasting Models](#7-notebook-4--forecasting-models)
8. [Notebook 5 — Carbon-Aware Scheduling](#8-notebook-5--carbon-aware-scheduling)
9. [Notebook 6 — Conformal Prediction (Uncertainty)](#9-notebook-6--conformal-prediction-uncertainty)
10. [Dashboard — All 12 Pages Explained In Detail](#10-dashboard--all-12-pages-explained-in-detail)
11. [All Graphs & Visualisations Explained](#11-all-graphs--visualisations-explained)
12. [Key Formulas & Equations](#12-key-formulas--equations)
13. [BDA Concepts Used (Big Data Analytics)](#13-bda-concepts-used-big-data-analytics)
14. [TSA Concepts Used (Time Series Analysis)](#14-tsa-concepts-used-time-series-analysis)
15. [What Makes This Project Novel (Research Contribution)](#15-what-makes-this-project-novel)
16. [Project Structure & Files](#16-project-structure--files)
17. [How to Run the Project](#17-how-to-run-the-project)
18. [Frequently Asked Questions](#18-frequently-asked-questions)

---

## 1. What Is This Project About?

### The Problem

Data centers consume about **1-2% of total global electricity** and produce millions of tonnes of CO₂ every year. Most data centers run batch jobs (like backups, model training, data processing) **immediately** when they arrive — without thinking about whether the electricity grid is clean or dirty at that moment.

The electricity grid is NOT always equally dirty:
- **During midday (12:00–15:00):** Solar panels are producing lots of clean energy → **low carbon emissions**
- **During evening (19:00–22:00):** Gas power plants fire up to meet demand → **high carbon emissions**
- **During weekends:** Overall demand is lower → moderately cleaner

### The Solution

**What if we delay non-urgent batch jobs to run when the grid is cleanest?**

That's exactly what this project does:

1. **Collect server data** — CPU usage, memory, disk, network from 4,023 real servers (Alibaba)
2. **Clean and process it** — Using Apache Spark (handles 95 million rows)
3. **Analyse the patterns** — Time series analysis to find daily cycles, trends, correlations
4. **Predict future CPU usage** — Using 4 different ML models (SARIMAX, SETAR, MS-AR, LSTM)
5. **Schedule jobs smartly** — Move batch jobs to low-carbon time windows
6. **Quantify the uncertainty** — Conformal Prediction gives **mathematically guaranteed** confidence intervals

### The Result

The system can reduce CO₂ emissions by **up to 4.25%** by simply shifting when batch jobs run — without buying new hardware, without reducing workloads. Just smarter timing.

---

## 2. The Big Picture — End-to-End Pipeline

```
                          THE COMPLETE PIPELINE
                          =====================

  ┌─────────────────────────────────────────────────────────────┐
  │  STAGE 1: BIG DATA INGESTION (Notebook 01)                  │
  │  ─────────────────────────────────────────                   │
  │  • Load 95M rows from Alibaba Cluster Trace (CSV)           │
  │  • Apache Spark: schema enforcement, null removal            │
  │  • 5-minute temporal bucketing and aggregation               │
  │  • Select top-10 representative machines                     │
  │  • Export to Parquet (columnar, compressed)                  │
  │  • 95M rows → 22,430 rows (4,238× compression)              │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STAGE 2: TIME SERIES RECONSTRUCTION (Notebook 02)           │
  │  ─────────────────────────────────────────────               │
  │  • Pivot long format → wide matrix (2243 × 56 features)     │
  │  • Build proper DatetimeIndex (5-min intervals)              │
  │  • Forward-fill imputation (LOCF)                            │
  │  • Generate synthetic carbon intensity signal (CAISO)        │
  │  • Min-Max scaling to [0,1] for ML models                    │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STAGE 3: STATISTICAL DIAGNOSTICS (Notebook 03)              │
  │  ─────────────────────────────────────────                   │
  │  • ADF test → Confirm stationarity                           │
  │  • STL decomposition → Reveal 24-hour seasonal pattern       │
  │  • ACF/PACF → Decide AR order for models                    │
  │  • Kurtosis → Prove heavy tails → need non-linear models     │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STAGE 4: FORECASTING MODELS (Notebook 04)                   │
  │  ─────────────────────────────────────────                   │
  │  • SARIMAX (linear + seasonal + exogenous carbon signal)     │
  │  • SETAR (threshold autoregressive, 2 regimes)               │
  │  • MS-AR (Markov Switching, hidden states)                   │
  │  • LSTM (deep learning, PyTorch)                             │
  │  • Compare: RMSE, MAE, MAPE                                 │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STAGE 5: CARBON-AWARE SCHEDULING (Notebook 05)              │
  │  ─────────────────────────────────────────                   │
  │  • FIFO baseline (run immediately, no optimization)          │
  │  • Carbon-Aware: shift jobs to lowest-CI window              │
  │  • Risk-Aware: use conformal intervals for safe deferral     │
  │  • Pareto analysis: CO₂ savings vs. delay trade-off          │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STAGE 6: CONFORMAL PREDICTION (Notebook 06)                 │
  │  ─────────────────────────────────────────                   │
  │  • Split Conformal: guaranteed 95% prediction intervals      │
  │  • MAPIE Jackknife+: adaptive (varying width) intervals      │
  │  • Empirical coverage verification                           │
  │  • Conditional coverage analysis (per hour of day)           │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset — Alibaba Cluster Trace v2018

### What Is This Dataset?

Alibaba released telemetry data from their production data center running **4,023 physical machines** over **8 days**. This dataset is one of the largest publicly available server monitoring traces.

### Dataset Numbers

| Property | Value |
|----------|-------|
| **Total rows** | ~95 million |
| **Machines** | 4,023 |
| **Duration** | 8 days (Jan 1–8, 2018) |
| **Sampling rate** | Every ~10 seconds (raw) |
| **File size** | ~3.4 GB (CSV) |
| **Columns** | machine_id, timestamp, cpu_util_percent, mem_util_percent, mem_gps, mkpi, net_in, net_out, disk_io_percent |

### What Each Column Means

| Column | What It Measures | Units |
|--------|-----------------|-------|
| `machine_id` | Unique ID for each physical server | String (e.g., m_103) |
| `ts` | Unix timestamp | Seconds since epoch |
| `cpu_util_percent` | How much of the CPU is being used | 0–100% |
| `mem_util_percent` | How much RAM is being used | 0–100% |
| `mem_gps` | Memory bandwidth | GB/s |
| `mkpi` | Kernel performance index | Unitless |
| `net_in` | Network traffic coming in | Mbps |
| `net_out` | Network traffic going out | Mbps |
| `disk_io_percent` | How busy the disk is | 0–100% |

### Why 10 Machines?

From the original 4,023 machines, Spark selects the **top 10 machines with the most complete data** (fewest missing values, consistent timestamps). Working with 10 machines keeps computation fast while still representing the diversity of the cluster.

**Selected machines:** m_103, m_639, m_694, m_1508, m_2014, m_2366, m_2773, m_2967, m_3244, m_3330

---

## 4. Notebook 1 — Spark ETL (Big Data Ingestion)

### What Does This Notebook Do?

This is the **Big Data Analytics (BDA)** core. It uses Apache Spark to process 95 million rows of raw server data, clean it, aggregate it, and export it in an efficient format.

### Why Do We Need Spark?

95 million rows × 9 columns = **too much for pandas** to handle on a single machine's RAM. Apache Spark distributes the work across all CPU cores and manages memory intelligently.

### Step-by-Step Process

#### Step 1: Create a Spark Session
```python
spark = SparkSession.builder \
    .appName("AlibabaClusterETL") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
```
Think of this as starting up the Spark engine. We give it 8 GB of memory to work with.

#### Step 2: Load the CSV with Strict Schema
```python
schema = StructType([
    StructField("machine_id", StringType()),
    StructField("ts", LongType()),
    StructField("cpu_util_percent", DoubleType()),
    ...
])
df = spark.read.csv(path, schema=schema, mode="DROPMALFORMED")
```
**Schema enforcement** means we tell Spark exactly what type each column should be. If a row doesn't match (corrupted data), it gets dropped. This is a BDA best practice — never trust raw data.

#### Step 3: Remove Invalid Values
```python
# Sensor readings of -1 mean "sensor failure" → replace with NULL
# Readings > 100% are physically impossible → replace with NULL
df = df.withColumn("cpu_util_percent",
    F.when((F.col("cpu_util_percent") < 0) | (F.col("cpu_util_percent") > 100), None)
     .otherwise(F.col("cpu_util_percent")))
```

#### Step 4: Temporal Bucketing (5-Minute Windows)
```python
# Round timestamps to 5-minute boundaries
df = df.withColumn("ts_bucket", F.floor(F.col("ts") / 300) * 300)
```
Raw data arrives every ~10 seconds, but that level of detail creates noise. Grouping into 5-minute windows (300 seconds) smooths the data while keeping enough resolution.

#### Step 5: Aggregate per Machine per Time Window
```python
agg_df = df.groupBy("machine_id", "ts_bucket").agg(
    F.mean("cpu_util_percent").alias("cpu_util_percent"),
    F.mean("mem_util_percent").alias("mem_util_percent"),
    ...
)
```
For each machine and each 5-minute window, compute the **average** of all sensor readings in that window.

#### Step 6: Select Top-10 Machines
Spark ranks machines by data completeness (fewest nulls, most timestamps) and selects the top 10.

#### Step 7: Export to Parquet
```python
agg_df.write.partitionBy("machine_id").parquet("data/clean_parquet/subset")
```
**Parquet** is a columnar storage format that compresses data dramatically (95M rows → ~22K rows after aggregation).

### Key Numbers

| What | Before | After |
|------|--------|-------|
| Total rows | 95,000,000 | 22,430 |
| Compression ratio | — | 4,238× |
| Machines | 4,023 | 10 |
| Time resolution | ~10 sec | 5 min |
| Storage format | CSV (3.4 GB) | Parquet (compressed) |
| Trace duration | 8 days | 7 days 22 hours |

### Output Files
- `data/clean_parquet/subset/` — Partitioned Parquet files (one folder per machine)
- `data/etl_summary.json` — Statistics about the ETL process
- `data/selected_machines.json` — List of chosen machine IDs

---

## 5. Notebook 2 — Time Series Reconstruction

### What Does This Notebook Do?

Takes the clean Parquet data from Notebook 1 and transforms it into a **proper time series** — a table where each row is a timestamp and each column is a sensor measurement for a specific machine.

### Why Is This Step Needed?

The Spark output is in "long format" (one row per machine per timestamp). Time series models need "wide format" (one row per timestamp, columns for each machine's sensors).

### Step-by-Step Process

#### Step 1: Pivot from Long to Wide
```
BEFORE (Long Format):                    AFTER (Wide Format):
┌────────────┬───────────┬──────┐        ┌────────────┬──────────┬──────────┐
│ timestamp  │ machine   │ cpu  │        │ timestamp  │ cpu_m103 │ cpu_m694 │
├────────────┼───────────┼──────┤   →    ├────────────┼──────────┼──────────┤
│ 00:00      │ m_103     │ 42.5 │        │ 00:00      │ 42.5     │ 31.0     │
│ 00:00      │ m_694     │ 31.0 │        │ 00:05      │ 44.1     │ 29.8     │
│ 00:05      │ m_103     │ 44.1 │        └────────────┴──────────┴──────────┘
│ 00:05      │ m_694     │ 29.8 │
└────────────┴───────────┴──────┘
```

The final wide matrix has **2,243 timestamps × 56 features** (50 machine-metric pairs + 5 cluster averages + 1 carbon intensity).

#### Step 2: Build DatetimeIndex
Converts raw Unix timestamps to proper Python datetime objects starting from `2018-01-01`. This enables time-based operations like resampling, hourly grouping, etc.

#### Step 3: Forward-Fill Imputation (LOCF)
If a sensor value is missing at a timestamp, we use the **Last Observation Carried Forward** method — i.e., use the previous known value.

**Why LOCF instead of interpolation?** Linear interpolation creates artificial smooth transitions that don't exist. If a server was offline, LOCF honestly says "the last thing we knew was X" without inventing fake data.

#### Step 4: Synthetic Carbon Intensity Signal
Since the Alibaba dataset doesn't include electricity grid data, we generate a **realistic synthetic carbon intensity signal** modeled after the California CAISO grid:

```
Carbon Intensity = Baseline (400 gCO₂/kWh)
                 - Solar dip at 1 PM (peak solar = cleanest)
                 + Evening gas peak at 7 PM (dirtiest)
                 + Weekend reduction (-40 gCO₂/kWh)
                 + AR(1) correlated noise (realistic variation)
```

The signal realistically ranges from **100 to 700 gCO₂/kWh** with a daily cycle.

#### Step 5: Min-Max Scaling
Squish all values to [0, 1] range so all features are on equal footing for ML models:
```
x_scaled = (x - x_min) / (x_max - x_min)
```

### Output Files
- `data/timeseries_ready.csv` / `.parquet` — Unscaled wide-format time series (2243 × 56)
- `data/timeseries_scaled.csv` — Scaled [0,1] version for ML models
- `data/scaler_params.csv` — Min/max per feature (for reversing the scaling later)

---

## 6. Notebook 3 — Statistical Diagnostics (TSA)

### What Does This Notebook Do?

Before building any forecasting model, we need to **prove** that the data has predictable patterns. This notebook runs 4 major statistical tests that answer:

1. **Is the data stationary?** (ADF Test)
2. **Is there a daily pattern?** (STL Decomposition)
3. **How far back does memory extend?** (ACF / PACF)
4. **Does the data have sudden spikes?** (Kurtosis)

### Test 1: ADF Stationarity Test

**What is stationarity?** A time series is stationary if its average value, spread, and patterns don't change over time. Think of it like the weather in a city — temperature changes day to day, but the overall pattern stays consistent year to year. That's stationary.

**Why does it matter?** Most forecasting models ONLY work on stationary data. If the data has a growing trend, the model gets confused.

**How it works — Augmented Dickey-Fuller (ADF) Test:**
- **Null hypothesis (H₀):** The series is NOT stationary (has a unit root)
- **Alternative (H₁):** The series IS stationary
- **Decision:** If p-value < 0.05 → reject H₀ → data IS stationary

**Results from this project:**

| Series | Test Statistic | p-value | Stationary? |
|--------|---------------|---------|-------------|
| CPU Cluster Average (raw) | -4.88 | 0.000039 | Yes |
| CPU Machine m_103 (raw) | -5.45 | 0.0000026 | Yes |
| Carbon Intensity (raw) | -3.90 | 0.002 | Yes |
| CPU (after differencing) | -13.55 | ≈ 0 | Very strongly yes |

**Plain English:** All our series are stationary — they don't drift upward or downward over time. This means ARIMA models can be safely applied.

### Test 2: STL Seasonal Decomposition

**What is STL?** STL stands for "Seasonal and Trend decomposition using Loess." It splits any time series into three parts:

```
Observed = Trend + Seasonal + Residual

    Observed:  What we see — the raw data
    Trend:     The slow, long-term direction (increasing or decreasing)
    Seasonal:  The repeating pattern (e.g., daily cycle)
    Residual:  The random noise that's left over
```

**Period = 288** (because 24 hours × 12 five-minute intervals per hour = 288 data points per day)

**Seasonal Strength (Fs):** Measures how strong the daily pattern is (0 = no pattern, 1 = perfect pattern).

```
Fs = max(0, 1 − Variance(Residual) / Variance(Seasonal + Residual))
```

**Results:**

| Series | Fs Value | Interpretation |
|--------|----------|---------------|
| CPU Cluster Average | 0.7322 | **Strong** seasonal pattern (> 0.64 threshold) |
| Carbon Intensity | 0.8160 | **Very strong** seasonal pattern |

**Plain English:** CPUs are busier during business hours (9 AM – 6 PM) and quieter at night. The carbon grid is cleanest midday (solar peak) and dirtiest in the evening. Both patterns are very consistent day-to-day.

### Test 3: ACF / PACF Analysis

**What is ACF?** Auto-Correlation Function measures how similar the data is to a copy of itself shifted back in time.

- **Lag 1 (5 minutes ago):** ACF ≈ 0.97 → The previous reading is almost identical to the current one
- **Lag 12 (1 hour ago):** ACF ≈ 0.80 → Still very correlated
- **Lag 288 (24 hours ago):** ACF ≈ 0.70 → **Strong daily pattern detected!**

**What is PACF?** Partial ACF removes the indirect effects. PACF cuts off at lag 2, telling us: "CPU at time t directly depends on time t-1 and t-2, but lags beyond that are indirect."

**Why this matters:**
- ACF lag-288 spike → Use **seasonal ARIMA** with period s=24 (hourly) or s=288 (5-min)
- PACF cut-off at 2 → Use **AR(2)** (autoregressive order 2) in SARIMAX
- High ACF at all lags → The data is **very predictable** — good news for forecasting

### Test 4: Kurtosis (Distribution Shape)

**What is kurtosis?** It measures how "spiky" the data distribution is compared to a normal bell curve.

| Series | Kurtosis | Meaning |
|--------|----------|---------|
| CPU Cluster Average | +0.597 | **Heavier tails than normal** — occasional sudden spikes |
| Carbon Intensity | -0.552 | **Lighter tails** — more bounded, fewer extremes |

**Why this matters:** Kurtosis = +0.597 means CPU usage has **unexpected sudden spikes** (e.g., batch job arrivals, flash traffic). Linear models like SARIMAX can't handle these spikes well. We need **non-linear models** like SETAR and MS-AR that can switch between "normal mode" and "spike mode."

### Model Selection Summary

| Diagnostic Finding | What It Tells Us | Model Decision |
|-------------------|-------------------|----------------|
| Stationary (p ≪ 0.05) | Safe to model directly | ARIMA models are applicable |
| Strong seasonality (Fs = 0.73) | Daily CPU cycle exists | Use seasonal component: SARIMAX(2,1,1)(1,1,1,24) |
| ACF spike at lag 288 | 24-hour memory | Include lag_288 as a feature |
| PACF cut-off at lag 2 | Direct AR(2) structure | Set p=2 for SARIMAX |
| Kurtosis = 0.597 | Heavy tails / spikes | Need non-linear: SETAR, MS-AR, LSTM |

### Output Files
- `data/diagnostics_summary.json` — All test results in machine-readable format
- `data/spark_diagnostics.json` — Per-machine ADF tests run via Spark
- `figures/adf_stationarity.png`, `stl_decomposition_cpu.png`, `acf_pacf_plots.png`, `distributions.png`

---

## 7. Notebook 4 — Forecasting Models

### What Does This Notebook Do?

Trains **4 completely different forecasting models** to predict future CPU utilisation, then compares them to find the best one.

### Data Split (Train / Validate / Test)

Data is split **in time order** (no shuffling — you can't use future data to predict the past!):

```
|──────── Train (70%) ────────|── Validation (15%) ──|──── Test (15%) ────|
Day 1                        Day 5.5               Day 6.7              Day 8
```

**Target variable:** `cpu_cluster_avg` (average CPU utilisation across all 10 machines)  
**Exogenous variable:** `carbon_intensity_gCO2_kWh` (grid carbon signal)

---

### Model A: SARIMAX (Seasonal ARIMA with Exogenous Variables)

**What is it?** A traditional statistical model that combines:
- **AR (Auto-Regressive):** Use past values to predict the future
- **I (Integrated):** Differencing to handle non-stationarity
- **MA (Moving Average):** Use past forecast errors to improve
- **S (Seasonal):** Captures the daily repeating cycle
- **X (Exogenous):** Includes external data (carbon intensity)

**Configuration:** SARIMAX(2, 1, 1)(1, 1, 1, 24)
- Non-seasonal: p=2 (use 2 past values), d=1 (one differencing), q=1 (one error term)
- Seasonal: P=1, D=1, Q=1, s=24 (24-hour cycle on hourly data)

**Strengths:** Well-understood statistical theory, interpretable coefficients  
**Weaknesses:** Linear model — cannot handle sudden CPU spikes, had to resample to hourly (loses 5-min detail)

---

### Model B: SETAR (Self-Exciting Threshold Autoregressive)

**What is it?** A model that divides data into two "regimes" based on a threshold value and uses different equations for each regime.

```
If CPU(t-1) ≤ threshold:
    CPU(t) = c₁ + φ₁·CPU(t-1) + φ₂·CPU(t-2) + noise    ← "Low CPU regime"
If CPU(t-1) > threshold:
    CPU(t) = c₂ + φ₃·CPU(t-1) + φ₄·CPU(t-2) + noise    ← "High CPU regime"
```

**How threshold is found:** Grid search over the 15th–85th percentiles of CPU values, selecting the one with the lowest BIC.

**Strengths:** Naturally captures low/high CPU states, interpretable (you can explain "normal mode" vs "busy mode")  
**Weaknesses:** Only two regimes, threshold is fixed during prediction

**This is the best-performing non-LSTM model (RMSE = 4.21).**

---

### Model C: MS-AR (Markov Switching Autoregressive)

**What is it?** Like SETAR, but instead of a hard threshold, the regime switching is governed by **hidden probabilities**. The system can be in either state and switches between them with some probability.

```
State 0: "Idle/Low"    →  Low mean CPU, low variance
State 1: "Busy/High"   →  High mean CPU, high variance

Transition probabilities:
  P(stay in low)  = p₀₀ ≈ 0.95  (95% chance of staying idle)
  P(stay in high) = p₁₁ ≈ 0.92  (92% chance of staying busy)
```

**Strengths:** Probabilistic regime assignment (smoother transitions), can estimate expected duration in each state  
**Weaknesses:** Slower to train, harder to interpret

---

### Model D: LSTM (Long Short-Term Memory — Deep Learning)

**What is it?** A neural network designed specifically for sequential data. It has a "memory cell" that can remember patterns across long time spans.

**Architecture:**
```
Input (24 time steps × features)
    ↓
LSTM Layer 1 (64 hidden units, dropout 0.2)
    ↓
LSTM Layer 2 (64 hidden units, dropout 0.2)
    ↓
Fully Connected Layer → Single output (predicted CPU %)
```

**Training:** Adam optimizer, learning rate 0.001, 100 epochs, early stopping when validation loss stops improving.

**Sliding window:** Uses the past 24 time steps (= past 2 hours of data) to predict the next time step.

**Strengths:** Captures long-range dependencies, learns complex non-linear patterns automatically  
**Weaknesses:** Black box (hard to interpret), needs GPU for efficient training, requires more data

---

### Model Comparison Results

| Model | RMSE ↓ | MAE ↓ | MAPE ↓ | Notes |
|-------|--------|-------|--------|-------|
| **LSTM** | **4.41** | **3.20** | **7.51%** | **Best overall** — deep learning advantage |
| **SETAR** | 4.21 | 3.25 | 7.93% | Best interpretable model |
| MS-AR | 4.85 | 3.72 | 9.02% | Good regime detection |
| SARIMAX | 16.31 | 12.79 | 29.67% | Linear limitation |

**Metric Definitions:**
- **RMSE (Root Mean Squared Error):** Average prediction error in CPU %. Lower = better. Penalises big errors more.
- **MAE (Mean Absolute Error):** Average absolute prediction error in CPU %. Lower = better.
- **MAPE (Mean Absolute Percentage Error):** Error as a percentage of the actual value. Lower = better.

**Why SARIMAX is worst:** It's a linear model forced to resample to hourly data (losing 5-min detail). It simply cannot handle the sudden CPU spikes (kurtosis = 0.597).

**Why SETAR is preferred for the scheduler:** Even though LSTM has slightly better RMSE, SETAR's threshold mechanism is **interpretable** — you can explain to engineers exactly *why* a prediction was made. The scheduler needs explainability, not just accuracy.

### Output Files
- `data/model_comparison.csv` — RMSE, MAE, MAPE for all models
- `data/forecast_results.csv` — SETAR's test-set predictions
- `data/lstm_forecast_results.csv` — LSTM's test-set predictions
- `data/msar_forecast_results.csv` — MS-AR's test-set predictions
- `models/setar_model.json`, `models/lstm_best.pt`, `models/sarimax_model.pkl`

---

## 8. Notebook 5 — Carbon-Aware Scheduling

### What Does This Notebook Do?

This is where everything comes together. It simulates a data center scheduler that decides **when** to run batch jobs based on CPU forecasts and carbon intensity predictions.

### Three Scheduling Strategies

#### Strategy 1: FIFO (Carbon-Blind Baseline)
Run every job **immediately** when it arrives. No optimization at all.
- ⏱ **Delay:** 0 minutes
- 🏭 **CO₂:** Highest (baseline)

#### Strategy 2: Carbon-Aware (Naive)
For each batch job, check the next N hours of carbon intensity forecasts. Schedule the job at the time window with the **lowest carbon intensity**, within the job's deadline.
- ⏱ **Delay:** Moderate (jobs wait for green windows)
- 🏭 **CO₂:** Significantly lower

#### Strategy 3: Risk-Aware (Uses Conformal Intervals)
Same as Strategy 2, but **only defer a job if the forecast is confident** (conformal interval is narrow). If the forecast is uncertain, run the job immediately to avoid missing the deadline.
- ⏱ **Delay:** Balanced
- 🏭 **CO₂:** Lowest — best of both worlds

### How the Power Model Works

Each server's power consumption is modeled as:
```
P(u) = P_idle + (P_max - P_idle) × u

Where:
  u     = CPU utilisation (0 to 1)
  P_idle = 200W (power when doing nothing)
  P_max  = 500W (power at full load)
```

Total CO₂ for a time period = Power × Duration × Carbon Intensity:
```
CO₂ = Σ P(uₜ) × Δt × CI(t)
```

### Pareto Analysis (Trade-off Curve)

The "Pareto frontier" shows the **optimal trade-off** between CO₂ savings and job delay:

| Max Job Flexibility | CO₂ Reduction | Average Delay |
|--------------------:|:--------------|:-------------|
| 1 hour | 0.93% | 27 min |
| 2 hours | 2.25% | 64 min |
| 4 hours | 3.23% | 176 min |
| 6 hours | 4.25% | 262 min |

**Interpretation:** If you're willing to let batch jobs wait up to 4 hours, you can cut CO₂ by 3.23%. More flexibility → more savings, but with diminishing returns.

### Output Files
- `data/scheduling_results.csv` — CO₂ totals for each strategy
- `data/pareto_analysis.csv` — Flexibility vs. savings data
- `data/spark_sql_scheduling.csv` — Spark SQL optimized scheduling

---

## 9. Notebook 6 — Conformal Prediction (Uncertainty)

### What Does This Notebook Do?

This is the **research novelty** of the project. It takes the SETAR model's predictions and wraps them with **mathematically guaranteed prediction intervals**.

### The Problem with Plain Forecasts

When SETAR predicts "CPU will be 45%", how much should we trust that? It could be off by 2% or 20%. Without knowing the uncertainty, the scheduler can't make safe decisions.

### What Is Conformal Prediction?

Conformal Prediction is a **distribution-free** uncertainty quantification method. Unlike traditional confidence intervals that assume the data follows a normal distribution, conformal prediction works with **any data distribution** and provides a **mathematical guarantee**.

**The Guarantee:**
```
P(actual value falls within [lower, upper]) ≥ 1 - α

For α = 0.05 → 95% of the time, the actual value will be inside the interval.
```

This isn't just a hope — it's a **theorem** with a mathematical proof.

### How It Works — Split Conformal (Simple Version)

```
Step 1: Split data into Train (60%) + Calibration (25%) + Test (15%)

Step 2: Train the SETAR model on the Train set

Step 3: On the Calibration set, compute residuals:
         score_i = |actual_i - predicted_i|
         These are "nonconformity scores" — how wrong was the model?

Step 4: Find the quantile:
         Q = the (1-α) quantile of all calibration scores
         For 95% coverage: Q = 95th percentile of |errors|

Step 5: For any new prediction:
         Interval = [predicted - Q,  predicted + Q]
```

### How It Works — MAPIE Jackknife+ (Advanced Version)

MAPIE gives **adaptive intervals** — wider when the model is uncertain, narrower when confident:

```python
from mapie.regression import CrossConformalRegressor

mapie = CrossConformalRegressor(
    estimator=SETARRegressor(),
    confidence_level=[0.90, 0.95],
    method="plus",    # Jackknife+ — tighter intervals
    cv=5              # 5-fold cross-conformal
)
```

### Coverage Results

| Method | Target | Achieved | Average Width |
|--------|--------|----------|--------------|
| Split Conformal (90%) | 90% | ~91% | ±6.25% CPU |
| Split Conformal (95%) | 95% | ~96% | ±7.87% CPU |
| Split Conformal (99%) | 99% | ~99% | ±11.0% CPU |
| MAPIE Jackknife+ (95%) | 95% | ~96% | ~±7.75% CPU (adaptive) |

**Plain English:** When we say "95% coverage with ±7.87%", it means: if the model predicts CPU = 45%, the actual value will be between 37.13% and 52.87% at least 95% of the time.

### How the Scheduler Uses This

```
For each batch job:
    1. Get CPU forecast + conformal interval for candidate time windows
    2. IF interval_width < 5% (= high confidence):
         → Safe to defer to lowest-CI window
    3. ELSE (forecast is uncertain):
         → Run immediately (conservative — protect SLA)
```

### Output Files
- `data/conformal_intervals.csv` — Predictions + 95% intervals for every test timestamp
- `data/spark_conformal_per_machine.csv` — Per-machine conformal results (via Spark)
- `models/mapie_model.pkl` — Saved MAPIE model

---

## 10. Dashboard — All 12 Pages Explained In Detail

The dashboard is built with **Streamlit** and uses **Plotly** for interactive charts. It has a premium dark theme with glassmorphism effects, animations, and a sidebar navigation.

### Sidebar Features

- **CarbonDC Logo** with animated float effect and version badge (v3.0)
- **Live Carbon Clock:** Shows current grid carbon intensity based on hour of day, with a color-coded progress bar (green = clean, yellow = moderate, red = dirty) and the best scheduling window
- **Pipeline Status:** 11 green/grey dots showing which data files exist — a real-time health check for the pipeline
- **Three Navigation Sections:**
  - Big Data Analytics (4 pages)
  - Time Series Analysis (4 pages)
  - Applied Analytics (4 pages)

---

### PAGE 1: Overview

**What it shows:** A high-level summary of the entire project in one screen.

**Components:**
- **Impact Summary Banner:** A green-gradient hero card showing the maximum CO₂ reduction achievable (e.g., "This data center can cut CO₂ emissions by 4.2% — saving X metric tons")
- **Hero Banner:** Project title "Carbon-Aware Data Center Scheduling" with a gradient-animated top border
- **6 KPI Cards:**
  1. Raw rows ingested (e.g., 95,000,000)
  2. Compression ratio (e.g., 4,238×)
  3. Best model RMSE (e.g., 4.2100)
  4. Max CO₂ reduction percentage
  5. Number of models trained (3+)
  6. Conformal coverage (95%)

**Purpose:** Give anyone (including non-technical viewers) an instant snapshot of what this project achieved.

---

### PAGE 2: Data Ingestion (How Data Was Loaded)

**What it shows:** The Spark ETL pipeline process and data reduction.

**Components:**
- **5 KPI Cards:** Raw rows, rows after cleaning, aggregated rows, compression ratio, trace duration
- **Data Reduction Waterfall Chart:** A waterfall chart showing how 95M rows get reduced step by step (raw → nulls removed → aggregated → subset). Each bar shows how many rows were removed at each stage.
- **Configuration Info Box:** Bucket size (300s), aggregation method (mean), invalid sensor handling, number of machines
- **Selected Machines Pills:** Visual pills showing the 10 selected machine IDs
- **Metric Columns Table:** Lists all sensor columns (CPU, memory, network, disk) with descriptions
- **Data Volume Funnel Chart:** An inverted funnel showing data shrinking at each ETL stage
- **Machine Treemap:** A treemap showing the 10 selected machines as tiles

---

### PAGE 3: Data Quality Check

**What it shows:** How clean the data is after Spark ETL processing.

**Components:**
- **3 KPI Cards:** Total rows in subset, number of machines, average null rate
- **4 Tabs:**
  - **Null Heatmap:** A color-coded heatmap with machines on the Y-axis and metrics on the X-axis. Colors go from dark blue (0% nulls = perfect) to red (high nulls = problem). After ETL, this should be nearly all dark.
  - **CPU Distribution (Violin Plots):** One violin per machine showing the spread of CPU values. Wide violins = high variation. The inner box shows median, Q1, Q3. The inner line shows the mean.
  - **Memory Distribution (Violin Plots):** Same as CPU but for memory utilisation.
  - **Other Sensors (Histogram):** Select any sensor (net_in, net_out, disk_io) and see its distribution.

---

### PAGE 4: Feature Deep Dive (Data Profiling)

**What it shows:** Statistical properties and correlations of all features.

**Components:**
- **4 KPI Cards:** Number of features, timestamps, memory usage, total nulls
- **4 Tabs:**
  - **Correlation Matrix:** A 56×56 heatmap showing Pearson correlation between every pair of features. Red = negative correlation, Blue = positive. CPU columns of different machines are positively correlated (they follow the same daily pattern).
  - **Summary Statistics Table:** Mean, std, min, 25th/50th/75th percentile, max for every feature.
  - **Scaler Parameters:** Shows min/max values used for Min-Max scaling, with a grouped bar chart comparing feature ranges before scaling.
  - **MLlib BDA Stats:** Spark MLlib distributed statistics (mean, variance, min, max) plus the Spark MLlib correlation matrix computed using distributed processing.

---

### PAGE 5: Explore Signals (Time Series Explorer)

**What it shows:** Interactive exploration of the raw time series data.

**Components:**
- **4 KPI Cards:** Timestamps, number of CPU series, number of memory series, duration in hours
- **3 Tabs:**
  - **CPU Series:** Interactive line chart. Use the multi-select dropdown to choose which machines' CPU values to plot. Each line is a different color. Hover to see exact values. Zoom, pan, select ranges.
  - **Memory Series:** Same concept but for memory utilisation.
  - **Carbon Intensity Tab:**
    - 4 KPI cards: Mean, Min, Max, Std dev of carbon intensity
    - Full time series plot with green fill-to-zero
    - **Hourly profile bar chart:** Shows average carbon intensity per hour of day. Green bars = low-CI hours (best for scheduling), amber = moderate. The clear "U-shape" shows midday is cleanest.

---

### PAGE 6: Trends & Seasonality (Stationarity & Trends)

**What it shows:** The results of statistical diagnostics from Notebook 3.

**Components:**
- **5 Tabs:**
  - **ADF Tests:** A table showing the ADF test statistic, p-value, and stationarity verdict for each series. With an info box explaining what this means.
  - **STL — CPU Load:** KPI for seasonal strength (Fs), then a 4-panel interactive STL decomposition:
    - Panel 1 (white): Original observed CPU series
    - Panel 2 (blue): Extracted trend (slow-moving average)
    - Panel 3 (green): Seasonal component (daily oscillation)
    - Panel 4 (amber): Residual (random noise)
  - **STL — Carbon:** Same 4-panel decomposition but for carbon intensity.
  - **Distributions:** A table of skewness/kurtosis values, plus an overlaid histogram showing CPU (blue) and carbon intensity (green) distributions.
  - **Spark Diagnostics:** Per-machine ADF statistics (bar chart), Spark SQL per-machine stats table, ACF(1) bar chart, kurtosis & skewness grouped bar chart — all computed via distributed Spark processing.

---

### PAGE 7: Pattern Detection (Autocorrelation)

**What it shows:** ACF and PACF analysis for understanding data memory.

**Components:**
- **2 Tabs:**
  - **Key Lags:** A bar chart showing ACF values at 4 important lags:
    - Lag 1 (5 min): ~0.97 — extremely high
    - Lag 12 (1 hour): ~0.80 — strong
    - Lag 144 (12 hours): ~0.50 — moderate
    - Lag 288 (24 hours): ~0.70 — **daily seasonal spike**
    - Info box explaining what each lag means
  - **Full ACF / PACF:** Two-panel chart:
    - Top: ACF with 576 bars (2 days of lags) — shows the slow decay and the 24h spike
    - Bottom: PACF with 100 bars — shows the sharp cut-off at lag 2

---

### PAGE 8: Model Performance (Forecasting Models)

**What it shows:** Comparison and visualisation of all 4 forecasting models.

**Components:**
- **Ranking KPI Cards:** Models ranked by RMSE (1st, 2nd, 3rd...)
- **9 Tabs:**
  - **Metrics Comparison:** Side-by-side bar charts for RMSE, MAE, MAPE, plus the raw comparison table, plus an RMSE Improvement Waterfall showing how each model improves over the worst.
  - **SETAR Forecast:** Time series plot with actual (white) vs predicted (blue dashed). If carbon intensity is available, a dual-axis chart shows forecast error alongside CI.
  - **MS-AR Forecast:** MS-AR prediction vs actual, plus error histogram and actual-vs-predicted scatter plot.
  - **LSTM Forecast:** LSTM prediction vs actual, error distribution histogram, actual-vs-predicted scatter, plus an info box about the LSTM architecture.
  - **Residuals:** Error histogram + actual-vs-predicted scatter plot + Cumulative Error Distribution (CDF) comparing SETAR and MS-AR.
  - **Model Radar:** A polar radar chart where each axis is a metric (inverted: higher = better). Larger area = better model. Shows SETAR and MS-AR dominating SARIMAX.
  - **Feature Importance:** Explains which input features matter most (lag_1 ~50%, rolling_mean_12 ~23%, hour, carbon_intensity, lag_288).
  - **Error Over Time:** Line chart of absolute prediction error over the test period, with mean error line.
  - **Per-Machine SETAR (Spark):** Shows SETAR models trained independently on each machine via Spark's `applyInPandas`. Includes per-machine RMSE bar chart and threshold bar chart.

---

### PAGE 9: Carbon Footprint (Power & Emissions)

**What it shows:** Carbon intensity patterns and CO₂ by scheduling strategy.

**Components:**
- **4 KPI Cards:** Mean, Min (cleanest), Max (dirtiest), Std dev of carbon intensity
- **3 Tabs:**
  - **CI Time Series:** Full carbon intensity over 8 days with the mean line.
  - **Hourly Profile:** Bar chart with error bars (±std dev) showing average CI by hour. Green bars (low CI) cluster around midday = solar peak.
  - **Day-Hour Heatmap:** A grid with days on Y-axis and hours on X-axis. Color represents CI. Makes the daily solar pattern visually obvious.
- **CO₂ by Scheduling Strategy:** KPI cards for each strategy's total CO₂, plus a bar chart comparing them (red = FIFO baseline, amber = carbon-aware, green = risk-aware).

---

### PAGE 10: Cluster Load (CPU Load Analysis)

**What it shows:** Deep dive into CPU utilisation patterns across the cluster.

**Components:**
- **4 KPI Cards:** Mean CPU, Peak load, Min load, Std dev
- **3 Tabs:**
  - **Machine Heatmap:** A 2D heatmap with time on X-axis and machines on Y-axis. Color = CPU utilisation. Shows which machines are busiest at which times.
  - **Rolling Average:** Three lines on one chart — raw 5-min data (faint grey), 1-hour rolling average (blue), 4-hour rolling average (amber). Shows how smoothing reveals the daily pattern.
  - **Load Distributions (Violin):** One violin per machine showing its CPU distribution. Wider = more variation. Helps identify consistently busy vs. idle machines.
- **Peak Load Events Table:** Lists all timestamps where CPU exceeded the 90th percentile, with a count of how many such events exist.

---

### PAGE 11: Try the Simulator (Scheduling Strategies)

**What it shows:** An interactive scheduling simulator with detailed strategy comparison.

**Components:**
- **Strategy KPI Cards + CO₂ Saved Card**
- **7 Tabs:**
  - **Comparison:** Bar chart of CO₂ by strategy
  - **Strategy Detail:** Full scheduling results table + delay comparison bar chart + info box explaining each strategy
  - **Pareto Front:** The CO₂ savings vs. flexibility trade-off curve
  - **Strategy Radar:** Polar chart comparing strategies on Low-CO₂, Low-Delay, Low-CI axes
  - **CO₂ Savings Waterfall:** Shows how much CO₂ each strategy saves compared to baseline, plus a donut chart (CO₂ saved vs remaining)
  - **Raw Data:** Full scheduling results tables
  - **Spark SQL Optimization:** Shows Spark SQL scheduling results — jobs optimized, average CI, distribution histogram

- **Interactive Job Scheduler Simulator:**
  - **3 Sliders:** Number of batch jobs (10–500), Average CPU per job (5–80%), Max delay allowed (0–12 hours)
  - **4 Result KPI Cards:** FIFO emissions, Carbon-Aware emissions, Risk-Aware emissions, Max CO₂ saved %
  - **Comparison Bar Chart:** Updated live as you move sliders
  - **24h Carbon Intensity Curve:** Shows the daily CI profile with the optimal scheduling window shaded in green

---

### PAGE 12: Prediction Confidence (Uncertainty & Conformal)

**What it shows:** Conformal prediction intervals and coverage analysis.

**Components:**
- **4 KPI Cards:** Empirical coverage (%), average half-width, test points, number of violations
- **4 Tabs:**
  - **Prediction Intervals:** The main chart — a shaded band (light blue) showing the 95% prediction interval, with actual (white line) and predicted (blue dashed) overlaid. Red X markers show violations (where actual fell outside the band).
  - **Rolling Coverage:** A line chart showing coverage rate over a sliding window, with a red dashed line at 95% target. Coverage should hover above 95%.
  - **Width Analysis:** Two charts — histogram of interval widths (left) and width over time (right). Wider intervals = more uncertainty.
  - **Conditional Coverage:** Bar chart of coverage by hour of day. All bars should be ≥95% (green). Any below 95% (red) indicates hours where the model is less reliable.

- **Calibration & Interval Analysis Section:**
  - **Calibration Plot (left):** Predicted vs Actual scatter. Blue dots = covered, Red X = violations. Diagonal line = perfect model.
  - **Interval Efficiency Plot (right):** Width vs absolute error scatter. Points below the Width/2 line = efficient intervals (not unnecessarily wide).

- **Distributed Per-Machine Conformal (Spark):**
  - 4 KPI cards: Machines, avg coverage, avg width, avg RMSE
  - Per-machine results table
  - Two bar charts: coverage per machine (with 95% target line) and width per machine

---

## 11. All Graphs & Visualisations Explained

### Static Figures (Pre-rendered PNGs in `figures/`)

| Figure | What It Shows |
|--------|---------------|
| `adf_stationarity.png` | Side-by-side plots of raw series vs first-differenced series. The differenced version looks more "stable" (stationary). |
| `stl_decomposition_cpu.png` | 4-panel STL: Observed → Trend → Seasonal → Residual for CPU |
| `stl_decomposition_ci.png` | Same 4-panel STL for carbon intensity |
| `acf_pacf_plots.png` | 2×2 grid: ACF and PACF for both CPU and carbon intensity |
| `distributions.png` | Histograms + KDE curves for CPU and carbon intensity — shows the shape of data |
| `ci_and_cpu_signal.png` | CPU cluster average and carbon intensity plotted together over 3 days |
| `cpu_timeseries_sample.png` | Individual machine CPU time series (all 10 machines) |
| `correlation_heatmap.png` | Full 56×56 Pearson correlation matrix of all features |
| `sarimax_forecast.png` | SARIMAX forecast vs actual on the test set |
| `setar_forecast.png` | SETAR forecast vs actual |
| `ml_ensemble_forecast.png` | SETAR forecast with broader context |
| `msar_forecast.png` | MS-AR forecast vs actual |
| `lstm_forecast.png` | LSTM forecast vs actual |
| `lstm_training_curves.png` | LSTM training and validation loss over epochs |
| `model_comparison.png` | Bar chart comparing RMSE/MAE/MAPE across all models |
| `conformal_prediction_intervals.png` | 3 panels: 90%, 95%, 99% conformal intervals |
| `mapie_prediction_intervals.png` | MAPIE 95% adaptive intervals |
| `conditional_coverage.png` | Coverage % by hour of day |
| `pareto_front.png` | CO₂ savings vs flexibility Pareto frontier |
| `scheduling_timeline.png` | Timeline showing when jobs are scheduled under each strategy |
| `batch_ci_distribution.png` | Distribution of carbon intensity at the time batch jobs run |
| `spark_distributed_diagnostics.png` | Per-machine Spark diagnostic charts |
| `spark_mllib_correlation.png` | Spark MLlib computed correlation matrix |
| `spark_per_machine_setar.png` | Per-machine SETAR results via Spark |
| `spark_conformal_per_machine.png` | Per-machine conformal coverage via Spark |

---

## 12. Key Formulas & Equations

### Augmented Dickey-Fuller (ADF) Test
```
Δyₜ = α + βt + γy_{t-1} + Σᵢ δᵢΔy_{t-i} + εₜ

If p-value < 0.05 → Stationary ✓
```

### STL Decomposition
```
yₜ = Tₜ + Sₜ + Rₜ     (Trend + Seasonal + Residual)
```

### Seasonal Strength
```
Fs = max(0, 1 − Var(Rₜ) / Var(Sₜ + Rₜ))
Fs > 0.64 → Strong seasonality
```

### Autocorrelation (ACF)
```
ρ(k) = Cov(yₜ, y_{t-k}) / Var(yₜ)
```

### SARIMAX Model
```
φₚ(L) · Φₚ(Lˢ) · (1-L)ᵈ · (1-Lˢ)ᴰ · yₜ = β·xₜ + θ_q(L) · Θ_Q(Lˢ) · εₜ

This project: SARIMAX(2,1,1)(1,1,1,24) with exog=carbon_intensity
```

### SETAR Model
```
yₜ = { c₁ + φ₁₁·y_{t-1} + φ₁₂·y_{t-2} + ε₁ₜ   if y_{t-1} ≤ γ   (low regime)
     { c₂ + φ₂₁·y_{t-1} + φ₂₂·y_{t-2} + ε₂ₜ   if y_{t-1} > γ   (high regime)
```

### MS-AR Model
```
Y_t | S_t=k ~ N(μ_k + φ_k·Y_{t-1}, σ²_k)
P(S_t=j | S_{t-1}=i) = p_ij     (transition probabilities)
```

### LSTM (simplified)
```
fₜ = σ(Wf·[h_{t-1}, xₜ] + bf)      (forget gate)
iₜ = σ(Wi·[h_{t-1}, xₜ] + bi)      (input gate)
Cₜ = fₜ·C_{t-1} + iₜ·tanh(Wc·[h_{t-1}, xₜ] + bc)   (cell state)
oₜ = σ(Wo·[h_{t-1}, xₜ] + bo)      (output gate)
hₜ = oₜ·tanh(Cₜ)                    (hidden state)
```

### Power Model
```
P(u) = P_idle + (P_max - P_idle) × u
CO₂ = Σ P(uₜ) × Δt × CI(t)
```

### Min-Max Scaling
```
x_scaled = (x - x_min) / (x_max - x_min)     Output ∈ [0, 1]
```

### Split Conformal Prediction
```
1. Train model on Train set (60%)
2. Calibration scores: sᵢ = |yᵢ - ŷᵢ|
3. Quantile: Q = (1-α) percentile of {s₁, ..., sₙ}
4. Interval: [ŷ - Q,  ŷ + Q]

Guarantee: P(y_new ∈ [lower, upper]) ≥ 1 - α
```

### Evaluation Metrics
```
RMSE = √(1/n · Σ(yᵢ - ŷᵢ)²)
MAE  = 1/n · Σ|yᵢ - ŷᵢ|
MAPE = 1/n · Σ|yᵢ - ŷᵢ|/|yᵢ| × 100%
```

---

## 13. BDA Concepts Used (Big Data Analytics)

### 1. Distributed Processing with Apache Spark
- **SparkSession:** Entry point for all Spark operations, configured with 8 GB driver memory
- **Lazy evaluation:** Spark doesn't execute anything until an action (like `.count()` or `.write()`) is called
- **Partitioning:** Data is split across CPU cores for parallel processing
- **Why Spark:** 95M rows don't fit in pandas RAM; Spark handles it across distributed partitions

### 2. Schema Enforcement
- Define exact data types (`StringType`, `LongType`, `DoubleType`) BEFORE loading
- Malformed rows are dropped (`DROPMALFORMED`)
- This prevents type errors and corrupted data from entering the pipeline

### 3. Data Cleaning at Scale
- `F.when` / `F.otherwise` for conditional column transformations
- Replace sensor failures (-1) and impossible values (>100%) with NULL
- Spark SQL `Window` functions for per-machine statistics

### 4. Temporal Aggregation
- `F.floor(col / 300) * 300` for 5-minute bucketing
- `groupBy` + `agg(F.mean(...))` for per-bucket averaging
- This Reduces noise and aligns all machines to the same time grid

### 5. Parquet Storage
- Columnar format → only reads columns you need (not entire rows)
- Built-in compression (Snappy by default)
- Partition-by-machine layout for efficient per-machine queries

### 6. Spark MLlib (Distributed Statistics)
- `VectorAssembler` + `Summarizer` for distributed mean/variance/min/max
- Spark `Correlation.corr()` for distributed correlation matrix
- `applyInPandas` for running per-machine Python functions in parallel

### 7. Spark SQL
- Window functions (`ROW_NUMBER() OVER PARTITION BY`)
- `JOIN + GROUP BY` for optimal scheduling window assignment
- Declarative SQL on distributed DataFrames

---

## 14. TSA Concepts Used (Time Series Analysis)

### 1. Stationarity
A series whose mean and variance don't change over time. Required by ARIMA-type models. Tested via ADF.

### 2. Differencing
Transforms a non-stationary series into a stationary one:
```
y'ₜ = yₜ - y_{t-1}
```

### 3. Seasonality
A repeating pattern over a fixed period. This project: 24-hour (diurnal) cycle caused by business hours.

### 4. Autocorrelation
How a series correlates with its own lagged values. High ACF at lag 1 = strong short-term memory. Spike at lag 288 = 24h pattern.

### 5. ARIMA Family
Auto-Regressive Integrated Moving Average — the workhorse of classical time series.

### 6. Threshold Models (SETAR)
Non-linear extension of AR — different equations for different regimes.

### 7. Regime-Switching Models (MS-AR)
Hidden Markov Model applied to time series — probabilistic regime assignment.

### 8. Deep Learning (LSTM)
Neural network with memory cells that can learn long-range dependencies in sequences.

### 9. Conformal Prediction
Distribution-free uncertainty quantification with guaranteed coverage.

### 10. Feature Engineering for Time Series
- Lag features: y_{t-1}, y_{t-2}, y_{t-288}
- Rolling statistics: rolling mean and std over 12 windows (1 hour)
- Calendar features: hour, day of week
- Exogenous signals: carbon intensity

---

## 15. What Makes This Project Novel

### Novelty 1: Conformal Prediction for Carbon Scheduling (PRIMARY)

All existing carbon-aware schedulers (Microsoft's Carbon Aware SDK, Google's Carbon-Intelligent Platform) use **plain forecasts** — they assume the prediction is always correct. This fails in practice.

This project is the **first** to:
- Attach distribution-free, mathematically guaranteed prediction intervals to workload forecasts
- Use those intervals for risk-calibrated scheduling decisions
- Only shift jobs when the forecast is provably reliable

```
Traditional:   forecast → schedule at lowest CI   (assumes forecast is perfect)
This project:  forecast + interval → if confident: schedule at lowest CI
                                     if uncertain: run now (safe)
```

### Novelty 2: End-to-End Integrated Pipeline

Instead of separate studies, this project connects:
- Diagnostic findings → Model hyperparameters
- Forecast outputs → Scheduler inputs
- Conformal intervals → Scheduling gate

### Novelty 3: Pareto Trade-off Quantification

Exact numbers for the CO₂ savings vs. delay trade-off at every flexibility level.

---

## 16. Project Structure & Files

```
bda-and-tsa/
│
├── 01_spark_etl.ipynb              ← Notebook 1: Spark ETL pipeline
├── 02_timeseries_reconstruction.ipynb  ← Notebook 2: Build time series
├── 03_tsa_diagnostics.ipynb        ← Notebook 3: Statistical tests
├── 04_forecasting_models.ipynb     ← Notebook 4: Train & compare models
├── 05_carbon_scheduling.ipynb      ← Notebook 5: Scheduling simulation
├── 06_conformal_prediction.ipynb   ← Notebook 6: Uncertainty quantification
│
├── dashboard/
│   └── app.py                      ← Streamlit dashboard (12 pages)
│
├── data/
│   ├── _spark_input.csv            ← Raw input for Spark
│   ├── clean_parquet/subset/       ← Partitioned Parquet output from Spark
│   ├── timeseries_ready.csv        ← Wide-format time series
│   ├── timeseries_scaled.csv       ← Scaled [0,1] version
│   ├── scaler_params.csv           ← Min/max per feature
│   ├── etl_summary.json            ← ETL statistics
│   ├── selected_machines.json      ← The 10 chosen machines
│   ├── diagnostics_summary.json    ← ADF, STL, ACF, kurtosis results
│   ├── spark_diagnostics.json      ← Per-machine Spark diagnostics
│   ├── mllib_summary_stats.json    ← Spark MLlib statistics
│   ├── spark_correlation_matrix.csv ← Spark MLlib correlation
│   ├── spark_machine_summary.csv   ← Per-machine summary from Spark
│   ├── model_comparison.csv        ← RMSE/MAE/MAPE comparison
│   ├── forecast_results.csv        ← SETAR predictions
│   ├── lstm_forecast_results.csv   ← LSTM predictions
│   ├── msar_forecast_results.csv   ← MS-AR predictions
│   ├── scheduling_results.csv      ← CO₂ per strategy
│   ├── pareto_analysis.csv         ← Flexibility vs savings
│   ├── spark_sql_scheduling.csv    ← Spark SQL optimized scheduling
│   ├── conformal_intervals.csv     ← 95% prediction intervals
│   └── spark_conformal_per_machine.csv ← Per-machine conformal
│
├── figures/                        ← 25 pre-rendered PNG charts
│   ├── adf_stationarity.png
│   ├── stl_decomposition_cpu.png
│   ├── acf_pacf_plots.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── conformal_prediction_intervals.png
│   └── ... (25 total)
│
├── models/
│   ├── setar_model.json            ← SETAR parameters
│   ├── lstm_best.pt                ← LSTM PyTorch weights
│   ├── sarimax_model.pkl           ← SARIMAX model
│   ├── mapie_model.pkl             ← MAPIE conformal model
│   ├── random_forest_model.pkl     ← RF model
│   └── xgboost_model.pkl          ← XGBoost model
│
├── report format/
│   └── main.tex                    ← LaTeX report template
│
├── requirements.txt                ← Python dependencies
├── TSA_DOCUMENTATION.md            ← TSA-focused technical documentation
└── README.md                       ← THIS FILE
```

---

## 17. How to Run the Project

### Prerequisites
- Python 3.9+
- Java 8 or 11 (for PySpark)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebooks (in order)
```bash
# 1. Start with Spark ETL
jupyter notebook 01_spark_etl.ipynb

# 2. Then time series reconstruction
jupyter notebook 02_timeseries_reconstruction.ipynb

# 3. Statistical diagnostics
jupyter notebook 03_tsa_diagnostics.ipynb

# 4. Forecasting models (takes the longest)
jupyter notebook 04_forecasting_models.ipynb

# 5. Carbon scheduling simulation
jupyter notebook 05_carbon_scheduling.ipynb

# 6. Conformal prediction
jupyter notebook 06_conformal_prediction.ipynb
```

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

### Libraries Used

| Library | Purpose |
|---------|---------|
| `pyspark` | Distributed ETL, Spark SQL, MLlib |
| `pandas` | DataFrames, time series operations |
| `numpy` | Numerical computation |
| `statsmodels` | ADF, STL, SARIMAX, MS-AR, ACF/PACF |
| `scikit-learn` | MinMaxScaler, metrics, base estimator |
| `torch` (PyTorch) | LSTM neural network |
| `mapie` | Conformal prediction (Jackknife+) |
| `streamlit` | Dashboard web app |
| `plotly` | Interactive charts |
| `matplotlib` / `seaborn` | Static charts in notebooks |
| `scipy` | Statistical functions (skew, kurtosis) |
| `pyarrow` | Parquet read/write |

---

## 18. Frequently Asked Questions

### General Questions

**Q: What problem does this project solve?**  
A: It reduces CO₂ emissions from data centers by smartly scheduling batch jobs during low-carbon electricity periods, using CPU forecasting and uncertainty quantification.

**Q: What's the maximum CO₂ reduction achieved?**  
A: Up to 4.25% with 6 hours of scheduling flexibility. Even 1 hour of flexibility saves ~1%.

**Q: Why not just use renewable energy?**  
A: This is complementary. Even with renewables, the grid mix varies hourly. Smart scheduling extracts maximum value from whatever clean energy is already available — no new investment needed.

---

### BDA Questions

**Q: Why use Apache Spark instead of Pandas?**  
A: The raw dataset has 95 million rows (~3.4 GB). Pandas loads everything into RAM and would crash on most machines. Spark distributes the workload across CPU cores and manages memory efficiently.

**Q: What does "schema enforcement" mean?**  
A: We tell Spark the exact type of each column before loading. If a row has corrupted data that doesn't match the schema, it's automatically dropped. This is a BDA best practice for data quality.

**Q: What is Parquet and why use it?**  
A: Parquet is a columnar storage format. Instead of reading full rows, it reads only the columns you need. It also compresses data well. Result: faster reads, smaller files.

**Q: What is `applyInPandas`?**  
A: A Spark function that lets you apply a Python/Pandas function to each partition (e.g., each machine) in parallel. We use it to train SETAR models and run ADF tests on each machine simultaneously.

**Q: What are Window functions in Spark SQL?**  
A: They let you compute values across related rows (like "rank this row within its group" or "compute running average within each machine"). Used for optimal scheduling window selection.

---

### TSA Questions

**Q: What does "stationary" mean in simple terms?**  
A: The data doesn't trend upward or downward over time. Its average and spread stay roughly constant. Like the temperature in a city — it goes up and down daily but doesn't keep climbing forever.

**Q: Why is the ADF test important?**  
A: ARIMA models only work on stationary data. The ADF test mathematically proves our data is suitable for these models.

**Q: What does STL decomposition tell us?**  
A: It shows that CPU usage has three parts: a slow trend, a daily repeating pattern (seasonal), and random noise. The seasonal component (business hours = busy, nights = quiet) is what the scheduler exploits.

**Q: What is ACF and why is the lag-288 spike important?**  
A: ACF measures how today's data correlates with data from the past. Lag 288 = 24 hours (at 5-min intervals). A spike at lag 288 proves the 24-hour daily cycle exists in the data, confirming that seasonal models should work well.

**Q: Why does SARIMAX perform worst?**  
A: SARIMAX is a linear model that can't handle sudden CPU spikes (kurtosis = 0.597 means heavy tails). It also had to be resampled to hourly data (losing 5-minute detail). Non-linear models like SETAR handle both regime switching and fine-grained data.

**Q: What's the difference between SETAR and MS-AR?**  
A: Both split data into regimes, but differently. SETAR uses a **hard threshold** (if CPU > X, switch to high regime). MS-AR uses **hidden probabilities** (there's a 5% chance of switching to high regime each timestep). SETAR is simpler and more interpretable.

**Q: Why use LSTM if SETAR is preferred?**  
A: LSTM achieves the best raw accuracy (lowest RMSE) by learning complex patterns automatically. But SETAR is preferred for the scheduler because it's **interpretable** — you can explain exactly why a prediction was made, which is important for operational trust.

---

### Conformal Prediction Questions

**Q: What is conformal prediction in simple terms?**  
A: It wraps any prediction with a "safety band." Instead of saying "CPU will be 45%", it says "CPU will be between 37% and 53%, and I'm 95% sure about that." The key is: that 95% is mathematically guaranteed, not just a hope.

**Q: How is this different from normal confidence intervals?**  
A: Normal confidence intervals assume your data follows a bell curve (normal distribution). Conformal prediction makes NO assumptions about the data distribution. It works for any shape of data.

**Q: What does "95% coverage" mean?**  
A: Out of 100 test predictions, at least 95 will have the actual value inside the predicted interval. The other ≤5 are "violations" — permitted by the guarantee.

**Q: What is MAPIE Jackknife+?**  
A: An advanced version that gives **adaptive** intervals — wider when the model is uncertain, narrower when it's confident. This is more efficient than fixed-width intervals.

**Q: How does the scheduler use conformal prediction?**  
A: If the prediction interval is narrow (model is confident), defer the job to a low-carbon window. If the interval is wide (model is uncertain), run the job immediately to avoid risking an SLA violation. This is the "risk-aware" strategy.

---

### Dashboard Questions

**Q: How do I change pages?**  
A: Use the sidebar on the left. Click any page name to navigate.

**Q: What do the green/grey dots in the sidebar mean?**  
A: Each dot represents a data file. Green = the file exists (that pipeline stage has been run). Grey = missing (you need to run the corresponding notebook first).

**Q: Can I interact with the charts?**  
A: Yes! All Plotly charts support zoom (drag), pan (shift+drag), hover (see values), and download (camera icon).

**Q: What's the "Live Carbon Clock" in the sidebar?**  
A: A simulated real-time display showing the current grid carbon intensity based on the hour of day. It updates when you refresh the page.

**Q: Are the simulator results real or simulated?**  
A: The simulator on Page 11 uses a simplified model (not the actual trained models) to give instant results. The actual results from the notebooks are shown on other pages.

---

> **Built with:** Apache Spark · PySpark · Pandas · NumPy · Statsmodels · PyTorch · MAPIE · Scikit-learn · Streamlit · Plotly  
> **Dataset:** Alibaba Cluster Trace v2018  
> **Version:** CarbonDC v3.0
