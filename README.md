# Wildfire-Prediction-Classification

A supervised multi-class classifier that predicts fire-weather risk bands directly from raw daily weather observations, without implementing or maintaining the Canadian Fire Weather Index (FWI) accumulator formulas.

## Project Scope and Motivation

The Canadian Forest Fire Weather Index (FWI) System is the standard tool used across Canada to estimate fire danger from weather conditions. Its official calculation requires implementing and maintaining a chain of accumulator sub-indices (FFMC, DMC, DC, ISI, BUI, DSR) that build on prior days' values. This project asks a narrower, practical question: can a standard supervised learning model learn to reproduce the *risk band* that the FWI formula would assign, using only the five raw daily weather inputs, without reimplementing the formula itself?

**This project does not predict wildfire occurrence, ignition probability, or fire spread.** It predicts which FWI-derived risk category (`no_risk`, `moderate`, `high`, `very_high`, `extreme`) a given day's weather conditions would fall into, using the official FWI value as ground truth during training. This distinction matters: a high predicted risk band means the weather resembles conditions the FWI system would flag as dangerous, not that a fire is expected to occur.

### How could it be helpful?

If a machine learning model can reliably classify FWI-derived risk bands from raw weather alone, it means practitioners could estimate fire-weather risk without implementing the multi-stage FWI accumulator chain, provided the underlying raw weather features are available. This is the core contribution the project aims to demonstrate.

### Known Limitation 

One of the five input features, `rndays` (rain days), may itself require historical tracking to compute accurately, depending on how it is derived from the raw station record.

## Data

- **Source:** Canadian Wildland Fire Information System (CWFIS), Canadian Forest Service (CFS), Natural Resources Canada (`cwfis.cfs.nrcan.gc.ca`)
- **File:** `cwfis_fwi2020s.csv`
- **Size:** Approximately 3.16 million rows
- **Coverage:** 2,673 Automated Environmental Stations (AES) across Canada
- **Station identifiers:** AES identifiers are used in preference to WMO identifiers, because WMO identifiers are subject to retirement and reuse over time, which would introduce ambiguity into station-level grouping and analysis.

### Input features (five raw daily weather readings)

| Feature | Description |
|---|---|
| `temp` | Daily temperature |
| `rh` | Relative humidity |
| `ws` | Wind speed |
| `precip` | Precipitation |
| `rndays` | Rain days |

### Target variable

The target is a categorical risk band derived from the official FWI value, binned into five classes:

- `no_risk`
- `moderate`
- `high`
- `very_high`
- `extreme`

### Excluded features

All FWI sub-components (`FFMC`, `DMC`, `DC`, `BUI`, `ISI`, `DSR`) and the `fwi` value itself are deliberately excluded from the model's input features. These are excluded because they are derived, in part or in whole, from the same information used to construct the target label, and including them would leak target information into the model rather than testing whether raw weather alone is sufficient.

## Repository Structure

```
Wildfire-Prediction-Classification/
├── 01_eda.ipynb            Exploratory analysis: reasoning and feature keep/drop documentation only
├── 02_preprocessing.ipynb  Executes data transformations
├── 03_modeling.ipynb       Model fitting and evaluation
├── data/                   Raw and processed CWFIS data
└── README.md
```

Reasoning and transformation are intentionally kept in separate notebooks. `01_eda.ipynb` documents *why* a feature or transformation decision was made; `02_preprocessing.ipynb` is where that decision is actually executed against the data. This separation keeps the analytical record distinct from the reproducible pipeline.

## Methodology

### Preprocessing

Preprocessing decisions are reasoned through in the EDA notebook and executed in the preprocessing notebook. Key considerations addressed during this phase include handling of missing values prior to any label construction, and correct bin-edge definition when discretizing the continuous FWI value into risk bands (bin edges must extend to the full range of the data to avoid top-percentile rows falling outside all bins).

### Models Evaluated

Four classifier families were fit and evaluated, drawing on standard classification methods (as covered in ISLP Chapter 4) but implemented using `scikit-learn` and `statsmodels`:

1. **Logistic Regression**
 - Baseline (unweighted)
 - `class_weight='balanced'` variant

2. **Linear Discriminant Analysis (LDA)**
 - Default class priors
 - Balanced priors variant (uniform `[0.2, 0.2, 0.2, 0.2, 0.2]` across the five classes)

3. **Quadratic Discriminant Analysis (QDA)**
 - Default class priors
 - Balanced priors variant

4. **K-Nearest Neighbors (KNN)**
 - Tuned via a stratified subsample sweep across `k = 5` to `k = 120`
 - Best value confirmed at full scale: `k = 100`

### Evaluation Convention

Model performance is reported using macro-averaged F1 score, computed two ways for every model:

- **All five classes included**
- **`no_risk` class excluded** (via the `labels` parameter)

This dual reporting convention exists because `no_risk` is a large and trivially separable class. Reporting macro-F1 across all five classes alone can make a model appear stronger than it actually is at distinguishing between the harder, more consequential classes (`moderate` through `extreme`). Excluding `no_risk` gives a clearer picture of how well the model performs on the classes that matter most operationally.

Note also that scikit-learn orders classes alphabetically (`extreme, high, moderate, no_risk, very_high`) rather than by logical severity. `target_names` in classification reports is display-only; the `labels` parameter is what actually controls which classes are scored.

## Results

### Final Model Standings (macro-F1: all 5 classes / excluding `no_risk`)

| Rank | Model | Macro-F1 (all 5) | Macro-F1 (excl. no_risk) |
|---|---|---|---|
| 1 | KNN (k=100) | 0.689 | 0.656 |
| 2 | Logistic Regression (baseline) | 0.67 | 0.63 |
| 3 | Logistic Regression (balanced) | 0.66 | 0.62 |
| 4 | LDA (balanced priors) | 0.61 | 0.56 |
| 5 | LDA (default) | 0.58 | 0.52 |
| 6 | QDA (default) | 0.54 | 0.48 |
| 7 | QDA (balanced priors) | 0.52 | 0.42 |

**KNN (k=100) is the best-performing model overall** on both evaluation conventions.

