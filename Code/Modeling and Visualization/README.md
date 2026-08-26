# Modelling and Visualization

Black-carbon **exploratory analysis** and **predictive modelling** on merged dashcam feature tables. Logic lives in importable Python modules.

**No bundled datasets.** Point each notebook at CSV paths on your machine. A conventional folder is `data/` next to the notebooks (create it yourself); any `Path` or string path works.

## Notebooks


| Notebook                                                       | Purpose                                                                                                           |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `[bc_eda_visualization.ipynb](bc_eda_visualization.ipynb)`     | Spatial BC maps, midday traffic–BC temporal analysis, raw vs. cleaned BC distributions                            |
| `[bc_predictive_modeling.ipynb](bc_predictive_modeling.ipynb)` | Gradient Boosting and GRU evaluation, feature scenarios, sampling-interval and transfer experiments, SHAP exports |


## Python modules


| Module           | Role                                                             |
| ---------------- | ---------------------------------------------------------------- |
| `bc_config.py`   | Shared constants (threshold, excluded columns, vehicle lists)    |
| `bc_eda.py`      | EDA data prep and plotting functions                             |
| `bc_modeling.py` | Splits, metrics, cross-validation, personalization, SHAP helpers |


## Input CSV format

Both notebooks expect **merged model feature tables**: one row per time bin, produced by aligning environmental/traffic rows with vehicle proximity counts (see `Code/build_model_features.ipynb` in the parent project).

### Minimum columns (all notebooks)


| Column                    | Role                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `timestamp` or `datetime` | Row time (parsed as datetimes; `datetime` is created from `timestamp` in modelling if needed)                                                                            |
| `bc`                      | Raw black-carbon reading                                                                                                                                                 |
| `lat`, `lon`              | GPS (required for the spatial EDA map)                                                                                                                                   |
| Vehicle count columns     | Per-class proximity bins, e.g. `car_close`, `car_medium`, `car_far`, and matching `*_stopped` variants for `car`, `bus`, `truck` (see `bc_config.STANDARD_VEHICLE_COLS`) |


### Recommended columns (modelling & richer EDA)


| Column                                              | Role                                                                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `bc_cleaned`                                        | Target for regressors (if absent, derived automatically from `bc` via 95th-percentile cap. See `ensure_bc_cleaned()` in `bc_eda.py`) |
| `Temperature`, `Humidity`, `Wind_speed`             | Weather features (`bc_config.DEFAULT_WEATHER_COLS`)                                                                                  |
| `TT_`* (e.g. `TT_currentSpeed`, `TT_freeFlowSpeed`) | TomTom traffic features; used when present                                                                                           |
| `Traffic_flow`                                      | Optional; preferred over `TT_currentSpeed` for spatial map marker size when both exist                                               |
| Lag columns (optional)                              | Names containing `_30s`, `_1m`, `_5m`, `_10m`, etc. (e.g. `car_close_10m`) for lag feature scenarios; skipped if missing             |


Other columns from the merge pipeline (`bc_categorical`, extra `TT_*`, etc.) are ignored or excluded via `bc_config.DEFAULT_EXCLUDE_FEATURES`.

### File naming

Use any filenames. The notebooks only read the paths you set in their **Configuration** cells. A common layout:

```text
Code/Modeling and Visualization/
├── data/                          # create locally; not in git
│   ├── model_features_5s.csv
│   └── model_features_5s_day2.csv
└── outputs/                       # figures & SHAP exports (gitignored)
```

If you run `build_model_features.ipynb`, its output is typically named `model_features_<YYYYMMDD>_<N>s.csv`. Copy or symlink those files into `data/` and set the notebook paths accordingly.

---

## Notebook inputs (configure in each notebook)

Open the notebook, set paths in the **Configuration** / **Setup** cells (`DATA_DIR` is optional; absolute paths are fine), then run from the top.

### `bc_eda_visualization.ipynb`


| Variable      | Used for                                      | Suggested content                                                                |
| ------------- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| `SPATIAL_CSV` | §1 Spatial map                                | One merged table with valid `lat`/`lon`/`bc`                                     |
| `DAY2_CSV`    | §2–3 Midday activity & BC cleaning comparison | Midday session table with `timestamp` and vehicle columns; `bc_cleaned` optional |



| Setting           | Default in notebook | Notes                                     |
| ----------------- | ------------------- | ----------------------------------------- |
| `USE_MAP_OVERLAY` | `True`              | Set `False` without `contextily` or in CI |
| `OUTPUT_DIR`      | `outputs/eda/`      | PNG exports                               |


Midday filtering uses `LOCAL_UTC_OFFSET_HOURS` and `MIDDAY_LOCAL_*` in `bc_config.py` (edit for your study timezone).

### `bc_predictive_modeling.ipynb`


| Variable             | Used for                             | Suggested content                                                                                            |
| -------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `PRIMARY_CSV`        | Main training, CV, holdout, SHAP     | Primary collection day at your chosen bin width (e.g. 5 s)                                                   |
| `TRANSFER_CSV`       | Cross-day transfer & personalization | Second collection day, **same columns and bin width** as primary                                             |
| `INTERVAL_CSV_PATHS` | §4 Sampling-interval comparison      | Dict mapping `"1s"`, `"5s"`, `"10s"`, `"15s"`, `"30s"` → one CSV per interval (same day/campaign is typical) |


Sections that concatenate two days (§7 combined training) reuse `PRIMARY_CSV` and `TRANSFER_CSV`.


| Setting      | Default in notebook              | Notes                                                                                  |
| ------------ | -------------------------------- | -------------------------------------------------------------------------------------- |
| `threshold`  | `bc_config.DEFAULT_BC_THRESHOLD` | High-BC classification for AUC/sensitivity metrics; re-estimate on your data if needed |
| `OUTPUT_DIR` | `outputs/`                       | Metrics tables, SHAP CSVs, correlation figure                                          |


If an interval path in `INTERVAL_CSV_PATHS` is missing on disk, that interval is skipped; at least one existing path is required for §4 to run.

---

## Integration with the upstream pipeline

```text
video_manipulation_utility.ipynb
        ↓
dashcam_vehicle_proximity_analysis.ipynb  →  proximity counts CSV
        ↓
extract_environment_and_traffic_features.ipynb  →  extracted features CSV
        ↓
build_model_features.ipynb  →  model_features_<date>_<N>s.csv
        ↓
bc_eda_visualization.ipynb  →  outputs/eda/*.png
bc_predictive_modeling.ipynb  →  outputs/ (metrics tables, SHAP CSVs)
```

1. **Upstream** — Run the core notebooks under `Code/` (see [project README](../../README.md)). Configure **input and output paths inside each notebook**; artifacts are not shipped with this repository.
2. **Targets and lags** — Modelling expects `bc_cleaned` when you have it; otherwise `prepare_data()` / `ensure_bc_cleaned()` derive it from `bc`. Lag scenarios are skipped automatically when lag columns are absent.
3. **Paths** — Set `PRIMARY_CSV`, `TRANSFER_CSV`, `SPATIAL_CSV`, etc. to your CSV locations before running.

## Outputs


| Path                                 | Contents                                                          |
| ------------------------------------ | ----------------------------------------------------------------- |
| `outputs/eda/`                       | Spatial map, activity/correlation figures, BC cleaning comparison |
| `outputs/shap/`                      | `predicted_black_carbon_results.csv`, `summary.csv`               |
| `outputs/updated_vehicle_impact.png` | Vehicle–prediction correlation bar chart                          |


## Dependencies

Install on top of the base pipeline environment:

```bash
conda activate bc-pipeline
pip install -r requirements.txt
```

- **Required:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- **EDA map overlay:** `contextily` (optional; disable `USE_MAP_OVERLAY` if not installed)
- **Modelling (GRU):** `tensorflow`
- **Interpretability:** `shap`
