"""Black-carbon predictive modeling utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from bc_config import (
    DEFAULT_BC_THRESHOLD,
    DEFAULT_EXCLUDE_FEATURES,
    DEFAULT_TARGET_COL,
    DEFAULT_WEATHER_COLS,
)
from bc_eda import ensure_bc_cleaned

_TF_AVAILABLE = True
try:
    from tensorflow.keras.layers import GRU, Dense
    from tensorflow.keras.models import Sequential
except ImportError:
    _TF_AVAILABLE = False
    GRU = Dense = Sequential = None

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def calculate_metrics(y_true, y_pred, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)

    try:
        auc = roc_auc_score(y_true_bin, y_pred)
    except ValueError:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "MAE": mae,
        "R2": r2,
        "AUC": auc,
        "Sensitivity": sens,
        "Specificity": spec,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns and "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out

def prepare_data(
    filepath: Union[Path, str],
    exclude_features: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL,
    *,
    ensure_cleaned_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load features and target for modeling.

    Sorts by datetime, drops rows missing target/time, and optionally
    creates ``bc_cleaned`` from raw ``bc`` when only pipeline CSVs are available.
    """
    exclude_features = list(exclude_features or DEFAULT_EXCLUDE_FEATURES)
    df = pd.read_csv(filepath)
    df = _ensure_time_column(df)

    if ensure_cleaned_target and target_col not in df.columns:
        df = ensure_bc_cleaned(df, cleaned_col=target_col)

    if "datetime" in df.columns:
        df = df.dropna(subset=["datetime", target_col]).sort_values("datetime").reset_index(drop=True)

    drop_cols = [target_col, "datetime"] + [c for c in exclude_features if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], df[target_col]

def classify_feature_columns(X: pd.DataFrame) -> dict[str, List[str]]:
    """Group columns into vehicles, traffic, weather, and lag buckets."""
    lag_tokens = ("_30s", "_1m", "_5m", "_10m", "_15m", "_20m", "_30m")
    vehicle_cols = [
        c
        for c in X.columns
        if any(v in c for v in ("car", "bus", "truck")) and not any(t in c for t in lag_tokens)
    ]
    return {
        "traffic_cols": [c for c in X.columns if c.startswith("TT_")],
        "vehicle_cols": vehicle_cols,
        "lag_30s": [c for c in X.columns if "_30s" in c],
        "lag_1m": [c for c in X.columns if "_1m" in c],
        "lag_5m": [c for c in X.columns if "_5m" in c],
        "lag_10m": [c for c in X.columns if "_10m" in c],
        "lag_15m": [c for c in X.columns if "_15m" in c],
        "lag_20m": [c for c in X.columns if "_20m" in c],
        "lag_30m": [c for c in X.columns if "_30m" in c],
    }

def improved_time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
    block_size: int = 20,
    n_splits: int = 1,
    random_state: int = 42,
):
    """Block-stratified train/test split preserving local temporal structure."""
    df_temp = X.copy()
    car_cols = [c for c in X.columns if "car" in c]
    truck_cols = [c for c in X.columns if "truck" in c]
    stop_cols = [c for c in X.columns if "stopped" in c]

    df_temp["traffic_intensity"] = X[car_cols + truck_cols].sum(axis=1) if (car_cols or truck_cols) else 0
    df_temp["stop_activity"] = X[stop_cols].sum(axis=1) if stop_cols else 0
    df_temp["block_id"] = np.arange(len(df_temp)) // block_size

    block_summary = df_temp.groupby("block_id").agg(
        {"traffic_intensity": "mean", "stop_activity": "mean"}
    )
    score = block_summary["traffic_intensity"] + block_summary["stop_activity"] * 2

    n_blocks = len(block_summary)
    n_strata = min(4, max(1, n_blocks // max(2, n_splits)))
    try:
        block_summary["strata"] = pd.qcut(score, q=n_strata, labels=False, duplicates="drop")
    except ValueError:
        block_summary["strata"] = 0

    max_splits = max(1, n_blocks - int(block_summary["strata"].nunique()))
    effective_splits = min(n_splits, max_splits) if n_splits > 1 else 1
    sss = StratifiedShuffleSplit(
        n_splits=effective_splits, train_size=train_ratio, random_state=random_state
    )
    cv_splits = []

    for train_block_idx, test_block_idx in sss.split(block_summary, block_summary["strata"]):
        train_indices = df_temp[df_temp["block_id"].isin(block_summary.index[train_block_idx])].index
        test_indices = df_temp[df_temp["block_id"].isin(block_summary.index[test_block_idx])].index
        if n_splits == 1:
            return (
                X.loc[train_indices],
                X.loc[test_indices],
                y.loc[train_indices],
                y.loc[test_indices],
            )
        cv_splits.append((train_indices, test_indices))

    return cv_splits

def create_sequences(data, target, window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    data_array = getattr(data, "values", data)
    target_array = getattr(target, "values", target)
    X, y = [], []
    for i in range(len(data_array) - window_size):
        X.append(data_array[i : (i + window_size)])
        y.append(target_array[i + window_size])
    return np.array(X), np.array(y)

def build_summary_table(
    results_metrics: dict[str, list[dict]],
    test_metric: str = "MAE",
    index_name: str = "Model",
) -> pd.DataFrame:
    metric_map = {"Specificity": "Spec", "Sensitivity": "Sens", "R2": "R2", "MAE": "MAE", "AUC": "AUC"}
    sort_col = metric_map.get(test_metric, test_metric)
    summary_list = []

    for name, folds in results_metrics.items():
        if not folds:
            continue
        df_m = pd.DataFrame(folds)
        summary_list.append(
            {
                index_name: name,
                "MAE": round(df_m["MAE"].mean(), 4),
                "R2": round(df_m["R2"].mean(), 4),
                "AUC": round(df_m["AUC"].mean(), 4),
                "Sens": round(df_m["Sensitivity"].mean(), 4),
                "Spec": round(df_m["Specificity"].mean(), 4),
                "TP": int(df_m["TP"].sum()),
                "TN": int(df_m["TN"].sum()),
                "FP": int(df_m["FP"].sum()),
                "FN": int(df_m["FN"].sum()),
            }
        )

    is_ascending = test_metric.upper() in ("MAE", "MSE", "RMSE", "MAPE")
    return pd.DataFrame(summary_list).set_index(index_name).sort_values(sort_col, ascending=is_ascending)

def report_and_test(
    results_metrics: dict[str, list[dict]],
    test_metric: str = "MAE",
    index_name: str = "Model",
    *,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    df_summary = build_summary_table(results_metrics, test_metric, index_name)

    if verbose:
        print(f"\n{index_name} performance (sorted by {test_metric})")
        print(df_summary)

        best_name = df_summary.index[0]
        best_scores = [f[test_metric] for f in results_metrics[best_name]]
        print(f"\nStatistical significance (vs {best_name} on {test_metric})")
        for name in df_summary.index:
            if name == best_name:
                continue
            current_scores = [f[test_metric] for f in results_metrics[name]]
            _, p_val = stats.ttest_rel(best_scores, current_scores)
            sig = " (Significant)" if p_val < 0.05 else " (Not Significant)"
            print(f"{best_name} vs {name}: p-value = {p_val:.6f}{sig}")

    return results_metrics

_report_and_test = report_and_test

def evaluate_algorithms(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Iterable[Tuple[Any, Any]],
    threshold: float,
    test_metric: str = "MAE",
    *,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    models = {"Gradient Boosting": GradientBoostingRegressor(random_state=42)}
    results = {name: [] for name in models}

    for train_idx, test_idx in cv_splits:
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
        for name, model_obj in models.items():
            model = clone(model_obj)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            results[name].append(calculate_metrics(y_te, preds, threshold))

    return report_and_test(results, test_metric, index_name="Model", verbose=verbose)

def evaluate_time_series_cross_val(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Iterable[Tuple[Any, Any]],
    threshold: float,
    test_metric: str = "MAE",
    window_size: int = 30,
    *,
    verbose: bool = True,
    epochs: int = 15,
) -> dict[str, list[dict]]:
    if not _TF_AVAILABLE:
        raise ImportError("tensorflow is required for GRU evaluation.")

    results: dict[str, list[dict]] = {"GRU": []}

    for train_idx, test_idx in cv_splits:
        sc_x, sc_y = MinMaxScaler(), MinMaxScaler()
        X_tr_s = sc_x.fit_transform(X.loc[train_idx])
        y_tr_s = sc_y.fit_transform(y.loc[train_idx].values.reshape(-1, 1)).flatten()
        X_te_s = sc_x.transform(X.loc[test_idx])

        X_train_ts, y_train_ts = create_sequences(X_tr_s, y_tr_s, window_size)
        X_test_ts, _ = create_sequences(X_te_s, np.zeros(len(X_te_s)), window_size)
        y_te_aligned = y.loc[test_idx].values[window_size:]

        model = Sequential([GRU(50, input_shape=(window_size, X_train_ts.shape[2])), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_ts, y_train_ts, epochs=epochs, batch_size=32, verbose=0)

        preds = sc_y.inverse_transform(model.predict(X_test_ts, verbose=0)).flatten()
        results["GRU"].append(calculate_metrics(y_te_aligned, preds, threshold))

    return report_and_test(results, test_metric, verbose=verbose)

def evaluate_feature_scenarios(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Iterable[Tuple[Any, Any]],
    best_model_obj,
    scenarios: dict[str, List[str]],
    threshold: float,
    test_metric: str = "MAE",
    *,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    scenario_metrics: dict[str, list[dict]] = {name: [] for name in scenarios}

    for name, cols in scenarios.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            if verbose:
                print(f"Skipping {name}: No matching columns.")
            continue

        X_scenario = X[valid_cols]
        for train_idx, test_idx in cv_splits:
            model = clone(best_model_obj)
            model.fit(X_scenario.loc[train_idx], y.loc[train_idx])
            y_pred = model.predict(X_scenario.loc[test_idx])
            scenario_metrics[name].append(calculate_metrics(y.loc[test_idx], y_pred, threshold))

    return report_and_test(scenario_metrics, test_metric, index_name="Scenario", verbose=verbose)

def default_feature_scenarios(groups: dict[str, List[str]]) -> dict[str, List[str]]:
    v = groups["vehicle_cols"]
    return {
        "Vehicles Only": v,
        "Vehicles + Lag 30s": v + groups["lag_30s"],
        "Vehicles + Lag 1m": v + groups["lag_1m"],
        "Vehicles + Lag 5m": v + groups["lag_5m"],
        "Vehicles + Lag 10m": v + groups["lag_10m"],
        "Vehicles + Lag 15m": v + groups["lag_15m"],
        "Vehicles + Lag 20m": v + groups["lag_20m"],
        "Vehicles + Lag 30m": v + groups["lag_30m"],
        "Vehicles + Lag 10m + Weather": v + groups["lag_10m"] + DEFAULT_WEATHER_COLS,
        "Vehicles + Lag 10m + Weather + Traffic": v
        + groups["lag_10m"]
        + DEFAULT_WEATHER_COLS
        + groups["traffic_cols"],
        "Weather + Traffic": DEFAULT_WEATHER_COLS + groups["traffic_cols"],
    }

def compare_sampling_intervals(
    data_files: dict[str, Union[Path, str]],
    exclude_features: List[str],
    feature_cols: List[str],
    threshold: float,
    *,
    evaluator: Callable = evaluate_algorithms,
    test_metric: str = "MAE",
    train_ratio: float = 0.8,
    evaluator_kwargs: Optional[dict] = None,
) -> dict[str, list[float]]:
    """Run CV per sampling interval and print inter-interval significance tests."""
    evaluator_kwargs = evaluator_kwargs or {}
    interval_fold_scores: dict[str, list[float]] = {}
    is_lower_better = test_metric.upper() in ("MAE", "MSE", "RMSE")

    for interval_name, filepath in data_files.items():
        print(f"\n{'=' * 30}\n PROCESSING INTERVAL: {interval_name} \n{'=' * 30}")
        X, y = prepare_data(filepath, exclude_features)
        X_train, X_test, y_train, y_test = improved_time_based_split(X, y, train_ratio=train_ratio, n_splits=1)
        cv_splits = improved_time_based_split(X_train, y_train, n_splits=5)

        results = evaluator(
            X_train[feature_cols],
            y_train,
            cv_splits,
            threshold,
            test_metric=test_metric,
            verbose=False,
            **evaluator_kwargs,
        )

        best_model = build_summary_table(results, test_metric).index[0]
        interval_fold_scores[interval_name] = [fold[test_metric] for fold in results[best_model]]

    avg_scores = {k: float(np.mean(v)) for k, v in interval_fold_scores.items()}
    best_interval = min(avg_scores, key=avg_scores.get) if is_lower_better else max(avg_scores, key=avg_scores.get)

    print(f"\n{'#' * 40}\n INTER-INTERVAL SIGNIFICANCE ({test_metric}) \n{'#' * 40}")
    print(f"Best Performing Interval: {best_interval} (Mean {test_metric}: {avg_scores[best_interval]:.4f})")

    for interval_name in data_files:
        if interval_name == best_interval:
            continue
        _, p_val = stats.ttest_ind(
            interval_fold_scores[best_interval],
            interval_fold_scores[interval_name],
            equal_var=False,
        )
        sig = " (Significant)" if p_val < 0.05 else " (Not Significant)"
        print(f"{best_interval} vs {interval_name}: p-value = {p_val:.6f}{sig}")

    return interval_fold_scores

def run_personalization_curve(
    X_base_train: pd.DataFrame,
    y_base_train: pd.Series,
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_fixed_test: pd.DataFrame,
    y_fixed_test: pd.Series,
    threshold: float,
    *,
    ratios: Optional[List[float]] = None,
    model_factory: Optional[Callable] = None,
) -> pd.DataFrame:
    """Gradient-boosting personalization curve (fixed test set)."""
    ratios = ratios or [0.25, 0.5, 0.75, 1.0]
    model_factory = model_factory or (lambda: GradientBoostingRegressor(random_state=42))
    rows = []

    for r in ratios:
        slice_idx = int(len(X_pool) * r)
        X_personal = X_pool.iloc[:slice_idx]
        y_personal = y_pool.iloc[:slice_idx]
        X_train = pd.concat([X_base_train, X_personal], ignore_index=True)
        y_train = pd.concat([y_base_train, y_personal], ignore_index=True)

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_fixed_test)
        metrics = calculate_metrics(y_fixed_test, preds, threshold)
        metrics["Total_Day2_Percent"] = f"{r * 20:.1f}%"
        metrics["Train_Rows_Day2"] = len(X_personal)
        rows.append(metrics)

    return pd.DataFrame(rows)

def export_shap_outputs(
    model,
    X_test: pd.DataFrame,
    preds: np.ndarray,
    *,
    output_dir: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export SHAP summaries for vehicle and non-vehicle feature groups."""
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    full_shap = explainer.shap_values(X_test)

    vehicle_keywords = ("car", "bus", "truck")
    vehicle_cols = [c for c in X_test.columns if any(v in c.lower() for v in vehicle_keywords)]
    non_vehicle_cols = [c for c in X_test.columns if c not in vehicle_cols]

    vehicle_df = X_test[vehicle_cols].copy()
    vehicle_df["Predicted_Black_Carbon"] = preds
    vehicle_path = output_dir / "predicted_black_carbon_results.csv"
    vehicle_df.to_csv(vehicle_path, index=True)

    nv_indices = [X_test.columns.get_loc(c) for c in non_vehicle_cols]
    nv_shap = full_shap[:, nv_indices]
    X_nv = X_test[non_vehicle_cols]

    nv_rows = []
    for i, name in enumerate(non_vehicle_cols):
        actual = X_nv.iloc[:, i]
        if np.std(actual) == 0 or np.std(nv_shap[:, i]) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(actual, nv_shap[:, i])[0, 1])
        nv_rows.append(
            {
                "Feature": name,
                "Importance_Weight": float(np.abs(nv_shap[:, i]).mean()),
                "Influence_Direction": "Increases BC" if correlation > 0 else "Decreases BC",
                "Impact_Strength_Score": correlation,
            }
        )

    summary_df = pd.DataFrame(nv_rows).sort_values(by="Importance_Weight", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    return vehicle_df, summary_df

def plot_vehicle_prediction_correlations(
    vehicle_results_path: Path | str,
    output_path: Path | str,
) -> pd.Series:
    """Bar chart of vehicle-feature correlation with predicted BC."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    from bc_config import STANDARD_VEHICLE_COLS
    from bc_eda import vehicle_color_map

    df = pd.read_csv(vehicle_results_path)
    vehicle_cols = [c for c in STANDARD_VEHICLE_COLS if c in df.columns]
    correlations = (
        df[vehicle_cols + ["Predicted_Black_Carbon"]]
        .corr()["Predicted_Black_Carbon"]
        .drop("Predicted_Black_Carbon")
        .sort_values(ascending=False)
    )

    color_map = vehicle_color_map()

    def get_color(col_name: str):
        if "car" in col_name:
            return color_map["cars_stopped"] if "stopped" in col_name else color_map["cars_moving"]
        if "bus" in col_name:
            return color_map["buses_stopped"] if "stopped" in col_name else color_map["buses_moving"]
        if "truck" in col_name:
            return color_map["trucks_stopped"] if "stopped" in col_name else color_map["trucks_moving"]
        return "gray"

    plot_colors = [get_color(c) for c in correlations.index]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=correlations.values, y=correlations.index, palette=plot_colors)
    legend_elements = [
        Line2D([0], [0], color=color_map["cars_moving"], lw=4, label="Cars Moving"),
        Line2D([0], [0], color=color_map["cars_stopped"], lw=4, label="Cars Stopped"),
        Line2D([0], [0], color=color_map["buses_moving"], lw=4, label="Buses Moving"),
        Line2D([0], [0], color=color_map["buses_stopped"], lw=4, label="Buses Stopped"),
        Line2D([0], [0], color=color_map["trucks_moving"], lw=4, label="Trucks Moving"),
        Line2D([0], [0], color=color_map["trucks_stopped"], lw=4, label="Trucks Stopped"),
    ]
    plt.legend(handles=legend_elements, title="Vehicle State", loc="lower right")
    plt.title("Vehicle Impact on Black Carbon (Color Coded by Type & State)")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Vehicle Feature")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return correlations
