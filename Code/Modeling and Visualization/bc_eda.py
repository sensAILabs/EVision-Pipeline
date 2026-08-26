"""Exploratory analysis and figures for black-carbon dashcam data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from bc_config import (
    LOCAL_UTC_OFFSET_HOURS,
    MIDDAY_LOCAL_END,
    MIDDAY_LOCAL_START,
    SPATIAL_BC_MAX,
    SPATIAL_BC_MIN,
    SPATIAL_COLOR_VMAX,
    VEHICLE_AGG_COLS,
)

try:
    import contextily as cx
except ImportError:
    cx = None

def load_spatial_data(
    filepath: Path | str,
    bc_min: float = SPATIAL_BC_MIN,
    bc_max: float = SPATIAL_BC_MAX,
) -> pd.DataFrame:
    """Load rows with valid lat/lon/bc within the BC range used in the spatial map."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["lat", "lon", "bc"])
    return df[(df["bc"] > bc_min) & (df["bc"] < bc_max)].copy()

def traffic_size_column(df: pd.DataFrame) -> str:
    """Return column used for marker size (Traffic_flow or TT_currentSpeed)."""
    return "Traffic_flow" if "Traffic_flow" in df.columns else "TT_currentSpeed"

def normalized_marker_sizes(series: pd.Series, min_size: float = 5, max_size: float = 45) -> np.ndarray:
    s_min, s_max = series.min(), series.max()
    if s_max > s_min:
        norm = (series - s_min) / (s_max - s_min)
    else:
        norm = pd.Series(0.5, index=series.index)
    return norm.to_numpy() * (max_size - min_size) + min_size

def add_vehicle_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving/stopped aggregates for cars, buses, and trucks."""
    out = df.copy()
    specs = [
        ("car", "cars_moving", "cars_stopped"),
        ("bus", "buses_moving", "buses_stopped"),
        ("truck", "trucks_moving", "trucks_stopped"),
    ]
    for prefix, moving, stopped in specs:
        out[moving] = (
            out[f"{prefix}_close"]
            + out[f"{prefix}_medium"]
            + out[f"{prefix}_far"]
            - (
                out[f"{prefix}_close_stopped"]
                + out[f"{prefix}_medium_stopped"]
                + out[f"{prefix}_far_stopped"]
            )
        ).clip(lower=0)
        out[stopped] = (
            out[f"{prefix}_close_stopped"]
            + out[f"{prefix}_medium_stopped"]
            + out[f"{prefix}_far_stopped"]
        )
    return out

def prepare_midday_resampled(
    df: pd.DataFrame,
    *,
    start_local: str = MIDDAY_LOCAL_START,
    end_local: str = MIDDAY_LOCAL_END,
    freq: str = "5min",
    bc_col: str = "bc_cleaned",
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Filter to a midday local-time window and resample vehicle sums + mean BC.

    Returns (resampled_df, target_start, target_end).
    """
    work = df.copy()
    if "timestamp" not in work.columns:
        raise KeyError("DataFrame must contain a 'timestamp' column.")

    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work["local_time"] = (
        work["timestamp"] + pd.Timedelta(hours=LOCAL_UTC_OFFSET_HOURS)
    ).dt.tz_localize(None)
    work = work.sort_values("local_time")
    work = add_vehicle_aggregates(work)

    data_date = work["local_time"].dt.date.iloc[0]
    target_start = pd.Timestamp.combine(data_date, pd.Timestamp(start_local).time())
    target_end = pd.Timestamp.combine(data_date, pd.Timestamp(end_local).time())

    filtered = work[(work["local_time"] >= target_start) & (work["local_time"] <= target_end)].copy()
    filtered = filtered.set_index("local_time")

    agg = {c: "sum" for c in VEHICLE_AGG_COLS}
    if bc_col in filtered.columns:
        agg[bc_col] = "mean"

    resampled = filtered.resample(freq).agg(agg)
    return resampled, target_start, target_end

def vehicle_color_map() -> dict[str, tuple]:
    palette = sns.color_palette("Paired", 6)
    return {
        "cars_moving": palette[1],
        "cars_stopped": palette[0],
        "buses_moving": palette[3],
        "buses_stopped": palette[2],
        "trucks_moving": palette[5],
        "trucks_stopped": palette[4],
    }

def plot_spatial_bc_map(
    df: pd.DataFrame,
    *,
    output_path: Path | str,
    use_map_overlay: bool = True,
    figsize: tuple[float, float] = (12, 9),
    map_style=None,
) -> plt.Figure:
    """Scatter map of BC colored by concentration; marker size reflects traffic."""
    if use_map_overlay and cx is None:
        raise ImportError("contextily is required for map overlays. pip install contextily")

    size_col = traffic_size_column(df)
    sizes = normalized_marker_sizes(df[size_col])

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["bc"],
        s=sizes,
        cmap="YlOrRd",
        vmin=0,
        vmax=SPATIAL_COLOR_VMAX,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.2,
        zorder=3,
    )

    if use_map_overlay:
        source = map_style or cx.providers.CartoDB.Positron
        cx.add_basemap(ax, crs="EPSG:4326", source=source, zorder=1)
        ax.set_facecolor("none")
    else:
        ax.set_facecolor("#fdfdfd")
        ax.grid(True, linestyle=":", alpha=0.6, color="gray", zorder=1)

    margin = 0.002
    ax.set_xlim(df["lon"].min() - margin, df["lon"].max() + margin)
    ax.set_ylim(df["lat"].min() - margin, df["lat"].max() + margin)
    ax.set_xlabel("Longitude ($^\\circ$E)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude ($^\\circ$N)", fontsize=12, fontweight="bold")
    title_type = "Map Overlay" if use_map_overlay else "Coordinate Plot"
    ax.set_title(f"Spatial Analysis: BC Peaks vs. Traffic ({title_type})", fontsize=14, pad=20)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Black Carbon ($ng/m^3$)", size=11, labelpad=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig

def plot_midday_activity_and_correlations(
    resampled: pd.DataFrame,
    *,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    output_dir: Path | str,
    bc_col: str = "bc_cleaned",
) -> dict[str, Path]:
    """Create stacked activity plot, correlation bars, and heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = vehicle_color_map()
    cols = VEHICLE_AGG_COLS
    paths: dict[str, Path] = {}

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bottom_val = pd.Series(0, index=resampled.index)
    bar_width = 5 / (24 * 60) * 0.7

    for col in cols:
        ax1.bar(
            resampled.index,
            resampled[col],
            bottom=bottom_val,
            label=col.replace("_", " ").title(),
            color=color_map[col],
            width=bar_width,
            align="center",
        )
        bottom_val += resampled[col]

    ax1.set_ylabel("Sum of Vehicles (5min bins)")
    ax1.set_xlabel("Time of Day (local)")
    ax2 = ax1.twinx()
    if bc_col in resampled.columns:
        ax2.plot(resampled.index, resampled[bc_col], color="#333333", label="BC Median", linewidth=2)
    ax2.set_ylabel("BC in ng/m3")
    ax1.set_xlim(target_start, target_end)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))

    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labs + labs2,
        loc="best",
        bbox_to_anchor=(1, 1),
        handlelength=0.5,
        borderpad=0.3,
        labelspacing=0.3,
    )
    plt.tight_layout()
    paths["activity"] = output_dir / "unified_activity_refined.png"
    fig.savefig(paths["activity"])

    plt.figure(figsize=(5, 5))
    renamed = resampled.rename(columns={bc_col: "BC Level"})
    bc_corrs = renamed.corr()["BC Level"].drop("BC Level").sort_values(ascending=False)
    bar_colors = [color_map[c] for c in bc_corrs.index]
    sns.barplot(
        x=bc_corrs.values,
        y=bc_corrs.index.to_series().str.replace("_", " ").str.title(),
        palette=bar_colors,
    )
    plt.axvline(0, color="black", lw=1)
    plt.xlim(-0.5, 0.6)
    plt.xlabel("Pearson Correlation Coefficient (r)")
    plt.ylabel("Vehicle Features")
    plt.tight_layout()
    paths["corr_bars"] = output_dir / "correlation_bars_refined.png"
    plt.savefig(paths["corr_bars"])

    plt.figure(figsize=(10, 5))
    mask = np.triu(np.ones_like(renamed.corr(), dtype=bool))
    sns.heatmap(
        renamed.corr(),
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdGy",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.tight_layout()
    paths["corr_heatmap"] = output_dir / "correlation_heatmap_refined.png"
    plt.savefig(paths["corr_heatmap"])

    plt.close("all")
    return paths

def plot_bc_raw_vs_cleaned(
    df: pd.DataFrame,
    *,
    output_path: Path | str,
    raw_col: str = "bc",
    cleaned_col: str = "bc_cleaned",
) -> tuple[plt.Figure, float, float]:
    """Boxplot and KDE comparing raw and cleaned BC; returns (fig, p95, max_cleaned)."""
    if cleaned_col not in df.columns:
        raise KeyError(f"Column '{cleaned_col}' is required for cleaning comparison plots.")

    p95_raw = df[raw_col].dropna().quantile(0.95)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    melted = df[[raw_col, cleaned_col]].melt(var_name="Version", value_name="BC_Value").dropna()
    sns.boxplot(x="Version", y="BC_Value", data=melted, ax=axes[0], palette=["lightgrey", "grey"])
    axes[0].axhline(p95_raw, color="black", linestyle="--", linewidth=1.5, label="95th Percentile Cap")
    axes[0].set_ylabel("Black Carbon (ng/m³)")
    axes[0].set_xlabel("Data Version")
    axes[0].legend()

    sns.kdeplot(df[raw_col].dropna(), label=f"Raw BC ({raw_col})", ax=axes[1], fill=True, color="black", alpha=0.2)
    sns.kdeplot(
        df[cleaned_col],
        label="Cleaned BC (DEMA)",
        ax=axes[1],
        fill=True,
        color="grey",
        alpha=0.5,
        bw_adjust=0.5,
    )
    axes[1].axvline(
        p95_raw,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"95th Percentile Cap ({p95_raw:.0f})",
    )
    axes[1].set_xlabel("Black Carbon (ng/m³)")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0, 150_000)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig, float(p95_raw), float(df[cleaned_col].max())

def ensure_bc_cleaned(df: pd.DataFrame, raw_col: str = "bc", cleaned_col: str = "bc_cleaned") -> pd.DataFrame:
    """
    Ensure bc_cleaned exists (cap raw BC at 95th percentile).

    Use when running against build_model_features output that only has raw bc.
    """
    out = df.copy()
    if cleaned_col in out.columns:
        return out
    if raw_col not in out.columns:
        raise KeyError(f"Neither '{cleaned_col}' nor '{raw_col}' found in data.")
    cap = out[raw_col].quantile(0.95)
    out[cleaned_col] = out[raw_col].clip(upper=cap)
    return out
