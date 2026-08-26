"""Shared configuration for Modelling and Visualization."""

from __future__ import annotations

DEFAULT_EXCLUDE_FEATURES: list[str] = [
    "timestamp",
    "lat",
    "lon",
    "TT_frc",
    "bc",
    "bc_categorical",
    "bc_clean_smooth_median",
    "bc_clean",
    "bc_clean_smooth_ma",
    "bc_clean_smooth_gaussian",
    "bc_clean_smooth_ema",
]

DEFAULT_TARGET_COL = "bc_cleaned"
DEFAULT_WEATHER_COLS = ["Temperature", "Humidity", "Wind_speed"]

DEFAULT_BC_THRESHOLD = 15904.877169777865

SPATIAL_BC_MIN = 0
SPATIAL_BC_MAX = 50_000
SPATIAL_COLOR_VMAX = 15_000
MIDDAY_LOCAL_START = "11:25:00"
MIDDAY_LOCAL_END = "14:30:00"
LOCAL_UTC_OFFSET_HOURS = 8

VEHICLE_AGG_COLS = [
    "cars_moving",
    "cars_stopped",
    "buses_moving",
    "buses_stopped",
    "trucks_moving",
    "trucks_stopped",
]

STANDARD_VEHICLE_COLS = [
    "car_close",
    "car_close_stopped",
    "car_medium",
    "car_medium_stopped",
    "car_far",
    "car_far_stopped",
    "bus_close",
    "bus_close_stopped",
    "bus_medium",
    "bus_medium_stopped",
    "bus_far",
    "bus_far_stopped",
    "truck_close",
    "truck_close_stopped",
    "truck_medium",
    "truck_medium_stopped",
    "truck_far",
    "truck_far_stopped",
]
