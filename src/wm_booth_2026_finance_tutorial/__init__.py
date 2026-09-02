"""Reusable helpers for the Booth finance tutorial notebooks."""

from importlib.metadata import PackageNotFoundError, version

from .market import (
    FEATURE_COLUMNS,
    cross_stock_pair,
    dense_grid,
    load_market_stream,
    random_resized_crop_pair,
    reshape_market_row,
    same_date_pair,
    same_stock_pair,
)
from .model import LeJEPAEncoder, load_lejepa_encoder
from .plotting import (
    plot_channel_drop,
    plot_cross_stock,
    plot_dense_grid,
    plot_market_sample,
    plot_views,
)

try:
    __version__ = version("wm-booth-2026-finance-tutorial")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "FEATURE_COLUMNS",
    "LeJEPAEncoder",
    "__version__",
    "cross_stock_pair",
    "dense_grid",
    "load_lejepa_encoder",
    "load_market_stream",
    "plot_channel_drop",
    "plot_cross_stock",
    "plot_dense_grid",
    "plot_market_sample",
    "plot_views",
    "random_resized_crop_pair",
    "reshape_market_row",
    "same_date_pair",
    "same_stock_pair",
]
