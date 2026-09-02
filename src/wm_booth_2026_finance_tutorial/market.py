"""Streaming market-data preparation and augmentations used by the tutorial."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
from datasets import IterableDataset, load_dataset

FEATURE_COLUMNS = [
    "bid_price",
    "vwap_all",
    "high",
    "low",
    "ask_price",
    "bid_size",
    "ask_size",
    "volume",
    "n",
]

PRICE_COLUMNS = [0, 1, 2, 3, 4]
NORMALIZATION_GROUPS = [
    (PRICE_COLUMNS, False),
    ([5, 6], True),
    ([7], True),
    ([8], True),
]
SESSION_SECONDS = 6.5 * 60 * 60
EPSILON = 1e-5
EASTERN = ZoneInfo("America/New_York")


def load_market_stream(
    repo_id: str,
    *,
    split: str = "train",
    token: str | None = None,
) -> IterableDataset:
    """Create a sequential Hugging Face stream without downloading the dataset."""
    return load_dataset(
        repo_id,
        data_files="parquet/*.parquet",
        split=split,
        streaming=True,
        token=token,
    )


def reshape_market_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a flattened Hub row to the array layout used by Market-JEPA."""
    n_steps = int(row["n_steps"])
    values = np.asarray(row["features"], dtype=np.float32)
    expected = n_steps * len(FEATURE_COLUMNS)
    if values.size != expected:
        raise ValueError(f"Expected {expected} feature values, found {values.size}")
    return {
        **row,
        "features": values.reshape(n_steps, len(FEATURE_COLUMNS)),
        "ts_interval": np.asarray(row["ts_interval"], dtype=np.int64),
    }


def market_bounds(date: str) -> tuple[int, int]:
    """Return the regular-session interval [09:30:01, 16:00:01) in Unix seconds."""
    day = dt.date.fromisoformat(date)
    midnight = dt.datetime.combine(day, dt.time(), tzinfo=EASTERN)
    market_open = midnight + dt.timedelta(hours=9, minutes=30)
    market_close = midnight + dt.timedelta(hours=16)
    return int(market_open.timestamp()) + 1, int(market_close.timestamp()) + 1


def dense_grid(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the canonical regular-hours 1 Hz grid for one sparse row."""
    ts_open, ts_close = market_bounds(str(sample["date"]))
    timestamps = np.arange(ts_open, ts_close, dtype=np.int64)
    features = np.asarray(sample["features"], dtype=np.float32)
    sparse_timestamps = np.asarray(sample["ts_interval"], dtype=np.int64)
    dense = np.full((len(timestamps), len(FEATURE_COLUMNS)), np.nan, dtype=np.float64)

    premarket_end = int(np.searchsorted(sparse_timestamps, ts_open))
    if premarket_end:
        dense[0, :7] = features[premarket_end - 1, :7]

    positions = np.searchsorted(timestamps, sparse_timestamps)
    clipped = np.minimum(positions, len(timestamps) - 1)
    valid = (positions < len(timestamps)) & (timestamps[clipped] == sparse_timestamps)
    dense[positions[valid]] = features[valid]

    for column in range(7):
        values = dense[:, column]
        missing = np.isnan(values)
        indices = np.arange(len(values))
        indices[missing] = 0
        np.maximum.accumulate(indices, out=indices)
        values[:] = values[indices]
    dense[:, 7:9] = np.nan_to_num(dense[:, 7:9], nan=0.0)

    complete = ~np.any(np.isnan(dense), axis=1)
    if not complete.any():
        raise ValueError(f"No complete market rows for {sample['ticker']} on {sample['date']}")
    first = int(np.argmax(complete))
    return timestamps[first:], dense[first:]


def aggregate(features: np.ndarray, scale: int, *, offset: int = 0) -> np.ndarray:
    """Aggregate 1 Hz rows with market-aware last/max/min/sum/VWAP semantics."""
    values = features[offset:]
    n_full = len(values) // scale
    if n_full < 2:
        raise ValueError("The requested aggregation produces fewer than two tokens")
    values = values[: n_full * scale].reshape(n_full, scale, -1)
    result = np.empty((n_full, len(FEATURE_COLUMNS)), dtype=np.float64)
    result[:, 0] = values[:, -1, 0]
    result[:, 2] = values[:, :, 2].max(axis=1)
    result[:, 3] = values[:, :, 3].min(axis=1)
    result[:, 4:7] = values[:, -1, 4:7]
    result[:, 7:9] = values[:, :, 7:9].sum(axis=1, dtype=np.float64)
    volume = values[:, :, 7]
    volume_sum = volume.sum(axis=1, dtype=np.float64)
    products = np.multiply(values[:, :, 1], volume, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result[:, 1] = np.where(
            volume_sum > 0,
            np.nansum(products, axis=1) / volume_sum,
            np.nan,
        )
    return result


def _prior_vwap(features: np.ndarray, start: int) -> float | None:
    if start <= 0:
        return None
    earlier = features[:start]
    valid = (earlier[:, 8] > 0) & ~np.isnan(earlier[:, 1])
    return float(earlier[np.flatnonzero(valid)[-1], 1]) if valid.any() else None


def _fill_vwap(view: np.ndarray, prior: float | None) -> None:
    values = view[:, 1]
    if np.isnan(values[0]) and prior is not None:
        values[0] = prior
    missing = np.isnan(values)
    indices = np.arange(len(values))
    indices[missing] = 0
    np.maximum.accumulate(indices, out=indices)
    values[:] = values[indices]
    missing = np.isnan(values)
    values[missing] = (view[missing, 0] + view[missing, 4]) / 2


def _normalize(view: np.ndarray, *, with_stats: bool) -> tuple[np.ndarray, list[float]]:
    stats: list[float] = []
    for indices, apply_log in NORMALIZATION_GROUPS:
        if apply_log:
            view[:, indices] = np.log1p(view[:, indices])
        values = view[:, indices]
        mean = float(np.mean(values))
        std = float(np.sqrt(np.var(values) + EPSILON))
        view[:, indices] = (values - mean) / std
        if with_stats:
            stats.extend([np.sign(mean) * np.log1p(abs(mean)), np.log(std + EPSILON)])
    return view, stats


def finish_view(
    view: np.ndarray,
    full_features: np.ndarray,
    timestamps: np.ndarray,
    start: int,
    aggregation: float,
    *,
    with_stats: bool = False,
    with_window: bool = False,
) -> torch.Tensor:
    """Apply VWAP filling, normalization, and optional checkpoint information channels."""
    view = view.copy()
    _fill_vwap(view, _prior_vwap(full_features, start))
    view, stats = _normalize(view, with_stats=with_stats)
    information = stats
    if with_window:
        date = dt.datetime.fromtimestamp(int(timestamps[start]), tz=EASTERN).date()
        market_open, _ = market_bounds(date.isoformat())
        start_seconds = int(timestamps[start]) - market_open
        end_seconds = start_seconds + len(view) * aggregation
        information.extend(
            [
                start_seconds / SESSION_SECONDS,
                end_seconds / SESSION_SECONDS,
                np.log(max(float(aggregation), EPSILON)),
            ]
        )
    if information:
        info = np.asarray(information, dtype=np.float64)
        view = np.concatenate([view, np.broadcast_to(info, (len(view), len(info)))], axis=1)
    np.nan_to_num(view, copy=False, nan=0.0)
    return torch.from_numpy(view.T.astype(np.float32))


def random_resized_crop_pair(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    n_global: int = 2,
    n_local: int = 2,
    global_length: int = 256,
    local_length: int = 96,
    global_scale: tuple[float, float] = (0.10, 0.20),
    local_scale: tuple[float, float] = (0.03, 0.08),
    with_stats: bool = False,
    with_window: bool = False,
    seed: int = 7,
) -> dict[str, Any]:
    """Draw independently located and scaled global/local views."""
    rng = np.random.RandomState(seed)
    views: list[torch.Tensor] = []
    aggregations: list[int] = []
    for count, scale_range, target_length in (
        (n_global, global_scale, global_length),
        (n_local, local_scale, local_length),
    ):
        for _ in range(count):
            scale = float(rng.uniform(*scale_range))
            aggregation = max(1, round(scale * len(features) / target_length))
            window = aggregation * target_length
            start = int(rng.randint(0, len(features) - window + 1))
            raw_view = aggregate(features[start : start + window], aggregation)
            views.append(
                finish_view(
                    raw_view,
                    features,
                    timestamps,
                    start,
                    aggregation,
                    with_stats=with_stats,
                    with_window=with_window,
                )
            )
            aggregations.append(aggregation)
    return {
        "views": views,
        "n_global_views": n_global,
        "agg_factors": aggregations,
    }


def _shared_crop(
    features: np.ndarray,
    rng: np.random.RandomState,
    length: int,
    scale_range: tuple[float, float] = (0.10, 0.20),
) -> tuple[int, int, int]:
    scale = float(rng.uniform(*scale_range))
    aggregation = max(1, round(scale * len(features) / length))
    window = aggregation * length
    start = int(rng.randint(0, len(features) - window + 1))
    return start, window, aggregation


def _warped_aggregate(
    features: np.ndarray,
    length: int,
    rng: np.random.RandomState,
    knots: int,
    strength: float,
) -> np.ndarray:
    width = len(features)
    knots_x = np.linspace(0.0, width, knots + 1)
    knots_y = knots_x.copy()
    knots_y[1:-1] += rng.uniform(-strength, strength, size=knots - 1) * (width / knots)
    np.maximum.accumulate(knots_y, out=knots_y)
    boundaries = np.round(np.interp(np.linspace(0.0, width, length + 1), knots_x, knots_y)).astype(
        np.int64
    )
    indices = np.arange(length + 1)
    boundaries = np.clip(boundaries, indices, width - length + indices)
    boundaries = np.maximum.accumulate(boundaries - indices) + indices
    starts, last = boundaries[:-1], boundaries[1:] - 1
    result = np.empty((length, len(FEATURE_COLUMNS)), dtype=np.float64)
    result[:, 0] = features[last, 0]
    result[:, 2] = np.maximum.reduceat(features[:, 2], starts)
    result[:, 3] = np.minimum.reduceat(features[:, 3], starts)
    result[:, 4:7] = features[last, 4:7]
    result[:, 7:9] = np.add.reduceat(features[:, 7:9], starts, axis=0)
    products = np.nan_to_num(features[:, 1] * features[:, 7], nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result[:, 1] = np.where(
            result[:, 7] > 0,
            np.add.reduceat(products, starts) / result[:, 7],
            np.nan,
        )
    return result


def same_stock_pair(
    name: str,
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    length: int = 256,
    seed: int = 7,
    **options: float | int,
) -> dict[str, Any]:
    """Build two views for time warp, noise, price jitter, or channel drop."""
    rng = np.random.RandomState(seed)
    start, window, aggregation = _shared_crop(features, rng, length)
    raw_window = features[start : start + window]
    views: list[torch.Tensor] = []
    if name == "time_warp":
        for _ in range(2):
            view = _warped_aggregate(
                raw_window,
                length,
                rng,
                int(options.get("warp_knots", 8)),
                float(options.get("warp_strength", 0.75)),
            )
            views.append(finish_view(view, features, timestamps, start, aggregation))
    elif name in {"gaussian_noise", "channel_drop"}:
        base = finish_view(
            aggregate(raw_window, aggregation), features, timestamps, start, aggregation
        )
        for _ in range(2):
            view = base.clone()
            if name == "gaussian_noise":
                sigma = float(options.get("noise_sigma", 0.35))
                noise = torch.from_numpy(rng.standard_normal(view.shape).astype(np.float32))
                view += sigma * noise
            else:
                probability = float(options.get("channel_drop_p", 0.35))
                while True:
                    dropped = rng.uniform(size=len(FEATURE_COLUMNS)) < probability
                    if not dropped.all():
                        break
                view[np.flatnonzero(dropped).tolist()] = 0.0
            views.append(view)
    elif name in {"volume_noise", "price_jitter"}:
        base = aggregate(raw_window, aggregation)
        for _ in range(2):
            view = base.copy()
            if name == "volume_noise":
                fraction = float(options.get("vol_noise_frac", 0.50))
                for column in (7, 8):
                    mean = float(np.nanmean(view[:, column]))
                    if mean > 0:
                        view[:, column] += rng.exponential(fraction * mean, size=len(view))
            else:
                fraction = float(options.get("price_jitter_frac", 1.0))
                half_spread = np.clip((view[:, 4] - view[:, 0]) / 2, 0, None)
                shifts = rng.standard_normal(len(view)) * fraction * half_spread
                view[:, :5] += shifts[:, None]
            views.append(finish_view(view, features, timestamps, start, aggregation))
    else:
        raise ValueError(f"Unknown augmentation: {name}")
    return {"views": views, "n_global_views": 2, "agg_factors": [aggregation] * 2}


def same_date_pair(
    iterator: Iterator[dict[str, Any]],
    existing: Iterable[dict[str, Any]] = (),
    *,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], int]:
    """Stream until two distinct tickers from one date have appeared."""
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in existing:
        by_date.setdefault(str(sample["date"]), {})[str(sample["ticker"])] = sample
    for consumed in range(1, limit + 1):
        sample = reshape_market_row(next(iterator))
        group = by_date.setdefault(str(sample["date"]), {})
        group[str(sample["ticker"])] = sample
        if len(group) >= 2:
            return list(group.values())[:2], consumed
    raise RuntimeError(f"No same-date pair found after streaming {limit} additional rows")


def cross_stock_pair(
    samples: list[dict[str, Any]],
    *,
    length: int = 256,
    aggregation: int = 8,
) -> dict[str, Any]:
    """Align distinct stocks to one shared wall-clock window."""
    grids = [(sample, *dense_grid(sample)) for sample in samples]
    overlap_start = max(int(timestamps[0]) for _, timestamps, _ in grids)
    overlap_end = min(int(timestamps[-1]) for _, timestamps, _ in grids)
    window = aggregation * length
    if overlap_end - overlap_start + 1 < window:
        raise ValueError("The stocks do not share a sufficiently long window")
    absolute_start = overlap_start + (overlap_end - overlap_start + 1 - window) // 2
    views = []
    for _, timestamps, features in grids:
        start = int(np.searchsorted(timestamps, absolute_start))
        raw_view = aggregate(features[start : start + window], aggregation)
        views.append(finish_view(raw_view, features, timestamps, start, aggregation))
    n_views = len(views)
    weights = torch.ones((n_views, n_views)) - torch.eye(n_views)
    return {
        "views": views,
        "n_global_views": n_views,
        "pair_weights": weights,
        "labels": [str(sample["ticker"]) for sample in samples],
    }
