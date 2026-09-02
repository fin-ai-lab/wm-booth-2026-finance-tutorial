"""Plotting helpers for the finance tutorial notebook."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .market import FEATURE_COLUMNS


def plot_market_sample(sample: dict[str, Any]):
    """Plot midpoint, trade VWAP, and volume from one sparse market row."""
    frame = pd.DataFrame(sample["features"], columns=FEATURE_COLUMNS)
    frame.insert(
        0,
        "time",
        pd.to_datetime(sample["ts_interval"], unit="s", utc=True).tz_convert("America/New_York"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    midpoint = (frame["bid_price"] + frame["ask_price"]) / 2
    axes[0].plot(frame["time"], midpoint, linewidth=0.8, label="quoted midpoint")
    axes[0].plot(
        frame["time"],
        frame["vwap_all"],
        linewidth=0,
        marker=".",
        markersize=1.5,
        alpha=0.5,
        label="trade VWAP",
    )
    axes[0].set_ylabel("price ($)")
    axes[0].legend()
    axes[1].plot(frame["time"], frame["volume"], linewidth=0.6)
    axes[1].set_ylabel("shares")
    axes[1].set_xlabel("New York time")
    axes[0].set_title(f"Raw sparse observations: {sample['ticker']} — {sample['date']}")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_dense_grid(timestamps: np.ndarray, features: np.ndarray):
    """Plot the regular-hours midpoint and volume on a canonical 1 Hz grid."""
    times = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York")
    midpoint = (features[:, 0] + features[:, 4]) / 2
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(times, midpoint, linewidth=0.8)
    axes[0].set_ylabel("midpoint ($)")
    axes[1].plot(times, features[:, 7], linewidth=0.6)
    axes[1].set_ylabel("volume")
    axes[1].set_xlabel("New York time")
    axes[0].set_title(f"Canonical 1 Hz training grid ({len(features):,} rows)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_views(
    pair: dict[str, Any],
    title: str,
    *,
    channels: tuple[str, ...] = ("vwap_all", "volume"),
):
    """Plot normalized channel-by-token tensors returned by an augmentation."""
    figure, axes = plt.subplots(
        len(channels),
        1,
        figsize=(11, 2.8 * len(channels)),
        squeeze=False,
    )
    for row, channel in enumerate(channels):
        channel_index = FEATURE_COLUMNS.index(channel)
        for view_index, view in enumerate(pair["views"]):
            values = view[channel_index].detach().cpu().numpy()
            relative_time = np.linspace(0, 1, len(values))
            kind = (
                "global" if view_index < pair.get("n_global_views", len(pair["views"])) else "local"
            )
            label = (
                pair.get("labels", [])[view_index]
                if pair.get("labels")
                else f"view {view_index + 1}"
            )
            axes[row, 0].plot(
                relative_time,
                values,
                linewidth=1.1,
                alpha=0.85,
                label=f"{label} ({kind}, {len(values)} tokens)",
            )
        axes[row, 0].set_ylabel(f"normalized {channel}")
        axes[row, 0].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("relative position within each view")
    axes[0, 0].set_title(title)
    axes[0, 0].legend(ncols=2, fontsize=8)
    figure.tight_layout()
    return figure


def plot_channel_drop(pair: dict[str, Any]):
    """Show all channels for the two channel-drop views."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    image = None
    for index, (axis, view) in enumerate(zip(axes, pair["views"], strict=True)):
        image = axis.imshow(view.numpy(), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        axis.set_title(f"channel-drop view {index + 1}")
        axis.set_xlabel("token")
        axis.set_yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    figure.colorbar(image, ax=axes, shrink=0.8, label="normalized value")
    return figure


def plot_cross_stock(pair: dict[str, Any]):
    """Plot aligned stock views alongside their structured matching graph."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    vwap_index = FEATURE_COLUMNS.index("vwap_all")
    for label, view in zip(pair["labels"], pair["views"], strict=True):
        axes[0].plot(view[vwap_index].numpy(), linewidth=1, label=label)
    axes[0].set_title("Aligned global stock views")
    axes[0].set_xlabel("token")
    axes[0].set_ylabel("normalized VWAP")
    axes[0].legend()
    axes[1].imshow(pair["pair_weights"].numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Structured matching edges")
    axes[1].set_xlabel("view")
    axes[1].set_ylabel("view")
    figure.tight_layout()
    return figure
