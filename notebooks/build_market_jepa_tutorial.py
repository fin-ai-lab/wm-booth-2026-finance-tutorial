"""Regenerate the Market-JEPA tutorial notebook."""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    """
# Market-JEPA: data, time-series augmentations, and a LeJEPA encoder

This notebook is a compact tour of the Market-JEPA representation-learning pipeline. It streams
real ticker-days from the public `fin-ai-lab/Market-1T-1Hz-2019-07-demo` dataset, constructs the
views used for self-supervised training, and loads a trained LeJEPA encoder from the Hugging Face
Hub.

Reusable implementation details live in the installed `wm_booth_2026_finance_tutorial` package so
the notebook can concentrate on the ideas and plots.
"""
)

markdown(
    """
## 0. Setup

Run `uv sync` and launch Jupyter with `uv run jupyter lab`. Both Hub repositories are public, so no
token or local market-data directory is required; Hugging Face caches only the model checkpoint and
the Parquet ranges actually read from the streaming dataset.
"""
)

code(
    """
import pandas as pd
import torch

import wm_booth_2026_finance_tutorial as tutorial

DATASET_ID = "fin-ai-lab/Market-1T-1Hz-2019-07-demo"
MODEL_ID = "fin-ai-lab/lejepa-demo-ecoder"
SEED = 7
"""
)

markdown(
    """
## 1. Stream and plot market data

Each Hugging Face row is one ticker-day containing sparse Unix timestamps and a flattened
`n_steps` by 9 feature array. Passing `streaming=True` creates an `IterableDataset`: only rows we
iterate are fetched through Parquet range reads, and the full dataset is never downloaded.
"""
)

code(
    """
market_stream = tutorial.load_market_stream(DATASET_ID)
market_iterator = iter(market_stream)
raw_sample = tutorial.reshape_market_row(next(market_iterator))
streamed_rows = [raw_sample]

print(market_stream)
print(
    f"Example: {raw_sample['ticker']} on {raw_sample['date']} "
    f"({raw_sample['n_steps']:,} stored rows)"
)
pd.DataFrame(raw_sample["features"], columns=tutorial.FEATURE_COLUMNS).head()
"""
)

code("tutorial.plot_market_sample(raw_sample)")

markdown(
    """
The source row is sparse: seconds with no new quote or trade are absent. Market-JEPA reconstructs
a regular-hours 1 Hz grid, forward-filling state variables such as quotes and sizes while filling
event variables—volume and trade count—with zeros.
"""
)

code(
    """
dense_seconds, dense_features = tutorial.dense_grid(raw_sample)
print(f"Dense grid shape: {dense_features.shape}")
tutorial.plot_dense_grid(dense_seconds, dense_features)
"""
)

markdown(
    """
## 2. Augmentations

The helpers below use the same market-aware rules as the training pipeline: bid/ask and sizes take
the last observation, highs and lows use extrema, volume and trade count sum, and VWAP is
volume-weighted. The plots show normalized tensors in `(channel, token)` layout—the encoder's input.
"""
)

markdown(
    """
### 2.1 Random resized crop

The time-series analogue of vision's random resized crop draws windows at random temporal scales
and reaggregates each one to a fixed token length. Global and local views may cover different spans
while retaining encoder-friendly shapes.
"""
)

code(
    """
rrc_pair = tutorial.random_resized_crop_pair(dense_features, dense_seconds, seed=SEED)
print("Seconds per token:", rrc_pair["agg_factors"])
tutorial.plot_views(rrc_pair, "Random resized crop: two global and two local views")
"""
)

markdown(
    """
### 2.2 Cross-stock

Cross-stock forms positive groups from distinct tickers over the same wall-clock window; the
training implementation also supports arbitrary `K`, same-industry selection, matched local views,
and structured edge matching. For this lightweight demo, we stream only until two stocks from one
date appear and then align their dense grids.
"""
)

code(
    """
cross_samples, rows_consumed = tutorial.same_date_pair(market_iterator, streamed_rows)
cross_pair = tutorial.cross_stock_pair(cross_samples)
print("Aligned pair:", [(row["ticker"], row["date"]) for row in cross_samples])
print("Additional streamed rows consumed:", rows_consumed)
tutorial.plot_cross_stock(cross_pair)
"""
)

markdown(
    """
### 2.3 Time warp

Time warp keeps a shared raw window but aggregates it through independently sampled monotone
piecewise-linear clocks. `warp_strength` controls knot displacement and `warp_knots` controls how
many locally faster or slower regions can appear.
"""
)

code(
    """
time_warp_pair = tutorial.same_stock_pair(
    "time_warp", dense_features, dense_seconds, warp_strength=0.75, warp_knots=8
)
tutorial.plot_views(time_warp_pair, "Time warp: independent monotone clocks")
"""
)

markdown(
    """
### 2.4 Gaussian noise

Gaussian noise is added independently after per-view normalization, so `noise_sigma` is measured
in normalized standard deviations. Only market-series channels are corrupted; checkpoint
information channels are deliberately exempt.
"""
)

code(
    """
gaussian_pair = tutorial.same_stock_pair(
    "gaussian_noise", dense_features, dense_seconds, noise_sigma=0.35
)
tutorial.plot_views(gaussian_pair, "Post-normalization Gaussian noise")
"""
)

markdown(
    """
### 2.5 Volume noise

Volume noise injects nonnegative exponential perturbations into volume and trade count in raw units,
scaled to the view's activity level. Prices and VWAP remain untouched before normalization.
"""
)

code(
    """
volume_pair = tutorial.same_stock_pair(
    "volume_noise", dense_features, dense_seconds, vol_noise_frac=0.50
)
tutorial.plot_views(volume_pair, "Volume noise", channels=("volume", "n"))
"""
)

markdown(
    """
### 2.6 Price jitter

Price jitter moves all five price-ladder channels in a bucket by the same random amount, scaled by
that bucket's half-spread. This preserves the spread and high/low ordering and cannot create a
crossed book.
"""
)

code(
    """
price_pair = tutorial.same_stock_pair(
    "price_jitter", dense_features, dense_seconds, price_jitter_frac=1.0
)
tutorial.plot_views(
    price_pair,
    "Spread-preserving price-ladder jitter",
    channels=("bid_price", "ask_price"),
)
"""
)

markdown(
    """
### 2.7 Channel drop

Channel drop independently masks entire normalized feature channels while resampling the all-dropped
case, so at least one channel survives. A zero row in the heatmap is a channel flattened to its
window mean in normalized space.
"""
)

code(
    """
drop_pair = tutorial.same_stock_pair(
    "channel_drop", dense_features, dense_seconds, channel_drop_p=0.35
)
tutorial.plot_channel_drop(drop_pair)
"""
)

markdown(
    """
## 3. Load the trained LeJEPA encoder

The reusable loader downloads `config.json`, `train_meta.json`, and `model.pt` from the model
repository, reconstructs the transformer backbone, and restores only the frozen encoder. The
projection head used by the LeJEPA pretraining loss is not needed for downstream representations.
"""
)

code(
    """
encoder, train_metadata = tutorial.load_lejepa_encoder(MODEL_ID)
n_parameters = sum(parameter.numel() for parameter in encoder.parameters())
trained_augmentation = train_metadata["config"]["dataset"]["augmentations"]["0"]["name"]

print(f"Encoder: {type(encoder).__name__}")
print(f"Input channels: {encoder.n_features}")
print(f"Embedding dimension: {encoder.d_embedding}")
print(f"Parameters: {n_parameters:,}")
print(f"Training augmentation: {trained_augmentation}")
"""
)

markdown(
    """
### Encode a market view

The checkpoint expects the nine market channels plus eight normalization-stat channels and three
window-information channels. We reconstruct that exact 20-channel wire format before running one
view through the frozen encoder.
"""
)

code(
    """
dataset_config = train_metadata["config"]["dataset"]
eval_config = dataset_config["eval_augmentations"]["0"]
model_pair = tutorial.random_resized_crop_pair(
    dense_features,
    dense_seconds,
    n_global=1,
    n_local=0,
    global_length=int(eval_config["global_seq_len"]),
    global_scale=tuple(eval_config["global_scale_range"]),
    with_stats=bool(train_metadata["info_norm_stats"]),
    with_window=bool(train_metadata["info_window"]),
    seed=SEED + 1,
)
model_input = model_pair["views"][0].unsqueeze(0)
lengths = torch.tensor([model_input.shape[-1]], dtype=torch.long)
with torch.inference_mode():
    embedding = encoder(model_input, lengths)

print("Encoder input shape:", tuple(model_input.shape))
print("Embedding shape:", tuple(embedding.shape))
print("First eight values:", embedding[0, :8])
"""
)

markdown(
    """
## Takeaways

- Hugging Face `datasets` streams sparse ticker-days from Parquet without a full data download.
- Market-aware aggregation preserves the semantics of prices, VWAP, sizes, volume, and trade count.
- Augmentations vary temporal scale, clock geometry, stock identity, activity, price levels, or
  available channels while respecting domain invariants.
- The public LeJEPA checkpoint supplies a reusable 384-dimensional market encoder.
"""
)

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (wm-booth-2026-finance-tutorial)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}
notebook["nbformat"] = 4
notebook["nbformat_minor"] = 5

destination = HERE / "01_market_jepa.ipynb"
nbf.write(notebook, destination)
print(f"Wrote {destination}")
