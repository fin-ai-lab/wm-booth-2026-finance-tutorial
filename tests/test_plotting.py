import matplotlib.pyplot as plt
import numpy as np

from wm_booth_2026_finance_tutorial.plotting import plot_market_sample


def test_plot_market_sample_returns_unregistered_figure() -> None:
    sample = {
        "ticker": "TEST",
        "date": "2026-01-02",
        "ts_interval": np.array([1_767_363_000, 1_767_363_001]),
        "features": np.array(
            [
                [100.0, 100.5, 100.5, 100.5, 101.0, 10.0, 12.0, 20.0, 1.0],
                [100.1, 100.6, 100.6, 100.6, 101.1, 11.0, 13.0, 25.0, 2.0],
            ]
        ),
    }

    figure = plot_market_sample(sample)

    assert not plt.fignum_exists(figure.number)
