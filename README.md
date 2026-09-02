# Booth 2026 Finance Tutorial

Reusable Python code and hands-on notebooks for the 2026 Booth Wealth Management Conference CS
Workshop. The project uses a `src` layout so notebook examples exercise the same installed package
that participants can build on.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and
run:

```bash
uv sync
uv run jupyter lab
```

Open a notebook from `notebooks/`. `uv sync` creates a local `.venv`, installs the reusable package
in editable mode, and installs the workshop and development dependencies from the committed lockfile.

The first notebook, `notebooks/01_market_jepa.ipynb`, streams its demo market data from Hugging Face
and is configured to download the public LeJEPA encoder on first use. It does not require the
original Market-JEPA checkout or a local market-data archive.

## Layout

```text
notebooks/                               narrative workshop notebooks
src/wm_booth_2026_finance_tutorial/     reusable Python package
tests/                                   package tests
```

Notebook code can import shared helpers directly:

```python
import wm_booth_2026_finance_tutorial as tutorial

print(tutorial.__version__)
```

Add shared functions and classes under `src/wm_booth_2026_finance_tutorial/`, rather than copying
them between notebooks. Add project dependencies with `uv add <package>` and development-only tools
with `uv add --dev <package>`.

## Development

```bash
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv run mypy                            # type-check src/
uv run pytest                          # run unit tests
uv run pytest --nbmake notebooks/      # execute all notebooks
uv run pre-commit install              # enable linting and notebook cleanup on commit
```

CI runs linting, formatting, type checking, unit tests, and every notebook on pushes and pull
requests. Notebook output is stripped by pre-commit so repository diffs remain readable.
