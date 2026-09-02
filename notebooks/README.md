# Notebooks

Put the workshop's numbered, narrative notebooks here. Code shared between notebooks belongs in
`src/wm_booth_2026_finance_tutorial/` and can be imported normally after running `uv sync`:

```python
import wm_booth_2026_finance_tutorial as tutorial
```

Keep notebooks executable from top to bottom. Before committing, clear generated output with
`uv run nbstripout notebooks/*.ipynb`; the repository's pre-commit hooks also do this automatically.

`01_market_jepa.ipynb` is generated from `build_market_jepa_tutorial.py`; edit the latter and rerun
it when changing the notebook structure.
