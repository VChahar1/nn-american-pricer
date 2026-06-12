## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train          # generate data + train (~2-5 min CPU)
python -m src.evaluate       # accuracy, speed, monotonicity
python -m src.make_figure    # prediction scatter + error plot
pytest -v                    # ~7 tests
```

## What this demonstrates vs what it doesn't

**Demonstrates**: the full methodology of building a neural surrogate for an expensive pricing function — sampling, training, evaluation against ground truth, speed comparison, and arbitrage-constraint checking.
