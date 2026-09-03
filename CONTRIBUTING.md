# Contributing

Keep source photographs, generated output, local configuration, and experiment
artifacts out of Git. Add or update tests with behavioral changes, then run:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 scripts/check_public_tree.py
```

Please keep detector, selector, and GUI version identities independent. Do not
rewrite the historical `auto-v4` record identifier when changing display text.
