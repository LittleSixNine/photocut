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

V8.4 is the default detector and every valid candidate requires user
confirmation. Keep this separate from the legacy selector's automatic-acceptance
contract. Public model files are limited to the explicitly released V8.4 and
V8.2 ONNX assets; do not add training checkpoints, source photos, annotations, or
per-image experimental logs. Report training-set adaptation separately from
source-group evaluation; see [V8.4](docs/V8.4.md).
