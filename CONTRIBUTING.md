# Contributing

Thanks for taking the time to contribute.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research]"     # what CI installs
ruff check src tests app.py           # lint
pytest -q -m "not network"            # offline tests (what CI gates on)
pytest -q -m network                  # live Sentinel-2 smoke test, needs internet
```

The offline suite needs no API key and no network. The live test depends on public
imagery services being reachable, so CI reports it but does not fail on it.

## Ground rules

- **One logical change per PR.** Keep diffs reviewable.
- **Tests are not optional.** New behaviour needs a test; bug fixes need a test
  that fails before the fix.
- **The build must be green:** `ruff check src tests app.py` and
  `pytest -q -m "not network"` pass, and the Docker image still builds.
- **Numbers need evidence.** A change that moves a validation or calibration
  figure in the README must ship the script and output that produced it.
- **Failures stay visible.** A new data source or hazard degrades the answer and
  says so in the output; it never fails silently or guesses.

## Commit messages

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Reporting bugs / proposing features

Use the issue templates. For anything security-related, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
