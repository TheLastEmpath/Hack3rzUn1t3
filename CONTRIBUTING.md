# Contributing

Thanks for helping. The `main` branch is the single source of truth for the
build. Changes land only through reviewed pull requests.

## Start here

1. Read [docs/handoff.md](docs/handoff.md) for where the project stands.
2. Pick an open issue (look for `good first issue`) and comment that you are
   taking it, so two people don't do the same work.
3. Fork the repo, create a branch, make your change, open a pull request.

## Run the tests

No GPU is needed for the harness core:

```bash
PYTHONPATH=src python3 -m unittest tests.test_harness tests.test_coordinator tests.test_net tests.test_worker
```

`tests/test_math.py` needs torch and a GPU machine. GPU stages (layer
streaming, shard editing, model runs) need a GPU machine; say in your pull
request what you ran and on what hardware class.

## What a good pull request has

- One focused change, with tests for new behaviour.
- Docs updated in the same pull request (`README.md`, `docs/`).
- Standard library only in the harness core (`src/hack3rzun1t3/harness/`) so it
  stays torch-free.
- A short description of what changed and how you checked it.

## Never commit

- Model weights, edited shards, checkpoints or run outputs.
- Model replies from evaluation runs.
- Hostnames, IP addresses, usernames, local paths, keys, tokens or passwords.
- Inventory or benchmark output that names a machine (use `--redact-host`).

A CI check (`scripts/check_safety.py`) fails pull requests that add IPs, local
paths, emails, key-like strings, weight files or files over 5 MB. For a
deliberate example, put `safety-ok` on that line.

## Licensing

The project is Apache-2.0 (see [LICENSE](LICENSE)). By submitting a
contribution you agree it is licensed under the same terms. Third-party code or
data keeps its own license and must be credited in [NOTICES.md](NOTICES.md).

## Scope and responsible use

This is a research and learning project about how refusal behaviour is
mediated in language models. Contributions that aim at harming people, or at
evading detection for malicious ends, will be declined.

## Security issues

Do not open a public issue for a vulnerability. Use the repository's
"Report a vulnerability" button on the Security tab (private reporting).

See also the [Code of Conduct](CODE_OF_CONDUCT.md).
