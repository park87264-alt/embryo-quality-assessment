# Experiments

Experiment numbers preserve the original server chronology. Each directory may contain:

- `src/`: model, training, evaluation, or inference entry points;
- `tests/`: focused logic tests when available;
- `results/`: aggregate, de-identified metrics only.

Raw images, per-embryo predictions, checkpoints, logs, smoke runs, and preliminary runs are intentionally excluded. The authoritative status and headline metrics are in [`docs/experiment_registry.md`](../docs/experiment_registry.md).

Experiments 06-12 are early provenance snapshots and have missing helper or training files in the local archive. Experiments 13-27 contain their primary entry point, but still require external datasets, derived features, or checkpoints described in the registry.
