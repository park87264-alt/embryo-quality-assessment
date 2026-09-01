# Data access and privacy

## Dataset roles

| Dataset | Role in this project | Included here |
|---|---|---|
| Nantes | Continuous embryo trajectories, 16-stage supervision, multifocal pretraining | No |
| SFU | ICM/TE/ZP segmentation supervision | No |
| Kromp Blastocyst Dataset | Static images with Gardner Expansion/ICM/TE labels | No |

Nantes and Kromp are not joined by embryo identity. Kromp images therefore cannot activate the real temporal branch used for Nantes trajectories. A repeated Kromp feature token is an interface compatibility device, not a reconstructed time series.

## Files that must remain private

- embryo and patient identifiers;
- image paths or manifests that expose identifiers;
- raw or processed embryo images;
- per-embryo predictions and annotation contact sheets;
- private dataset archives and derived feature tables;
- checkpoints trained on data whose redistribution terms are unclear.

Only de-identified aggregate metrics may be committed. Before every release, inspect `git status` and scan text files for credentials, local usernames, server paths, and identifiers.

PowerShell scan without extra tools:

```powershell
Get-ChildItem -Recurse -File -Include *.py,*.ps1,*.sh,*.md,*.txt,*.json,*.yaml,*.yml,*.toml |
  Select-String -Pattern "password|passwd|token|secret|server-host|server-port|private-user|private-home" -CaseSensitive:$false
```

The placeholder `/path/to/embryo_data` in scripts is intentional. Supply authorized local paths with command-line arguments; do not replace it with a private server path in a committed file.
