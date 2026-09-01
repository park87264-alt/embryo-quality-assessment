# Release checklist

Run these checks before each push.

## 1. Inspect files

```powershell
git status --short
git diff --check
git diff --stat
```

## 2. Scan text for private information

```powershell
Get-ChildItem -Recurse -File -Include *.py,*.ps1,*.sh,*.md,*.txt,*.json,*.yaml,*.yml,*.toml |
  Select-String -Pattern "password|passwd|token|secret|server-host|server-port|private-user|private-home" -CaseSensitive:$false
```

Review every match manually. Do not commit raw images, identifiers, manifests, per-embryo predictions, private paths, or checkpoints.

## 3. Review the staged snapshot

```powershell
git add .
git status
git diff --cached --stat
git diff --cached
```

If the staged snapshot is correct:

```powershell
git commit -m "Organize experiment code and documentation"
git push
```

Keep the GitHub repository private until data licenses, third-party licenses, and the incomplete early experiment files have been reviewed.
