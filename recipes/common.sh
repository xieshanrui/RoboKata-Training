#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
IG_ROOT="$ROOT/environments/imitator_game"
export IG_ROOT PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
need() { for name in "$@"; do if [[ -z "${!name:-}" ]]; then printf 'Set %s in your local environment.
' "$name" >&2; exit 2; fi; done; }
run() { printf '+ '; printf '%q ' "$@"; printf '
'; if [[ "${TRAIN_DRY_RUN:-0}" != 1 ]]; then "$@"; fi; }
export PYTHONPATH="$ROOT/models/act:$ROOT/models/osvi:$ROOT/models/lingbot:$IG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
