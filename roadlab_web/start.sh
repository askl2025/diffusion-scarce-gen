#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$app_dir")"
env_file="${ROADLAB_ENV_FILE:-$repo_dir/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi
python_bin="${ROADLAB_PYTHON:-python3}"
cd -- "$app_dir"
exec "$python_bin" -u app.py
