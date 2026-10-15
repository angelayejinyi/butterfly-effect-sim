#!/usr/bin/env bash
# Compatibility entry point: run the homophily family with explicit paired seeds.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${SCRIPT_DIR}/run_experiments.py" homophily "$@"
