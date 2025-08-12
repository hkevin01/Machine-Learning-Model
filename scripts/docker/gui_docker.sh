#!/usr/bin/env bash
# Relocated docker GUI runner (wraps original run.sh logic)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT_DIR}/run.sh" "$@"
