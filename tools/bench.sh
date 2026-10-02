#!/usr/bin/env bash
# One-command benchmark against a running server: tools/bench.sh [--base URL] [--model ID] [--reps N]
exec python3 "$(dirname "$(readlink -f "$0")")/bench.py" "$@"
