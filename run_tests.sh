#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
poetry run python -m backend.run_tests "$@" 