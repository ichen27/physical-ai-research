#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements-dev.txt
python -W error -m unittest discover -s tests -v
# understudy_isaac needs Isaac Lab to import, so CI only checks that it compiles.
python -m compileall -q understudy understudy_isaac scripts
