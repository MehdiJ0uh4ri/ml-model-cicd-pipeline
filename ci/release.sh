#!/usr/bin/env bash
# The CD path, run only after the eval gate passed on main:
#
#   register (challenger) -> shadow -> canary -> promote (champion) -> baseline
#
# Each step is a separate process with its own exit code so the workflow can
# stop at exactly the stage that failed, and so a human re-running one stage
# by hand does the same thing CI did.
set -euo pipefail

export PYTHONPATH=.
TARGET=${TARGET:-inprocess}

echo "::group::register challenger"
python -m src.model.registry register
echo "::endgroup::"

echo "::group::build replay traffic"
python -m deploy.make_traffic --out traffic/replay.jsonl --n "${TRAFFIC_N:-8000}"
echo "::endgroup::"

echo "::group::shadow deploy"
python -m deploy.shadow --target "$TARGET" --traffic traffic/replay.jsonl
echo "::endgroup::"

echo "::group::canary"
# canary.py exits non-zero on ROLLBACK and HOLD; `set -e` turns that into a
# failed deploy, which is the correct outcome for both.
python -m deploy.canary --target "$TARGET" --traffic traffic/replay.jsonl
echo "::endgroup::"

echo "::group::promote baseline"
python -m src.model.promote_baseline
echo "::endgroup::"

python -m src.model.registry status
