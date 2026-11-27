#!/usr/bin/env bash
# The PR pipeline: data gate -> train -> evaluate -> eval gate.
#
# Exits non-zero at the first gate that blocks, but always leaves a report
# behind in artifacts/ so the PR comment can explain what happened. The two
# gates write separate markdown files on purpose: "your data is broken" and
# "your model got worse" are different conversations.
set -uo pipefail

export PYTHONPATH=.
mkdir -p artifacts

echo "::group::data quality gate"
python -m data.validate \
  --data data/raw/applications.parquet \
  --json-out artifacts/data_validation.json \
  --markdown artifacts/data_validation.md
data_rc=$?
echo "::endgroup::"

if [[ $data_rc -ne 0 ]]; then
  echo "[ci] data gate blocked; not spending a training run on bad data"
  exit $data_rc
fi

echo "::group::train"
python -m src.model.train || exit 1
echo "::endgroup::"

echo "::group::evaluate"
python -m src.model.evaluate || exit 1
echo "::endgroup::"

echo "::group::evaluation gate"
python -m src.model.gate
gate_rc=$?
echo "::endgroup::"

exit $gate_rc
