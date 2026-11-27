#!/usr/bin/env bash
# Install dependencies and make the DVC-tracked data present. Idempotent:
# safe to run twice, safe to run in a warm cache.
set -euo pipefail

pip install -q --upgrade pip
pip install -q -r requirements-dev.txt

if [[ -d .dvc ]]; then
  # `dvc pull` is the real path in a repo with a remote. In this template the
  # snapshot is generated instead of downloaded, so a missing remote is not an
  # error -- but a *failing* remote must still be loud.
  if dvc remote list | grep -q .; then
    dvc pull --allow-missing || echo "[ci] dvc pull found nothing to fetch"
  fi
fi

if [[ ! -f data/raw/applications.parquet ]]; then
  echo "[ci] no snapshot from DVC; generating the seeded one"
  python -m data.generate
fi

python -c "import pandas, sklearn; print('[ci] pandas', pandas.__version__, '| sklearn', sklearn.__version__)"
