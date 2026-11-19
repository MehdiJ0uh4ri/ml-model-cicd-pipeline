SHELL := /bin/bash
.DEFAULT_GOAL := help

PY        ?= python
VENV      ?= .venv
BIN       := $(VENV)/bin
# venvs put their entry points in Scripts/ on Windows. Detected rather than
# documented so `make` works the same from Git Bash as it does in CI.
ifeq ($(OS),Windows_NT)
BIN       := $(VENV)/Scripts
endif
ARTIFACTS ?= artifacts
DATA      ?= data/raw/applications.parquet
TRAFFIC   ?= traffic/replay.jsonl
TARGET    ?= inprocess
THRESHOLDS?= thresholds.yaml

export PYTHONPATH := .

# Every target below is exactly what CI runs. If a job in .github/workflows/
# does something this Makefile cannot, that is a bug in the workflow.

.PHONY: help
help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

$(BIN)/activate:
	$(PY) -m venv $(VENV)

.PHONY: install
install: $(BIN)/activate ## install runtime + MLOps toolchain
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements-dev.txt

.PHONY: lint
lint: ## ruff
	$(BIN)/ruff check data src serving deploy monitoring tests

.PHONY: test
test: ## unit tests (offline; no model training)
	$(BIN)/python -m pytest tests

# ----------------------------------------------------------------- data

.PHONY: snapshot
snapshot: ## materialise the raw snapshot DVC versions
	$(BIN)/python -m data.generate --params params.yaml --out $(DATA)

.PHONY: validate
validate: ## Great Expectations gate -- blocks everything downstream
	$(BIN)/python -m data.validate --data $(DATA) \
	  --json-out $(ARTIFACTS)/data_validation.json \
	  --markdown $(ARTIFACTS)/data_validation.md

.PHONY: pull
pull: ## dvc pull: sync data + model artifacts from the remote
	$(BIN)/dvc pull --allow-missing || \
	  echo "[dvc] nothing to pull (no remote configured in this clone)"

.PHONY: repro
repro: ## run the pipeline through DVC (respects the cache)
	$(BIN)/dvc repro

# ------------------------------------------------------------- training

.PHONY: train
train: ## train + log an MLflow run
	$(BIN)/python -m src.model.train --data $(DATA)

.PHONY: evaluate
evaluate: ## score the held-out split, write the scorecard
	$(BIN)/python -m src.model.evaluate --data $(DATA)

.PHONY: gate
gate: ## apply thresholds.yaml; non-zero exit blocks promotion
	$(BIN)/python -m src.model.gate --thresholds $(THRESHOLDS)

.PHONY: gate-warn
gate-warn: ## same checks, never blocks (nightly)
	$(BIN)/python -m src.model.gate --thresholds $(THRESHOLDS) --warn-only

.PHONY: pipeline
pipeline: snapshot validate train evaluate gate ## the full training pipeline, no DVC needed

# ------------------------------------------------------------- registry

.PHONY: register
register: ## register a gate-approved model as challenger
	$(BIN)/python -m src.model.registry register

.PHONY: status
status: ## what is champion / challenger right now
	$(BIN)/python -m src.model.registry status

.PHONY: baseline
baseline: ## promote the current scorecard to the tracked baseline (main only)
	$(BIN)/python -m src.model.promote_baseline

.PHONY: rollback
rollback: ## swap champion back to the previous version
	$(BIN)/python -m src.model.registry rollback

# --------------------------------------------------------------- deploy

.PHONY: traffic
traffic: ## build the replay file from the held-out split
	$(BIN)/python -m deploy.make_traffic --data $(DATA) --out $(TRAFFIC)

.PHONY: shadow
shadow: ## shadow deploy: challenger scores real traffic, nobody sees it
	$(BIN)/python -m deploy.shadow --target $(TARGET) --traffic $(TRAFFIC)

.PHONY: canary
canary: ## 5% canary -> promote on pass, auto-rollback on regression
	$(BIN)/python -m deploy.canary --target $(TARGET) --traffic $(TRAFFIC)

.PHONY: release
release: register traffic shadow canary ## registry -> shadow -> canary -> promote

.PHONY: serve
serve: ## run the inference service on :8080
	$(BIN)/uvicorn serving.app:app --host 0.0.0.0 --port 8080

# ----------------------------------------------------------- monitoring

.PHONY: drift
drift: ## PSI / KS drift report over recent traffic
	$(BIN)/python -m monitoring.drift --current $(TRAFFIC)

# ------------------------------------------------------------------ demos
# Each demo proves one gate actually blocks. They are the fastest way to see
# what this repo does without reading a workflow file.

.PHONY: demo-bad-data
demo-bad-data: ## prove the data gate blocks: 4% null income
	$(BIN)/python -m data.generate --out $(ARTIFACTS)/demo/bad.parquet --null-rate 0.04
	-$(BIN)/python -m data.validate --data $(ARTIFACTS)/demo/bad.parquet \
	  --json-out $(ARTIFACTS)/demo/data_validation.json
	@echo ">>> expected: BLOCKED (expect_column_values_to_not_be_null on income)"

.PHONY: demo-bad-model
demo-bad-model: ## prove the eval gate blocks: a deliberately crippled model
	$(BIN)/python -m src.model.train --out $(ARTIFACTS)/demo/model.joblib \
	  --metrics-out $(ARTIFACTS)/demo/train_metrics.json \
	  --params ci/params-crippled.yaml
	$(BIN)/python -m src.model.evaluate --model $(ARTIFACTS)/demo/model.joblib \
	  --params ci/params-crippled.yaml --out $(ARTIFACTS)/demo/scorecard.json
	-$(BIN)/python -m src.model.gate --scorecard $(ARTIFACTS)/demo/scorecard.json \
	  --json-out $(ARTIFACTS)/demo/verdict.json --markdown $(ARTIFACTS)/demo/report.md
	@echo ">>> expected: BLOCKED (floors and/or regression vs baseline)"

.PHONY: demo-rollback
demo-rollback: ## prove the canary catches what the eval gate cannot
	# A challenger that clears every offline threshold and still moves the live
	# decision mix by 6%. It passes the gate, passes shadow, and is rolled back
	# by the canary's business guard. Leaves the challenger alias pointing at
	# the rejected version -- exactly as a real rollback does, for triage.
	$(BIN)/python -m src.model.train --params ci/params-loose-threshold.yaml \
	  --out $(ARTIFACTS)/demo/model.joblib --metrics-out $(ARTIFACTS)/demo/train_metrics.json
	$(BIN)/python -m src.model.evaluate --params ci/params-loose-threshold.yaml \
	  --model $(ARTIFACTS)/demo/model.joblib --out $(ARTIFACTS)/demo/scorecard.json
	$(BIN)/python -m src.model.gate --scorecard $(ARTIFACTS)/demo/scorecard.json \
	  --json-out $(ARTIFACTS)/demo/verdict.json --markdown $(ARTIFACTS)/demo/report.md
	@echo ">>> eval gate: PROMOTE (it cannot see a decision-mix change)"
	$(BIN)/python -m src.model.registry register --model $(ARTIFACTS)/demo/model.joblib \
	  --verdict $(ARTIFACTS)/demo/verdict.json --scorecard $(ARTIFACTS)/demo/scorecard.json \
	  --train-metrics $(ARTIFACTS)/demo/train_metrics.json \
	  --json-out $(ARTIFACTS)/demo/registered.json
	$(BIN)/python -m deploy.make_traffic --n 8000 --out $(ARTIFACTS)/demo/replay.jsonl
	$(BIN)/python -m deploy.shadow --traffic $(ARTIFACTS)/demo/replay.jsonl \
	  --json-out $(ARTIFACTS)/demo/shadow.json --markdown $(ARTIFACTS)/demo/shadow.md
	@echo ">>> shadow: PASS (it runs fine; it just decides differently)"
	-$(BIN)/python -m deploy.canary --traffic $(ARTIFACTS)/demo/replay.jsonl \
	  --shadow-report $(ARTIFACTS)/demo/shadow.json \
	  --json-out $(ARTIFACTS)/demo/canary.json --markdown $(ARTIFACTS)/demo/canary.md
	@echo ">>> expected: ROLLBACK on decision_rate_shift"

.PHONY: demo-drift
demo-drift: ## prove the drift monitor fires: replay a severe macro shift
	$(BIN)/python -m deploy.make_traffic --shift severe --out $(ARTIFACTS)/demo/shifted.jsonl
	-$(BIN)/python -m monitoring.drift --current $(ARTIFACTS)/demo/shifted.jsonl \
	  --json-out $(ARTIFACTS)/demo/drift.json --markdown $(ARTIFACTS)/demo/drift.md \
	  --fail-on-alert
	@echo ">>> expected: ALERT on income, utilization, late_payments_2y"

# ------------------------------------------------------------- container

.PHONY: docker-build
docker-build:
	docker build -t credit-default-risk:local -f serving/Dockerfile .

.PHONY: up
up: ## service + prometheus + grafana + mlflow
	docker compose up --build -d

.PHONY: down
down:
	docker compose down -v

.PHONY: clean
clean:
	rm -rf $(ARTIFACTS) registry mlruns mlruns-local .pytest_cache .ruff_cache traffic/*.jsonl

# ------------------------------------------------------------------- ci

.PHONY: ci
ci: lint test pipeline ## everything the PR job runs
