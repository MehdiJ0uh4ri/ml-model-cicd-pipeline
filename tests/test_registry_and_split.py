"""Registry alias semantics, the split rule, and Makefile/DVC parity."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from src.model.features import stable_split_mask
from src.model.registry import LocalRegistry

# ------------------------------------------------------------------ split


def test_split_is_stable_when_the_snapshot_grows():
    ids = pd.Series([f"APP-{i:07d}" for i in range(10000)])
    first = set(ids[stable_split_mask(ids, 0.2)])

    grown = pd.Series([f"APP-{i:07d}" for i in range(15000)])
    second = set(grown[stable_split_mask(grown, 0.2)])

    # Every row held out last week is still held out this week. Without this,
    # week-over-week metric comparisons are noise.
    assert first <= second


def test_split_ratio_is_close_to_requested():
    ids = pd.Series([f"APP-{i:07d}" for i in range(20000)])
    share = stable_split_mask(ids, 0.2).mean()
    assert 0.19 <= share <= 0.21


def test_split_is_independent_of_row_order():
    ids = pd.Series([f"APP-{i:07d}" for i in range(5000)])
    forward = set(ids[stable_split_mask(ids, 0.2)])
    reversed_ids = ids[::-1].reset_index(drop=True)
    backward = set(reversed_ids[stable_split_mask(reversed_ids, 0.2)])
    assert forward == backward


# --------------------------------------------------------------- registry


def make_registry(tmp_path: Path, model_file: Path) -> LocalRegistry:
    model_file.write_text("not-really-a-model", encoding="utf-8")
    return LocalRegistry(tmp_path, "test-model")


def test_register_then_promote_moves_aliases(tmp_path):
    model = tmp_path / "model.joblib"
    reg = make_registry(tmp_path, model)

    v1 = reg.register(model, {"git_sha": "aaa"})
    reg.set_alias("champion", v1)
    v2 = reg.register(model, {"git_sha": "bbb"})
    reg.set_alias("challenger", v2)

    assert reg.alias_version("champion") == "1"
    assert reg.alias_version("challenger") == "2"


def test_rollback_is_a_swap_not_a_delete(tmp_path):
    model = tmp_path / "model.joblib"
    reg = make_registry(tmp_path, model)
    v1 = reg.register(model, {"git_sha": "aaa"})
    v2 = reg.register(model, {"git_sha": "bbb"})

    reg.set_alias("champion", v1)
    reg.set_alias("rollback", v1)
    reg.set_alias("champion", v2)

    # Both versions remain addressable after a promotion; nothing is removed.
    assert reg.alias_version("rollback") == "1"
    assert reg.alias_version("champion") == "2"
    assert (Path(reg.root) / "v1" / "model.joblib").exists()


def test_history_records_every_alias_move(tmp_path):
    model = tmp_path / "model.joblib"
    reg = make_registry(tmp_path, model)
    v1 = reg.register(model, {"git_sha": "aaa"})
    reg.set_alias("champion", v1)
    actions = [h["action"] for h in reg.status()["history"]]
    assert actions == ["register", "set_alias"]


def test_unknown_version_is_refused(tmp_path):
    model = tmp_path / "model.joblib"
    reg = make_registry(tmp_path, model)
    try:
        reg.set_alias("champion", "99")
    except SystemExit as exc:
        assert "no version 99" in str(exc)
    else:
        raise AssertionError("aliasing a nonexistent version must fail loudly")


# ----------------------------------------------------------------- parity


def test_dvc_stages_and_make_targets_run_the_same_modules():
    """`make pipeline` and `dvc repro` must not drift apart.

    Both are documented as the way to run the pipeline; if they invoke
    different entry points, one of the two documented paths is a lie.
    """
    dvc = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for stage in ("train", "evaluate", "gate"):
        module = re.search(r"python -m ([\w.]+)", dvc["stages"][stage]["cmd"]).group(1)
        assert module in makefile, f"dvc stage {stage} runs {module}, which no make target does"


def test_every_threshold_section_is_consumed():
    """A threshold nobody reads is a false sense of safety."""
    thresholds = yaml.safe_load(Path("thresholds.yaml").read_text(encoding="utf-8"))
    sources = "".join(
        Path(p).read_text(encoding="utf-8")
        for p in ("src/model/gate.py", "deploy/shadow.py", "deploy/canary.py")
    )
    for section in thresholds:
        assert f'"{section}"' in sources or f"'{section}'" in sources, (
            f"thresholds.yaml:{section} is never read by a gate"
        )
