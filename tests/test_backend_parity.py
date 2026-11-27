"""The two validation backends must reach the same verdict.

The repo claims `--backend local` is a faithful stand-in for Great
Expectations, which is a claim worth a test rather than a paragraph. Skipped
when GE is not installed; CI installs it, so the claim is checked on every PR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.generate import build_frame
from data.validate import run_local

pytest.importorskip("great_expectations", reason="GE backend not installed")

from data.validate import run_ge  # noqa: E402  (must come after importorskip)

SUITE = json.loads(Path("data/expectations/suite.json").read_text(encoding="utf-8"))
CFG = {
    "seed": 5,
    "n_rows": 12000,
    "snapshot": "2026-01-01",
    "inject": {"null_rate": 0.0, "label_flip": 0.0},
}


def verdicts(results) -> dict[str, bool]:
    return {
        f"{r['expectation_type']}:{r['kwargs'].get('column', '-')}": r["success"] for r in results
    }


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return build_frame(CFG)


def test_backends_agree_on_a_clean_snapshot(clean):
    assert verdicts(run_ge(clean, SUITE)) == verdicts(run_local(clean, SUITE))


def test_backends_agree_on_excess_nulls(clean):
    df = clean.copy()
    df.loc[df.index[: int(len(df) * 0.05)], "income"] = None
    ge, local = verdicts(run_ge(df, SUITE)), verdicts(run_local(df, SUITE))
    assert ge == local
    assert ge["expect_column_values_to_not_be_null:income"] is False


def test_backends_agree_on_an_unexpected_category(clean):
    df = clean.copy()
    df["employment"] = df["employment"].cat.add_categories(["gig_work"])
    df.loc[df.index[:100], "employment"] = "gig_work"
    ge, local = verdicts(run_ge(df, SUITE)), verdicts(run_local(df, SUITE))
    assert ge == local
    assert ge["expect_column_values_to_be_in_set:employment"] is False
