"""The expectation evaluator must fail on the defects it claims to catch.

These tests assert on *blocking*, not on wording. A suite that quietly passes
broken data is the only failure mode that matters here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.generate import build_frame
from data.validate import run_local

SUITE = json.loads(Path("data/expectations/suite.json").read_text(encoding="utf-8"))

BASE_CFG = {
    "seed": 1,
    "n_rows": 12000,
    "snapshot": "2026-01-01",
    "inject": {"null_rate": 0.0, "label_flip": 0.0},
}


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return build_frame(BASE_CFG)


def failures(df: pd.DataFrame) -> set[str]:
    return {
        f"{r['expectation_type']}:{r['kwargs'].get('column', '-')}"
        for r in run_local(df, SUITE)
        if not r["success"]
    }


def test_clean_snapshot_passes(clean):
    assert failures(clean) == set()


def test_excess_nulls_block(clean):
    df = clean.copy()
    # 4% > the 1% `mostly` tolerance on income.
    df.loc[df.index[: int(len(df) * 0.04)], "income"] = None
    assert "expect_column_values_to_not_be_null:income" in failures(df)


def test_nulls_within_tolerance_pass(clean):
    df = clean.copy()
    df.loc[df.index[: int(len(df) * 0.005)], "income"] = None
    assert "expect_column_values_to_not_be_null:income" not in failures(df)


def test_missing_column_blocks(clean):
    assert "expect_table_columns_to_match_set:-" in failures(clean.drop(columns=["utilization"]))


def test_new_category_blocks(clean):
    df = clean.copy()
    df["employment"] = df["employment"].cat.add_categories(["gig_work"])
    df.loc[df.index[:50], "employment"] = "gig_work"
    assert "expect_column_values_to_be_in_set:employment" in failures(df)


def test_unit_change_blocks(clean):
    # The classic upstream break: utilization arrives as 0-100 instead of 0-1.
    df = clean.copy()
    df["utilization"] = df["utilization"] * 100
    assert "expect_column_values_to_be_between:utilization" in failures(df)


def test_duplicate_ids_block(clean):
    df = pd.concat([clean, clean.iloc[:200]], ignore_index=True)
    assert "expect_column_values_to_be_unique:application_id" in failures(df)


def test_truncated_extract_blocks(clean):
    assert "expect_table_row_count_to_be_between:-" in failures(clean.iloc[:500])


def test_unknown_expectation_type_fails_loudly(clean):
    suite = {"expectations": [{"expectation_type": "expect_something_invented", "kwargs": {}}]}
    results = run_local(clean, suite)
    assert results[0]["success"] is False
    assert "no local implementation" in results[0]["error"]


def test_generator_is_deterministic():
    a, b = build_frame(BASE_CFG), build_frame(BASE_CFG)
    pd.testing.assert_frame_equal(a, b)
