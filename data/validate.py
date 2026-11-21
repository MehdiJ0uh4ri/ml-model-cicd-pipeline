"""The data quality gate. Nothing downstream runs until this exits 0.

Two backends, one suite file:

  * ``--backend ge``    hands data/expectations/suite.json to the installed
    great_expectations library (what CI uses).
  * ``--backend local`` evaluates the same expectation types with the small
    implementations below.

The local backend exists because a data gate that can only run inside CI is a
data gate developers route around. `make validate` on a laptop with no GE
install produces the same verdict JSON, the same exit code and the same
report, so "it passed locally" means something.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# local evaluator: one function per expectation type used by the suite
# --------------------------------------------------------------------------

Result = dict[str, Any]
_HANDLERS: dict[str, Callable[[pd.DataFrame, dict], Result]] = {}


def handles(name: str):
    def deco(fn):
        _HANDLERS[name] = fn
        return fn

    return deco


def _ok(success: bool, observed: Any, **extra) -> Result:
    return {"success": bool(success), "observed_value": observed, **extra}


@handles("expect_table_row_count_to_be_between")
def _rows(df, kw):
    n = len(df)
    lo, hi = kw.get("min_value"), kw.get("max_value")
    return _ok((lo is None or n >= lo) and (hi is None or n <= hi), n)


@handles("expect_table_columns_to_match_set")
def _cols(df, kw):
    expected, actual = set(kw["column_set"]), set(df.columns)
    return _ok(
        expected == actual,
        sorted(actual),
        missing=sorted(expected - actual),
        unexpected=sorted(actual - expected),
    )


@handles("expect_column_values_to_be_unique")
def _unique(df, kw):
    dupes = int(df[kw["column"]].duplicated().sum())
    return _ok(dupes == 0, dupes)


@handles("expect_column_values_to_not_be_null")
def _not_null(df, kw):
    col = df[kw["column"]]
    filled = float(col.notna().mean()) if len(col) else 1.0
    return _ok(filled >= float(kw.get("mostly", 1.0)), round(filled, 6))


@handles("expect_column_values_to_be_between")
def _between(df, kw):
    col = df[kw["column"]].dropna()
    if col.empty:
        return _ok(True, 1.0)
    lo, hi = kw.get("min_value"), kw.get("max_value")
    inside = pd.Series(True, index=col.index)
    if lo is not None:
        inside &= col >= lo
    if hi is not None:
        inside &= col <= hi
    frac = float(inside.mean())
    return _ok(frac >= float(kw.get("mostly", 1.0)), round(frac, 6))


@handles("expect_column_values_to_be_in_set")
def _in_set(df, kw):
    col = df[kw["column"]].dropna()
    allowed = set(kw["value_set"])
    seen = set(pd.unique(col))
    frac = float(col.isin(allowed).mean()) if len(col) else 1.0
    return _ok(
        frac >= float(kw.get("mostly", 1.0)),
        round(frac, 6),
        unexpected_values=sorted(str(v) for v in seen - allowed)[:10],
    )


@handles("expect_column_mean_to_be_between")
def _mean(df, kw):
    m = float(df[kw["column"]].astype("float64").mean())
    lo, hi = kw.get("min_value"), kw.get("max_value")
    return _ok((lo is None or m >= lo) and (hi is None or m <= hi), round(m, 6))


@handles("expect_column_proportion_of_unique_values_to_be_between")
def _prop_unique(df, kw):
    col = df[kw["column"]]
    prop = float(col.nunique()) / max(len(col), 1)
    lo, hi = kw.get("min_value"), kw.get("max_value")
    return _ok((lo is None or prop >= lo) and (hi is None or prop <= hi), round(prop, 8))


def run_local(df: pd.DataFrame, suite: dict) -> list[Result]:
    results = []
    for exp in suite["expectations"]:
        kind, kwargs = exp["expectation_type"], exp.get("kwargs", {})
        handler = _HANDLERS.get(kind)
        if handler is None:
            # Unknown expectation types fail loudly. Silently skipping one is
            # how a suite quietly stops protecting anything.
            res = _ok(False, None, error=f"no local implementation for {kind}")
        else:
            try:
                res = handler(df, kwargs)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                res = _ok(False, None, error=f"{type(exc).__name__}: {exc}")
        results.append(
            {
                "expectation_type": kind,
                "kwargs": kwargs,
                "meta": exp.get("meta", {}),
                **res,
            }
        )
    return results


# --------------------------------------------------------------------------
# great_expectations backend
# --------------------------------------------------------------------------


def _gx_class(expectation_type: str):
    """`expect_column_values_to_be_unique` -> `gx.expectations.ExpectColumnValuesToBeUnique`.

    GE 1.x replaced the string-keyed ExpectationConfiguration with typed
    classes. The suite file stays in the portable snake_case format both
    backends read, and the mapping happens here.
    """
    import great_expectations.expectations as gxe

    name = "".join(part.title() for part in expectation_type.split("_"))
    try:
        return getattr(gxe, name)
    except AttributeError as exc:
        raise SystemExit(
            f"[validate] great_expectations has no expectation {name} "
            f"(from {expectation_type}); check the suite against the installed GE version"
        ) from exc


def run_ge(df: pd.DataFrame, suite: dict) -> list[Result]:
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    gx_suite = context.suites.add(gx.ExpectationSuite(name=suite["expectation_suite_name"]))
    for exp in suite["expectations"]:
        gx_suite.add_expectation(_gx_class(exp["expectation_type"])(**exp.get("kwargs", {})))

    source = context.data_sources.add_pandas("snapshot")
    asset = source.add_dataframe_asset(name="applications")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name="raw", data=batch_def, suite=gx_suite)
    )
    out = validation.run(batch_parameters={"dataframe": df})

    # GE does not return results in suite order, so they are matched by
    # (type, column) rather than by position. Zipping the two lists silently
    # attributes one expectation's verdict to another -- the kind of bug that
    # makes a data gate report a failure on the wrong column.
    by_key: dict[tuple[str, str], list] = {}
    for r in out.results:
        cfg = r.expectation_config
        key = (cfg.type, str(cfg.kwargs.get("column", "-")))
        by_key.setdefault(key, []).append(r)

    results = []
    for exp in suite["expectations"]:
        key = (exp["expectation_type"], str(exp.get("kwargs", {}).get("column", "-")))
        matches = by_key.get(key)
        if not matches:
            raise SystemExit(f"[validate] great_expectations returned no result for {key}")
        r = matches.pop(0)
        results.append(
            {
                "expectation_type": exp["expectation_type"],
                "kwargs": exp.get("kwargs", {}),
                "meta": exp.get("meta", {}),
                "success": bool(r.success),
                "observed_value": (r.result or {}).get("observed_value"),
            }
        )
    return results


# --------------------------------------------------------------------------


def to_markdown(report: dict) -> str:
    lines = [
        "## Data quality gate",
        "",
        f"**{report['decision']}** - {report['passed']}/{report['total']} expectations passed "
        f"on `{report['source']}` ({report['rows']:,} rows, backend `{report['backend']}`).",
        "",
    ]
    failed = [r for r in report["results"] if not r["success"]]
    if failed:
        lines += ["| expectation | column | observed | why it matters |", "|---|---|---|---|"]
        for r in failed:
            col = r["kwargs"].get("column", "-")
            why = r.get("meta", {}).get("why", "")
            lines.append(
                f"| `{r['expectation_type']}` | `{col}` | `{r.get('observed_value')}` | {why} |"
            )
    else:
        lines.append("No expectation failed. Training may proceed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="run the expectation suite against a snapshot")
    ap.add_argument("--data", type=Path, default=Path("data/raw/applications.parquet"))
    ap.add_argument("--suite", type=Path, default=Path("data/expectations/suite.json"))
    ap.add_argument("--backend", choices=["auto", "ge", "local"], default="auto")
    ap.add_argument("--json-out", type=Path, default=Path("artifacts/data_validation.json"))
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--warn-only", action="store_true", help="report but always exit 0")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))

    backend = args.backend
    if backend == "auto":
        try:
            import great_expectations  # noqa: F401

            backend = "ge"
        except ImportError:
            backend = "local"

    results = run_ge(df, suite) if backend == "ge" else run_local(df, suite)
    passed = sum(1 for r in results if r["success"])
    report = {
        "suite": suite["expectation_suite_name"],
        "source": str(args.data),
        "backend": backend,
        "rows": len(df),
        "total": len(results),
        "passed": passed,
        "decision": "PASS" if passed == len(results) else "BLOCKED",
        "results": results,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(report), encoding="utf-8")

    print(f"[{backend}] {report['decision']}: {passed}/{len(results)} expectations passed")
    for r in results:
        if not r["success"]:
            col = r["kwargs"].get("column", "-")
            print(f"  FAIL {r['expectation_type']}({col}) observed={r.get('observed_value')}")

    if report["decision"] == "BLOCKED" and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
