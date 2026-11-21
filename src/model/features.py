"""Feature pipeline and the split rule.

Two things in here are load-bearing for reproducibility:

* the split is a hash of `application_id`, not `train_test_split(random_state=)`.
  A row keeps its side of the split when the snapshot grows, so week-over-week
  metrics compare the same held-out population instead of a reshuffled one.
* the preprocessing lives inside the sklearn Pipeline, so the artifact that
  the registry promotes is the same object that serving calls. There is no
  separate transform step to drift out of sync.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "default"
ID_COLUMN = "application_id"


def stable_split_mask(ids: pd.Series, test_size: float, salt: str = "credit-default") -> np.ndarray:
    """True where the row belongs to the test set.

    md5 of (salt, id) mapped into [0, 1). Deterministic across machines,
    Python versions and snapshot sizes -- unlike hash(), which is salted per
    process, and unlike a shuffled split, which moves rows between sides when
    new data arrives.
    """
    buckets = ids.astype(str).map(
        lambda v: int(hashlib.md5(f"{salt}:{v}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    )
    return (buckets < test_size).to_numpy()


def split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = stable_split_mask(df[ID_COLUMN], test_size)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


def feature_columns(params: dict) -> list[str]:
    """Model inputs. `age_group` is deliberately absent: it is a slicing key
    for the fairness gate, never a feature."""
    return list(params["features"]["numeric"]) + list(params["features"]["categorical"])


def build_pipeline(params: dict) -> Pipeline:
    numeric = list(params["features"]["numeric"])
    categorical = list(params["features"]["categorical"])
    model_cfg = params["model"]

    pre = ColumnTransformer(
        transformers=[
            # median imputation matches the <=1% missing-income tolerance the
            # expectation suite enforces upstream. Beyond that the data gate
            # blocks the run rather than letting the imputer paper over it.
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = HistGradientBoostingClassifier(
        max_iter=model_cfg["max_iter"],
        learning_rate=model_cfg["learning_rate"],
        max_leaf_nodes=model_cfg["max_leaf_nodes"],
        min_samples_leaf=model_cfg["min_samples_leaf"],
        l2_regularization=model_cfg["l2_regularization"],
        early_stopping=model_cfg["early_stopping"],
        validation_fraction=model_cfg["validation_fraction"],
        random_state=model_cfg["random_state"],
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def as_frame(records: list[dict], params: dict) -> pd.DataFrame:
    """Coerce serving payloads into exactly the training column order.

    Serving receives JSON; training receives parquet with categorical dtypes.
    Reindexing here -- rather than trusting the caller -- is what stops a
    reordered or partial payload from silently shifting a one-hot block.
    """
    cols = feature_columns(params)
    df = pd.DataFrame.from_records(records)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]

    # Every numeric column becomes float64, including the counts that are
    # integers in the warehouse. A payload with a missing `credit_lines` would
    # otherwise arrive as float in one request and int in the next, which
    # breaks schema enforcement on the registered signature -- and an integer
    # column cannot represent the missing value the imputer exists to handle.
    numeric = list(params["features"]["numeric"])
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").astype("float64")
    return df
