from __future__ import annotations
import json
import os
from functools import lru_cache
import joblib
import pandas as pd
import config


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the model artefact is missing on disk."""


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load and cache the fitted scikit-learn pipeline."""
    if not os.path.exists(config.MODEL_PATH):
        raise ModelNotTrainedError(
            f"Model not found at '{config.MODEL_PATH}'. "
            "Train it first with `python models/train_model.py`."
        )
    return joblib.load(config.MODEL_PATH)


def load_metrics() -> dict:
    """Return the saved training metrics, or an empty dict if unavailable."""
    if not os.path.exists(config.METRICS_PATH):
        return {}
    with open(config.METRICS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _coerce_and_validate(raw: dict) -> dict:
    """Cast incoming form values to the correct types and validate ranges.

    Raises
    ------
    ValueError
        If a required field is missing or a value is out of range.
    """
    # ``is_weekend`` is derivable from ``day_of_week`` so it is not required
    # from the caller; every other feature must be supplied.
    required = [
        f
        for f in config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES
        if f != "is_weekend"
    ]
    missing = [f for f in required if f not in raw or raw[f] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    cleaned: dict = {}

    # Categorical values ----------------------------------------------------
    for col in config.CATEGORICAL_FEATURES:
        cleaned[col] = str(raw[col]).strip()

    # Numeric values with sensible bounds -----------------------------------
    def _num(key, cast, lo, hi):
        try:
            val = cast(raw[key])
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number.")
        if not (lo <= val <= hi):
            raise ValueError(f"'{key}' must be between {lo} and {hi}.")
        return val

    cleaned["post_hour"] = _num("post_hour", int, 0, 23)
    cleaned["followers"] = _num("followers", int, 1, 100_000_000)
    cleaned["hashtags_count"] = _num("hashtags_count", int, 0, 60)
    cleaned["caption_length"] = _num("caption_length", int, 0, 5000)

    # is_weekend can be derived from day_of_week if not supplied explicitly.
    if "is_weekend" in raw and raw["is_weekend"] not in (None, ""):
        cleaned["is_weekend"] = int(raw["is_weekend"])
    else:
        cleaned["is_weekend"] = int(
            cleaned["day_of_week"] in ("Saturday", "Sunday")
        )

    return cleaned


def predict_engagement(raw: dict) -> dict:
    """Predict engagement rate for a single post described by ``raw``.

    Parameters
    ----------
    raw:
        Mapping of feature name -> value (typically request.form).

    Returns
    -------
    dict
        ``{"engagement_rate": float, "band": str, "inputs": dict}``
    """
    cleaned = _coerce_and_validate(raw)
    pipeline = _load_pipeline()

    features = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES
    row = pd.DataFrame([{k: cleaned[k] for k in features}])

    rate = float(pipeline.predict(row)[0])
    rate = max(0.0, round(rate, 3))  # engagement rate can't be negative

    return {
        "engagement_rate": rate,
        "band": config.band_for_rate(rate),
        "inputs": cleaned,
    }
