"""Core response-surface modelling utilities for injection moulding data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


INPUT_COLUMNS = ["射出速度", "保圧圧力", "金型温度"]
OUTPUT_COLUMNS = ["ヒケ深さ", "反り量", "光沢度"]
ALL_COLUMNS = INPUT_COLUMNS + OUTPUT_COLUMNS


@dataclass
class ResponseModel:
    name: str
    pipeline: Pipeline
    train_r2: float
    cv_r2: float
    cv_rmse: float

    def predict(self, conditions: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(conditions[INPUT_COLUMNS])


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a clean modelling data frame."""
    missing = [column for column in ALL_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("不足している列: " + ", ".join(missing))

    clean = data[ALL_COLUMNS].copy()
    for column in ALL_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    bad_rows = clean.index[clean.isna().any(axis=1)].tolist()
    if bad_rows:
        display_rows = ", ".join(str(index + 2) for index in bad_rows[:10])
        raise ValueError(f"数値でない値または欠損値があります（CSV行: {display_rows}）")
    if len(clean) < 10:
        raise ValueError("二次モデルには少なくとも10実験点が必要です（15点以上を推奨）。")
    constant = [column for column in INPUT_COLUMNS if clean[column].nunique() < 2]
    if constant:
        raise ValueError("条件が変化していない列: " + ", ".join(constant))
    return clean.reset_index(drop=True)


def fit_models(data: pd.DataFrame) -> dict[str, ResponseModel]:
    """Fit one full quadratic response surface for every quality response."""
    clean = validate_data(data)
    x = clean[INPUT_COLUMNS]
    models: dict[str, ResponseModel] = {}
    for response in OUTPUT_COLUMNS:
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
                ("regression", LinearRegression()),
            ]
        )
        y = clean[response].to_numpy()
        pipeline.fit(x, y)
        fitted = pipeline.predict(x)
        cv_prediction = cross_val_predict(pipeline, x, y, cv=LeaveOneOut())
        models[response] = ResponseModel(
            name=response,
            pipeline=pipeline,
            train_r2=float(r2_score(y, fitted)),
            cv_r2=float(r2_score(y, cv_prediction)),
            cv_rmse=float(mean_squared_error(y, cv_prediction) ** 0.5),
        )
    return models


def coefficient_table(model: ResponseModel) -> pd.DataFrame:
    """Return coefficients in standardized (coded factor) units."""
    polynomial = model.pipeline.named_steps["quadratic"]
    regression = model.pipeline.named_steps["regression"]
    names = polynomial.get_feature_names_out(["速度", "圧力", "温度"])
    return pd.DataFrame(
        {"項（標準化変数）": ["切片", *names],
         "係数": [regression.intercept_, *regression.coef_]}
    )


def make_slice_grid(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    fixed_values: Mapping[str, float],
    resolution: int = 70,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create a rectangular in-range grid for a two-factor slice."""
    x_values = np.linspace(data[x_name].min(), data[x_name].max(), resolution)
    y_values = np.linspace(data[y_name].min(), data[y_name].max(), resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = pd.DataFrame({x_name: xx.ravel(), y_name: yy.ravel()})
    for name in INPUT_COLUMNS:
        if name not in grid:
            grid[name] = float(fixed_values[name])
    return xx, yy, grid[INPUT_COLUMNS]


def feasible_mask(
    predictions: Mapping[str, np.ndarray],
    limits: Mapping[str, tuple[float | None, float | None]],
) -> np.ndarray:
    """Calculate points satisfying all enabled lower/upper quality limits."""
    first = next(iter(predictions.values()))
    mask = np.ones(np.asarray(first).shape, dtype=bool)
    for response, values in predictions.items():
        lower, upper = limits[response]
        if lower is not None:
            mask &= values >= lower
        if upper is not None:
            mask &= values <= upper
    return mask


def factor_pairs() -> list[tuple[str, str]]:
    return list(combinations(INPUT_COLUMNS, 2))


def suggest_optimum(
    data: pd.DataFrame,
    models: Mapping[str, ResponseModel],
    limits: Mapping[str, tuple[float | None, float | None]],
    resolution: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search an in-range 3-D grid and return feasible conditions and predictions."""
    axes = [
        np.linspace(data[column].min(), data[column].max(), resolution)
        for column in INPUT_COLUMNS
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    conditions = pd.DataFrame(
        {column: values.ravel() for column, values in zip(INPUT_COLUMNS, mesh)}
    )
    predicted = pd.DataFrame(
        {response: model.predict(conditions) for response, model in models.items()}
    )
    mask = feasible_mask(
        {column: predicted[column].to_numpy() for column in OUTPUT_COLUMNS}, limits
    )
    return conditions.loc[mask].reset_index(drop=True), predicted.loc[mask].reset_index(drop=True)

