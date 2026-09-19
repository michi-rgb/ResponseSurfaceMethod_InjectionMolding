"""Design-of-experiments generators for three-factor response surfaces."""

from __future__ import annotations

from itertools import combinations, product
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from rsm import OUTPUT_COLUMNS


DesignName = Literal["box-behnken", "central-composite"]
CCDType = Literal["face-centered", "rotatable"]


def _validate_settings(
    levels: Mapping[str, tuple[float, float]], center_runs: int
) -> None:
    if len(levels) < 2:
        raise ValueError("実験計画には2因子以上が必要です。")
    for factor in levels:
        low, high = levels[factor]
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"{factor}は－1水準より＋1水準を大きくしてください。")
    if center_runs < 1:
        raise ValueError("中心点の反復数は1以上にしてください。")


def _box_behnken_points(
    factor_count: int, center_runs: int
) -> tuple[list[list[float]], list[str]]:
    if factor_count < 3:
        raise ValueError("Box–Behnken計画には3因子以上が必要です。")
    points: list[list[float]] = []
    point_types: list[str] = []
    for first, second in combinations(range(factor_count), 2):
        for signs in product((-1.0, 1.0), repeat=2):
            point = [0.0] * factor_count
            point[first], point[second] = signs
            points.append(point)
            point_types.append("辺中心点")
    points.extend([[0.0] * factor_count] * center_runs)
    point_types.extend(["中心点"] * center_runs)
    return points, point_types


def _central_composite_points(
    factor_count: int, center_runs: int, ccd_type: CCDType
) -> tuple[list[list[float]], list[str], float]:
    alpha = 1.0 if ccd_type == "face-centered" else float((2**factor_count) ** 0.25)
    factorial = [
        list(values) for values in product((-1.0, 1.0), repeat=factor_count)
    ]
    axial: list[list[float]] = []
    for factor_index in range(factor_count):
        for sign in (-1.0, 1.0):
            point = [0.0] * factor_count
            point[factor_index] = sign * alpha
            axial.append(point)
    points = factorial + axial + [[0.0] * factor_count] * center_runs
    point_types = (
        ["要因点"] * len(factorial)
        + ["軸点"] * len(axial)
        + ["中心点"] * center_runs
    )
    return points, point_types, alpha


def generate_design(
    design: DesignName,
    levels: Mapping[str, tuple[float, float]],
    center_runs: int = 3,
    ccd_type: CCDType = "face-centered",
    randomize: bool = True,
    seed: int = 42,
    response_columns: list[str] = OUTPUT_COLUMNS,
) -> pd.DataFrame:
    """Generate a run sheet containing coded levels, real levels, and blank responses.

    ``levels`` specifies the real values corresponding to coded levels -1 and +1.
    For a rotatable CCD, axial points therefore extend beyond those two values.
    """
    _validate_settings(levels, center_runs)
    input_columns = list(levels)
    factor_count = len(input_columns)
    if design == "box-behnken":
        points, point_types = _box_behnken_points(factor_count, center_runs)
    elif design == "central-composite":
        points, point_types, _ = _central_composite_points(
            factor_count, center_runs, ccd_type
        )
    else:
        raise ValueError(f"未対応の実験計画: {design}")

    coded = np.asarray(points, dtype=float)
    frame = pd.DataFrame(
        {
            "標準順": np.arange(1, len(coded) + 1),
            "点種別": point_types,
        }
    )
    for index, factor in enumerate(input_columns):
        low, high = levels[factor]
        centre = (low + high) / 2.0
        half_range = (high - low) / 2.0
        frame[factor] = centre + coded[:, index] * half_range
        frame[f"コード_{factor}"] = coded[:, index]

    if randomize:
        frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    frame.insert(0, "実行順", np.arange(1, len(frame) + 1))
    for response in response_columns:
        frame[response] = np.nan
    return frame


def design_summary(
    design: DesignName, factor_count: int, center_runs: int
) -> str:
    if design == "box-behnken":
        edge_count = 2 * factor_count * (factor_count - 1)
        return (
            f"辺中心点{edge_count}回＋中心点{center_runs}回"
            f"＝合計{edge_count + center_runs}回"
        )
    factorial_count = 2**factor_count
    axial_count = 2 * factor_count
    return (
        f"要因点{factorial_count}回＋軸点{axial_count}回＋中心点{center_runs}回"
        f"＝合計{factorial_count + axial_count + center_runs}回"
    )
