"""Deterministic demonstration data; replace it with measured moulding data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from doe import generate_design
from rsm import INPUT_COLUMNS, OUTPUT_COLUMNS


def create_sample_data(
    input_columns: list[str] = INPUT_COLUMNS,
    output_columns: list[str] = OUTPUT_COLUMNS,
) -> pd.DataFrame:
    known_levels = {
        "射出速度": (40.0, 100.0),
        "保圧圧力": (40.0, 80.0),
        "金型温度": (50.0, 90.0),
    }
    levels = {
        factor: known_levels.get(factor, (0.0, 100.0))
        for factor in input_columns
    }
    design_name = "box-behnken" if len(input_columns) >= 3 else "central-composite"
    design = generate_design(
        design_name,
        levels,
        center_runs=1,
        ccd_type="face-centered",
        randomize=False,
        response_columns=output_columns,
    )
    rng = np.random.default_rng(20260919)
    coded = np.column_stack(
        [design[f"コード_{factor}"].to_numpy() for factor in input_columns]
    )
    responses: dict[str, np.ndarray] = {}
    for response_index, response in enumerate(output_columns):
        linear = sum(
            (0.8 + 0.25 * ((index + response_index) % 3)) * coded[:, index]
            for index in range(len(input_columns))
        )
        quadratic = sum(
            (0.45 + 0.1 * ((index + response_index) % 2)) * coded[:, index] ** 2
            for index in range(len(input_columns))
        )
        interaction = (
            0.35 * coded[:, 0] * coded[:, 1]
            if len(input_columns) >= 2 else 0.0
        )
        base = 10.0 * (response_index + 1)
        responses[response] = (
            base + linear + quadratic + interaction
            + rng.normal(0, 0.08, len(design))
        )
    return pd.concat(
        [design[input_columns].reset_index(drop=True), pd.DataFrame(responses)], axis=1
    )
