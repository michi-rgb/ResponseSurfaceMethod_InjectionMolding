"""Deterministic demonstration data; replace it with measured moulding data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from doe import generate_design
from rsm import ALL_COLUMNS


def create_sample_data() -> pd.DataFrame:
    design = generate_design(
        "box-behnken",
        {
            "射出速度": (40.0, 100.0),
            "保圧圧力": (40.0, 80.0),
            "金型温度": (50.0, 90.0),
        },
        center_runs=1,
        randomize=False,
    )
    rows = []
    rng = np.random.default_rng(20260919)
    for _, condition in design.iterrows():
        speed = condition["射出速度"]
        pressure = condition["保圧圧力"]
        temperature = condition["金型温度"]
        xs = (speed - 70.0) / 30.0
        xp = (pressure - 60.0) / 20.0
        xt = (temperature - 70.0) / 20.0
        sink = 0.18 - 0.040 * xp - 0.018 * xs + 0.020 * xt**2 + 0.012 * xs * xt
        warp = 0.48 + 0.080 * xs**2 + 0.060 * xt**2 - 0.035 * xp + 0.025 * xp * xt
        gloss = 72.0 + 5.0 * xt + 2.5 * xs - 2.0 * xp**2 + 1.8 * xs * xt
        rows.append(
            [
                speed,
                pressure,
                temperature,
                sink + rng.normal(0, 0.004),
                warp + rng.normal(0, 0.008),
                gloss + rng.normal(0, 0.5),
            ]
        )
    return pd.DataFrame(rows, columns=ALL_COLUMNS)
