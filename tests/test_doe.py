import numpy as np

from doe import generate_design
from rsm import INPUT_COLUMNS, OUTPUT_COLUMNS


LEVELS = {
    "射出速度": (40.0, 100.0),
    "保圧圧力": (40.0, 80.0),
    "金型温度": (50.0, 90.0),
}


def test_box_behnken_has_expected_points_and_centres():
    design = generate_design(
        "box-behnken", LEVELS, center_runs=3, randomize=False
    )
    assert len(design) == 15
    assert (design["点種別"] == "辺中心点").sum() == 12
    assert (design["点種別"] == "中心点").sum() == 3
    assert (design.loc[design["点種別"] == "中心点", INPUT_COLUMNS]
            == [70.0, 60.0, 70.0]).all().all()
    assert design[OUTPUT_COLUMNS].isna().all().all()


def test_face_centred_ccd_stays_between_factorial_levels():
    design = generate_design(
        "central-composite", LEVELS, center_runs=4,
        ccd_type="face-centered", randomize=False,
    )
    assert len(design) == 18
    assert (design["点種別"] == "要因点").sum() == 8
    assert (design["点種別"] == "軸点").sum() == 6
    for factor, (low, high) in LEVELS.items():
        assert design[factor].between(low, high).all()


def test_rotatable_ccd_uses_alpha_and_randomization_is_reproducible():
    first = generate_design(
        "central-composite", LEVELS, center_runs=3,
        ccd_type="rotatable", randomize=True, seed=7,
    )
    second = generate_design(
        "central-composite", LEVELS, center_runs=3,
        ccd_type="rotatable", randomize=True, seed=7,
    )
    assert first.equals(second)
    alpha = (2**3) ** 0.25
    assert np.isclose(first["コード_射出速度"].abs().max(), alpha)
    assert first["射出速度"].min() < LEVELS["射出速度"][0]
    assert first["射出速度"].max() > LEVELS["射出速度"][1]


def test_four_factor_box_behnken_scales_run_count():
    levels = {**LEVELS, "冷却時間": (5.0, 15.0)}
    design = generate_design(
        "box-behnken", levels, center_runs=3, randomize=False,
        response_columns=["寸法誤差"],
    )
    assert len(design) == 27
    assert (design["点種別"] == "辺中心点").sum() == 24
    assert "寸法誤差" in design


def test_two_factor_ccd_is_supported():
    levels = {"温度": (100.0, 140.0), "時間": (10.0, 30.0)}
    design = generate_design(
        "central-composite", levels, center_runs=1,
        ccd_type="face-centered", randomize=False,
        response_columns=["強度"],
    )
    assert len(design) == 9
    assert (design["点種別"] == "要因点").sum() == 4
    assert (design["点種別"] == "軸点").sum() == 4
