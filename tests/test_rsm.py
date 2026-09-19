import numpy as np

from rsm import OUTPUT_COLUMNS, feasible_mask, fit_models, suggest_optimum
from sample_data import create_sample_data


def test_models_fit_and_predict_all_responses():
    data = create_sample_data()
    assert len(data) == 13
    centre = data[
        (data["射出速度"] == 70.0)
        & (data["保圧圧力"] == 60.0)
        & (data["金型温度"] == 70.0)
    ]
    assert len(centre) == 1
    models = fit_models(data)
    assert set(models) == set(OUTPUT_COLUMNS)
    for model in models.values():
        prediction = model.predict(data.iloc[:2])
        assert prediction.shape == (2,)
        assert np.isfinite(prediction).all()


def test_feasible_mask_combines_limits():
    prediction = {
        "ヒケ深さ": np.array([0.1, 0.3, 0.1]),
        "反り量": np.array([0.4, 0.4, 0.8]),
        "光沢度": np.array([70.0, 70.0, 70.0]),
    }
    limits = {"ヒケ深さ": (None, 0.2), "反り量": (None, 0.5), "光沢度": (65, None)}
    assert feasible_mask(prediction, limits).tolist() == [True, False, False]


def test_grid_search_stays_inside_experimental_range():
    data = create_sample_data()
    models = fit_models(data)
    limits = {response: (None, None) for response in OUTPUT_COLUMNS}
    conditions, quality = suggest_optimum(data, models, limits, resolution=5)
    assert len(conditions) == 125
    assert len(quality) == 125
    for column in conditions:
        assert conditions[column].between(data[column].min(), data[column].max()).all()
