"""Interactive response-surface analysis for injection moulding."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from doe import design_summary, generate_design
from rsm import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    coefficient_table,
    feasible_mask,
    fit_models,
    make_slice_grid,
    suggest_optimum,
    validate_data,
)
from sample_data import create_sample_data


st.set_page_config(page_title="射出成形プロセスウィンドウ", page_icon="🧩", layout="wide")
st.title("樹脂射出成形｜応答曲面とプロセスウィンドウ")
st.caption("射出速度・保圧圧力・金型温度から、ヒケ深さ・反り量・光沢度の二次応答曲面を推定します。")


@st.cache_data
def read_csv(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSVの文字コードを読み取れませんでした。UTF-8またはCP932で保存してください。")


@st.cache_resource
def cached_fit(csv_text: str):
    frame = pd.read_csv(io.StringIO(csv_text))
    return fit_models(frame)


with st.sidebar:
    st.header("1. 実験データ")
    uploaded = st.file_uploader("CSVを選択", type="csv")
    sample = create_sample_data()
    st.download_button(
        "入力テンプレートをダウンロード",
        sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="射出成形_RSM_入力例.csv",
        mime="text/csv",
    )
    st.caption("未選択時はBox–Behnken計画（中心点1回）の13点デモデータを使用します。列名と単位はプロジェクト内で統一してください。")

try:
    raw_data = read_csv(uploaded.getvalue()) if uploaded else sample
    data = validate_data(raw_data)
    models = cached_fit(data.to_csv(index=False))
except Exception as exc:
    st.error(f"データを解析できません: {exc}")
    st.stop()

with st.sidebar:
    st.header("2. 品質規格")
    limits: dict[str, tuple[float | None, float | None]] = {}
    default_modes = {"ヒケ深さ": "上限", "反り量": "上限", "光沢度": "下限"}
    default_values = {
        "ヒケ深さ": float(data["ヒケ深さ"].quantile(0.55)),
        "反り量": float(data["反り量"].quantile(0.55)),
        "光沢度": float(data["光沢度"].quantile(0.45)),
    }
    for response in OUTPUT_COLUMNS:
        mode = st.selectbox(
            f"{response}の判定", ["上限", "下限", "範囲", "判定しない"],
            index=["上限", "下限", "範囲", "判定しない"].index(default_modes[response]),
            key=f"mode_{response}",
        )
        spread = float(data[response].max() - data[response].min()) or 1.0
        if mode == "上限":
            upper = st.number_input(f"{response} ≤", value=default_values[response], step=spread / 50, format="%.4f")
            limits[response] = (None, upper)
        elif mode == "下限":
            lower = st.number_input(f"{response} ≥", value=default_values[response], step=spread / 50, format="%.4f")
            limits[response] = (lower, None)
        elif mode == "範囲":
            lower = st.number_input(f"{response} 下限", value=float(data[response].quantile(0.25)), key=f"lo_{response}")
            upper = st.number_input(f"{response} 上限", value=float(data[response].quantile(0.75)), key=f"hi_{response}")
            limits[response] = (lower, upper)
        else:
            limits[response] = (None, None)

tab_doe, tab_data, tab_model, tab_surface, tab_window, tab_predict = st.tabs(
    ["実験計画", "データ", "モデル精度", "応答曲面", "プロセスウィンドウ", "条件予測"]
)

with tab_doe:
    st.subheader("応答曲面用の実験条件表を作成")
    design_label = st.radio(
        "実験計画",
        ["Box–Behnken計画", "中心複合計画（CCD）"],
        horizontal=True,
        help=(
            "極端な条件の組み合わせを避けたい場合はBox–Behnken、"
            "条件範囲の境界や角を詳しく評価したい場合は面中心型CCDが目安です。"
        ),
    )
    design_name = "box-behnken" if design_label.startswith("Box") else "central-composite"

    with st.popover("？ 使い分けワンポイント"):
        st.markdown(
            """
            **Box–Behnken計画**

            - 各因子の両端は測定しますが、3因子すべてが同時に端となる角点は作りません。
            - バリ、焼け、ショートショット、過大圧力など、極端条件の組み合わせを避けたい場合に向きます。

            **面中心型CCD**

            - 設定範囲の角点と、1因子だけを端にした軸点を含みます。
            - プロセスウィンドウの境界付近を詳しく確認したい場合に向きます。

            **回転可能型CCD**

            - 中心から同じ距離にある条件の予測精度をそろえやすい設計です。
            - 軸点が設定した±1水準の外側に出るため、成形可能範囲と設備上限の確認が必要です。

            **中心点を繰り返す理由**

            - 同じ設定でも生じる成形・測定のばらつき（純粋誤差）を推定します。
            - 二次応答曲面の当てはまり不足や、実験中の工程変動を確認しやすくなります。
            - 繰り返しによって工程自体のばらつきが小さくなるわけではありません。

            **条件数の決まり方**

            - このアプリのBox–Behnken計画は3因子固定です。
            - 3つの因子ペアそれぞれに4条件を作るため、中心点以外は `3 × 4 = 12条件` です。
            - 総条件数は `12 ＋ 中心点の反復数`。中心点3回なら15条件です。
            - CCDは要因点8条件＋軸点6条件なので、総条件数は `14 ＋ 中心点の反復数` です。

            **迷った場合:** 極端条件が危険ならBox–Behnken、境界把握を優先するなら面中心型CCDを選びます。
            """
        )

    level_columns = st.columns(3)
    levels: dict[str, tuple[float, float]] = {}
    for column, factor in zip(level_columns, INPUT_COLUMNS):
        default_low = float(data[factor].min())
        default_high = float(data[factor].max())
        step = (default_high - default_low) / 20 or 1.0
        column.markdown(f"**{factor}**")
        low = column.number_input(
            "−1水準",
            value=default_low,
            step=step,
            key=f"doe_low_{factor}",
        )
        high = column.number_input(
            "＋1水準",
            value=default_high,
            step=step,
            key=f"doe_high_{factor}",
        )
        column.caption(f"中心水準: {(low + high) / 2:g}")
        levels[factor] = (low, high)

    setting_columns = st.columns(4)
    center_runs = setting_columns[0].number_input(
        "中心点の反復数",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help=(
            "同一の中心条件を独立して繰り返し、成形・測定の純粋誤差を推定します。"
            "3回が最低限の目安で、工程変動をより確実に確認する場合は5回程度を検討します。"
            "Box–Behnkenの総条件数は12＋反復数、CCDは14＋反復数です。"
        ),
    )
    ccd_type = "face-centered"
    if design_name == "central-composite":
        ccd_label = setting_columns[1].selectbox(
            "CCD形式", ["面中心型", "回転可能型"]
        )
        ccd_type = "face-centered" if ccd_label == "面中心型" else "rotatable"
    randomize = setting_columns[2].checkbox("実行順をランダム化", value=True)
    seed = setting_columns[3].number_input(
        "乱数シード", min_value=0, max_value=999999, value=42, step=1,
        disabled=not randomize,
    )

    try:
        design_table = generate_design(
            design_name,
            levels,
            center_runs=int(center_runs),
            ccd_type=ccd_type,
            randomize=randomize,
            seed=int(seed),
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.info(design_summary(design_name, int(center_runs)))
        if design_name == "central-composite" and ccd_type == "rotatable":
            st.warning(
                "回転可能型では軸点が±1水準の外側（コード値±1.682）になります。"
                "成形機・材料・金型の安全範囲内であることを確認してください。"
            )
        show_coded = st.checkbox("コード化水準を表示", value=False)
        coded_columns = [f"コード_{factor}" for factor in INPUT_COLUMNS]
        display_columns = [
            column for column in design_table.columns
            if show_coded or column not in coded_columns
        ]
        st.dataframe(
            design_table[display_columns], width="stretch", hide_index=True
        )
        st.download_button(
            "実験条件表をCSVでダウンロード",
            design_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=(
                "Box-Behnken_実験条件.csv"
                if design_name == "box-behnken"
                else "CCD_実験条件.csv"
            ),
            mime="text/csv",
        )
        st.caption(
            "品質測定後、CSVのヒケ深さ・反り量・光沢度を入力し、"
            "先頭の実行順・標準順・点種別・コード列を残したまま本アプリへ読み込めます。"
        )

with tab_data:
    source = "アップロードデータ" if uploaded else "デモデータ"
    st.subheader(f"{source}（{len(data)}点）")
    st.dataframe(data, width="stretch", hide_index=True)
    st.caption("モデルは入力データの最小～最大範囲内で使用してください。範囲外への外挿は行いません。")

with tab_model:
    metrics = pd.DataFrame(
        [
            {
                "品質指標": response,
                "学習 R²": model.train_r2,
                "LOOCV R²": model.cv_r2,
                "LOOCV RMSE": model.cv_rmse,
            }
            for response, model in models.items()
        ]
    )
    st.dataframe(metrics.style.format({"学習 R²": "{:.3f}", "LOOCV R²": "{:.3f}", "LOOCV RMSE": "{:.4g}"}), width="stretch", hide_index=True)
    st.info("LOOCV R²を予測性能の目安にしてください。負の値は、平均値で予測するより性能が低く、追加実験やモデル見直しが必要なサインです。")
    selected_coefficient = st.selectbox("係数を表示", OUTPUT_COLUMNS)
    st.dataframe(coefficient_table(models[selected_coefficient]), hide_index=True, width="stretch")
    st.caption("係数は平均0・標準偏差1に標準化した因子に対する値です。絶対値が大きい項ほど影響が強い目安になります。")

with tab_surface:
    col_a, col_b, col_c = st.columns(3)
    x_name = col_a.selectbox("横軸", INPUT_COLUMNS, index=0, key="surface_x")
    y_options = [name for name in INPUT_COLUMNS if name != x_name]
    y_name = col_b.selectbox("縦軸", y_options, index=0, key="surface_y")
    fixed_name = next(name for name in INPUT_COLUMNS if name not in (x_name, y_name))
    fixed_value = col_c.slider(
        f"固定: {fixed_name}", float(data[fixed_name].min()), float(data[fixed_name].max()),
        float(data[fixed_name].median()), key="surface_fixed"
    )
    xx, yy, grid = make_slice_grid(data, x_name, y_name, {fixed_name: fixed_value})
    figure = make_subplots(rows=1, cols=3, subplot_titles=OUTPUT_COLUMNS)
    colorbar_centres = [0.144, 0.500, 0.856]
    for column_index, response in enumerate(OUTPUT_COLUMNS, start=1):
        zz = models[response].predict(grid).reshape(xx.shape)
        figure.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=zz,
                colorscale="Viridis",
                contours_coloring="heatmap",
                colorbar=dict(
                    title=dict(text=response, side="top"),
                    orientation="h",
                    x=colorbar_centres[column_index - 1],
                    xanchor="center",
                    y=-0.22,
                    yanchor="top",
                    len=0.25,
                    thickness=12,
                ),
            ),
            row=1, col=column_index,
        )
        figure.add_trace(
            go.Scatter(x=data[x_name], y=data[y_name], mode="markers", marker=dict(color="white", line=dict(color="black", width=1), size=6), showlegend=False, hovertemplate="実験点<extra></extra>"),
            row=1, col=column_index,
        )
    figure.update_xaxes(title_text=x_name)
    figure.update_yaxes(title_text=y_name)
    figure.update_layout(height=610, margin=dict(t=60, b=125))
    st.plotly_chart(figure, width="stretch")
    st.caption(f"{fixed_name} = {fixed_value:g} の断面。白丸は全実験点の{x_name}–{y_name}投影です。")

with tab_window:
    col_a, col_b, col_c = st.columns(3)
    wx = col_a.selectbox("横軸", INPUT_COLUMNS, index=0, key="window_x")
    wy_options = [name for name in INPUT_COLUMNS if name != wx]
    wy = col_b.selectbox("縦軸", wy_options, index=0, key="window_y")
    wf = next(name for name in INPUT_COLUMNS if name not in (wx, wy))
    wvalue = col_c.slider(
        f"固定: {wf}", float(data[wf].min()), float(data[wf].max()),
        float(data[wf].median()), key="window_fixed"
    )
    wxx, wyy, wgrid = make_slice_grid(data, wx, wy, {wf: wvalue}, resolution=100)
    predictions = {response: models[response].predict(wgrid) for response in OUTPUT_COLUMNS}
    mask = feasible_mask(predictions, limits).reshape(wxx.shape)
    window_figure = go.Figure(
        go.Contour(
            x=wxx[0], y=wyy[:, 0], z=mask.astype(int),
            zmin=0, zmax=1, contours=dict(start=0, end=1, size=1, coloring="heatmap"),
            colorscale=[[0, "#ef4444"], [0.499, "#ef4444"], [0.5, "#22c55e"], [1, "#22c55e"]],
            colorbar=dict(tickvals=[0, 1], ticktext=["規格外", "規格内"]),
            hovertemplate=f"{wx}: %{{x:.3g}}<br>{wy}: %{{y:.3g}}<br>%{{z}}<extra></extra>",
        )
    )
    window_figure.add_trace(go.Scatter(x=data[wx], y=data[wy], mode="markers", name="実験点", marker=dict(color="white", line=dict(color="black", width=1))))
    window_figure.update_layout(
        xaxis_title=wx,
        yaxis_title=wy,
        width=700,
        height=700,
        autosize=False,
        margin=dict(l=80, r=80, t=80, b=80),
        title=f"{wf} = {wvalue:g} の同時規格適合領域",
    )
    st.plotly_chart(
        window_figure,
        width="content",
        config={"responsive": False},
    )
    st.metric("この断面の規格内面積率（グリッド近似）", f"{mask.mean():.1%}")

    feasible_conditions, feasible_quality = suggest_optimum(data, models, limits)
    total_grid_points = 24 ** 3
    st.metric("3因子空間の規格内体積率（グリッド近似）", f"{len(feasible_conditions) / total_grid_points:.1%}")
    if len(feasible_conditions):
        centre = pd.Series({name: data[name].median() for name in INPUT_COLUMNS})
        span = pd.Series({name: data[name].max() - data[name].min() for name in INPUT_COLUMNS}).replace(0, 1)
        distance = (((feasible_conditions - centre) / span) ** 2).sum(axis=1)
        best_index = distance.idxmin()
        recommendation = pd.concat([feasible_conditions.loc[best_index], feasible_quality.loc[best_index]])
        st.subheader("規格内の推奨条件")
        st.dataframe(recommendation.rename("推定値").to_frame(), width="stretch")
        st.caption("探索範囲の中央に最も近い規格内点です。ロバスト性の最終確認には確認実験を行ってください。")
    else:
        st.warning("現在の品質規格を同時に満たす条件は探索グリッド内で見つかりませんでした。")

with tab_predict:
    st.subheader("任意条件での品質予測")
    condition = {}
    columns = st.columns(3)
    for column, factor in zip(columns, INPUT_COLUMNS):
        condition[factor] = column.slider(
            factor, float(data[factor].min()), float(data[factor].max()), float(data[factor].median()), key=f"predict_{factor}"
        )
    query = pd.DataFrame([condition])
    predicted_values = {response: float(model.predict(query)[0]) for response, model in models.items()}
    columns = st.columns(3)
    for column, response in zip(columns, OUTPUT_COLUMNS):
        lower, upper = limits[response]
        ok = (lower is None or predicted_values[response] >= lower) and (upper is None or predicted_values[response] <= upper)
        column.metric(response, f"{predicted_values[response]:.4g}", "規格内" if ok else "規格外", delta_color="normal" if ok else "inverse")
    st.caption("予測値は統計モデルによる推定です。量産条件の決定前に、推奨点と境界付近で確認実験を実施してください。")
