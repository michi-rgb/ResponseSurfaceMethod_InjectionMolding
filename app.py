"""Interactive response-surface analysis for injection moulding."""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
st.caption("任意の入力因子と出力応答について、二次応答曲面とプロセスウィンドウを推定します。")


@st.cache_data
def read_csv(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSVの文字コードを読み取れませんでした。UTF-8またはCP932で保存してください。")


@st.cache_resource
def cached_fit(
    csv_text: str,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
):
    frame = pd.read_csv(io.StringIO(csv_text))
    return fit_models(frame, list(input_columns), list(output_columns))


with st.sidebar:
    st.header("1. 入出力項目")
    factor_count = st.number_input(
        "入力因子数", min_value=2, max_value=6, value=3, step=1,
        help="応答曲面の断面表示には2因子以上が必要です。",
    )
    default_inputs = INPUT_COLUMNS + [f"入力因子{i}" for i in range(4, 7)]
    input_columns = [
        st.text_input(
            f"入力因子 {index + 1}",
            value=default_inputs[index],
            key=f"input_name_{index}",
        ).strip()
        for index in range(int(factor_count))
    ]
    response_count = st.number_input(
        "出力応答数", min_value=1, max_value=6, value=3, step=1
    )
    default_outputs = OUTPUT_COLUMNS + [f"出力応答{i}" for i in range(4, 7)]
    output_columns = [
        st.text_input(
            f"出力応答 {index + 1}",
            value=default_outputs[index],
            key=f"output_name_{index}",
        ).strip()
        for index in range(int(response_count))
    ]

    all_names = input_columns + output_columns
    if any(not name for name in all_names):
        st.error("項目名を空欄にはできません。")
        st.stop()
    if len(set(all_names)) != len(all_names):
        st.error("入力因子と出力応答には重複しない名前を付けてください。")
        st.stop()

    st.header("2. 実験データ")
    uploaded = st.file_uploader("CSVを選択", type="csv")
    sample = create_sample_data(input_columns, output_columns)
    st.download_button(
        "入力テンプレートをダウンロード",
        sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="射出成形_RSM_入力例.csv",
        mime="text/csv",
    )
    demo_design = "Box–Behnken" if len(input_columns) >= 3 else "面中心型CCD"
    st.caption(
        f"未選択時は{demo_design}計画（中心点1回）のデモデータを使用します。"
        "列名と単位はプロジェクト内で統一してください。"
    )

try:
    raw_data = read_csv(uploaded.getvalue()) if uploaded else sample
    data = validate_data(raw_data, input_columns, output_columns)
    models = cached_fit(
        data.to_csv(index=False), tuple(input_columns), tuple(output_columns)
    )
except Exception as exc:
    st.error(f"データを解析できません: {exc}")
    st.stop()

with st.sidebar:
    st.header("3. 品質規格")
    limits: dict[str, tuple[float | None, float | None]] = {}
    for response_index, response in enumerate(output_columns):
        default_mode = "下限" if "光沢" in response else "上限"
        default_value = float(
            data[response].quantile(0.45 if default_mode == "下限" else 0.55)
        )
        mode = st.selectbox(
            f"{response}の判定", ["上限", "下限", "範囲", "判定しない"],
            index=["上限", "下限", "範囲", "判定しない"].index(default_mode),
            key=f"mode_{response_index}",
        )
        spread = float(data[response].max() - data[response].min()) or 1.0
        if mode == "上限":
            upper = st.number_input(f"{response} ≤", value=default_value, step=spread / 50, format="%.4f", key=f"upper_{response_index}")
            limits[response] = (None, upper)
        elif mode == "下限":
            lower = st.number_input(f"{response} ≥", value=default_value, step=spread / 50, format="%.4f", key=f"lower_{response_index}")
            limits[response] = (lower, None)
        elif mode == "範囲":
            lower = st.number_input(f"{response} 下限", value=float(data[response].quantile(0.25)), key=f"lo_{response_index}")
            upper = st.number_input(f"{response} 上限", value=float(data[response].quantile(0.75)), key=f"hi_{response_index}")
            limits[response] = (lower, upper)
        else:
            limits[response] = (None, None)

tab_labels = ["実験計画", "データ", "モデル精度", "応答曲面", "プロセスウィンドウ", "条件予測"]
selected_tab = st.segmented_control(
    "表示画面",
    tab_labels,
    default=tab_labels[0],
    key="selected_tab",
    label_visibility="collapsed",
    width="stretch",
)

if selected_tab == tab_labels[0]:
    st.subheader("応答曲面用の実験条件表を作成")
    design_options = (
        ["Box–Behnken計画", "中心複合計画（CCD）"]
        if len(input_columns) >= 3
        else ["中心複合計画（CCD）"]
    )
    design_label = st.radio(
        "実験計画",
        design_options,
        horizontal=True,
        help=(
            "極端な条件の組み合わせを避けたい場合はBox–Behnken、"
            "条件範囲の境界や角を詳しく評価したい場合は面中心型CCDが目安です。"
        ),
    )
    design_name = "box-behnken" if design_label.startswith("Box") else "central-composite"

    with st.popover("？ 使い分けワンポイント"):
        edge_count = 2 * len(input_columns) * (len(input_columns) - 1)
        ccd_base_count = 2 ** len(input_columns) + 2 * len(input_columns)
        st.markdown(
            f"""
            **Box–Behnken計画**

            - 各因子の両端は測定しますが、3つ以上の因子が同時に端となる角点は作りません。
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

            - 現在は{len(input_columns)}因子です。
            - Box–Behnkenは各因子ペアに4条件を作るため、総条件数は `{edge_count} ＋ 中心点反復数` です。
            - CCDは要因点と軸点を合わせ、総条件数は `{ccd_base_count} ＋ 中心点反復数` です。

            **迷った場合:** 極端条件が危険ならBox–Behnken、境界把握を優先するなら面中心型CCDを選びます。
            """
        )

    level_columns = st.columns(min(3, len(input_columns)))
    levels: dict[str, tuple[float, float]] = {}
    for factor_index, factor in enumerate(input_columns):
        column = level_columns[factor_index % len(level_columns)]
        default_low = float(data[factor].min())
        default_high = float(data[factor].max())
        step = (default_high - default_low) / 20 or 1.0
        column.markdown(f"**{factor}**")
        low = column.number_input(
            "−1水準",
            value=default_low,
            step=step,
            key=f"doe_low_{factor_index}",
        )
        high = column.number_input(
            "＋1水準",
            value=default_high,
            step=step,
            key=f"doe_high_{factor_index}",
        )
        column.caption(f"中心水準: {(low + high) / 2:g}")
        levels[factor] = (low, high)

    setting_columns = st.columns(4)
    center_runs = setting_columns[0].number_input(
        "中心点の反復数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help=(
            "同一の中心条件を独立して繰り返し、成形・測定の純粋誤差を推定します。"
            "3回が最低限の目安で、工程変動をより確実に確認する場合は5回程度を検討します。"
            f"現在の因子数ではBox–Behnkenは{edge_count}＋反復数、"
            f"CCDは{ccd_base_count}＋反復数です。"
        ),
    )
    ccd_type = "face-centered"
    if design_name == "central-composite":
        ccd_label = setting_columns[1].selectbox(
            "CCD形式", ["面中心型", "回転可能型"]
        )
        ccd_type = "face-centered" if ccd_label == "面中心型" else "rotatable"
    randomize = setting_columns[2].checkbox("実行順をランダム化", value=False)
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
            response_columns=output_columns,
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.info(design_summary(design_name, len(input_columns), int(center_runs)))
        if design_name == "central-composite" and ccd_type == "rotatable":
            alpha = (2 ** len(input_columns)) ** 0.25
            st.warning(
                f"回転可能型では軸点が±1水準の外側（コード値±{alpha:.3f}）になります。"
                "成形機・材料・金型の安全範囲内であることを確認してください。"
            )
        show_coded = st.checkbox("コード化水準を表示", value=False)
        coded_columns = [f"コード_{factor}" for factor in input_columns]
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
            f"品質測定後、CSVの{'・'.join(output_columns)}を入力し、"
            "先頭の実行順・標準順・点種別・コード列を残したまま本アプリへ読み込めます。"
        )

if selected_tab == tab_labels[1]:
    source = "アップロードデータ" if uploaded else "デモデータ"
    st.subheader(f"{source}（{len(data)}点）")
    st.dataframe(data, width="stretch", hide_index=True)
    st.caption("モデルは入力データの最小～最大範囲内で使用してください。範囲外への外挿は行いません。")

if selected_tab == tab_labels[2]:
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
    selected_coefficient = st.selectbox("係数を表示", output_columns)
    st.dataframe(
        coefficient_table(models[selected_coefficient]),
        hide_index=True,
        width="content",
    )
    st.caption("係数は平均0・標準偏差1に標準化した因子に対する値です。絶対値が大きい項ほど影響が強い目安になります。")

if selected_tab == tab_labels[3]:
    col_a, col_b, col_c = st.columns(3)
    selected_response = col_a.selectbox(
        "表示する出力応答", output_columns, key="surface_response"
    )
    x_name = col_b.selectbox("横軸", input_columns, index=0, key="surface_x")
    y_options = [name for name in input_columns if name != x_name]
    y_name = col_c.selectbox("縦軸", y_options, index=0, key="surface_y")
    fixed_names = [name for name in input_columns if name not in (x_name, y_name)]
    fixed_values: dict[str, float] = {}
    if fixed_names:
        st.markdown("**固定する入力因子**")
        fixed_columns = st.columns(min(3, len(fixed_names)))
        for index, factor in enumerate(fixed_names):
            fixed_values[factor] = fixed_columns[index % len(fixed_columns)].slider(
                factor,
                float(data[factor].min()),
                float(data[factor].max()),
                float(data[factor].median()),
                key=f"surface_fixed_{factor}",
            )
    xx, yy, grid = make_slice_grid(
        data, x_name, y_name, fixed_values, input_columns=input_columns
    )
    zz = models[selected_response].predict(grid).reshape(xx.shape)
    figure = go.Figure(
        go.Contour(
            x=xx[0], y=yy[:, 0], z=zz, colorscale="Viridis",
            contours_coloring="heatmap",
            colorbar=dict(title=selected_response),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data[x_name], y=data[y_name], mode="markers",
            marker=dict(color="white", line=dict(color="black", width=1), size=7),
            name="実験点", hovertemplate="実験点<extra></extra>",
        )
    )
    figure.update_layout(
        width=700,
        height=700,
        autosize=False,
        margin=dict(l=80, r=80, t=80, b=80),
        xaxis_title=x_name,
        yaxis_title=y_name,
        title=f"{selected_response}の応答曲面",
    )
    st.plotly_chart(
        figure,
        width="content",
        config={"responsive": False},
    )
    fixed_text = "、".join(
        f"{name}={value:g}" for name, value in fixed_values.items()
    ) or "固定因子なし"
    st.caption(f"{fixed_text}。白丸は全実験点の{x_name}–{y_name}投影です。")

if selected_tab == tab_labels[4]:
    col_a, col_b = st.columns(2)
    wx = col_a.selectbox("横軸", input_columns, index=0, key="window_x")
    wy_options = [name for name in input_columns if name != wx]
    wy = col_b.selectbox("縦軸", wy_options, index=0, key="window_y")
    window_fixed_names = [name for name in input_columns if name not in (wx, wy)]
    window_fixed_values: dict[str, float] = {}
    if window_fixed_names:
        st.markdown("**固定する入力因子**")
        fixed_columns = st.columns(min(3, len(window_fixed_names)))
        for index, factor in enumerate(window_fixed_names):
            window_fixed_values[factor] = fixed_columns[index % len(fixed_columns)].slider(
                factor,
                float(data[factor].min()),
                float(data[factor].max()),
                float(data[factor].median()),
                key=f"window_fixed_{factor}",
            )
    wxx, wyy, wgrid = make_slice_grid(
        data, wx, wy, window_fixed_values,
        input_columns=input_columns, resolution=100,
    )
    predictions = {response: models[response].predict(wgrid) for response in output_columns}
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
        title="全出力応答の同時規格適合領域",
    )
    st.plotly_chart(
        window_figure,
        width="content",
        config={"responsive": False},
    )
    st.metric("この断面の規格内面積率（グリッド近似）", f"{mask.mean():.1%}")

    search_resolution = max(
        4, min(24, int(20000 ** (1 / len(input_columns))))
    )
    feasible_conditions, feasible_quality = suggest_optimum(
        data, models, limits,
        input_columns=input_columns,
        output_columns=output_columns,
        resolution=search_resolution,
    )
    total_grid_points = search_resolution ** len(input_columns)
    st.metric(
        f"{len(input_columns)}因子空間の規格内体積率（グリッド近似）",
        f"{len(feasible_conditions) / total_grid_points:.1%}",
    )
    if len(feasible_conditions):
        centre = pd.Series({name: data[name].median() for name in input_columns})
        span = pd.Series({name: data[name].max() - data[name].min() for name in input_columns}).replace(0, 1)
        distance = (((feasible_conditions - centre) / span) ** 2).sum(axis=1)
        best_index = distance.idxmin()
        recommendation = pd.concat([feasible_conditions.loc[best_index], feasible_quality.loc[best_index]])
        st.subheader("規格内の推奨条件")
        st.dataframe(recommendation.rename("推定値").to_frame(), width="stretch")
        st.caption("探索範囲の中央に最も近い規格内点です。ロバスト性の最終確認には確認実験を行ってください。")
    else:
        st.warning("現在の品質規格を同時に満たす条件は探索グリッド内で見つかりませんでした。")

if selected_tab == tab_labels[5]:
    st.subheader("任意条件での品質予測")
    condition = {}
    columns = st.columns(min(3, len(input_columns)))
    for factor_index, factor in enumerate(input_columns):
        column = columns[factor_index % len(columns)]
        condition[factor] = column.slider(
            factor, float(data[factor].min()), float(data[factor].max()),
            float(data[factor].median()), key=f"predict_{factor_index}"
        )
    query = pd.DataFrame([condition])
    predicted_values = {response: float(model.predict(query)[0]) for response, model in models.items()}
    prediction_rows = []
    for response in output_columns:
        lower, upper = limits[response]
        ok = (lower is None or predicted_values[response] >= lower) and (upper is None or predicted_values[response] <= upper)
        prediction_rows.append(
            {
                "出力応答": response,
                "予測値": predicted_values[response],
                "判定": "✔ 規格内" if ok else "✖ 規格外",
            }
        )
    st.dataframe(pd.DataFrame(prediction_rows), width="content", hide_index=True)
    st.caption("予測値は統計モデルによる推定です。量産条件の決定前に、推奨点と境界付近で確認実験を実施してください。")
