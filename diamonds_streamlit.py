"""Streamlit dashboard that loads pre-trained XGBoost models and clean datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from xgboost import XGBRegressor

# Paths
MODEL_DIR = Path(__file__).parent / "models"


@st.cache_data(show_spinner="Loading data...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_lab = pd.read_csv(Path(__file__).parent / "df_lab.csv")
    df_nat = pd.read_csv(Path(__file__).parent / "df_natural.csv")
    # Combine into a single view for charts/UI convenience.
    df_full = pd.concat(
        [
            df_lab.assign(diamond_type="lab"),
            df_nat.assign(diamond_type="natural"),
        ],
        ignore_index=True,
    )
    return df_full, df_lab, df_nat


def load_category_mapping(mapping_path: Path, df_lab: pd.DataFrame, df_nat: pd.DataFrame, feature_order: List[str]) -> Dict[str, Dict[str, int]]:
    """Load saved category mappings; if missing/empty, derive from provided data."""
    mapping: Dict[str, Dict[str, int]] = {}
    if mapping_path.exists():
        loaded = joblib.load(mapping_path)
        if isinstance(loaded, dict):
            mapping = loaded
    cat_cols = [col for col in feature_order if col not in ["carat", "qualityScore"]]
    if not mapping:
        combined = pd.concat([df_lab[cat_cols], df_nat[cat_cols]], ignore_index=True)
        for col in cat_cols:
            categories = sorted(combined[col].dropna().astype(str).unique())
            mapping[col] = {val: idx for idx, val in enumerate(categories)}
    return mapping


def predict_with_model(model: XGBRegressor, mapping: Dict[str, Dict[str, int]], feature_order: List[str], input_df: pd.DataFrame) -> np.ndarray:
    cat_cols = [col for col in feature_order if col not in ["carat", "qualityScore"]]
    num_cols = [col for col in feature_order if col in ["carat", "qualityScore"]]

    rows = []
    for _, row in input_df.iterrows():
        encoded_row = []
        for feature in feature_order:
            if feature in cat_cols:
                value = str(row.get(feature, ""))
                encoded_row.append(mapping.get(feature, {}).get(value, -1))
            else:
                encoded_row.append(float(pd.to_numeric(row.get(feature, np.nan), errors="coerce")))
        rows.append(encoded_row)

    return model.predict(np.asarray(rows, dtype=np.float32))


@st.cache_resource(show_spinner="Loading XGBoost models...")
def build_models(df_lab: pd.DataFrame, df_nat: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    feature_order = joblib.load(MODEL_DIR / "features.pkl")
    category_mapping = load_category_mapping(MODEL_DIR / "category_mapping.pkl", df_lab, df_nat, feature_order)

    lab_model = XGBRegressor()
    lab_model.load_model(MODEL_DIR / "lab_model.json")

    nat_model = XGBRegressor()
    nat_model.load_model(MODEL_DIR / "natural_model.json")

    return {
        "lab": {
            "model": lab_model,
            "label": "Lab-Grown Model",
            "feature_order": feature_order,
            "category_mapping": category_mapping,
        },
        "natural": {
            "model": nat_model,
            "label": "Natural Model",
            "feature_order": feature_order,
            "category_mapping": category_mapping,
        },
    }


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def build_overview_tab(df_full: pd.DataFrame, df_lab: pd.DataFrame, df_nat: pd.DataFrame):
    st.subheader("Dataset Snapshot")
    price_col = "price_2025"

    col_full, col_lab, col_nat = st.columns(3)
    col_full.metric("All diamonds", f"{len(df_full):,}")
    col_lab.metric("Lab-grown", f"{len(df_lab):,}")
    col_nat.metric("Natural", f"{len(df_nat):,}")

    plot_df = df_full.dropna(subset=["carat", price_col]).copy()
    plot_df["price_per_carat"] = plot_df[price_col] / plot_df["carat"]

    st.markdown("#### Carat vs. Price")
    st.caption("Brush to zoom, hover for details.")
    price_chart = (
        alt.Chart(plot_df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("carat:Q", title="Carat"),
            y=alt.Y(f"{price_col}:Q", title="Price (2025 USD)"),
            color=alt.Color("diamond_type:N", legend=alt.Legend(title="Diamond type")),
            tooltip=[
                alt.Tooltip("retailer:N"),
                alt.Tooltip("diamond_type:N", title="Type"),
                alt.Tooltip("shape:N"),
                alt.Tooltip("cut:N"),
                alt.Tooltip("color:N"),
                alt.Tooltip("clarity:N"),
                alt.Tooltip("carat:Q", format=".2f"),
                alt.Tooltip(f"{price_col}:Q", title="Price (2025)", format="$.0f"),
            ],
        )
        .properties(height=380)
        .interactive()
    )
    st.altair_chart(price_chart, use_container_width=True)

    st.markdown("#### Distribution explorer")
    dist_option = st.selectbox(
        "Choose a distribution view",
        [
            "Cut",
            "Shapes",
            "Color Grades",
            "Clarity Grades",
        ],
    )
    field_map = {
        "Cut": "cut:N",
        "Shapes": "shape:N",
        "Color Grades": "color:N",
        "Clarity Grades": "clarity:N",
    }
    dist_chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(field_map[dist_option], sort="-y", title=dist_option.split()[0]),
            y=alt.Y("count()", title="Count", stack=None),
            color=alt.Color("diamond_type:N", title="Diamond type"),
            xOffset=alt.XOffset("diamond_type:N"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=320)
    )
    st.altair_chart(dist_chart, use_container_width=True)

    st.markdown("#### Price per carat by quality features")
    quality_options = {
        "Cut": ("cut:N", "Cut"),
        "Color": ("color:N", "Color"),
        "Clarity": ("clarity:N", "Clarity"),
    }
    quality_choice = st.selectbox("Choose quality feature", list(quality_options.keys()))
    quality_field, quality_label = quality_options[quality_choice]

    quality_box = (
        alt.Chart(plot_df)
        .mark_boxplot(outliers=True)
        .encode(
            x=alt.X(quality_field, title=quality_label),
            y=alt.Y("price_per_carat:Q", title="Price per carat (2025 USD)"),
            color=alt.Color("diamond_type:N", title="Diamond type"),
            xOffset=alt.XOffset("diamond_type:N"),
            tooltip=[
                alt.Tooltip(quality_field, title=quality_label),
                alt.Tooltip("diamond_type:N", title="Type"),
                alt.Tooltip("price_per_carat:Q", title="Price per carat", format="$.0f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(quality_box, use_container_width=True)


def build_predictor_tab(models: Dict[str, Dict[str, object]], df_full: pd.DataFrame):
    st.subheader("Price Estimator")
    st.write("Configure the characteristics to estimate lab-grown or natural diamond pricing.")

    feature_order = next(iter(models.values()))["feature_order"]
    cat_cols = [col for col in feature_order if col not in ["carat", "qualityScore"]]

    cat_options = {col: sorted(df_full[col].dropna().astype(str).unique()) for col in cat_cols}

    carat_series = pd.to_numeric(df_full["carat"], errors="coerce").dropna()
    quality_series = pd.to_numeric(df_full["qualityScore"], errors="coerce").dropna()
    carat_min, carat_max = float(carat_series.min()), float(carat_series.max())
    quality_min, quality_max = float(quality_series.min()), float(quality_series.max())

    with st.form("predictor"):
        diamond_type = st.selectbox(
            "Diamond type",
            options=[("natural", "Natural diamond"), ("lab", "Lab-grown diamond")],
            format_func=lambda x: x[1],
        )[0]

        col_left, col_right = st.columns(2)
        with col_left:
            carat = st.slider(
                "Carat",
                min_value=round(carat_min, 2),
                max_value=round(carat_max, 2),
                value=float(np.clip(carat_series.median(), carat_min, carat_max)),
                step=0.01,
            )
            cut = st.selectbox("Cut", options=cat_options["cut"])
            color = st.selectbox("Color", options=cat_options["color"])
            clarity = st.selectbox("Clarity", options=cat_options["clarity"])
        with col_right:
            quality = st.slider(
                "Quality score",
                min_value=round(quality_min, 1),
                max_value=round(quality_max, 1),
                value=float(np.clip(quality_series.median(), quality_min, quality_max)),
                step=0.1,
            )
            shape = st.selectbox("Shape", options=cat_options["shape"])
            report = st.selectbox("Report", options=cat_options["report"])
            polish = st.selectbox("Polish", options=cat_options["polish"])
            symmetry = st.selectbox("Symmetry", options=cat_options["symmetry"])
            fluorescence = st.selectbox("Fluorescence", options=cat_options["fluorescence"])

        submitted = st.form_submit_button("Estimate price")

    if not submitted:
        return

    input_row = pd.DataFrame(
        [
            {
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "shape": shape,
                "report": report,
                "qualityScore": quality,
                "polish": polish,
                "symmetry": symmetry,
                "fluorescence": fluorescence,
            }
        ]
    )

    model_key = "natural" if diamond_type == "natural" else "lab"
    bundle = models[model_key]
    prediction = predict_with_model(
        bundle["model"],
        bundle["category_mapping"],
        bundle["feature_order"],
        input_row,
    )[0]

    col_a, col_b = st.columns(2)
    col_a.metric("Estimated Price", format_currency(prediction))


def build_model_tab(models: Dict[str, Dict[str, object]]):
    st.subheader("Models Loaded")
    for key, bundle in models.items():
        st.markdown(f"#### {bundle['label']}")
        st.caption(f"Loaded from {MODEL_DIR / f'{key}_model.json'}")

        importances = pd.DataFrame(
            {
                "feature": bundle["feature_order"],
                "importance": bundle["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        chart = (
            alt.Chart(importances)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Feature importance"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("importance:Q", title="Importance", format=".4f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="Diamond Pricing Explorer", layout="wide")
    st.title("Diamond Pricing Explorer")
    st.caption(
        "Streamlit interface using pre-trained XGBoost models for lab-grown and natural diamond pricing."
    )

    df_full, df_lab, df_nat = load_data()
    models = build_models(df_lab, df_nat)

    tab_predictor, tab_overview, tab_models = st.tabs(
        ["Price Estimator", "Overview", "Models"]
    )

    with tab_predictor:
        build_predictor_tab(models, df_full)

    with tab_overview:
        build_overview_tab(df_full, df_lab, df_nat)

    with tab_models:
        build_model_tab(models)


if __name__ == "__main__":
    main()
