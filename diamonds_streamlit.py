"""Interactive Streamlit app for diamond price exploration and prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import altair as alt


FEATURE_ORDER: List[str] = [
    "carat",
    "cut",
    "color",
    "clarity",
    "shape",
    "report",
    "qualityScore",
    "polish",
    "symmetry",
    "fluorescence",
]

CAT_FEATURES: List[str] = [
    "cut",
    "color",
    "clarity",
    "shape",
    "report",
    "polish",
    "symmetry",
    "fluorescence",
]

NUM_FEATURES: List[str] = ["carat", "qualityScore"]

TARGET_COL = "price_2025"

FAIR_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
}

LAB_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
}

NAT_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
}


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add any missing columns with NaN so downstream selection never fails."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
    return df


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Prepare feature matrix by filling missing values and enforcing order."""
    processed = _ensure_columns(df.copy(), FEATURE_ORDER)

    # Fill categorical features with a catch-all bucket to avoid unseen values.
    for col in CAT_FEATURES:
        processed[col] = processed[col].fillna("Unknown").astype(str)

    # Fill numeric features with median values for stability.
    num_fill: Dict[str, float] = {}
    for col in NUM_FEATURES:
        processed[col] = pd.to_numeric(processed[col], errors="coerce")
        median_val = processed[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        num_fill[col] = float(median_val)
        processed[col] = processed[col].fillna(num_fill[col])

    return processed[FEATURE_ORDER], num_fill


def assemble_feature_matrix(
    num_data: np.ndarray,
    cat_data: np.ndarray,
    feature_order: List[str],
    num_cols: List[str],
    cat_cols: List[str],
) -> np.ndarray:
    """Combine numeric and categorical encoded arrays into model input order."""
    num_index = {col: idx for idx, col in enumerate(num_cols)}
    cat_index = {col: idx for idx, col in enumerate(cat_cols)}

    stacked = []
    for feature in feature_order:
        if feature in num_index:
            stacked.append(num_data[:, [num_index[feature]]])
        elif feature in cat_index:
            stacked.append(cat_data[:, [cat_index[feature]]])
        else:
            raise KeyError(f"Unexpected feature column: {feature}")

    return np.hstack(stacked).astype(np.float32)


def train_regression_model(
    df: pd.DataFrame,
    params: Dict[str, float],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train an XGBoost regressor with ordinal encoding and return artifacts."""
    feature_df, num_fill = preprocess_features(df[FEATURE_ORDER])
    target = pd.to_numeric(df[TARGET_COL], errors="coerce")

    mask = target.notna()
    feature_df = feature_df.loc[mask].reset_index(drop=True)
    target = target.loc[mask].astype(np.float32).reset_index(drop=True)

    cat_cols = [col for col in FEATURE_ORDER if col in CAT_FEATURES]
    num_cols = [col for col in FEATURE_ORDER if col in NUM_FEATURES]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )
    cat_encoded = encoder.fit_transform(feature_df[cat_cols])
    num_data = feature_df[num_cols].to_numpy(dtype=np.float32)
    features = assemble_feature_matrix(num_data, cat_encoded, FEATURE_ORDER, num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target.values,
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        n_jobs=-1,
        eval_metric="rmse",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }

    return {
        "model": model,
        "encoder": encoder,
        "feature_order": FEATURE_ORDER,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "num_fill": num_fill,
        "cat_fill": "Unknown",
        "metrics": metrics,
        "params": params,
    }


def predict_dataframe(bundle: Dict[str, object], df: pd.DataFrame) -> np.ndarray:
    """Generate predictions for the provided dataframe using stored artifacts."""
    feature_df = _ensure_columns(df.copy(), bundle["feature_order"])

    for col in bundle["cat_cols"]:
        feature_df[col] = feature_df[col].fillna(bundle["cat_fill"]).astype(str)

    for col in bundle["num_cols"]:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        fill_value = bundle["num_fill"].get(col, 0.0)
        feature_df[col] = feature_df[col].fillna(fill_value)

    feature_df = feature_df[bundle["feature_order"]]

    cat_encoded = bundle["encoder"].transform(feature_df[bundle["cat_cols"]])
    num_data = feature_df[bundle["num_cols"]].to_numpy(dtype=np.float32)
    features = assemble_feature_matrix(
        num_data,
        cat_encoded,
        bundle["feature_order"],
        bundle["num_cols"],
        bundle["cat_cols"],
    )
    return bundle["model"].predict(features)


@st.cache_data(show_spinner="Loading data...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load full, lab-grown, and natural diamond datasets."""
    base_path = Path(__file__).parent
    full = pd.read_csv(base_path / "df.csv")
    lab = pd.read_csv(base_path / "df_lab.csv")
    natural = pd.read_csv(base_path / "df_natural.csv")
    return full, lab, natural


@st.cache_resource(show_spinner="Training XGBoost models...")
def build_models(
    df_full: pd.DataFrame,
    df_lab: pd.DataFrame,
    df_nat: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """Train fair, lab-grown, and natural price models and enrich the dataset."""
    # Fair model operates on grouped combinations to mimic notebook workflow.
    fair_grouped = (
        df_full.drop(columns=["retailer"])
        .groupby(FEATURE_ORDER, as_index=False)[TARGET_COL]
        .mean()
    )
    fair_bundle = train_regression_model(fair_grouped, FAIR_MODEL_PARAMS)
    fair_bundle["label"] = "Fair Market Price"

    lab_bundle = train_regression_model(df_lab, LAB_MODEL_PARAMS)
    lab_bundle["label"] = "Lab-Grown Model"

    nat_bundle = train_regression_model(df_nat, NAT_MODEL_PARAMS)
    nat_bundle["label"] = "Natural Model"

    enriched = df_full.copy()
    enriched[TARGET_COL] = pd.to_numeric(enriched[TARGET_COL], errors="coerce")
    enriched["fair_price"] = predict_dataframe(fair_bundle, enriched)

    enriched["markup_vs_fair"] = enriched[TARGET_COL] - enriched["fair_price"]
    fair_mask = enriched["fair_price"] != 0
    enriched["markup_pct"] = np.nan
    enriched.loc[fair_mask, "markup_pct"] = (
        100 * enriched.loc[fair_mask, "markup_vs_fair"] / enriched.loc[fair_mask, "fair_price"]
    )

    return {"fair": fair_bundle, "lab": lab_bundle, "natural": nat_bundle}, enriched


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def build_overview_tab(
    df_full: pd.DataFrame,
    df_lab: pd.DataFrame,
    df_nat: pd.DataFrame,
    df_enriched: pd.DataFrame,
):
    st.subheader("Dataset Snapshot")

    col_full, col_lab, col_nat = st.columns(3)
    col_full.metric("All diamonds", f"{len(df_full):,}")
    col_lab.metric("Lab-grown", f"{len(df_lab):,}")
    col_nat.metric("Natural", f"{len(df_nat):,}")

    plot_df = df_enriched.dropna(subset=["carat", TARGET_COL]).copy()
    plot_df["markup_pct"] = plot_df["markup_pct"].fillna(0.0)
    plot_df["price_per_carat"] = plot_df[TARGET_COL] / plot_df["carat"]

    st.markdown("#### Carat vs. Price")
    st.caption("Brush to zoom, hover for details.")
    price_chart = (
        alt.Chart(plot_df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("carat:Q", title="Carat"),
            y=alt.Y(f"{TARGET_COL}:Q", title="Price (2025 USD)"),
            color=alt.Color("diamond_type:N", legend=alt.Legend(title="Diamond type")),
            tooltip=[
                alt.Tooltip("retailer:N"),
                alt.Tooltip("diamond_type:N", title="Type"),
                alt.Tooltip("shape:N"),
                alt.Tooltip("cut:N"),
                alt.Tooltip("color:N"),
                alt.Tooltip("clarity:N"),
                alt.Tooltip("carat:Q", format=".2f"),
                alt.Tooltip(f"{TARGET_COL}:Q", title="Price (2025)", format="$.0f"),
                alt.Tooltip("markup_pct:Q", title="Markup %", format=".1f"),
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
    if dist_option == "Cut":
        dist_chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("cut:N", sort="-y", title="Cut"),
                y=alt.Y("count()", title="Count", stack=None),
                color=alt.Color("diamond_type:N", title="Diamond type"),
                xOffset=alt.XOffset("diamond_type:N"),
                tooltip=[alt.Tooltip("count()", title="Count")],
            )
            .properties(height=320)
        )
    elif dist_option == "Shapes":
        dist_chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("shape:N", sort="-y", title="Shape"),
                y=alt.Y("count()", title="Count", stack=None),
                color=alt.Color("diamond_type:N", title="Diamond type"),
                xOffset=alt.XOffset("diamond_type:N"),
                tooltip=[alt.Tooltip("count()", title="Count")],
            )
            .properties(height=320)
        )
    elif dist_option == "Color Grades":
        dist_chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("color:N", sort="-y", title="Color"),
                y=alt.Y("count()", title="Count", stack=None),
                color=alt.Color("diamond_type:N", title="Diamond type"),
                xOffset=alt.XOffset("diamond_type:N"),
                tooltip=[alt.Tooltip("count()", title="Count")],
            )
            .properties(height=320)
        )
    else:
        dist_chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("clarity:N", sort="-y", title="Clarity"),
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

    st.markdown("#### Average markup by retailer")
    markup_df = (
        df_enriched.dropna(subset=["markup_pct"])
        .groupby("retailer")["markup_pct"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .head(20)
    )
    markup_chart = (
        alt.Chart(markup_df)
        .mark_bar()
        .encode(
            x=alt.X("markup_pct:Q", title="Average markup vs fair (%)"),
            y=alt.Y("retailer:N", sort="-x", title="Retailer"),
            tooltip=[
                alt.Tooltip("retailer:N"),
                alt.Tooltip("markup_pct:Q", title="Markup %", format=".1f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(markup_chart, use_container_width=True)


def build_model_tab(models: Dict[str, Dict[str, object]]):
    st.subheader("Model Performance")
    descriptions = {
        "lab": "Trained solely on lab-grown diamond data to estimate prices specific to lab-grown stones.",
        "natural": "Trained on natural diamond listings to estimate prices for mined stones.",
    }
    for key in ["lab", "natural"]:
        bundle = models[key]
        metrics = bundle["metrics"]
        st.markdown(f"#### {bundle['label']}")
        st.caption(descriptions.get(key, ""))

        col_r2, col_mae, col_rmse = st.columns(3)
        col_r2.metric("R²", f"{metrics['r2']:.3f}")
        col_mae.metric("MAE", format_currency(metrics["mae"]))
        col_rmse.metric("RMSE", format_currency(metrics["rmse"]))
        st.caption(
            f"Trained on {metrics['train_size']:,} rows, evaluated on {metrics['test_size']:,} rows."
        )

        importance = pd.DataFrame(
            {
                "feature": bundle["feature_order"],
                "importance": bundle["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        st.markdown("Feature importance")
        st.caption(
            "Bars show the relative importance scores learned by the XGBoost model; longer bars indicate "
            "features that contributed more to predicting price."
        )
        st.bar_chart(
            importance.set_index("feature"),
            height=260,
        )


def build_predictor_tab(models: Dict[str, Dict[str, object]], df_full: pd.DataFrame):
    st.subheader("Price Estimator")
    st.write(
        "Configure the characteristics below to estimate lab-grown or natural diamond pricing "
        "and compare against the fair-market model."
    )

    cat_options = {
        col: sorted(df_full[col].dropna().astype(str).unique())
        for col in CAT_FEATURES
    }

    carat_series = pd.to_numeric(df_full["carat"], errors="coerce").dropna()
    quality_series = pd.to_numeric(df_full["qualityScore"], errors="coerce").dropna()
    carat_min, carat_max = float(carat_series.min()), float(carat_series.max())
    quality_min, quality_max = float(quality_series.min()), float(quality_series.max())

    with st.form("predictor"):
        diamond_type = st.selectbox(
            "Diamond type",
            options=[("natural", "Natural diamond"), ("lab", "Lab-grown diamond")],
            format_func=lambda x: x[1],
            help="Select whether you want a natural or lab-grown price estimate.",
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

    fair_prediction = predict_dataframe(models["fair"], input_row)[0]
    lab_prediction = predict_dataframe(models["lab"], input_row)[0]
    natural_prediction = predict_dataframe(models["natural"], input_row)[0]

    chosen_prediction = natural_prediction if diamond_type == "natural" else lab_prediction

    delta_value = chosen_prediction - fair_prediction
    delta_pct = (delta_value / fair_prediction * 100) if fair_prediction else np.nan

    col_a, col_b = st.columns(2)
    col_a.metric("Fair Price", format_currency(chosen_prediction))


def main():
    st.set_page_config(page_title="Diamond Pricing Explorer", layout="wide")
    st.title("Diamond Pricing Explorer")
    st.caption(
        "Streamlit interface derived from `DiamondsXGB.ipynb`, featuring tuned XGBoost "
        "models for fair-market, lab-grown, and natural diamond pricing."
    )

    df_full, df_lab, df_nat = load_data()
    models, df_enriched = build_models(df_full, df_lab, df_nat)

    tab_predictor, tab_overview, tab_models = st.tabs(
        ["Price Estimator", "Overview", "Models"]
    )

    with tab_predictor:
        build_predictor_tab(models, df_full)

    with tab_overview:
        build_overview_tab(df_full, df_lab, df_nat, df_enriched)

    with tab_models:
        build_model_tab(models)


if __name__ == "__main__":
    main()
