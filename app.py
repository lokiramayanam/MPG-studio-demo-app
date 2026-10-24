"""
MPG Studio - Automobile Fuel Efficiency Predictor
-------------------------------------------------
End-to-end ML application:
  1. Data extraction      -> load auto-mpg.csv (or an uploaded CSV)
  2. Data cleaning        -> '?' -> NaN, type fixing, de-duplication
  3. Preprocessing        -> missing-value imputation, feature scaling,
                             categorical encoding (all inside one sklearn Pipeline)
  4. Model selection      -> 5 algorithms compared with cross-validation,
                             best one is refit and used for deployment
  5. Dashboard            -> predict, explore, compare models, inspect pipeline

Run:  streamlit run app.py
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "auto-mpg.csv"
MODEL_PATH = APP_DIR / "mpg_model.joblib"
RANDOM_STATE = 42

TARGET = "mpg"
NUM_FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CAT_FEATURES = ["origin"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}

# Palette (instrument-cluster inspired)
INK, PANEL, AMBER, TEAL, ROSE, MUTED = "#0E1621", "#16222F", "#F2B134", "#5CC8C0", "#E5707E", "#8DA1B5"
ORIGIN_COLORS = {"USA": AMBER, "Europe": TEAL, "Japan": ROSE}


# --------------------------------------------------------------------------- #
# 1. DATA EXTRACTION + 2. CLEANING
# --------------------------------------------------------------------------- #
def extract_data(source) -> pd.DataFrame:
    """Read the raw CSV. `source` can be a path or an uploaded file object."""
    return pd.read_csv(source)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names/types and convert placeholder values to NaN."""
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Auto-MPG stores unknown horsepower as '?'  ->  real missing values
    df = df.replace("?", np.nan)
    for col in NUM_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["origin"] = pd.to_numeric(df["origin"], errors="coerce").map(ORIGIN_MAP)

    if "car_name" in df.columns:
        df["car_name"] = df["car_name"].astype(str).str.strip().str.title()
        df["brand"] = df["car_name"].str.split().str[0]
    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)  # target must exist
    return df


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        return clean_data(extract_data(io.BytesIO(uploaded_bytes)))
    return clean_data(extract_data(DATA_PATH))


# --------------------------------------------------------------------------- #
# 3. PREPROCESSING PIPELINE  (imputation + scaling + encoding)
# --------------------------------------------------------------------------- #
def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]  # missing values
    if scale:
        numeric_steps.append(("scale", StandardScaler()))  # feature scaling
    numeric = Pipeline(numeric_steps)

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),  # missing values
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # encoding
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)],
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, Pipeline]:
    """Five algorithms, each wrapped with the same preprocessing."""
    algos = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RANDOM_STATE
        ),
        "Support Vector Regression": SVR(C=20, epsilon=0.5, gamma="scale"),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("model", algo)]) for name, algo in algos.items()}


# --------------------------------------------------------------------------- #
# 4. TRAINING + MODEL SELECTION
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training and comparing models...")
def train_and_select(df: pd.DataFrame) -> dict:
    X, y = df[ALL_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, pipe in candidate_models().items():
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append(
            {
                "Model": name,
                "CV RMSE": cv_rmse,
                "Test RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "Test MAE": mean_absolute_error(y_te, pred),
                "Test R2": r2_score(y_te, pred),
            }
        )
        fitted[name] = pipe

    leaderboard = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    best_name = leaderboard.loc[0, "Model"]
    best_eval = fitted[best_name]

    # Which raw inputs matter most (permutation importance on held-out data)
    imp = permutation_importance(best_eval, X_te, y_te, n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1)
    importance = (
        pd.DataFrame({"Feature": ALL_FEATURES, "Importance": imp.importances_mean})
        .sort_values("Importance")
        .reset_index(drop=True)
    )

    # Deployment model: same recipe, refit on ALL data
    final_model = clone(best_eval).fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    return {
        "model": final_model,
        "best_name": best_name,
        "leaderboard": leaderboard,
        "importance": importance,
        "y_test": y_te.to_numpy(),
        "y_pred": best_eval.predict(X_te),
        "cv_rmse": float(leaderboard.loc[0, "CV RMSE"]),
        "n_train": len(df),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def efficiency_band(mpg: float) -> tuple[str, str]:
    if mpg >= 32:
        return "Excellent", TEAL
    if mpg >= 24:
        return "Good", "#9BD16B"
    if mpg >= 18:
        return "Average", AMBER
    return "Thirsty", ROSE


def gauge(value: float, avg: float) -> go.Figure:
    label, color = efficiency_band(value)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " mpg", "font": {"size": 54, "family": "Barlow Condensed", "color": "#E8EEF4"}},
            gauge={
                "axis": {"range": [0, 50], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": PANEL,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 18], "color": "#2A2230"},
                    {"range": [18, 24], "color": "#2B2A2A"},
                    {"range": [24, 32], "color": "#223029"},
                    {"range": [32, 50], "color": "#1D3236"},
                ],
                "threshold": {"line": {"color": "#E8EEF4", "width": 3}, "thickness": 0.8, "value": avg},
            },
        )
    )
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EEF4"})
    return fig


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EEF4", "family": "Barlow, sans-serif"},
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#223142", zeroline=False)
    fig.update_yaxes(gridcolor="#223142", zeroline=False)
    return fig


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Barlow', sans-serif; }}
h1, h2, h3 {{ font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .3px; }}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
.hero {{ border-left: 4px solid {AMBER}; padding: .2rem 0 .2rem 1rem; margin-bottom: 1rem; }}
.hero h1 {{ font-size: 2.6rem; margin: 0; line-height: 1.05; }}
.hero p {{ color: {MUTED}; margin: .3rem 0 0 0; font-size: 1.02rem; }}
.panel {{ background: {PANEL}; border: 1px solid #223142; border-radius: 10px; padding: 1rem 1.2rem; }}
.verdict {{ font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; }}
.small {{ color: {MUTED}; font-size: .92rem; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #223142; border-radius: 10px; padding: .7rem 1rem; }}
[data-testid="stMetricValue"] {{ font-family: 'Barlow Condensed', sans-serif; }}
button[data-baseweb="tab"] {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.15rem; }}
</style>
"""


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def tab_predict(df: pd.DataFrame, art: dict) -> None:
    model = art["model"]
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Vehicle specification")
        c1, c2 = st.columns(2)
        cylinders = c1.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        origin = c2.selectbox("Origin", list(ORIGIN_MAP.values()))
        year = st.slider("Model year", 1970, 1982, 1976)
        displacement = st.slider("Displacement (cu in)", int(df.displacement.min()), int(df.displacement.max()), 150)
        weight = st.slider("Weight (lbs)", int(df.weight.min()), int(df.weight.max()), 2800, step=10)
        acceleration = st.slider("Acceleration (0-60 mph, sec)", 8.0, 25.0, 15.0, step=0.1)
        hp_unknown = st.checkbox("I don't know the horsepower (let the pipeline impute it)")
        if hp_unknown:
            horsepower = np.nan
        else:
            horsepower = st.slider("Horsepower", int(df.horsepower.min()), int(df.horsepower.max()), 95)

        with st.expander("Running-cost assumptions"):
            k1, k2, k3 = st.columns(3)
            distance = k1.number_input("Distance / year (km)", 1000, 100000, 12000, step=1000)
            price = k2.number_input("Fuel price / litre", 0.1, 500.0, 100.0)
            cur = k3.selectbox("Currency", ["Rs.", "$", "EUR", "GBP"])

    row = pd.DataFrame(
        [
            {
                "cylinders": cylinders,
                "displacement": displacement,
                "horsepower": horsepower,
                "weight": weight,
                "acceleration": acceleration,
                "model_year": year - 1900,
                "origin": origin,
            }
        ]
    )
    mpg = float(model.predict(row)[0])
    label, color = efficiency_band(mpg)
    kmpl, l100 = mpg * 0.425144, 235.215 / mpg
    annual_cost = distance / kmpl * price
    pct = float((df[TARGET] < mpg).mean() * 100)

    with right:
        st.subheader("Predicted fuel efficiency")
        st.plotly_chart(gauge(mpg, df[TARGET].mean()), width="stretch")
        st.markdown(
            f"<div class='panel'><span class='verdict' style='color:{color}'>{label}</span>"
            f"<span class='small'> &nbsp;|&nbsp; better than {pct:.0f}% of cars in the dataset. "
            f"Typical model error is about {art['cv_rmse']:.1f} mpg, so expect "
            f"{max(mpg - art['cv_rmse'], 0):.1f} to {mpg + art['cv_rmse']:.1f} mpg. "
            f"The white marker on the gauge is the dataset average ({df[TARGET].mean():.1f} mpg).</span></div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Kilometres per litre", f"{kmpl:.1f}")
        m2.metric("Litres per 100 km", f"{l100:.1f}")
        m3.metric("Fuel cost / year", f"{cur} {annual_cost:,.0f}")
        if hp_unknown:
            st.caption(f"Horsepower was missing, so the pipeline filled in the training median ({df.horsepower.median():.0f} hp).")

    # Nearest real cars in the training data (distance in the pipeline's scaled space)
    st.subheader("Closest cars in the dataset")
    prep = model.named_steps["prep"]
    space = prep.transform(df[ALL_FEATURES])
    dist = np.linalg.norm(space - prep.transform(row), axis=1)
    near = df.assign(distance=dist).nsmallest(5, "distance")
    near = near[["car_name", "model_year", "origin", "cylinders", "weight", "horsepower", "mpg"]].copy()
    near["model_year"] = near["model_year"].astype(int) + 1900
    near.columns = ["Car", "Year", "Origin", "Cyl", "Weight", "HP", "Actual MPG"]
    st.dataframe(near, hide_index=True, width="stretch")
    if year < 1970 or year > 1982:
        st.warning("Model year is outside the training range.")


def tab_explore(df: pd.DataFrame) -> None:
    st.subheader("What drives fuel economy?")
    f1, f2 = st.columns([2, 1])
    origins = f1.multiselect("Origin", list(ORIGIN_MAP.values()), default=list(ORIGIN_MAP.values()))
    yr = f2.slider("Model years", 1970, 1982, (1970, 1982))
    d = df[df.origin.isin(origins) & df.model_year.between(yr[0] - 1900, yr[1] - 1900)].copy()
    d["year"] = d.model_year + 1900

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cars", f"{len(d)}")
    k2.metric("Average MPG", f"{d.mpg.mean():.1f}")
    k3.metric("Best MPG", f"{d.mpg.max():.1f}")
    k4.metric("Average weight", f"{d.weight.mean():,.0f} lb")

    a, b = st.columns(2)
    fig = px.scatter(
        d, x="weight", y="mpg", color="origin", color_discrete_map=ORIGIN_COLORS,
        hover_name="car_name", hover_data={"year": True, "horsepower": True}, opacity=.8,
        title="Heavier cars drink more",
    )
    a.plotly_chart(style(fig), width="stretch")

    trend = d.groupby(["year", "origin"], as_index=False).mpg.mean()
    fig = px.line(trend, x="year", y="mpg", color="origin", color_discrete_map=ORIGIN_COLORS, markers=True,
                  title="Average MPG by model year")
    b.plotly_chart(style(fig), width="stretch")

    c, e = st.columns(2)
    fig = px.box(d, x="cylinders", y="mpg", color_discrete_sequence=[AMBER], title="MPG by cylinder count")
    c.plotly_chart(style(fig), width="stretch")

    corr = d[NUM_FEATURES + [TARGET]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=["#E5707E", "#16222F", "#5CC8C0"],
                    zmin=-1, zmax=1, title="Correlation matrix")
    e.plotly_chart(style(fig), width="stretch")

    with st.expander("Browse the data"):
        st.dataframe(d.drop(columns=["year"]), hide_index=True, width="stretch")


def tab_models(art: dict) -> None:
    lb = art["leaderboard"]
    st.subheader("Model comparison")
    st.markdown(
        f"<div class='panel'>Five algorithms were compared with 5-fold cross-validation on the training split. "
        f"<b style='color:{AMBER}'>{art['best_name']}</b> had the lowest cross-validated error, so it was refit on "
        f"all {art['n_train']} cars and is the model serving the Predict tab.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    best = lb.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Test R2", f"{best['Test R2']:.3f}")
    m2.metric("Test RMSE", f"{best['Test RMSE']:.2f} mpg")
    m3.metric("Test MAE", f"{best['Test MAE']:.2f} mpg")

    st.dataframe(
        lb.style.format({"CV RMSE": "{:.3f}", "Test RMSE": "{:.3f}", "Test MAE": "{:.3f}", "Test R2": "{:.3f}"})
        .highlight_min(subset=["CV RMSE", "Test RMSE", "Test MAE"], color="#3a3320")
        .highlight_max(subset=["Test R2"], color="#3a3320"),
        hide_index=True, width="stretch",
    )

    a, b = st.columns(2)
    fig = px.bar(lb, x="CV RMSE", y="Model", orientation="h", color_discrete_sequence=[TEAL],
                 title="Cross-validated RMSE (lower is better)")
    fig.update_yaxes(categoryorder="total descending")
    a.plotly_chart(style(fig), width="stretch")

    yt, yp = art["y_test"], art["y_pred"]
    fig = go.Figure()
    fig.add_scatter(x=yt, y=yp, mode="markers", marker=dict(color=AMBER, opacity=.8), name="Test cars")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dash"), name="Perfect")
    fig.update_layout(title="Actual vs predicted (held-out cars)", xaxis_title="Actual MPG", yaxis_title="Predicted MPG")
    b.plotly_chart(style(fig), width="stretch")

    imp = art["importance"]
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[AMBER],
                 title="Permutation importance (drop in R2 when a feature is shuffled)")
    st.plotly_chart(style(fig, 340), width="stretch")


def tab_pipeline(raw_missing: pd.Series, df: pd.DataFrame) -> None:
    st.subheader("Pipeline and data quality")
    st.markdown(
        """
| Stage | What happens |
|---|---|
| Extraction | `auto-mpg.csv` is read with pandas (or the CSV you upload in the sidebar). |
| Cleaning | Column names are normalised, `?` placeholders become missing values, text numbers are converted, duplicates and rows without MPG are removed. |
| Missing values | Numeric columns use the **median**; the categorical column uses the **most frequent** value. |
| Scaling | `StandardScaler` puts numeric features on a common scale. |
| Encoding | `OneHotEncoder` turns origin (USA, Europe, Japan) into indicator columns and ignores unseen categories. |
| Model | Five regressors share this preprocessing; the best by cross-validated RMSE is deployed. |
        """
    )
    st.markdown("**Missing values found in the raw file (before imputation)**")
    miss = raw_missing[raw_missing > 0]
    if miss.empty:
        st.success("No missing values in the uploaded data.")
    else:
        st.dataframe(miss.rename("Missing").reset_index().rename(columns={"index": "Column"}), hide_index=True)
    st.markdown("**Summary of the cleaned numeric data**")
    st.dataframe(df[NUM_FEATURES + [TARGET]].describe().T.round(2), width="stretch")
    if MODEL_PATH.exists():
        st.download_button("Download trained pipeline (.joblib)", MODEL_PATH.read_bytes(), "mpg_model.joblib")


# --------------------------------------------------------------------------- #
# App entry
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="MPG Studio", page_icon=":material/speed:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Data source")
        up = st.file_uploader("Upload an Auto-MPG style CSV", type="csv", help="Leave empty to use the bundled dataset.")
        st.caption("Needs columns: mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin (1/2/3).")
        st.divider()
        st.caption("Built with scikit-learn and Streamlit.")

    try:
        uploaded = up.getvalue() if up else None
        df = load_data(uploaded)
        raw = extract_data(io.BytesIO(uploaded)) if uploaded else extract_data(DATA_PATH)
        raw_missing = raw.replace("?", np.nan).isna().sum()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the data: {exc}")
        st.stop()

    art = train_and_select(df)

    st.markdown(
        f"<div class='hero'><h1>MPG Studio</h1>"
        f"<p>Predict a car's fuel efficiency from its specification. Serving model: "
        f"<b>{art['best_name']}</b>, trained on {art['n_train']} vehicles.</p></div>",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["Predict", "Explore data", "Model lab", "Pipeline"])
    with t1:
        tab_predict(df, art)
    with t2:
        tab_explore(df)
    with t3:
        tab_models(art)
    with t4:
        tab_pipeline(raw_missing, df)


if __name__ == "__main__":
    main()
