# Run from the project folder with: streamlit run dashboard.py
# Requires: pip install plotly (for the donut chart on the Overview tab)
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Telecom Churn Analytics", page_icon=":bar_chart:", layout="wide")
DATA_PATH = Path(__file__).resolve().parent / "Churn in Telecom.csv"
RANDOM_STATE = 42

PALETTE = {
    "coral": "#F97066",
    "amber": "#FDB022",
    "navy": "#2E3A59",
    "teal": "#17B6A7",
    "violet": "#7A5AF8",
    "ink": "#15181F",
    "muted": "#6B7280",
    "line": "#E8EAEE",
    "panel": "#FFFFFF",
    "bg": "#F6F7F9",
}

# ---------------------------------------------------------------------------
# Theme — light, card-based, colorful accent dots (CRM dashboard style)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: {PALETTE['bg']}; color: {PALETTE['ink']}; }}

h1 {{ font-weight: 800 !important; letter-spacing: -0.02em; color: {PALETTE['ink']} !important; font-size: 2.1rem !important; }}
h2, h3 {{ font-weight: 700 !important; color: {PALETTE['ink']} !important; }}
p, span, label {{ color: {PALETTE['ink']}; }}

section[data-testid="stSidebar"] {{
    background: {PALETTE['panel']};
    border-right: 1px solid {PALETTE['line']};
}}
section[data-testid="stSidebar"] * {{ color: {PALETTE['ink']} !important; }}
.brand-mark {{
    width: 40px; height: 40px; border-radius: 12px;
    background: {PALETTE['navy']};
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 1.1rem;
    margin-bottom: 14px;
}}
.nav-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 8px; margin-bottom: 2px;
    font-size: 0.92rem; font-weight: 500; color: {PALETTE['muted']};
}}
.nav-item.active {{ background: #EEF1FD; color: {PALETTE['navy']}; font-weight: 700; }}
.nav-dot {{ width: 7px; height: 7px; border-radius: 50%; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {PALETTE['line']};
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {PALETTE['muted']};
    font-weight: 600;
    font-size: 0.95rem;
    padding: 10px 4px;
    margin-right: 22px;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE['navy']} !important;
    border-bottom: 2px solid {PALETTE['navy']} !important;
}}

div[data-testid="stMetric"] {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}
div[data-testid="stMetricLabel"] {{
    color: {PALETTE['muted']} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}}
div[data-testid="stMetricValue"] {{
    color: {PALETTE['ink']} !important;
    font-weight: 800 !important;
    font-size: 1.9rem !important;
}}

.card {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    margin-bottom: 18px;
}}
.card-title {{ font-weight: 700; font-size: 1rem; color: {PALETTE['ink']}; margin-bottom: 14px; }}

.legend-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid {PALETTE['line']};
    font-size: 0.9rem;
}}
.legend-row:last-child {{ border-bottom: none; }}
.legend-label {{ display: flex; align-items: center; gap: 10px; color: {PALETTE['ink']}; font-weight: 500; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
.legend-value {{ color: {PALETTE['muted']}; font-variant-numeric: tabular-nums; }}

div[data-testid="stForm"] {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['line']};
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}

.stButton button, .stFormSubmitButton button {{
    background: {PALETTE['navy']} !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    padding: 0.5rem 1.3rem !important;
}}
.stButton button:hover, .stFormSubmitButton button:hover {{ filter: brightness(1.15); }}
div[data-testid="stDownloadButton"] button {{
    background: {PALETTE['panel']} !important;
    color: {PALETTE['navy']} !important;
    border: 1px solid {PALETTE['navy']} !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
}}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input {{
    background: {PALETTE['panel']} !important;
    border-color: {PALETTE['line']} !important;
    color: {PALETTE['ink']} !important;
    border-radius: 8px !important;
}}
label, .stCheckbox label p {{ color: {PALETTE['muted']} !important; font-size: 0.86rem !important; font-weight: 500 !important; }}

div[data-testid="stDataFrame"] {{ border: 1px solid {PALETTE['line']}; border-radius: 12px; overflow: hidden; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {PALETTE['muted']} !important; }}

.risk-badge {{
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-weight: 700; font-size: 0.85rem;
}}
.risk-low {{ background: #E6F7F4; color: {PALETTE['teal']}; }}
.risk-elevated {{ background: #FEF2E6; color: {PALETTE['amber']}; }}
.risk-high {{ background: #FDEEEC; color: {PALETTE['coral']}; }}
.risk-urgent {{ background: {PALETTE['coral']}; color: white; }}
</style>
""", unsafe_allow_html=True)


def fmt_currency(series: pd.Series) -> pd.Series:
    return series.map(lambda value: f"${value:,.2f}")


def fmt_percent(series: pd.Series) -> pd.Series:
    return series.map(lambda value: f"{value:.1%}")


def fmt_number(series: pd.Series, decimals: int = 2) -> pd.Series:
    return series.map(lambda value: f"{value:.{decimals}f}")


def risk_badge_html(tier: str) -> str:
    css_class = {"Low": "risk-low", "Elevated": "risk-elevated", "High": "risk-high", "Urgent": "risk-urgent"}[tier]
    return f'<span class="risk-badge {css_class}">{tier}</span>'


def donut_chart(labels, values, colors, center_label, center_value):
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.68,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
        textinfo="none", sort=False,
    )])
    fig.update_layout(
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=220,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b style='font-size:22px'>{center_value}</b><br><span style='font-size:12px;color=#6B7280'>{center_label}</span>",
            showarrow=False, font=dict(color=PALETTE["ink"]),
        )],
    )
    return fig


@st.cache_resource
def train_models(path: str, modified_ns: int) -> dict:
    raw_df = pd.read_csv(path)
    df = raw_df.rename(columns=lambda name: name.strip().replace(" ", "_")).drop_duplicates().copy()
    df = df.drop(columns="phone_number")
    for column in ["international_plan", "voice_mail_plan"]:
        df[column] = df[column].str.lower().map({"no": 0, "yes": 1}).astype("int8")
    df["churn"] = df["churn"].astype(bool).astype("int8")
    df["area_code"] = df["area_code"].astype(str)

    outlier_columns = [
        column for column in df.select_dtypes(include="number").columns
        if column.endswith(("_minutes", "_charge", "_calls"))
    ]
    outlier_caps = {}
    for column in outlier_columns:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_caps[column] = (float(lower), float(upper))
        df[column] = df[column].clip(lower=lower, upper=upper)
    df = add_engineered_features(df)

    target = "churn"
    feature_columns = [column for column in df.columns if column != target]
    categorical_columns = ["state", "area_code"]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    X, y = df[feature_columns], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        verbose_feature_names_out=False,
    )
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    encoded_feature_names = preprocessor.get_feature_names_out()

    classifier_estimator = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    classifier_estimator.fit(X_train_encoded, y_train)
    classifier = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier_estimator),
    ])
    churn_predictions = classifier_estimator.predict(X_test_encoded)
    churn_probabilities = classifier_estimator.predict_proba(X_test_encoded)[:, 1]
    classification_metrics = {
        "accuracy": float(accuracy_score(y_test, churn_predictions)),
        "precision": float(precision_score(y_test, churn_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, churn_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, churn_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, churn_probabilities)),
    }
    feature_importance = pd.Series(
        classifier_estimator.feature_importances_, index=encoded_feature_names
    ).sort_values(ascending=False)
    top_features = feature_importance.head(15).index.tolist()
    top_mask = np.isin(encoded_feature_names, top_features)
    top_classifier = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    top_classifier.fit(X_train_encoded[:, top_mask], y_train)
    top_predictions = top_classifier.predict(X_test_encoded[:, top_mask])
    top_probabilities = top_classifier.predict_proba(X_test_encoded[:, top_mask])[:, 1]
    classification_comparison = pd.DataFrame([
        {"features": "All features", **classification_metrics},
        {
            "features": "Top 15",
            "accuracy": float(accuracy_score(y_test, top_predictions)),
            "precision": float(precision_score(y_test, top_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, top_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, top_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, top_probabilities)),
        },
    ])

    regression_target = "total_charge"
    regression_columns = [
        "total_minutes", "total_calls",
        "international_plan", "voice_mail_plan", "customer_service_calls",
        "state", "area_code", "has_both_plans",
    ]
    forbidden_charge_forecast_features = {
        "total_day_minutes", "total_day_calls",
        "total_day_charge", "total_eve_minutes", "total_eve_calls", "total_eve_charge",
        "total_night_minutes", "total_night_calls", "total_night_charge",
        "total_intl_minutes", "total_intl_calls", "total_intl_charge",
        "avg_charge_per_minute", "avg_call_duration",
    }
    assert set(regression_columns) == {
        "total_minutes", "total_calls", "customer_service_calls",
        "international_plan", "voice_mail_plan", "state", "area_code", "has_both_plans",
    }
    assert not forbidden_charge_forecast_features.intersection(regression_columns)
    X_regression = df[regression_columns]
    y_regression = df[regression_target]
    X_reg_train, X_reg_test = X_regression.loc[X_train.index], X_regression.loc[X_test.index]
    y_reg_train, y_reg_test = y_regression.loc[y_train.index], y_regression.loc[y_test.index]
    regression_categorical = [column for column in categorical_columns if column in regression_columns]
    regression_numeric = [column for column in regression_columns if column not in regression_categorical]
    regression_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), regression_numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), regression_categorical),
        ],
        verbose_feature_names_out=False,
    )
    regression_model = Pipeline([
        ("preprocessor", regression_preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=2, loss="huber", random_state=RANDOM_STATE
        )),
    ])
    regression_model.fit(X_reg_train, y_reg_train)
    regression_predictions = regression_model.predict(X_reg_test)
    regression_metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_reg_test, regression_predictions))),
        "MAE": float(mean_absolute_error(y_reg_test, regression_predictions)),
        "R2": float(r2_score(y_reg_test, regression_predictions)),
    }

    high_value_columns = [
        "customer_service_calls", "total_minutes", "total_calls",
        "international_plan", "voice_mail_plan", "has_both_plans", "state",
    ]
    high_value_categorical = ["state"]
    high_value_numeric = [column for column in high_value_columns if column not in high_value_categorical]
    high_value_threshold = float(df.loc[X_train.index, "total_charge"].median())
    y_high_train = (df.loc[X_train.index, "total_charge"] >= high_value_threshold).astype("int8")
    y_high_test = (df.loc[X_test.index, "total_charge"] >= high_value_threshold).astype("int8")
    high_value_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), high_value_numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), high_value_categorical),
        ],
        verbose_feature_names_out=False,
    )
    high_value_classifier = Pipeline([
        ("preprocessor", high_value_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    high_value_classifier.fit(X_train[high_value_columns], y_high_train)
    high_predictions = high_value_classifier.predict(X_test[high_value_columns])
    high_probabilities = high_value_classifier.predict_proba(X_test[high_value_columns])[:, 1]
    high_value_classifier_metrics = {
        "accuracy": float(accuracy_score(y_high_test, high_predictions)),
        "precision": float(precision_score(y_high_test, high_predictions, zero_division=0)),
        "recall": float(recall_score(y_high_test, high_predictions, zero_division=0)),
        "f1": float(f1_score(y_high_test, high_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_high_test, high_probabilities)),
    }

    cluster_columns = [
        "total_minutes", "total_charge", "total_calls", "customer_service_calls",
        "international_plan", "voice_mail_plan", "number_vmail_messages", "total_intl_minutes",
    ]
    cluster_scaler = StandardScaler()
    cluster_matrix = cluster_scaler.fit_transform(df[cluster_columns])
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(cluster_matrix)
    cluster_profiles = df.assign(cluster=cluster_labels).groupby("cluster").agg(
        customers=(target, "size"),
        churn_rate=(target, "mean"),
        avg_total_minutes=("total_minutes", "mean"),
        avg_total_charge=("total_charge", "mean"),
        avg_customer_service_calls=("customer_service_calls", "mean"),
        international_plan_rate=("international_plan", "mean"),
        voice_mail_plan_rate=("voice_mail_plan", "mean"),
    ).sort_values("churn_rate", ascending=False)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    cluster_coordinates = pca.fit_transform(cluster_matrix)
    cluster_silhouette = float(silhouette_score(cluster_matrix, cluster_labels))

    recommendation_table = df.loc[X_test.index, ["state", "total_charge", target]].copy()
    recommendation_table.insert(0, "customer_id", [f"C{index + 1:04d}" for index in recommendation_table.index])
    recommendation_table["churn_probability"] = churn_probabilities
    recommendation_table["high_value_probability"] = high_probabilities
    recommendation_table["predicted_total_charge"] = regression_predictions
    recommendation_table["priority_score"] = churn_probabilities * high_probabilities
    recommendation_table["recommendation"] = np.select(
        [
            (churn_probabilities >= 0.50) & (high_probabilities >= 0.50),
            churn_probabilities >= 0.50,
            high_probabilities >= 0.50,
        ],
        ["Priority retention", "Churn prevention", "Value nurture"],
        default="Monitor",
    )
    recommendation_table = recommendation_table.sort_values("priority_score", ascending=False).head(20).reset_index(drop=True)
    recommendation_metrics = {
        "top_20_actual_churner_share": float(recommendation_table[target].mean()),
        "top_20_high_value_rate": float((recommendation_table["total_charge"] >= high_value_threshold).mean()),
        "top_20_mean_predicted_charge": float(recommendation_table["predicted_total_charge"].mean()),
    }

    customers = df[feature_columns + [target]].copy()
    customers.insert(0, "customer_id", [f"C{index + 1:04d}" for index in customers.index])
    customers["cluster"] = cluster_labels
    customers["churn_probability"] = classifier.predict_proba(X)[:, 1]
    customers["predicted_total_charge"] = regression_model.predict(X_regression)
    customers["high_value_probability"] = high_value_classifier.predict_proba(X[high_value_columns])[:, 1]
    customers["priority_score"] = customers["churn_probability"] * customers["high_value_probability"]

    return {
        "classifier": classifier,
        "regression_model": regression_model,
        "high_value_classifier": high_value_classifier,
        "feature_columns": feature_columns,
        "regression_columns": regression_columns,
        "regression_target": regression_target,
        "high_value_columns": high_value_columns,
        "outlier_caps": outlier_caps,
        "feature_importance": feature_importance,
        "top_features": top_features,
        "classification_comparison": classification_comparison,
        "classification_metrics": classification_metrics,
        "regression_metrics": regression_metrics,
        "high_value_classifier_metrics": high_value_classifier_metrics,
        "high_value_threshold": high_value_threshold,
        "recommendation_metrics": recommendation_metrics,
        "recommendation_table": recommendation_table,
        "cluster_columns": cluster_columns,
        "cluster_scaler": cluster_scaler,
        "kmeans": kmeans,
        "cluster_profiles": cluster_profiles,
        "cluster_coordinates": cluster_coordinates,
        "cluster_silhouette": cluster_silhouette,
        "customers": customers,
        "churn_rate": float(y.mean()),
    }


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["total_minutes"] = frame[["total_day_minutes", "total_eve_minutes", "total_night_minutes"]].sum(axis=1)
    frame["total_charge"] = frame[["total_day_charge", "total_eve_charge", "total_night_charge"]].sum(axis=1)
    frame["total_calls"] = frame[["total_day_calls", "total_eve_calls", "total_night_calls"]].sum(axis=1)
    frame["avg_charge_per_minute"] = frame["total_charge"].div(frame["total_minutes"].replace(0, np.nan)).fillna(0)
    frame["avg_call_duration"] = frame["total_minutes"].div(frame["total_calls"].replace(0, np.nan)).fillna(0)
    frame["has_both_plans"] = (
        (frame["international_plan"] == 1) & (frame["voice_mail_plan"] == 1)
    ).astype("int8")
    frame["day_minutes_ratio"] = frame["total_day_minutes"].div(frame["total_minutes"].replace(0, np.nan)).fillna(0)
    return frame


def make_prediction_row(values: dict, artifacts: dict) -> pd.DataFrame:
    row = pd.DataFrame([values])
    for column, (lower, upper) in artifacts["outlier_caps"].items():
        if column in row:
            row[column] = row[column].clip(lower=lower, upper=upper)

    row["area_code"] = row["area_code"].astype(str)
    row = add_engineered_features(row)
    return row[artifacts["feature_columns"]]


def make_value_forecast_row(values: dict, artifacts: dict) -> pd.DataFrame:
    row = pd.DataFrame([values])
    row["area_code"] = row["area_code"].astype(str)
    row["has_both_plans"] = (
        (row["international_plan"] == 1) & (row["voice_mail_plan"] == 1)
    ).astype("int8")
    return row[artifacts["regression_columns"]]


# ---------------------------------------------------------------------------
# Sidebar — brand + quick project facts (mirrors the reference's left nav)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-mark">TC</div>', unsafe_allow_html=True)
    st.markdown("**Telecom Churn Analytics**")
    st.caption("DS 4892 \u00b7 Data Analytics Lab")
    st.markdown("---")
    nav_items = [
        ("Overview", PALETTE["navy"]), ("EDA", PALETTE["teal"]), ("Prediction", PALETTE["coral"]),
        ("Segmentation", PALETTE["violet"]), ("Value Forecast", PALETTE["amber"]), ("Retention Decision", PALETTE["navy"]),
    ]
    for label, color in nav_items:
        st.markdown(
            f'<div class="nav-item"><span class="nav-dot" style="background:{color};"></span>{label}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Four model roles: churn classifier, K-Means segmentation, charge forecast, and a value-aware retention ranker.")

st.title("Dashboard")
st.caption("Customer retention, usage patterns, and value prioritization")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

with st.spinner("Training the four model roles from the dataset (cached for this app session)..."):
    artifacts = train_models(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
regression_model = artifacts["regression_model"]
value_model = artifacts["high_value_classifier"]
customers = artifacts["customers"].copy()

overview_tab, eda_tab, prediction_tab, segmentation_tab, forecast_tab, retention_tab = st.tabs(
    ["Overview", "EDA", "Prediction", "Segmentation", "Value Forecast", "Retention Decision"]
)

with overview_tab:
    total_customers = len(customers)
    churn_rate = customers["churn"].mean()
    average_value = customers["total_charge"].mean()
    metric_columns = st.columns(3)
    metric_columns[0].metric("Customers", f"{total_customers:,}")
    metric_columns[1].metric("Churn rate", f"{churn_rate:.1%}")
    metric_columns[2].metric("Average account value", f"${average_value:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top churn states</div>', unsafe_allow_html=True)
        state_rates = customers.groupby("state")["churn"].mean().sort_values(ascending=False).head(10)
        colors_cycle = [PALETTE["coral"], PALETTE["amber"], PALETTE["navy"], PALETTE["teal"], PALETTE["violet"]]
        rows_html = ""
        for index, (state, rate) in enumerate(state_rates.items()):
            dot_color = colors_cycle[index % len(colors_cycle)]
            rows_html += (
                f'<div class="legend-row"><span class="legend-label">'
                f'<span class="legend-dot" style="background:{dot_color};"></span>{state}</span>'
                f'<span class="legend-value">{rate:.1%}</span></div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Churn composition</div>', unsafe_allow_html=True)
        retained_count = int((customers["churn"] == 0).sum())
        churned_count = int((customers["churn"] == 1).sum())
        fig = donut_chart(
            ["Retained", "Churned"], [retained_count, churned_count],
            [PALETTE["navy"], PALETTE["coral"]], "Churn rate", f"{churn_rate:.1%}",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="legend-row"><span class="legend-label">'
            f'<span class="legend-dot" style="background:{PALETTE["navy"]};"></span>Retained</span>'
            f'<span class="legend-value">{retained_count:,}</span></div>'
            f'<div class="legend-row"><span class="legend-label">'
            f'<span class="legend-dot" style="background:{PALETTE["coral"]};"></span>Churned</span>'
            f'<span class="legend-value">{churned_count:,}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

with eda_tab:
    st.subheader("Explore churn patterns")
    filter_columns = st.columns(3)
    states = ["All"] + sorted(customers["state"].dropna().unique().tolist())
    selected_state = filter_columns[0].selectbox("State", states, key="eda_state")
    international_filter = filter_columns[1].selectbox(
        "International plan", ["All", "No", "Yes"], key="eda_international"
    )
    voicemail_filter = filter_columns[2].selectbox(
        "Voice mail plan", ["All", "No", "Yes"], key="eda_voicemail"
    )

    filtered = customers.copy()
    if selected_state != "All":
        filtered = filtered[filtered["state"] == selected_state]
    if international_filter != "All":
        filtered = filtered[filtered["international_plan"] == (international_filter == "Yes")]
    if voicemail_filter != "All":
        filtered = filtered[filtered["voice_mail_plan"] == (voicemail_filter == "Yes")]

    st.caption(f"{len(filtered):,} customers match these filters.")
    if filtered.empty:
        st.info("No customers match the selected filters.")
    else:
        chart_columns = st.columns(2)
        filtered_state_rates = filtered.groupby("state")["churn"].mean().sort_values(ascending=False)
        chart_columns[0].markdown("**Churn rate by state**")
        chart_columns[0].bar_chart(filtered_state_rates, horizontal=True, color=PALETTE["navy"])

        plan_rates = filtered.groupby("international_plan")["churn"].mean().rename(index={0: "No plan", 1: "International plan"})
        chart_columns[1].markdown("**Churn by international plan**")
        chart_columns[1].bar_chart(plan_rates, color=PALETTE["coral"])

        outcome_means = filtered.groupby("churn")[["customer_service_calls", "total_day_minutes", "total_charge"]].mean()
        outcome_means.index = outcome_means.index.map({0: "Retained", 1: "Churned"})
        st.markdown("**Usage and support indicators by outcome**")
        st.bar_chart(outcome_means)

        area_rates = filtered.groupby("area_code")["churn"].mean()
        st.markdown("**Churn rate by area code**")
        st.bar_chart(area_rates, color=PALETTE["teal"])

with prediction_tab:
    st.subheader("Estimate an individual customer's churn risk")
    states = sorted(customers["state"].dropna().unique().tolist())
    area_codes = sorted(customers["area_code"].astype(str).unique().tolist())
    with st.form("prediction_form"):
        categorical_inputs = st.columns(4)
        state = categorical_inputs[0].selectbox("State", states)
        area_code = categorical_inputs[1].selectbox("Area code", area_codes)
        international_plan = categorical_inputs[2].checkbox("International plan")
        voice_mail_plan = categorical_inputs[3].checkbox("Voice mail plan", value=True)

        field_specs = [
            ("account_length", "Account length", 128.0, 1.0, 300.0, 1.0),
            ("number_vmail_messages", "Voice mail messages", 25.0, 0.0, 100.0, 1.0),
            ("total_day_minutes", "Day minutes", 265.1, 0.0, 500.0, 0.1),
            ("total_day_calls", "Day calls", 110.0, 0.0, 250.0, 1.0),
            ("total_day_charge", "Day charge", 45.07, 0.0, 100.0, 0.01),
            ("total_eve_minutes", "Evening minutes", 197.4, 0.0, 500.0, 0.1),
            ("total_eve_calls", "Evening calls", 99.0, 0.0, 250.0, 1.0),
            ("total_eve_charge", "Evening charge", 16.78, 0.0, 100.0, 0.01),
            ("total_night_minutes", "Night minutes", 244.7, 0.0, 500.0, 0.1),
            ("total_night_calls", "Night calls", 91.0, 0.0, 250.0, 1.0),
            ("total_night_charge", "Night charge", 11.01, 0.0, 100.0, 0.01),
            ("total_intl_minutes", "International minutes", 10.0, 0.0, 100.0, 0.1),
            ("total_intl_calls", "International calls", 3.0, 0.0, 100.0, 1.0),
            ("total_intl_charge", "International charge", 2.7, 0.0, 100.0, 0.01),
            ("customer_service_calls", "Customer service calls", 1.0, 0.0, 20.0, 1.0),
        ]
        input_columns = st.columns(3)
        numeric_values = {}
        for index, (name, label, default, minimum, maximum, step) in enumerate(field_specs):
            numeric_values[name] = input_columns[index % 3].number_input(
                label, min_value=minimum, max_value=maximum, value=default, step=step
            )
        submitted = st.form_submit_button("Score churn risk", type="primary")

    if submitted:
        customer_values = {
            "state": state,
            "area_code": area_code,
            "international_plan": int(international_plan),
            "voice_mail_plan": int(voice_mail_plan),
            **numeric_values,
        }
        prediction_row = make_prediction_row(customer_values, artifacts)
        churn_probability = float(artifacts["classifier"].predict_proba(prediction_row)[0, 1])
        regression_row = prediction_row[artifacts["regression_columns"]]
        predicted_charge = float(regression_model.predict(regression_row)[0])
        high_value_row = prediction_row[artifacts["high_value_columns"]]
        high_value_probability = float(value_model.predict_proba(high_value_row)[0, 1])
        if churn_probability >= 0.65:
            tier = "Urgent"
        elif churn_probability >= 0.35:
            tier = "High"
        elif churn_probability >= 0.20:
            tier = "Elevated"
        else:
            tier = "Low"
        st.markdown("&nbsp;")
        result_columns = st.columns(4)
        result_columns[0].metric("Churn probability", f"{churn_probability:.1%}")
        with result_columns[1]:
            st.markdown(
                f'<div style="padding-top:6px;"><div style="color:{PALETTE["muted"]};font-size:0.82rem;margin-bottom:6px;">Risk tier</div>{risk_badge_html(tier)}</div>',
                unsafe_allow_html=True,
            )
        result_columns[2].metric("Forecast total charge", f"${predicted_charge:.2f}")
        result_columns[3].metric("High-value probability", f"{high_value_probability:.1%}")

with segmentation_tab:
    st.subheader("Customer segments")
    coordinates = pd.DataFrame(
        artifacts["cluster_coordinates"], columns=["PC1", "PC2"]
    )
    coordinates["cluster"] = customers["cluster"].astype(str).to_numpy()
    coordinates["churn"] = customers["churn"].map({0: "Retained", 1: "Churned"}).to_numpy()
    st.scatter_chart(coordinates, x="PC1", y="PC2", color="cluster")
    st.caption("PCA is a 2D visualization of the eight standardized inputs used by K-Means.")
    st.markdown("**Cluster profiles**")
    cluster_display = artifacts["cluster_profiles"].copy()
    cluster_display["churn_rate"] = fmt_percent(cluster_display["churn_rate"])
    cluster_display["international_plan_rate"] = fmt_percent(cluster_display["international_plan_rate"])
    cluster_display["voice_mail_plan_rate"] = fmt_percent(cluster_display["voice_mail_plan_rate"])
    cluster_display["avg_total_minutes"] = fmt_number(cluster_display["avg_total_minutes"], 1)
    cluster_display["avg_total_charge"] = fmt_currency(cluster_display["avg_total_charge"])
    cluster_display["avg_customer_service_calls"] = fmt_number(cluster_display["avg_customer_service_calls"], 2)
    st.dataframe(cluster_display, width="stretch")

with forecast_tab:
    st.subheader("Forecast expected customer charge")
    st.caption(
        "Forecasting expected customer charge from aggregate minutes/calls and plan/engagement profile, without day/evening/night usage splits. "
        f"Holdout: RMSE ${artifacts['regression_metrics']['RMSE']:.2f}, "
        f"MAE ${artifacts['regression_metrics']['MAE']:.2f}, R\u00b2 {artifacts['regression_metrics']['R2']:.3f}."
    )
    if artifacts["regression_metrics"]["R2"] <= 0:
        st.info("The aggregate-usage and plan-profile model did not outperform the holdout mean baseline; assess the metrics before operational use.")
    states = sorted(customers["state"].dropna().unique().tolist())
    area_codes = sorted(customers["area_code"].astype(str).unique().tolist())
    with st.form("value_forecast_form"):
        categorical_inputs = st.columns(4)
        forecast_state = categorical_inputs[0].selectbox("State", states, key="forecast_state")
        forecast_area_code = categorical_inputs[1].selectbox("Area code", area_codes, key="forecast_area_code")
        forecast_international = categorical_inputs[2].checkbox("International plan", key="forecast_international")
        forecast_voicemail = categorical_inputs[3].checkbox("Voice mail plan", value=True, key="forecast_voicemail")

        usage_columns = st.columns(3)
        forecast_values = {
            "total_minutes": usage_columns[0].number_input(
                "Total minutes", min_value=0.0, max_value=3000.0, value=600.0, step=10.0
            ),
            "total_calls": usage_columns[1].number_input(
                "Total calls", min_value=0.0, max_value=2000.0, value=300.0, step=10.0
            ),
            "customer_service_calls": usage_columns[2].number_input(
                "Customer service calls", min_value=0.0, max_value=20.0, value=1.0, step=1.0
            ),
        }
        forecast_submitted = st.form_submit_button("Forecast customer value", type="primary")

    if forecast_submitted:
        forecast_values.update({
            "state": forecast_state,
            "area_code": forecast_area_code,
            "international_plan": int(forecast_international),
            "voice_mail_plan": int(forecast_voicemail),
        })
        forecast_row = make_value_forecast_row(forecast_values, artifacts)
        predicted_charge = float(regression_model.predict(forecast_row)[0])
        st.metric("Predicted total charge", f"${predicted_charge:.2f}")

with retention_tab:
    st.subheader("Value-aware retention recommendations")
    st.caption(
        "Recommendations combine separate held-out churn and high-value classifier probabilities "
        "(priority score = churn probability \u00d7 high-value probability). "
        "No retention-offer response labels exist in this dataset, so this is customer prioritization, not treatment-effect prediction."
    )
    value_metrics = artifacts["high_value_classifier_metrics"]
    metric_columns = st.columns(5)
    for column, metric_name in zip(metric_columns, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        column.metric(f"Value model {metric_name.upper()}", f"{value_metrics[metric_name]:.3f}")
    recommendation_metrics = artifacts["recommendation_metrics"]
    ranking_columns = st.columns(3)
    ranking_columns[0].metric("Top-20 actual churner share", f"{recommendation_metrics['top_20_actual_churner_share']:.1%}")
    ranking_columns[1].metric("Top-20 high-value share", f"{recommendation_metrics['top_20_high_value_rate']:.1%}")
    ranking_columns[2].metric("Top-20 mean predicted charge", f"${recommendation_metrics['top_20_mean_predicted_charge']:.2f}")
    priority_display = artifacts["recommendation_table"].drop(columns=["churn"], errors="ignore").copy()
    priority_display["total_charge"] = fmt_currency(priority_display["total_charge"])
    priority_display["churn_probability"] = fmt_percent(priority_display["churn_probability"])
    priority_display["high_value_probability"] = fmt_percent(priority_display["high_value_probability"])
    priority_display["predicted_total_charge"] = fmt_currency(priority_display["predicted_total_charge"])
    priority_display["priority_score"] = fmt_number(priority_display["priority_score"], 3)
    st.dataframe(priority_display, width="stretch", hide_index=True)
    st.download_button(
        "Download priority list",
        data=artifacts["recommendation_table"].drop(columns=["churn"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="retention_recommendations.csv",
        mime="text/csv",
    )
