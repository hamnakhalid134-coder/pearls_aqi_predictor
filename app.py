import os
import glob
import joblib
import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import hopsworks
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT LINKS
# ============================================================

GITHUB_URL = (
    "https://github.com/hamnakhalid134-coder/"
    "pearls_aqi_predictor"
)

# IMPORTANT:
# Paste/keep your existing LinkedIn URL here.
LINKEDIN_URL = "http://linkedin.com/in/hamna-khalid-29a107319"

# Add after technical report is uploaded.
REPORT_URL = ""


# ============================================================
# PROFESSIONAL LIGHT CLOUD / ATMOSPHERIC THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN APPLICATION
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(255,255,255,0.98) 0%,
                rgba(255,255,255,0.45) 23%,
                transparent 42%
            ),
            radial-gradient(
                circle at 83% 13%,
                rgba(204,225,238,0.76) 0%,
                transparent 39%
            ),
            radial-gradient(
                circle at 55% 82%,
                rgba(245,249,251,0.90) 0%,
                transparent 43%
            ),
            linear-gradient(
                135deg,
                #eef7fb 0%,
                #dceaf3 41%,
                #edf4f7 72%,
                #d7e7ef 100%
            );

        color: #20384b;
    }


    .stAppViewContainer {
        background: transparent !important;
    }


    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }


    /* ======================================================
       REMOVE BLACK STREAMLIT HEADER AREA
       ====================================================== */

    header[data-testid="stHeader"] {
        background:
            rgba(235,245,250,0.96) !important;

        border-bottom:
            1px solid rgba(71,111,138,0.10) !important;
    }


    [data-testid="stToolbar"] {
        background: transparent !important;
    }


    [data-testid="stDecoration"] {
        display: none !important;
    }


    /* ======================================================
       TYPOGRAPHY
       ====================================================== */

    h1 {
        color: #17384e !important;
        font-weight: 800 !important;
        letter-spacing: -1.1px !important;
    }


    h2 {
        color: #18394f !important;
        font-weight: 780 !important;
    }


    h3 {
        color: #24485f !important;
        font-weight: 730 !important;
    }


    h4 {
        color: #567286 !important;
        font-weight: 800 !important;
        letter-spacing: 1.1px !important;
        text-transform: uppercase;
    }


    .stApp p,
    .stApp label {
        color: #344f63;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(219,235,245,0.99),
                rgba(240,247,250,0.99)
            );

        border-right:
            1px solid rgba(67,106,135,0.18);
    }


    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #30495c !important;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {
        background:
            rgba(255,255,255,0.82);

        border:
            1px solid rgba(68,111,143,0.18);

        border-radius: 18px;

        padding: 17px 18px;

        box-shadow:
            0 8px 24px rgba(70,95,115,0.08);
    }


    div[data-testid="stMetricLabel"] * {
        color: #607789 !important;
        font-weight: 720 !important;
        font-size: 0.96rem !important;
    }


    div[data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
    }


    div[data-testid="stMetricValue"] * {
        color: #17384e !important;
        font-weight: 820 !important;
    }


    div[data-testid="stMetricDelta"] * {
        font-weight: 650 !important;
    }


    /* ======================================================
       BORDERED CONTENT CARDS
       ====================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            rgba(255,255,255,0.73) !important;

        border:
            1px solid #c3d7e3 !important;

        border-radius:
            20px !important;

        box-shadow:
            0 9px 26px
            rgba(72,99,119,0.08) !important;
    }


    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] span {
        color: #304b5f !important;
    }


    /* ======================================================
       LINKS
       ====================================================== */

    a {
        color: #247fa6 !important;
        text-decoration: none !important;
        font-weight: 650 !important;
    }


    a:hover {
        color: #185f80 !important;
        text-decoration: underline !important;
    }


    /* ======================================================
       DATA TABLE
       ====================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }


    /* ======================================================
       EXPANDER
       ====================================================== */

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.48);
        border-radius: 13px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AQI CLASSIFICATION
# ============================================================

def aqi_category(aqi):

    if aqi <= 50:
        return "Good", "#2ca66d"

    if aqi <= 100:
        return "Moderate", "#c99600"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e37c2f"

    if aqi <= 200:
        return "Unhealthy", "#db4451"

    if aqi <= 300:
        return "Very Unhealthy", "#8556b4"

    return "Hazardous", "#9d2947"


def health_message(aqi):

    category, _ = aqi_category(aqi)

    if category == "Good":
        return (
            "Air quality is satisfactory and presents "
            "minimal health concern."
        )

    if category == "Moderate":
        return (
            "Air quality is generally acceptable. "
            "Unusually sensitive individuals may consider "
            "reducing prolonged outdoor exertion."
        )

    if category == "Unhealthy for Sensitive Groups":
        return (
            "Sensitive groups should reduce prolonged "
            "or heavy outdoor exertion."
        )

    if category == "Unhealthy":
        return (
            "Health effects may occur across the population. "
            "Limit prolonged outdoor activity."
        )

    if category == "Very Unhealthy":
        return (
            "Health alert conditions. "
            "Reduce unnecessary outdoor exposure."
        )

    return (
        "Health warning conditions. "
        "Avoid outdoor exposure where possible."
    )


# ============================================================
# HOPSWORKS API KEY
# ============================================================

def get_api_key():

    try:

        if "HOPSWORKS_API_KEY" in st.secrets:
            return st.secrets[
                "HOPSWORKS_API_KEY"
            ]

    except Exception:
        pass

    return os.getenv(
        "HOPSWORKS_API_KEY"
    )


# ============================================================
# CHART STYLE
# ============================================================

def style_chart(fig):

    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.56)",

        font=dict(
            color="#294154",
            size=14
        ),

        title_font=dict(
            color="#17384e",
            size=20
        ),

        legend=dict(
            font=dict(
                color="#294154"
            )
        ),

        margin=dict(
            l=35,
            r=25,
            t=60,
            b=35
        )
    )

    fig.update_xaxes(
        tickfont=dict(
            color="#425d70"
        ),

        title_font=dict(
            color="#314b5e"
        ),

        gridcolor=
            "rgba(76,108,130,0.14)"
    )

    fig.update_yaxes(
        tickfont=dict(
            color="#425d70"
        ),

        title_font=dict(
            color="#314b5e"
        ),

        gridcolor=
            "rgba(76,108,130,0.14)"
    )

    return fig


# ============================================================
# HOPSWORKS CONNECTION
# ============================================================

@st.cache_resource(show_spinner=False)
def connect_hopsworks():

    api_key = get_api_key()

    if not api_key:

        raise ValueError(
            "HOPSWORKS_API_KEY is not configured."
        )

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project="pearls_aqi_hk26",
        port=443,
        engine="python",
        api_key_value=api_key
    )

    fs = project.get_feature_store()
    mr = project.get_model_registry()

    return project, fs, mr


# ============================================================
# LIVE FEATURE STORE
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_live_features():

    _, fs, _ = connect_hopsworks()

    feature_group = fs.get_feature_group(
        name="aqi_live_features",
        version=1
    )

    data = feature_group.read(
        online=True,
        dataframe_type="pandas"
    )

    data = data[
        data["city"] == "lahore"
    ].copy()

    if data.empty:

        raise ValueError(
            "No Lahore live feature record was found."
        )

    return (
        data
        .sort_values("date_time")
        .tail(1)
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

@st.cache_resource(show_spinner=False)
def load_registry_model(model_name):

    _, _, model_registry = (
        connect_hopsworks()
    )

    model_metadata = (
        model_registry.get_model(
            name=model_name,
            version=1
        )
    )

    model_directory = (
        model_metadata.download()
    )

    files = glob.glob(
        os.path.join(
            model_directory,
            "**",
            "*.pkl"
        ),
        recursive=True
    )

    if not files:

        raise FileNotFoundError(
            f"No model artifact found for "
            f"{model_name}"
        )

    return joblib.load(
        files[0]
    )


def prepare_features(
    model,
    dataframe
):

    result = dataframe.copy()

    if hasattr(
        model,
        "feature_names_in_"
    ):

        result = result[
            list(
                model.feature_names_in_
            )
        ]

    return result


# ============================================================
# LIVE PREDICTIONS
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def make_predictions():

    latest = get_live_features()

    X = latest.drop(
        columns=[
            "city",
            "date_time"
        ],
        errors="ignore"
    )

    model_24 = load_registry_model(
        "aqi_24h_xgboost"
    )

    model_48 = load_registry_model(
        "aqi_48h_ridge"
    )

    model_72 = load_registry_model(
        "aqi_72h_ridge"
    )

    pred_24 = int(
        round(
            model_24.predict(
                prepare_features(
                    model_24,
                    X
                )
            )[0]
        )
    )

    pred_48 = int(
        round(
            model_48.predict(
                prepare_features(
                    model_48,
                    X
                )
            )[0]
        )
    )

    pred_72 = int(
        round(
            model_72.predict(
                prepare_features(
                    model_72,
                    X
                )
            )[0]
        )
    )

    return {
        "latest": latest,
        "X": X,
        "model_24": model_24,
        "pred_24": pred_24,
        "pred_48": pred_48,
        "pred_72": pred_72
    }


# ============================================================
# RECENT AQI / POLLUTION DATA
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def recent_air_data():

    url = (
        "https://air-quality-api.open-meteo.com/"
        "v1/air-quality"
    )

    params = {
        "latitude": 31.5204,
        "longitude": 74.3587,

        "hourly": [
            "us_aqi",
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone"
        ],

        "past_days": 10,
        "forecast_days": 1,

        "timezone":
            "Asia/Karachi"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    hourly = response.json()[
        "hourly"
    ]

    data = pd.DataFrame(
        {
            "Date_Time":
                pd.to_datetime(
                    hourly["time"]
                ),

            "AQI":
                hourly["us_aqi"],

            "PM2_5":
                hourly["pm2_5"],

            "PM10":
                hourly["pm10"],

            "NO2":
                hourly[
                    "nitrogen_dioxide"
                ],

            "O3":
                hourly["ozone"]
        }
    )

    now = (
        pd.Timestamp.now(
            tz="Asia/Karachi"
        )
        .tz_localize(None)
        .floor("h")
    )

    return (
        data[
            data["Date_Time"] <= now
        ]
        .dropna()
        .reset_index(drop=True)
    )


# ============================================================
# MODEL PERFORMANCE RESULTS
# ============================================================

@st.cache_data(show_spinner=False)
def get_model_results():

    filename = (
        "Hopsworks_Model_Results.csv"
    )

    if os.path.exists(filename):

        return pd.read_csv(
            filename
        )

    return pd.DataFrame(
        {
            "Target": [
                "aqi_24h",
                "aqi_48h",
                "aqi_72h"
            ],

            "MAE": [
                17.076,
                23.323,
                24.484
            ],

            "RMSE": [
                23.580,
                31.637,
                32.868
            ],

            "R2": [
                0.627,
                0.329,
                0.277
            ],

            "Baseline_MAE": [
                20.228,
                24.848,
                26.500
            ],

            "Baseline_RMSE": [
                31.224,
                37.059,
                37.889
            ],

            "Baseline_R2": [
                0.346,
                0.080,
                0.039
            ]
        }
    )


# ============================================================
# LOAD LIVE BACKEND
# ============================================================

try:

    with st.spinner(
        "Loading Lahore air-quality forecast..."
    ):

        output = make_predictions()

except Exception as error:

    st.error(
        "The live AQI prediction backend "
        "could not be reached."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# CURRENT VALUES
# ============================================================

latest = output[
    "latest"
]


current_time = pd.to_datetime(
    latest[
        "date_time"
    ].iloc[0]
)


current_aqi = int(
    round(
        latest[
            "aqi"
        ].iloc[0]
    )
)


pred_24 = output[
    "pred_24"
]

pred_48 = output[
    "pred_48"
]

pred_72 = output[
    "pred_72"
]


day1 = (
    current_time
    + pd.Timedelta(
        hours=24
    )
)

day2 = (
    current_time
    + pd.Timedelta(
        hours=48
    )
)

day3 = (
    current_time
    + pd.Timedelta(
        hours=72
    )
)


current_status, current_color = (
    aqi_category(
        current_aqi
    )
)


status_24, _ = (
    aqi_category(
        pred_24
    )
)


status_48, _ = (
    aqi_category(
        pred_48
    )
)


status_72, _ = (
    aqi_category(
        pred_72
    )
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Pearls AQI Predictor"
    )

    st.markdown(
        "### Hamna Khalid"
    )

    st.markdown(
        "**Data Science Intern**"
    )

    st.caption(
        "Electrical Engineer"
    )


    if LINKEDIN_URL:

        st.markdown(
            f"[LinkedIn Profile ↗]"
            f"({LINKEDIN_URL})"
        )

    else:

        st.caption(
            "LinkedIn profile link will be "
            "added before final submission."
        )


    st.divider()


    st.markdown(
        "### US EPA AQI Classification"
    )

    st.markdown(
        "🟢 **Good** · 0–50"
    )

    st.markdown(
        "🟡 **Moderate** · 51–100"
    )

    st.markdown(
        "🟠 **Unhealthy for Sensitive Groups** · 101–150"
    )

    st.markdown(
        "🔴 **Unhealthy** · 151–200"
    )

    st.markdown(
        "🟣 **Very Unhealthy** · 201–300"
    )

    st.markdown(
        "🟥 **Hazardous** · 301+"
    )


    st.divider()


    st.markdown(
        "### Project Resources"
    )


    st.markdown(
        f"[GitHub Repository ↗]"
        f"({GITHUB_URL})"
    )


    if REPORT_URL:

        st.markdown(
            f"[Technical Report ↗]"
            f"({REPORT_URL})"
        )

    else:

        st.caption(
            "Technical report link will "
            "be added after finalization."
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "Lahore Air Quality Forecast"
)

st.caption(
    "Current air-quality assessment with direct "
    "+24 h, +48 h and +72 h machine-learning forecasts."
)


# ============================================================
# CURRENT AIR QUALITY ASSESSMENT
# ============================================================

st.markdown(
    "#### Current Air Quality Assessment"
)


gauge_column, assessment_column = (
    st.columns(
        [
            1.08,
            1.15
        ],
        gap="large"
    )
)


# ============================================================
# AQI GAUGE
# ============================================================

with gauge_column:

    with st.container(
        border=True
    ):

        st.subheader(
            "Current US AQI"
        )


        gauge = go.Figure(
            go.Indicator(

                mode=
                    "gauge+number",

                value=
                    current_aqi,

                number={
                    "font": {
                        "size": 64,
                        "color": "#17384e"
                    }
                },

                gauge={
                    "axis": {
                        "range": [
                            0,
                            500
                        ],

                        "tickvals": [
                            0,
                            50,
                            100,
                            150,
                            200,
                            300,
                            400,
                            500
                        ],

                        "tickfont": {
                            "color":
                                "#5d7385"
                        }
                    },

                    "bar": {
                        "color":
                            current_color,

                        "thickness":
                            0.28
                    },

                    "bgcolor":
                        "rgba(255,255,255,0)",

                    "borderwidth":
                        0,

                    "steps": [
                        {
                            "range":
                                [0, 50],

                            "color":
                                "rgba(44,166,109,0.25)"
                        },

                        {
                            "range":
                                [50, 100],

                            "color":
                                "rgba(201,150,0,0.22)"
                        },

                        {
                            "range":
                                [100,150],

                            "color":
                                "rgba(227,124,47,0.22)"
                        },

                        {
                            "range":
                                [150,200],

                            "color":
                                "rgba(219,68,81,0.22)"
                        },

                        {
                            "range":
                                [200,300],

                            "color":
                                "rgba(133,86,180,0.22)"
                        },

                        {
                            "range":
                                [300,500],

                            "color":
                                "rgba(157,41,71,0.22)"
                        }
                    ],

                    "threshold": {
                        "line": {
                            "color":
                                current_color,

                            "width":
                                5
                        },

                        "thickness":
                            0.80,

                        "value":
                            current_aqi
                    }
                }
            )
        )


        gauge.update_layout(
            height=350,

            paper_bgcolor=
                "rgba(255,255,255,0)",

            margin=dict(
                l=25,
                r=25,
                t=20,
                b=10
            )
        )


        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


        st.markdown(
            f"### {current_status}"
        )

        st.caption(
            current_time.strftime(
                "%d %b %Y · %I:%M %p PKT"
            )
        )


# ============================================================
# CURRENT AQI ASSESSMENT CARD
# ============================================================

with assessment_column:

    with st.container(
        border=True
    ):

        st.subheader(
            "Current Air Quality Assessment"
        )

        st.caption(
            "US EPA CLASSIFICATION"
        )

        st.markdown(
            f"## {current_status}"
        )

        st.metric(
            "Current AQI",
            current_aqi
        )

        st.write(
            health_message(
                current_aqi
            )
        )

        st.progress(
            min(
                current_aqi / 300,
                1.0
            )
        )

        st.caption(
            f"Displayed AQI intensity · "
            f"{current_aqi} / 300"
        )

        st.caption(
            "AQI category is based on the "
            "US EPA classification scale."
        )


# ============================================================
# CURRENT CONDITIONS
# ============================================================

st.markdown(
    "#### Current Atmospheric Conditions"
)


m1, m2, m3, m4, m5, m6 = (
    st.columns(
        6,
        gap="small"
    )
)


with m1:

    st.metric(
        "🌡️ Temperature · °C",
        f"{latest['temperature'].iloc[0]:.1f}"
    )


with m2:

    st.metric(
        "💧 Humidity · %",
        f"{latest['humidity'].iloc[0]:.0f}"
    )


with m3:

    st.metric(
        "💨 Wind · km/h",
        f"{latest['wind_speed'].iloc[0]:.1f}"
    )


with m4:

    st.metric(
        "🌫️ PM2.5 · µg/m³",
        f"{latest['pm2_5'].iloc[0]:.1f}"
    )


with m5:

    st.metric(
        "🏙️ PM10 · µg/m³",
        f"{latest['pm10'].iloc[0]:.1f}"
    )


with m6:

    st.metric(
        "🫧 O₃ · µg/m³",
        f"{latest['o3'].iloc[0]:.1f}"
    )


# ============================================================
# OPTIONAL ADDITIONAL ATMOSPHERIC METRICS
# ============================================================

with st.expander(
    "Additional Atmospheric Observations"
):

    a1, a2, a3, a4, a5 = (
        st.columns(5)
    )


    with a1:

        st.metric(
            "Pressure · hPa",
            f"{latest['pressure'].iloc[0]:.1f}"
        )


    with a2:

        st.metric(
            "Precipitation · mm",
            f"{latest['precipitation'].iloc[0]:.1f}"
        )


    with a3:

        st.metric(
            "NO₂ · µg/m³",
            f"{latest['no2'].iloc[0]:.1f}"
        )


    with a4:

        st.metric(
            "SO₂ · µg/m³",
            f"{latest['so2'].iloc[0]:.1f}"
        )


    with a5:

        st.metric(
            "CO · µg/m³",
            f"{latest['co'].iloc[0]:.1f}"
        )


# ============================================================
# NEXT THREE DAYS
# ============================================================

st.markdown(
    "#### Three-Day AQI Forecast"
)


forecast_columns = st.columns(
    3,
    gap="medium"
)


forecast_data = [
    (
        forecast_columns[0],
        "Tomorrow · +24 h",
        day1,
        pred_24,
        status_24,
        "XGBoost"
    ),

    (
        forecast_columns[1],
        "Day 2 · +48 h",
        day2,
        pred_48,
        status_48,
        "Ridge Regression"
    ),

    (
        forecast_columns[2],
        "Day 3 · +72 h",
        day3,
        pred_72,
        status_72,
        "Ridge Regression"
    )
]


for (
    column,
    title,
    date,
    prediction,
    status,
    model_name
) in forecast_data:

    change = (
        prediction
        - current_aqi
    )


    with column:

        with st.container(
            border=True
        ):

            st.subheader(
                title
            )

            st.metric(
                "Predicted US AQI",
                prediction,
                delta=change,
                delta_color="inverse"
            )

            st.markdown(
                f"**Classification:** {status}"
            )


            if change > 0:

                st.caption(
                    f"Forecast increase · "
                    f"+{change} AQI points "
                    f"relative to current conditions"
                )


            elif change < 0:

                st.caption(
                    f"Forecast decrease · "
                    f"{abs(change)} AQI points "
                    f"relative to current conditions"
                )


            else:

                st.caption(
                    "Forecast remains stable "
                    "relative to current conditions."
                )


            st.caption(
                date.strftime(
                    "%A · %d %b %Y"
                )
            )


            st.caption(
                f"Selected model · "
                f"{model_name}"
            )


# ============================================================
# 72-HOUR FORECAST OUTLOOK
# ============================================================

worst_forecast = max(
    pred_24,
    pred_48,
    pred_72
)


worst_status, _ = (
    aqi_category(
        worst_forecast
    )
)


change_72 = (
    worst_forecast
    - current_aqi
)


st.markdown(
    "#### 72-Hour Forecast Outlook"
)


with st.container(
    border=True
):

    outlook_left, outlook_right = (
        st.columns(
            [
                1,
                2
            ]
        )
    )


    with outlook_left:

        st.caption(
            "FORECAST DIRECTION"
        )


        if change_72 > 0:

            st.markdown(
                "## Worsening"
            )

            st.write(
                f"+{change_72} AQI points "
                f"relative to current conditions"
            )


        elif change_72 < 0:

            st.markdown(
                "## Improving"
            )

            st.write(
                f"{abs(change_72)} AQI points "
                f"below current conditions"
            )


        else:

            st.markdown(
                "## Stable"
            )

            st.write(
                "No material change relative "
                "to current conditions"
            )


    with outlook_right:

        st.metric(
            "Maximum Predicted AQI",
            worst_forecast
        )

        st.write(
            f"**US EPA Classification:** "
            f"{worst_status}"
        )

        st.progress(
            min(
                worst_forecast / 300,
                1.0
            )
        )

        st.caption(
            f"Displayed AQI intensity · "
            f"{worst_forecast} / 300"
        )


# ============================================================
# MODEL PERFORMANCE BENCHMARK
# ============================================================

st.markdown(
    "#### Model Performance Benchmark"
)


results = get_model_results().copy()


results[
    "Forecast Horizon"
] = (
    results[
        "Target"
    ]
    .replace(
        {
            "aqi_24h":
                "+24 h",

            "aqi_48h":
                "+48 h",

            "aqi_72h":
                "+72 h"
        }
    )
)


results[
    "Best Model"
] = [
    "XGBoost",
    "Ridge Regression",
    "Ridge Regression"
]


results[
    "MAE Improvement (%)"
] = (
    (
        results[
            "Baseline_MAE"
        ]
        -
        results[
            "MAE"
        ]
    )
    /
    results[
        "Baseline_MAE"
    ]
    *
    100
)


results[
    "Selected"
] = [
    "✓ XGBoost",
    "✓ Ridge Regression",
    "✓ Ridge Regression"
]


performance_table = (
    results[
        [
            "Forecast Horizon",
            "Best Model",
            "MAE",
            "RMSE",
            "R2",
            "Baseline_MAE",
            "MAE Improvement (%)",
            "Selected"
        ]
    ]
    .copy()
)


performance_table.columns = [
    "Forecast Horizon",
    "Best Model",
    "MAE",
    "RMSE",
    "R²",
    "Persistence MAE",
    "MAE Improvement (%)",
    "Selected Model"
]


st.dataframe(
    performance_table.round(3),
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Lower MAE and RMSE indicate better predictive accuracy. "
    "Higher R² indicates greater explained variance."
)


# ============================================================
# FORECAST ANALYTICS
# ============================================================

st.markdown(
    "#### Forecast Analytics"
)


try:

    history = recent_air_data()


    # --------------------------------------------------------
    # AQI HISTORY AND FORECAST
    # --------------------------------------------------------

    forecast_fig = go.Figure()


    forecast_fig.add_trace(
        go.Scatter(
            x=history[
                "Date_Time"
            ],

            y=history[
                "AQI"
            ],

            mode="lines",

            name="Observed AQI",

            line=dict(
                color="#318cb7",
                width=2.5
            )
        )
    )


    forecast_fig.add_trace(
        go.Scatter(
            x=[
                current_time,
                day1,
                day2,
                day3
            ],

            y=[
                current_aqi,
                pred_24,
                pred_48,
                pred_72
            ],

            mode="lines+markers",

            name="ML Forecast",

            line=dict(
                color="#df7148",
                dash="dash",
                width=2
            ),

            marker=dict(
                size=11,
                symbol="diamond",
                color="#df7148"
            )
        )
    )


    forecast_fig.update_layout(
        height=480,
        title=(
            "Recent Observed AQI and "
            "Three-Day Forecast"
        ),
        xaxis_title="Date and Time",
        yaxis_title="US AQI",
        hovermode="x unified"
    )


    forecast_fig = style_chart(
        forecast_fig
    )


    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # MAE / RMSE
    # --------------------------------------------------------

    left_graph, right_graph = (
        st.columns(2)
    )


    mae_data = pd.DataFrame(
        {
            "Horizon": [
                "+24 h",
                "+48 h",
                "+72 h"
            ],

            "Selected Model":
                results[
                    "MAE"
                ].values,

            "Persistence Baseline":
                results[
                    "Baseline_MAE"
                ].values
        }
    )


    mae_long = mae_data.melt(
        id_vars="Horizon",
        var_name="Method",
        value_name="MAE"
    )


    mae_fig = px.bar(
        mae_long,
        x="Horizon",
        y="MAE",
        color="Method",
        barmode="group",
        title=(
            "Mean Absolute Error · "
            "Selected Model vs Persistence"
        )
    )


    mae_fig = style_chart(
        mae_fig
    )


    with left_graph:

        st.plotly_chart(
            mae_fig,
            use_container_width=True
        )


    rmse_data = pd.DataFrame(
        {
            "Horizon": [
                "+24 h",
                "+48 h",
                "+72 h"
            ],

            "Selected Model":
                results[
                    "RMSE"
                ].values,

            "Persistence Baseline":
                results[
                    "Baseline_RMSE"
                ].values
        }
    )


    rmse_long = rmse_data.melt(
        id_vars="Horizon",
        var_name="Method",
        value_name="RMSE"
    )


    rmse_fig = px.bar(
        rmse_long,
        x="Horizon",
        y="RMSE",
        color="Method",
        barmode="group",
        title=(
            "Root Mean Squared Error · "
            "Selected Model vs Persistence"
        )
    )


    rmse_fig = style_chart(
        rmse_fig
    )


    with right_graph:

        st.plotly_chart(
            rmse_fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # R2 / HOURLY PATTERN
    # --------------------------------------------------------

    left_graph2, right_graph2 = (
        st.columns(2)
    )


    r2_data = pd.DataFrame(
        {
            "Horizon": [
                "+24 h",
                "+48 h",
                "+72 h"
            ],

            "Selected Model":
                results[
                    "R2"
                ].values,

            "Persistence Baseline":
                results[
                    "Baseline_R2"
                ].values
        }
    )


    r2_long = r2_data.melt(
        id_vars="Horizon",
        var_name="Method",
        value_name="R2"
    )


    r2_fig = px.bar(
        r2_long,
        x="Horizon",
        y="R2",
        color="Method",
        barmode="group",
        title=(
            "Coefficient of Determination · R²"
        )
    )


    r2_fig = style_chart(
        r2_fig
    )


    with left_graph2:

        st.plotly_chart(
            r2_fig,
            use_container_width=True
        )


    history[
        "Hour"
    ] = (
        history[
            "Date_Time"
        ].dt.hour
    )


    hourly_aqi = (
        history
        .groupby(
            "Hour",
            as_index=False
        )[
            "AQI"
        ]
        .mean()
    )


    hourly_fig = px.line(
        hourly_aqi,
        x="Hour",
        y="AQI",
        markers=True,
        title=(
            "Mean AQI by Hour of Day"
        )
    )


    hourly_fig = style_chart(
        hourly_fig
    )


    with right_graph2:

        st.plotly_chart(
            hourly_fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # POLLUTANT CONCENTRATIONS
    # --------------------------------------------------------

    pollutant_data = pd.DataFrame(
        {
            "Pollutant": [
                "PM2.5",
                "PM10",
                "NO₂",
                "O₃"
            ],

            "Concentration": [
                float(
                    latest[
                        "pm2_5"
                    ].iloc[0]
                ),

                float(
                    latest[
                        "pm10"
                    ].iloc[0]
                ),

                float(
                    latest[
                        "no2"
                    ].iloc[0]
                ),

                float(
                    latest[
                        "o3"
                    ].iloc[0]
                )
            ]
        }
    )


    pollutant_fig = px.bar(
        pollutant_data,
        x="Pollutant",
        y="Concentration",
        title=(
            "Current Pollutant Concentrations"
        )
    )


    pollutant_fig.update_yaxes(
        title=(
            "Concentration · µg/m³"
        )
    )


    pollutant_fig = style_chart(
        pollutant_fig
    )


    st.plotly_chart(
        pollutant_fig,
        use_container_width=True
    )


except Exception as graph_error:

    st.warning(
        "Some recent analytical visualizations "
        "could not be loaded."
    )

    st.caption(
        str(graph_error)
    )


# ============================================================
# LOCAL MODEL EXPLAINABILITY - SHAP
# ============================================================

st.markdown(
    "#### Local Model Explainability · SHAP"
)


try:

    model_24 = output[
        "model_24"
    ]


    X_current = prepare_features(
        model_24,
        output[
            "X"
        ].copy()
    )


    explainer = shap.TreeExplainer(
        model_24
    )


    shap_result = explainer(
        X_current
    )


    shap_values = np.asarray(
        shap_result.values[0]
    )


    shap_df = pd.DataFrame(
        {
            "Feature":
                X_current.columns,

            "SHAP Value":
                shap_values
        }
    )


    shap_df[
        "Absolute Importance"
    ] = (
        shap_df[
            "SHAP Value"
        ].abs()
    )


    shap_df = (
        shap_df
        .sort_values(
            "Absolute Importance",
            ascending=False
        )
        .head(12)
        .sort_values(
            "SHAP Value"
        )
    )


    shap_fig = px.bar(
        shap_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title=(
            "Top Features Influencing "
            "the +24 h AQI Prediction"
        )
    )


    shap_fig.update_layout(
        height=540
    )


    shap_fig = style_chart(
        shap_fig
    )


    st.plotly_chart(
        shap_fig,
        use_container_width=True
    )


    st.caption(
        "Positive SHAP values increase the predicted AQI. "
        "Negative values reduce the predicted AQI."
    )


except Exception as shap_error:

    st.info(
        "SHAP explanation is temporarily unavailable."
    )

    st.caption(
        str(shap_error)
    )


# ============================================================
# TECHNICAL FOOTER
# ============================================================

st.divider()


st.caption(
    "Air-quality data · Open-Meteo / CAMS   |   "
    "Feature Store & Model Registry · Hopsworks   |   "
    "Workflow orchestration · GitHub Actions   |   "
    "Web application · Streamlit"
)