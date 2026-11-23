import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  

import streamlit as st  

from ui import api_client  
from ui.tabs import drift_tab, predict_tab  

st.set_page_config(page_title="Purchase Intent", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1400px; padding-top: 2rem; }
    html, body, [class*="css"], .stMarkdown p { font-size: 20px; }
    label, [data-testid="stWidgetLabel"] p { font-size: 1.15rem !important; font-weight: 500; }
    [data-testid="stCaptionContainer"] p { font-size: 1.05rem !important; }
    div[data-testid="stMetricValue"] { font-size: 3.5rem; }
    div[data-testid="stMetricLabel"] p { font-size: 1.3rem !important; }
    input, div[data-baseweb="select"] { font-size: 1.1rem !important; }
    button[data-baseweb="tab"] p { font-size: 1.3rem !important; }
    [data-testid="stExpander"] summary p { font-size: 1.2rem !important; }
    button[kind="primaryFormSubmit"], button[kind="primary"] {
        font-size: 1.3rem; padding: 0.7rem 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Purchase intent predictor")
st.caption("Learning project on 2017 data from one shop. Business costs behind the "
           "recommendations are assumptions.")


def _sidebar() -> None:
    st.sidebar.header("API status")
    try:
        info = api_client.health()
    except api_client.ApiError as err:
        st.sidebar.error(str(err))
        return
    if info["status"] == "ok":
        st.sidebar.success(f"Model {info['model_version']} is running")
        st.sidebar.caption(f"Trained: {info.get('trained_at')}")
    else:
        st.sidebar.warning(info.get("detail") or "Model unavailable")


_sidebar()
predict, drift = st.tabs(["Predict", "Drift"])
with predict:
    predict_tab.render()
with drift:
    drift_tab.render()