import streamlit as st

BRAND_COLOR = "#E63946"
TEAL = "#2EC4B6"
DARK_BG = "#0D1117"
CARD_BG = "#161B22"


def apply_global_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1117 0%, #161B22 100%); border-right: 1px solid #30363D; }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    [data-testid="metric-container"] { background: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 1rem 1.25rem; }
    [data-testid="stDataFrame"] { border: 1px solid #30363D; border-radius: 8px; }
    .stButton > button { background: #E63946; color: white; border: none; border-radius: 8px; font-weight: 600; }
    #MainMenu, footer { visibility: hidden; }
    /* Ensure header with sidebar toggle is visible */
    [data-testid="stAppHeader"] { visibility: visible !important; }
    header { visibility: visible !important; }
    </style>
    """, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:12px'>
      <div style='font-size:28px'>{icon}</div>
      <div>
        <div style='font-size:20px;font-weight:700'>{title}</div>
        <div style='color:#9AA4B2'>{subtitle}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card_row(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(label=m["label"], value=m["value"], delta=m.get("delta"), delta_color=m.get("delta_color", "normal"))


def insight_card(text: str):
    st.markdown(f"""
    <div style='background:{CARD_BG};border-left:4px solid {BRAND_COLOR};padding:10px;border-radius:6px;margin:8px 0'>
      <p style='margin:0;color:#C9D1D9'>{text}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f"""
    <div style='font-weight:600;margin:8px 0 12px 0'>{title}</div>
    """, unsafe_allow_html=True)
