import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import requests
import os
import random
import json
from datetime import datetime, timedelta
from io import BytesIO
from vedic_engine import (
    compute_chart, ChartData, calculate_ashtakoota, get_year_prediction,
    calculate_varshphal, analyze_career, analyze_marriage, analyze_children, analyze_health,
    ZODIAC, ZODIAC_SHORT, SIGN_SANSKRIT, SIGN_LORD, HOUSE_MEANINGS, NAKSHATRAS,
    generate_demo_chart, load_chart_from_file, longitude_to_sign,
    SWISSEPH_AVAILABLE, NAKSHATRA_GANA, NAKSHATRA_NADI, NAKSHATRA_YONI
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Jyotish · Vedic Astrology",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SESSION STATE ────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "birth_name": "Native",
        "birth_date": datetime(1991, 4, 12),
        "birth_time": datetime.strptime("10:26", "%H:%M").time(),
        "birth_lat": 25.42,
        "birth_lon": 86.13,
        "birth_tz": 5.5,
        "birth_city": "",
        "computed_chart": None,
        "computed_chart_name": "",
        "ai_ctx": "",
        "last_shalaka": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ─── GEOCODING ────────────────────────────────────────────────────
@st.cache_resource
def get_geolocator():
    try:
        from geopy.geocoders import Nominatim
        return Nominatim(user_agent="vedic-astro-suite/3.1", timeout=10)
    except Exception:
        return None

def geocode_city(name: str):
    geo = get_geolocator()
    if not geo or not name.strip():
        return None
    try:
        loc = geo.geocode(name, language="en")
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    return None

# ─── TIMEZONE MAP ─────────────────────────────────────────────────
TIMEZONES = {
    "IST (India, UTC+5:30)": 5.5,
    "GMT / UTC (UTC+0:00)": 0.0,
    "BST (London, UTC+1:00)": 1.0,
    "EST (New York, UTC−5:00)": -5.0,
    "CST (Chicago, UTC−6:00)": -6.0,
    "MST (Denver, UTC−7:00)": -7.0,
    "PST (Los Angeles, UTC−8:00)": -8.0,
    "CET (Berlin, UTC+1:00)": 1.0,
    "JST (Tokyo, UTC+9:00)": 9.0,
    "AEST (Sydney, UTC+10:00)": 10.0,
    "Custom Offset": None,
}
TZ_KEYS = list(TIMEZONES.keys())

# ─── PALETTE ──────────────────────────────────────────────────────
INK      = "#1a1714"
INK_SOFT = "#5c5650"
INK_MUTE = "#a09890"
CREAM    = "#f9f6f2"
WARM     = "#f2ece4"
GOLD     = "#c9a84c"
RUST     = "#b85c2a"
SAGE     = "#5a7a6a"
BORDER   = "#e4ddd5"

# ─── GLOBAL CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: {INK};
    background: {CREAM} !important;
    -webkit-font-smoothing: antialiased;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: #ffffff !important;
    border-right: 1px solid {BORDER};
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 400;
    color: {INK_SOFT};
    padding: 0.35rem 0;
    cursor: pointer;
    transition: color 0.15s;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color: {INK};
}}

/* Typography */
h1 {{ font-family: 'Cormorant Garamond', serif !important; font-weight: 500 !important; font-size: 2.4rem !important; color: {INK} !important; letter-spacing: -0.02em; line-height: 1.1; }}
h2 {{ font-family: 'Cormorant Garamond', serif !important; font-weight: 500 !important; font-size: 1.7rem !important; color: {INK} !important; letter-spacing: -0.01em; }}
h3 {{ font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; font-size: 0.8rem !important; color: {INK_MUTE} !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }}
p {{ color: {INK_SOFT}; line-height: 1.65; font-size: 0.95rem; }}

/* Buttons */
.stButton > button {{
    background: {INK} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.55rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.03em;
    transition: opacity 0.15s !important;
    box-shadow: none !important;
}}
.stButton > button:hover {{
    opacity: 0.82 !important;
    transform: none !important;
    box-shadow: none !important;
}}
.stDownloadButton > button {{
    background: transparent !important;
    color: {INK} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
}}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div > input,
.stTimeInput > div > div > input {{
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    background: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 300 !important;
    color: {INK} !important;
    padding: 0.5rem 0.75rem !important;
    box-shadow: none !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: {INK_SOFT} !important;
    box-shadow: none !important;
}}
.stSelectbox > div > div > div {{
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    background: #fff !important;
    font-size: 0.875rem !important;
    font-weight: 300 !important;
    box-shadow: none !important;
}}

/* DataFrames */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    overflow: hidden;
}}

/* Divider */
hr {{ border: none; border-top: 1px solid {BORDER}; margin: 2rem 0; }}

/* Spinner */
.stSpinner > div {{ color: {INK_MUTE} !important; }}

/* Alerts */
.stAlert {{ border-radius: 4px !important; border-left-width: 3px !important; font-size: 0.875rem; }}

/* Label */
.stLabel, label {{ font-size: 0.8rem !important; font-weight: 400 !important; color: {INK_MUTE} !important; letter-spacing: 0.02em; }}

/* Toggle */
.stToggle label {{ font-size: 0.85rem !important; }}

/* Section spacing */
.block-container {{ padding: 2.5rem 2rem 4rem !important; max-width: 1100px; }}

/* Metric */
[data-testid="metric-container"] {{
    background: #fff;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 1rem 1.25rem;
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    font-size: 0.75rem !important;
    color: {INK_MUTE} !important;
    font-weight: 400 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
    color: {INK} !important;
    font-weight: 500 !important;
    line-height: 1.2;
}}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 1.5rem 0 1rem;">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; font-weight: 500; color: {INK}; letter-spacing: -0.01em;">◎ Jyotish</div>
        <div style="font-size: 0.72rem; color: {INK_MUTE}; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;">Vedic Astrology · v5.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("", [
        "Horoscope",
        "Matchmaking",
        "Yearly Predictions",
        "Varshphal",
        "AI Astrologer",
        "Ram Shalaka",
    ], label_visibility="collapsed")

    st.divider()

    st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem;">Settings</div>', unsafe_allow_html=True)
    use_demo = st.toggle("Demo mode", value=False, help="Use synthetic chart data without pyswisseph")
    api_key  = st.text_input("OpenRouter API key", type="password", placeholder="sk-or-…", label_visibility="collapsed")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    if not api_key:
        st.markdown(f'<div style="font-size:0.78rem; color:{INK_MUTE};">Enter key for AI features</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top: 3rem; font-size: 0.75rem; color: {INK_MUTE}; line-height: 1.7;">
        Tip: Use <em>City, State, Country</em> format<br>e.g. Sitamarhi, Bihar, India
    </div>
    """, unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────

def label(text):
    st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.3rem; margin-top:1rem;">{text}</div>', unsafe_allow_html=True)

def rule_row(title, detail, score, sev="neutral"):
    colors = {
        "positive": ("#d4edda", "#1a6031", "#0f4021"),
        "caution":  ("#fef3cd", "#7d5a00", "#5c4000"),
        "warning":  ("#fde2e2", "#8b1a1a", "#6b0f0f"),
        "neutral":  ("#f0ede8", "#5c5650", "#3a3530"),
    }
    bg, fc, tc = colors.get(sev, colors["neutral"])
    sign = "+" if score > 0 else ""
    st.markdown(f"""
    <div style="border:1px solid {BORDER}; border-radius:5px; padding:0.85rem 1rem; margin-bottom:0.6rem; background:#fff; display:flex; gap:1rem; align-items:flex-start;">
        <div style="flex:1;">
            <div style="font-size:0.875rem; font-weight:500; color:{INK}; margin-bottom:0.2rem;">{title}</div>
            <div style="font-size:0.82rem; color:{INK_SOFT}; line-height:1.5;">{detail}</div>
        </div>
        <div style="background:{bg}; color:{tc}; font-size:0.78rem; font-weight:500; padding:3px 10px; border-radius:20px; white-space:nowrap; align-self:center;">{sign}{score}</div>
    </div>
    """, unsafe_allow_html=True)

def render_fired_rules(fired_rules):
    if not fired_rules:
        st.markdown(f'<div style="color:{INK_MUTE}; font-size:0.875rem; padding:1rem 0;">No significant indicators found.</div>', unsafe_allow_html=True)
        return
    for r in fired_rules:
        rule_row(r["title"], r["detail"], r["score"], r.get("severity","neutral"))

def section_title(text, sub=""):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem; margin-top:0.5rem;">
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.6rem; font-weight:500; color:{INK}; letter-spacing:-0.01em;">{text}</div>
        {f'<div style="font-size:0.82rem; color:{INK_MUTE}; margin-top:3px;">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)

def pill(text, color=INK_MUTE):
    return f'<span style="background:{WARM}; color:{INK}; font-size:0.75rem; padding:3px 10px; border-radius:20px; border:1px solid {BORDER};">{text}</span>'

# ─── BIRTH INPUT FORM ─────────────────────────────────────────────
def birth_input_form(key_prefix: str):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Name", st.session_state["birth_name"], key=f"{key_prefix}_name", placeholder="Full name")
        st.session_state["birth_name"] = name
    with c2:
        dob = st.date_input("Date of birth", st.session_state["birth_date"], key=f"{key_prefix}_date")
        st.session_state["birth_date"] = dob
    with c3:
        tob = st.time_input("Time of birth", st.session_state["birth_time"], step=60, key=f"{key_prefix}_time")
        st.session_state["birth_time"] = tob

    cc1, cc2 = st.columns([4, 1])
    with cc1:
        city_name = st.text_input("City", st.session_state["birth_city"], key=f"{key_prefix}_city", placeholder="City, State, Country")
        st.session_state["birth_city"] = city_name
    with cc2:
        st.write("")
        if st.button("Find", key=f"{key_prefix}_find"):
            with st.spinner("Locating…"):
                coords = geocode_city(city_name)
                if coords:
                    st.session_state["birth_lat"] = round(coords[0], 4)
                    st.session_state["birth_lon"] = round(coords[1], 4)
                    st.session_state["birth_geo_ok"] = True
                    st.toast(f"✓ {coords[0]:.4f}, {coords[1]:.4f}")
                else:
                    st.error("City not found. Enter coordinates manually.")

    tz_col, lat_col, lon_col = st.columns([2, 1, 1])
    with tz_col:
        cur_tz = st.session_state["birth_tz"]
        tz_idx = 0
        for i, (k, v) in enumerate(TIMEZONES.items()):
            if v == cur_tz:
                tz_idx = i; break
        tz_choice = st.selectbox("Timezone", TZ_KEYS, index=tz_idx, key=f"{key_prefix}_tz")
        tz_val = TIMEZONES[tz_choice]
        if tz_val is None:
            tz_val = st.number_input("UTC offset", -12.0, 14.0, cur_tz, 0.5, key=f"{key_prefix}_tz_custom")
        st.session_state["birth_tz"] = tz_val
    with lat_col:
        lat = st.number_input("Lat", -90.0, 90.0, value=st.session_state["birth_lat"], key=f"{key_prefix}_lat")
        st.session_state["birth_lat"] = lat
    with lon_col:
        lon = st.number_input("Lon", -180.0, 180.0, value=st.session_state["birth_lon"], key=f"{key_prefix}_lon")
        st.session_state["birth_lon"] = lon

    return name, dob, tob, lat, lon, tz_val

# ─── CHART DRAWING ────────────────────────────────────────────────
PLANET_ABBR = {
    "Sun": "Su", "Moon": "Mo", "Mars": "Ma", "Mercury": "Me",
    "Jupiter": "Ju", "Venus": "Ve", "Saturn": "Sa", "Rahu": "Ra", "Ketu": "Ke"
}
PLANET_COLORS = {
    "Sun": "#c9a84c", "Moon": "#7a8fa6", "Mars": "#c04a2a",
    "Mercury": "#4a8a6a", "Jupiter": "#8a5a2a", "Venus": "#b04880",
    "Saturn": "#5a5a6a", "Rahu": "#5a4a8a", "Ketu": "#5a4a8a"
}

def draw_north_indian_chart(chart: ChartData, title=""):
    """Minimal, elegant North Indian diamond chart."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal"); ax.axis("off")

    # House outline coords
    house_polys = {
        1:  [(5,10),(7.5,7.5),(5,5),(2.5,7.5)],
        2:  [(7.5,7.5),(10,5),(7.5,2.5),(5,5)],
        3:  [(10,10),(10,5),(7.5,7.5)],
        4:  [(10,5),(10,0),(5,0),(7.5,2.5)],
        5:  [(7.5,2.5),(5,0),(2.5,2.5),(5,5)],
        6:  [(0,0),(5,0),(2.5,2.5)],
        7:  [(0,5),(2.5,2.5),(5,5),(2.5,7.5)],
        8:  [(0,10),(2.5,7.5),(0,5)],
        9:  [(0,10),(5,10),(2.5,7.5)],
        10: [(5,10),(10,10),(7.5,7.5)],
        11: [(0,5),(5,5),(2.5,2.5)],  # Note: adjusted
        12: [(0,0),(0,5),(2.5,2.5)],
    }
    # Fix missing corner triangles
    house_polys[3]  = [(10,10),(10,5),(7.5,7.5)]
    house_polys[6]  = [(0,0),(5,0),(2.5,2.5)]
    house_polys[8]  = [(0,10),(2.5,7.5),(0,5)]  # fix
    house_polys[9]  = [(0,10),(5,10),(2.5,7.5)]
    house_polys[10] = [(5,10),(10,10),(7.5,7.5)]
    house_polys[11] = [(5,5),(2.5,2.5),(0,5)]
    house_polys[12] = [(0,0),(0,5),(2.5,2.5)]

    house_centers = {
        1:  (5.0, 8.0), 2:  (7.7, 5.0), 3:  (9.0, 8.5),
        4:  (8.5, 1.5), 5:  (5.0, 2.0), 6:  (1.5, 1.0),
        7:  (2.3, 5.0), 8:  (1.0, 8.5), 9:  (1.5, 9.2),
        10: (8.5, 9.2), 11: (1.5, 4.0), 12: (0.8, 1.2),
    }
    # Better centers for corner triangles
    house_centers[3]  = (9.3, 8.3)
    house_centers[6]  = (1.3, 0.7)
    house_centers[8]  = (0.7, 7.5)
    house_centers[9]  = (1.8, 9.1)
    house_centers[10] = (8.2, 9.1)
    house_centers[11] = (1.5, 4.2)
    house_centers[12] = (0.7, 2.5)

    lagna_idx = ZODIAC.index(chart.lagna_sign)

    # Draw each house
    for h, verts in house_polys.items():
        poly = plt.Polygon(verts, closed=True,
                           facecolor=CREAM, edgecolor=BORDER, linewidth=0.8, zorder=1)
        ax.add_patch(poly)

    # Central diamond border
    diamond = plt.Polygon([(5,10),(10,5),(5,0),(0,5)], closed=True,
                          fill=False, edgecolor="#c8bfb4", linewidth=1.0, zorder=2)
    ax.add_patch(diamond)
    # Cross lines
    ax.plot([5,5],[0,10], color=BORDER, lw=0.7, zorder=2)
    ax.plot([0,10],[5,5], color=BORDER, lw=0.7, zorder=2)
    ax.plot([5,10],[10,5], color=BORDER, lw=0.5, ls="--", alpha=0.4, zorder=2)
    ax.plot([5,0],[10,5], color=BORDER, lw=0.5, ls="--", alpha=0.4, zorder=2)
    ax.plot([10,5],[5,0], color=BORDER, lw=0.5, ls="--", alpha=0.4, zorder=2)
    ax.plot([0,5],[5,0], color=BORDER, lw=0.5, ls="--", alpha=0.4, zorder=2)

    # Place sign label + planets per house
    house_planets = {i: [] for i in range(1, 13)}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        h = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        house_planets[h].append(p)

    for h in range(1, 13):
        cx, cy = house_centers[h]
        sign = ZODIAC[(lagna_idx + h - 1) % 12]
        short = ZODIAC_SHORT[(lagna_idx + h - 1) % 12]
        skt = SIGN_SANSKRIT[sign][:3]

        # Lagna marker
        if h == 1:
            ax.text(cx, cy + 0.6, "▲", ha="center", va="center",
                    fontsize=7, color=RUST, zorder=4)

        # House number (tiny, muted)
        ax.text(cx, cy + 0.32, str(h), ha="center", va="center",
                fontsize=6, color=INK_MUTE, zorder=4, fontweight="normal")

        # Sign short name
        ax.text(cx, cy, short, ha="center", va="center",
                fontsize=9, color=INK_SOFT, zorder=4, fontweight="normal",
                fontfamily="serif")

        # Sanskrit name (tiny)
        ax.text(cx, cy - 0.28, skt, ha="center", va="center",
                fontsize=5.5, color=INK_MUTE, zorder=4)

        # Planets
        planets_here = house_planets[h]
        if planets_here:
            n = len(planets_here)
            xs = np.linspace(cx - (n-1)*0.3, cx + (n-1)*0.3, n)
            for i, p in enumerate(planets_here):
                abbr = PLANET_ABBR.get(p, p[:2])
                retro = chart.retrograde.get(p, False)
                label_p = f"{'℞' if retro else ''}{abbr}"
                ax.text(xs[i], cy - 0.65, label_p, ha="center", va="center",
                        fontsize=7.5, color=PLANET_COLORS.get(p, INK),
                        zorder=5, fontweight="normal")

    if title:
        ax.set_title(title, fontsize=10, color=INK_SOFT, pad=8,
                     fontfamily="serif", fontweight="normal", style="italic")

    plt.tight_layout(pad=0.3)
    return fig

def draw_navamsa_chart(chart: ChartData):
    """Draw compact D9 Navamsa chart."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal"); ax.axis("off")

    # Simple 3×4 south-indian style grid for D9
    # Use north-indian diamond same as main but with navamsa data
    house_polys = {
        1:  [(5,10),(7.5,7.5),(5,5),(2.5,7.5)],
        2:  [(7.5,7.5),(10,5),(7.5,2.5),(5,5)],
        3:  [(10,10),(10,5),(7.5,7.5)],
        4:  [(10,5),(10,0),(5,0),(7.5,2.5)],
        5:  [(7.5,2.5),(5,0),(2.5,2.5),(5,5)],
        6:  [(0,0),(5,0),(2.5,2.5)],
        7:  [(0,5),(2.5,2.5),(5,5),(2.5,7.5)],
        8:  [(0,10),(2.5,7.5),(0,5)],
        9:  [(0,10),(5,10),(2.5,7.5)],
        10: [(5,10),(10,10),(7.5,7.5)],
        11: [(5,5),(2.5,2.5),(0,5)],
        12: [(0,0),(0,5),(2.5,2.5)],
    }
    house_centers = {
        1:(5.0,8.0), 2:(7.7,5.0), 3:(9.3,8.3),
        4:(8.5,1.5), 5:(5.0,2.0), 6:(1.3,0.7),
        7:(2.3,5.0), 8:(0.7,7.5), 9:(1.8,9.1),
        10:(8.2,9.1), 11:(1.5,4.2), 12:(0.7,2.5),
    }

    for h, verts in house_polys.items():
        poly = plt.Polygon(verts, closed=True,
                           facecolor=CREAM, edgecolor=BORDER, linewidth=0.8, zorder=1)
        ax.add_patch(poly)
    diamond = plt.Polygon([(5,10),(10,5),(5,0),(0,5)], closed=True,
                          fill=False, edgecolor="#c8bfb4", linewidth=1.0, zorder=2)
    ax.add_patch(diamond)
    ax.plot([5,5],[0,10], color=BORDER, lw=0.7, zorder=2)
    ax.plot([0,10],[5,5], color=BORDER, lw=0.7, zorder=2)
    for line in [([5,10],[10,5]),([5,0],[10,5]),([10,5],[5,0]),([0,5],[5,0])]:
        ax.plot(line[0], line[1], color=BORDER, lw=0.5, ls="--", alpha=0.4, zorder=2)

    # Navamsa planet placement
    nav_planets = {i: [] for i in range(1, 13)}
    nav_lagna = chart.navamsa.get("Lagna", chart.navamsa.get("Moon", "Aries"))
    nav_lagna_idx = ZODIAC.index(nav_lagna) if nav_lagna in ZODIAC else 0

    for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]:
        nav_sign = chart.navamsa.get(p, "Aries")
        if nav_sign in ZODIAC:
            h = ((ZODIAC.index(nav_sign) - nav_lagna_idx) % 12) + 1
        else:
            h = 1
        nav_planets[h].append(p)

    for h in range(1, 13):
        cx, cy = house_centers[h]
        sign = ZODIAC[(nav_lagna_idx + h - 1) % 12]
        short = ZODIAC_SHORT[(nav_lagna_idx + h - 1) % 12]

        ax.text(cx, cy + 0.32, str(h), ha="center", va="center",
                fontsize=6, color=INK_MUTE, zorder=4)
        ax.text(cx, cy, short, ha="center", va="center",
                fontsize=9, color=INK_SOFT, zorder=4, fontfamily="serif")

        planets_here = nav_planets[h]
        if planets_here:
            n = len(planets_here)
            xs = np.linspace(cx-(n-1)*0.3, cx+(n-1)*0.3, n)
            for i, p in enumerate(planets_here):
                ax.text(xs[i], cy-0.6, PLANET_ABBR.get(p,"?"),
                        ha="center", va="center", fontsize=7.5,
                        color=PLANET_COLORS.get(p, INK), zorder=5)

    ax.set_title("Navamsa · D9", fontsize=10, color=INK_SOFT, pad=8,
                 fontfamily="serif", fontweight="normal", style="italic")
    plt.tight_layout(pad=0.3)
    return fig

# ─── PLANET TABLE ────────────────────────────────────────────────
def planet_table(chart: ChartData) -> pd.DataFrame:
    rows = []
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]:
        sign, deg = longitude_to_sign(chart.planets[p])
        nak = chart.nakshatras[p]
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        retro = "℞" if chart.retrograde.get(p, False) else ""
        rows.append({
            "Planet": f"{p} {retro}".strip(),
            "Sign": sign,
            "°": f"{deg:.1f}",
            "House": house,
            "Nakshatra": nak["nakshatra"],
            "Pada": nak["pada"],
            "Lord": nak["lord"],
            "Navamsa": chart.navamsa.get(p,"—"),
            "Dignity": chart.dignities.get(p,"—"),
        })
    return pd.DataFrame(rows)

# ─── SAVE CHART ──────────────────────────────────────────────────
def save_chart_ui(chart: ChartData, name: str):
    c1, c2 = st.columns(2)
    with c1:
        js = json.dumps(chart.to_dict(), indent=2, ensure_ascii=False)
        st.download_button("↓ Chart JSON", js,
                           file_name=f"{name.replace(' ','_')}_chart.json",
                           mime="application/json", use_container_width=True)
    with c2:
        lines = [f"VEDIC CHART — {name}", "="*50]
        if chart.birth_date:
            lines.append(f"Birth: {chart.birth_date.strftime('%d %b %Y, %I:%M %p')}")
        lines += [
            f"Lagna: {chart.lagna_sign}  Moon: {chart.moon_sign}  Sun: {chart.sun_sign}",
            f"Nakshatra: {chart.nakshatras['Moon']['nakshatra']} pada {chart.nakshatras['Moon']['pada']}",
            "", "PLANETS:"
        ]
        for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]:
            sign, deg = longitude_to_sign(chart.planets[p])
            lines.append(f"  {p}: {sign} {deg:.1f}° — {chart.nakshatras[p]['nakshatra']} p{chart.nakshatras[p]['pada']}")
        current = chart.get_current_dasha_info()
        if current:
            lines += ["", "DASHA:",
                      f"  MD: {current['mahadasha']} ({current['mahadasha_start']}→{current['mahadasha_end']})",
                      f"  AD: {current['antardasha']} ({current['antardasha_start']}→{current['antardasha_end']})"]
        st.download_button("↓ Summary TXT", "\n".join(lines),
                           file_name=f"{name.replace(' ','_')}_summary.txt",
                           mime="text/plain", use_container_width=True)

def _muntha_interpretation(muntha: str) -> str:
    interp = {
        "Aries": "New beginnings, courage, self-development.",
        "Taurus": "Financial growth, stability, material comfort.",
        "Gemini": "Communication, learning, networking and travel.",
        "Cancer": "Emotional growth, family matters, home improvements.",
        "Leo": "Recognition, creativity, leadership opportunities.",
        "Virgo": "Health focus, service, analytical success.",
        "Libra": "Relationships, partnerships, balance and deals.",
        "Scorpio": "Transformation, research, hidden gains.",
        "Sagittarius": "Wisdom, travel, fortune and higher education.",
        "Capricorn": "Hard work, discipline, long-term career gains.",
        "Aquarius": "Innovation, social causes, technology.",
        "Pisces": "Spirituality, foreign connections, creative pursuits.",
    }
    return interp.get(muntha, "Mixed results — maintain balance and adaptability.")

# ═══════════════════════════════════════════════════════════════
# HOROSCOPE PAGE
# ═══════════════════════════════════════════════════════════════
if page == "Horoscope":
    section_title("Free Horoscope", "Sidereal chart · Lahiri Ayanamsa")

    if "loaded_chart_data" in st.session_state:
        if st.button("Use loaded chart"):
            try:
                data = st.session_state["loaded_chart_data"]
                chart = ChartData(
                    planets=data["planets"], ascendant=data["ascendant"],
                    lagna_sign=data["lagna_sign"],
                    birth_date=datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None,
                    lat=data.get("lat",0), lon=data.get("lon",0), tz=data.get("tz",0),
                    retrograde=data.get("retrograde",{})
                )
                st.session_state["computed_chart"] = chart
                st.session_state["computed_chart_name"] = data.get("name","Loaded Chart")
                st.success("Chart loaded.")
            except Exception as e:
                st.error(str(e))

    name, date, time, lat, lon, tz = birth_input_form("chart")

    col_btn, col_style = st.columns([1, 2])
    with col_btn:
        generate = st.button("Calculate chart", use_container_width=True)
    with col_style:
        chart_style = st.selectbox("Style", ["North Indian (Diamond)", "South Indian (Circular)"],
                                   label_visibility="collapsed")

    if generate:
        with st.spinner("Computing sidereal positions…"):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except RuntimeError as e:
                st.error(str(e)); st.stop()
            st.session_state["computed_chart"] = chart
            st.session_state["computed_chart_name"] = name
        if use_demo:
            st.info("Demo mode active — install pyswisseph for live ephemeris.")

    if st.session_state.get("computed_chart"):
        chart = st.session_state["computed_chart"]
        name  = st.session_state["computed_chart_name"]

        st.divider()
        save_chart_ui(chart, name)
        st.divider()

        # ── CHARTS side by side ──
        ch1, ch2 = st.columns(2)
        with ch1:
            if "North" in chart_style:
                fig = draw_north_indian_chart(chart, f"{name} · D1 Rashi")
            else:
                fig = draw_north_indian_chart(chart, f"{name} · D1 Rashi")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with ch2:
            fig2 = draw_navamsa_chart(chart)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.divider()

        # ── Summary row ──
        current = chart.get_current_dasha_info()
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Lagna", f"{chart.lagna_sign}")
            st.caption(SIGN_SANSKRIT[chart.lagna_sign])
        with s2:
            st.metric("Moon sign", f"{chart.moon_sign}")
            st.caption(SIGN_SANSKRIT[chart.moon_sign])
        with s3:
            st.metric("Nakshatra", chart.nakshatras["Moon"]["nakshatra"])
            st.caption(f"Pada {chart.nakshatras['Moon']['pada']}")
        with s4:
            st.metric("Mahadasha", current["mahadasha"] if current else "—")
            if current:
                st.caption(f"till {current['mahadasha_end']}")

        # ── Dasha detail ──
        if current:
            st.markdown(f"""
            <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1rem 1.25rem; margin:1rem 0;">
                <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em; color:{INK_MUTE}; margin-bottom:0.6rem;">Current Dasha</div>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
                    <div>
                        <div style="font-size:0.8rem; color:{INK_MUTE};">Mahadasha</div>
                        <div style="font-size:1rem; font-weight:500; color:{INK};">{current['mahadasha']}</div>
                        <div style="font-size:0.75rem; color:{INK_MUTE};">{current['mahadasha_start']} → {current['mahadasha_end']}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem; color:{INK_MUTE};">Antardasha</div>
                        <div style="font-size:1rem; font-weight:500; color:{INK};">{current['antardasha']}</div>
                        <div style="font-size:0.75rem; color:{INK_MUTE};">{current['antardasha_start']} → {current['antardasha_end']}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem; color:{INK_MUTE};">Pratyantardasha</div>
                        <div style="font-size:1rem; font-weight:500; color:{INK};">{current['pratyantardasha']}</div>
                        <div style="font-size:0.75rem; color:{INK_MUTE};">{current['pd_start']} → {current['pd_end']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Planetary positions</div>', unsafe_allow_html=True)
        st.dataframe(planet_table(chart), hide_index=True, use_container_width=True)

        st.divider()
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Divisional charts (Varga)</div>', unsafe_allow_html=True)
        varga = []
        for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]:
            varga.append({
                "Planet": p,
                "D1": longitude_to_sign(chart.planets[p])[0],
                "D9 Navamsa": chart.navamsa.get(p,"—"),
                "D3 Drekkana": chart.drekkana.get(p,"—"),
                "D7 Saptamsa": chart.saptamsa.get(p,"—"),
                "D10 Dasamsa": chart.dasamsa.get(p,"—"),
                "D12 Dwadasamsa": chart.dwadasamsa.get(p,"—"),
            })
        st.dataframe(pd.DataFrame(varga), hide_index=True, use_container_width=True)

        st.divider()
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Vimshottari Dasha timeline</div>', unsafe_allow_html=True)
        dasha_df = pd.DataFrame([{
            "Planet": p.planet,
            "Start": p.start_date.strftime("%d %b %Y"),
            "End": p.end_date.strftime("%d %b %Y"),
            "Years": f"{p.years:.1f}",
            "": "← now" if p.start_date <= datetime.now() < p.end_date else "",
        } for p in chart.dasha_periods])
        st.dataframe(dasha_df, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# MATCHMAKING PAGE
# ═══════════════════════════════════════════════════════════════
elif page == "Matchmaking":
    section_title("Ashtakoota Matchmaking", "36-point Koota compatibility analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Person 1 · Groom</div>', unsafe_allow_html=True)
        n1 = st.text_input("Name", "Person 1", key="m1_name")
        d1 = st.date_input("Date of birth", datetime(1990,1,1), key="m1_date")
        t1 = st.time_input("Time", datetime.strptime("08:00","%H:%M").time(), key="m1_time", step=60)
        lat1 = st.number_input("Lat", -90.0, 90.0, 25.42, key="m1_lat")
        lon1 = st.number_input("Lon", -180.0, 180.0, 86.13, key="m1_lon")
        tz1  = st.number_input("TZ offset", -12.0, 14.0, 5.5, key="m1_tz")
    with c2:
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Person 2 · Bride</div>', unsafe_allow_html=True)
        n2 = st.text_input("Name", "Person 2", key="m2_name")
        d2 = st.date_input("Date of birth", datetime(1992,6,15), key="m2_date")
        t2 = st.time_input("Time", datetime.strptime("10:30","%H:%M").time(), key="m2_time", step=60)
        lat2 = st.number_input("Lat", -90.0, 90.0, 28.61, key="m2_lat")
        lon2 = st.number_input("Lon", -180.0, 180.0, 77.20, key="m2_lon")
        tz2  = st.number_input("TZ offset", -12.0, 14.0, 5.5, key="m2_tz")

    if st.button("Calculate compatibility", use_container_width=True):
        with st.spinner("Matching Kootas…"):
            try:
                if use_demo:
                    chart1 = generate_demo_chart()
                    chart2 = generate_demo_chart()
                    chart2.planets = {k:(v+55)%360 for k,v in chart2.planets.items()}
                    chart2._compute_derived()
                else:
                    chart1 = compute_chart(d1.year,d1.month,d1.day,t1.hour,t1.minute,lat1,lon1,tz1)
                    chart2 = compute_chart(d2.year,d2.month,d2.day,t2.hour,t2.minute,lat2,lon2,tz2)
            except RuntimeError as e:
                st.error(str(e)); st.stop()

            res = calculate_ashtakoota(chart1, chart2)

        pct = res["percentage"]
        if pct >= 75: score_color = "#1a6031"; score_bg = "#d4edda"
        elif pct >= 50: score_color = "#7d5a00"; score_bg = "#fef3cd"
        else: score_color = "#8b1a1a"; score_bg = "#fde2e2"

        st.markdown(f"""
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:2rem; text-align:center; margin:1.5rem 0;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:3.5rem; font-weight:500; color:{INK}; line-height:1;">{res['total']}<span style="font-size:1.5rem; color:{INK_MUTE};">/36</span></div>
            <div style="display:inline-block; background:{score_bg}; color:{score_color}; font-size:0.875rem; font-weight:500; padding:6px 20px; border-radius:20px; margin-top:0.75rem;">{res['verdict']} · {pct}%</div>
            <div style="background:{WARM}; border-radius:4px; height:6px; width:60%; margin:1.25rem auto 0;">
                <div style="background:{score_color}; width:{pct}%; height:100%; border-radius:4px; transition:width 0.4s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        kootas = ["varna","vashya","tara","yoni","graha_maitri","gana","bhakoot","nadi"]
        names  = ["Varna","Vashya","Tara","Yoni","Graha Maitri","Gana","Bhakoot","Nadi"]
        maxes  = [1, 2, 3, 4, 5, 6, 7, 8]
        cols = st.columns(4)
        for idx, (k, label_k, mx) in enumerate(zip(kootas, names, maxes)):
            s = res[k]["score"]
            with cols[idx % 4]:
                c = "#1a6031" if s == mx else "#7d5a00" if s >= mx*0.5 else "#8b1a1a"
                st.markdown(f"""
                <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:0.85rem; margin-bottom:0.6rem; text-align:center;">
                    <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">{label_k}</div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; color:{c}; font-weight:500; line-height:1;">{s}<span style="font-size:0.9rem; color:{INK_MUTE};">/{mx}</span></div>
                    <div style="font-size:0.72rem; color:{INK_MUTE}; margin-top:4px;">{res[k]['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown(f"""
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1rem 1.25rem;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.85rem; color:{INK_SOFT};">
                <div><span style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Moon signs</span><br/>{chart1.moon_sign} · {chart2.moon_sign}</div>
                <div><span style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Nakshatras</span><br/>{chart1.nakshatras['Moon']['nakshatra']} · {chart2.nakshatras['Moon']['nakshatra']}</div>
                <div><span style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Gana</span><br/>{NAKSHATRA_GANA[chart1.nakshatras['Moon']['nakshatra']]} · {NAKSHATRA_GANA[chart2.nakshatras['Moon']['nakshatra']]}</div>
                <div><span style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Nadi</span><br/>{NAKSHATRA_NADI[chart1.nakshatras['Moon']['nakshatra']]} · {NAKSHATRA_NADI[chart2.nakshatras['Moon']['nakshatra']]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# YEARLY PREDICTIONS
# ═══════════════════════════════════════════════════════════════
elif page == "Yearly Predictions":
    section_title("Yearly Predictions", "Dasha · Transit · Varshphal synthesis")

    name, date, time, lat, lon, tz = birth_input_form("pred")

    c1, c2 = st.columns([1, 2])
    with c1:
        year = st.selectbox("Year", list(range(2024, 2036)))
    with c2:
        topic = st.segmented_control("Topic", ["Career","Marriage","Children","Health","All"], default="All")

    if st.button("Generate predictions", use_container_width=True):
        with st.spinner("Analysing Dasha, Transit & Varshphal…"):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except RuntimeError as e:
                st.error(str(e)); st.stop()
            pred = get_year_prediction(chart, year)

        dasha = pred["dasha"]
        varsh = pred.get("varshphal", {})
        sade  = pred.get("sade_sati", {})
        sade_txt = sade.get("phase","") if isinstance(sade, dict) else str(sade)

        st.markdown(f"""
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:1.25rem 1.5rem; margin:1rem 0;">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em; color:{INK_MUTE}; margin-bottom:0.75rem;">Year {year} · Overview</div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; font-size:0.875rem;">
                <div><div style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Dasha</div><div style="color:{INK}; font-weight:500; margin-top:2px;">{dasha.get('mahadasha','—')} / {dasha.get('antardasha','—')}</div></div>
                <div><div style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Transit</div><div style="color:{INK}; font-weight:500; margin-top:2px;">♄ {pred.get('transit_saturn','—')}  ♃ {pred.get('transit_jupiter','—')}</div></div>
                <div><div style="color:{INK_MUTE}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Muntha</div><div style="color:{INK}; font-weight:500; margin-top:2px;">{varsh.get('muntha_sign','—')}</div></div>
            </div>
            {f'<div style="margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid {BORDER}; font-size:0.82rem; color:#8b1a1a;">{sade_txt}</div>' if sade_txt else ''}
            <div style="margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid {BORDER}; font-size:0.82rem; color:{INK_SOFT}; line-height:1.6;">{pred.get('overall_summary','')[:400]}</div>
        </div>
        """, unsafe_allow_html=True)

        topics_to_show = ["Career","Marriage","Children","Health"] if topic=="All" else [topic]
        rating_color = {"Excellent":"#1a6031","Good":"#3a6031","Average":"#7d5a00","Challenging":"#8b1a1a"}

        for t in topics_to_show:
            data = pred[t.lower()]
            rc = rating_color.get(data["rating"], INK_SOFT)
            st.markdown(f"""
            <div style="border-top:2px solid {BORDER}; padding-top:1.25rem; margin-top:1.25rem;">
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.75rem;">
                    <div style="font-family:'Cormorant Garamond',serif; font-size:1.3rem; font-weight:500; color:{INK};">{t}</div>
                    <div style="font-size:0.8rem; color:{rc}; font-weight:500;">{data['rating']} · {'+' if data['net_score']>0 else ''}{data['net_score']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_fired_rules(data.get("fired_rules",[]))

        if api_key:
            st.divider()
            st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Ask AI about this prediction</div>', unsafe_allow_html=True)
            user_query = st.text_input("Question", placeholder="e.g. What remedies for career challenges?", label_visibility="collapsed")
            if user_query and st.button("Ask AI", use_container_width=True):
                with st.spinner("Consulting…"):
                    ctx = f"Lagna {chart.lagna_sign}, Moon {chart.moon_sign}, MD {dasha.get('mahadasha','')}, Year {year}. {pred.get('overall_summary','')[:400]}"
                    try:
                        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                            json={"model":"google/gemini-2.0-flash-lite-preview-02-05:free","messages":[
                                {"role":"system","content":"You are an expert Vedic Astrologer. Answer concisely and practically."},
                                {"role":"user","content":f"Context: {ctx}\n\nQuestion: {user_query}"}]}, timeout=30)
                        ai = r.json()["choices"][0]["message"]["content"]
                        st.markdown(f"""
                        <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1.25rem; margin-top:0.75rem;">
                            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem;">AI guidance</div>
                            <div style="font-size:0.875rem; color:{INK_SOFT}; line-height:1.7; white-space:pre-wrap;">{ai}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"AI error: {e}")

# ═══════════════════════════════════════════════════════════════
# VARSHPHAL
# ═══════════════════════════════════════════════════════════════
elif page == "Varshphal":
    section_title("Varshphal", "Tajaka · Annual Solar Return chart")

    name, date, time, lat, lon, tz = birth_input_form("varsh")
    year = st.selectbox("Year for Varshphal", list(range(2024, 2036)))

    if st.button("Calculate Varshphal", use_container_width=True):
        with st.spinner("Calculating Solar Return…"):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except RuntimeError as e:
                st.error(str(e)); st.stop()
            varsh = calculate_varshphal(chart, year)

        if not varsh:
            st.error("Unable to calculate Varshphal. Check birth data."); st.stop()

        st.markdown(f"""
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:1.5rem; margin:1rem 0;">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em; color:{INK_MUTE}; margin-bottom:1rem;">Solar Return {year}</div>
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem;">
                <div>
                    <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.06em;">Return date</div>
                    <div style="font-size:1rem; font-weight:500; color:{INK}; margin-top:3px;">{varsh.get('varshphal_date','—')}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.06em;">Muntha</div>
                    <div style="font-size:1rem; font-weight:500; color:{INK}; margin-top:3px;">{varsh.get('muntha_sign','—')}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.06em;">Varsha Lagna</div>
                    <div style="font-size:1rem; font-weight:500; color:{INK}; margin-top:3px;">{varsh.get('varsha_lagna','—')}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.06em;">Years elapsed</div>
                    <div style="font-size:1rem; font-weight:500; color:{INK}; margin-top:3px;">{varsh.get('years_elapsed','—')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        themes = varsh.get("themes", [])
        if themes:
            st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin:1rem 0 0.5rem;">Annual themes</div>', unsafe_allow_html=True)
            cols_t = st.columns(min(len(themes), 3))
            for i, theme in enumerate(themes):
                with cols_t[i % 3]:
                    st.markdown(f"""
                    <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:0.85rem; font-size:0.85rem; color:{INK_SOFT}; line-height:1.5;">
                        {theme}
                    </div>
                    """, unsafe_allow_html=True)

        muntha = varsh.get("muntha_sign","")
        if muntha:
            lord = SIGN_LORD.get(muntha, "—")
            interp = _muntha_interpretation(muntha)
            st.markdown(f"""
            <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1rem 1.25rem; margin-top:1rem;">
                <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Muntha in {muntha} · Lord {lord}</div>
                <div style="font-size:0.9rem; color:{INK_SOFT}; line-height:1.65;">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

        if api_key:
            st.divider()
            st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Ask about your Solar Return</div>', unsafe_allow_html=True)
            q = st.text_input("Question", placeholder="e.g. What does Muntha in Gemini mean for finances?", label_visibility="collapsed")
            if q and st.button("Ask AI about Varshphal", use_container_width=True):
                with st.spinner("Analysing…"):
                    ctx = f"{name}: Lagna {chart.lagna_sign}, Year {year}. Muntha {varsh.get('muntha_sign','')}, Varsha Lagna {varsh.get('varsha_lagna','')}. Themes: {', '.join(themes)}"
                    try:
                        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                            json={"model":"google/gemini-2.0-flash-lite-preview-02-05:free","messages":[
                                {"role":"system","content":"You are a Vedic astrologer specialising in Tajaka (Varshphal)."},
                                {"role":"user","content":f"Context: {ctx}\n\nQuestion: {q}"}]}, timeout=30)
                        ai = r.json()["choices"][0]["message"]["content"]
                        st.markdown(f"""
                        <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1.25rem; margin-top:0.75rem;">
                            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.6rem;">AI guidance</div>
                            <div style="font-size:0.875rem; color:{INK_SOFT}; line-height:1.7; white-space:pre-wrap;">{ai}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"AI error: {e}")

# ═══════════════════════════════════════════════════════════════
# AI ASTROLOGER
# ═══════════════════════════════════════════════════════════════
elif page == "AI Astrologer":
    section_title("AI Astrologer", "Powered by OpenRouter · Gemini")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Your birth data</div>', unsafe_allow_html=True)
        name, date, time, lat, lon, tz = birth_input_form("ai")
        if st.button("Load chart context", use_container_width=True):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
                cur = chart.get_current_dasha_info()
                st.session_state["ai_ctx"] = (
                    f"{name}: Lagna {chart.lagna_sign}, Moon {chart.moon_sign} "
                    f"in {chart.nakshatras['Moon']['nakshatra']} pada {chart.nakshatras['Moon']['pada']}, "
                    f"MD {cur.get('mahadasha','—')}, AD {cur.get('antardasha','—')}, PD {cur.get('pratyantardasha','—')}."
                )
                st.success("Context loaded.")
            except RuntimeError as e:
                st.error(str(e))

        if st.session_state.get("ai_ctx"):
            st.markdown(f"""
            <div style="background:{WARM}; border-radius:5px; padding:0.75rem; font-size:0.78rem; color:{INK_SOFT}; margin-top:0.5rem; line-height:1.6;">
                {st.session_state["ai_ctx"]}
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Your question</div>', unsafe_allow_html=True)
        question = st.text_area("", "What does my chart say about career in 2026?",
                                height=120, label_visibility="collapsed")
        if st.button("Ask Astrologer", use_container_width=True):
            if not api_key:
                st.error("Enter your OpenRouter API key in the sidebar.")
            elif not st.session_state.get("ai_ctx"):
                st.warning("Load chart context first (left panel).")
            else:
                with st.spinner("Consulting the stars…"):
                    try:
                        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                            json={"model":"google/gemini-2.0-flash-lite-preview-02-05:free","messages":[
                                {"role":"system","content":f"You are a wise Vedic Astrologer. Chart: {st.session_state['ai_ctx']}"},
                                {"role":"user","content":question}]}, timeout=30)
                        ans = r.json()["choices"][0]["message"]["content"]
                        st.markdown(f"""
                        <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1.5rem; margin-top:1rem;">
                            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Guidance</div>
                            <div style="font-size:0.9rem; color:{INK_SOFT}; line-height:1.8; white-space:pre-wrap;">{ans}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"API Error: {e}")

# ═══════════════════════════════════════════════════════════════
# RAM SHALAKA
# ═══════════════════════════════════════════════════════════════
elif page == "Ram Shalaka":
    section_title("Ram Shalaka", "Divine guidance from Shri Ram Charit Manas")

    SHALAKA = [
        {"text": "Sunu siya satya aseesa hamari, pujahi mana kamana tumhari",
         "meaning": "Your wish will be fulfilled by divine grace.", "type": "Auspicious"},
        {"text": "Prabishi nagara keeje saba kaaja, hridaya rakhi koushalapur raaja",
         "meaning": "Begin without fear — success and protection are assured.", "type": "Auspicious"},
        {"text": "Hoeehai soee jo rama rachi raakhaa, ko kari taraka badhaavai saakhaa",
         "meaning": "What is destined shall happen — do not overthink.", "type": "Neutral"},
        {"text": "Garala sudha ripu karahi mitaee, gopada sindhu anala sitalaee",
         "meaning": "Even enemies turn to friends; the impossible becomes possible.", "type": "Very Auspicious"},
        {"text": "Sakala sumangala daayaka raghunandana, sadhubara nindaaka aridata bandana",
         "meaning": "The Lord brings all auspiciousness and destroys suffering.", "type": "Auspicious"},
        {"text": "Bhagati heti mori kara puja, hoyi siddhi millaahi mahatuja",
         "meaning": "Worship with devotion — perfection and valor shall be attained.", "type": "Auspicious"},
        {"text": "Rama charana rati mori man mahi, basahu sadaa siya sahita sadahi",
         "meaning": "Devotion to Lord Ram's feet fills the heart with eternal joy.", "type": "Auspicious"},
        {"text": "Suni siya pati ke bachana suhaae, hridaya harasha gayatri guna gaae",
         "meaning": "Pleasing words bring joy — the heart rejoices and sings virtues.", "type": "Auspicious"},
    ]

    type_colors = {
        "Very Auspicious": (SAGE, "#1a3d30"),
        "Auspicious": (GOLD, "#5c3d00"),
        "Neutral": (INK_MUTE, "#3a3530"),
    }

    st.markdown(f"""
    <div style="text-align:center; padding:2rem 0 1.5rem;">
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.1rem; color:{INK_MUTE}; font-style:italic; margin-bottom:1.5rem;">Concentrate your mind, ask your question silently, then seek the blessing</div>
    </div>
    """, unsafe_allow_html=True)

    col_btn = st.columns([1,2,1])[1]
    with col_btn:
        if st.button("Seek Blessing", use_container_width=True):
            verse = random.choice(SHALAKA)
            st.session_state["last_shalaka"] = verse
            st.balloons()

    if st.session_state.get("last_shalaka"):
        verse = st.session_state["last_shalaka"]
        tc, tt = type_colors.get(verse["type"], (INK_MUTE, INK_SOFT))
        st.markdown(f"""
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:10px; padding:2.5rem; text-align:center; margin:1.5rem 0; max-width:640px; margin-left:auto; margin-right:auto;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:1.4rem; font-weight:500; color:{INK}; line-height:1.6; margin-bottom:1.25rem; font-style:italic;">"{verse['text']}"</div>
            <div style="font-size:0.95rem; color:{INK_SOFT}; line-height:1.7; margin-bottom:1.5rem;">{verse['meaning']}</div>
            <div style="display:inline-block; background:{WARM}; color:{INK}; font-size:0.72rem; font-weight:500; padding:5px 16px; border-radius:20px; letter-spacing:0.06em; text-transform:uppercase; border:1px solid {BORDER};">{verse['type']}</div>
        </div>
        """, unsafe_allow_html=True)

        if api_key:
            st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin:1rem 0 0.5rem;">Ask for deeper guidance</div>', unsafe_allow_html=True)
            follow_up = st.text_input("Question", placeholder="Ask about this shloka or seek a remedy…", label_visibility="collapsed")
            if follow_up and st.button("Receive guidance", use_container_width=True):
                with st.spinner("Reflecting…"):
                    prompt = f"Ram Shalaka verse: '{verse['text']}' meaning: {verse['meaning']}. User asks: {follow_up}. Provide compassionate, practical Vedic guidance."
                    try:
                        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                            json={"model":"google/gemini-2.0-flash-lite-preview-02-05:free","messages":[
                                {"role":"user","content":prompt}]}, timeout=30)
                        ans = r.json()["choices"][0]["message"]["content"]
                        st.markdown(f"""
                        <div style="background:#fff; border:1px solid {BORDER}; border-radius:6px; padding:1.25rem; margin-top:0.75rem; max-width:640px; margin-left:auto; margin-right:auto;">
                            <div style="font-size:0.875rem; color:{INK_SOFT}; line-height:1.75;">{ans}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════
# HOME — Load chart page (accessible from sidebar)
# ═══════════════════════════════════════════════════════════════
# (If none matched we show a clean home)
else:
    section_title("Jyotish", "Vedic Astrology Suite · v5.0")
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.5rem;">
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:1.25rem;">
            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Horoscope</div>
            <div style="font-size:0.9rem; color:{INK_SOFT};">Sidereal chart with Lahiri Ayanamsa, Nakshatra, Navamsa, Dasha & divisionals.</div>
        </div>
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:1.25rem;">
            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Matchmaking</div>
            <div style="font-size:0.9rem; color:{INK_SOFT};">Full Ashtakoota (36 points) with 8 Kootas & detailed verdict.</div>
        </div>
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:1.25rem;">
            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Yearly Predictions</div>
            <div style="font-size:0.9rem; color:{INK_SOFT};">Year-wise analysis for Career, Marriage, Children & Health with rule cards.</div>
        </div>
        <div style="background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:1.25rem;">
            <div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Varshphal</div>
            <div style="font-size:0.9rem; color:{INK_SOFT};">Annual Solar Return chart (Tajaka) with Muntha analysis.</div>
        </div>
    </div>
    <div style="margin-top:1.5rem; font-size:0.85rem; color:{INK_MUTE}; line-height:1.8;">
        Select a section from the sidebar to begin. Enable Demo mode if pyswisseph is not installed.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div style="font-size:0.72rem; color:{INK_MUTE}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.75rem;">Load a saved chart</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload chart JSON", type=["json"], label_visibility="collapsed")
    if uploaded:
        try:
            data = json.load(uploaded)
            st.session_state["loaded_chart_data"] = data
            st.success("Chart loaded. Go to Horoscope to view.")
        except Exception as e:
            st.error(f"Error loading chart: {e}")
