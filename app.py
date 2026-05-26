import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Vedic Astrology — Jyotish",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------
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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ------------------------------------------------------------------
# GEOCODING
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# TIMEZONE MAP
# ------------------------------------------------------------------
TIMEZONES = {
    "IST (India, UTC+5:30)": 5.5,
    "GMT / UTC (UTC+0:00)": 0.0,
    "BST (London, UTC+1:00)": 1.0,
    "EST (New York, UTC-5:00)": -5.0,
    "CST (Chicago, UTC-6:00)": -6.0,
    "MST (Denver, UTC-7:00)": -7.0,
    "PST (Los Angeles, UTC-8:00)": -8.0,
    "CET (Berlin, UTC+1:00)": 1.0,
    "JST (Tokyo, UTC+9:00)": 9.0,
    "AEST (Sydney, UTC+10:00)": 10.0,
    "AEDT (Sydney DST, UTC+11:00)": 11.0,
    "Custom Offset": None
}

TZ_KEYS = list(TIMEZONES.keys())

# ------------------------------------------------------------------
# IMPROVED CUSTOM CSS — LIGHT THEME WITH ELEGANT SPACING
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #2d2a26;
    background: #faf8f5 !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cinzel', serif !important;
    color: #6b2d0f !important;
    letter-spacing: 0.3px;
    font-weight: 700;
}
.stButton>button {
    background: linear-gradient(90deg, #c25e00 0%, #9a4a00 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 2px 8px rgba(154, 74, 0, 0.15);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(154, 74, 0, 0.25);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff8f0 0%, #fef3c7 100%) !important;
    border-right: 1px solid #e8d5b7;
}
.card {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(212, 175, 55, 0.25);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 20px rgba(107, 45, 15, 0.04);
    transition: all 0.2s;
}
.card-title {
    font-family: 'Cinzel', serif;
    color: #92400e;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #fcd34d;
    padding-bottom: 0.4rem;
    display: inline-block;
    font-weight: 700;
}
.metric-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fbbf24;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.score-excellent { color: #15803d; font-weight: 800; }
.score-good { color: #65a30d; font-weight: 700; }
.score-average { color: #ca8a04; font-weight: 700; }
.score-challenging { color: #b91c1c; font-weight: 700; }
hr { border-color: #d4af37 !important; opacity: 0.35; margin: 1.5rem 0; }
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div, .stDateInput>div>div>input {
    border-radius: 12px !important;
    border: 1px solid #e2c28b !important;
    background: white;
}
.stDownloadButton>button {
    background: linear-gradient(90deg, #059669 0%, #047857 100%) !important;
}
.badge-positive {
    background: #dcfce7;
    color: #166534;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    border: 1px solid #bbf7d0;
}
.badge-caution {
    background: #fef3c7;
    color: #92400e;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-warning {
    background: #fee2e2;
    color: #991b1b;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-neutral {
    background: #f3f4f6;
    color: #4b5563;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-natal {
    background: #eff6ff;
    color: #1e40af;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-dasha {
    background: #f3e8ff;
    color: #6b21a8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}
.rule-card {
    background: #ffffff;
    border-left: 5px solid #d97706;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.rule-card.positive { border-left-color: #16a34a; }
.rule-card.caution { border-left-color: #ca8a04; }
.rule-card.warning { border-left-color: #dc2626; }
.rule-card.neutral { border-left-color: #6b7280; }
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    border: 1px solid #f0e6d2;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.markdown("<h1 style='text-align:center; font-family:Cinzel; color:#92400e;'>🕉️ Jyotish</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#78350f;'>Vedic Astrology Suite v3.1</p>", unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "🏠 Home", "📜 Horoscope", "💑 Matchmaking",
    "🔮 Yearly Predictions", "📊 Varshphal", "❓ AI Astrologer", "🎲 Ram Shalaka"
])

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
use_demo = st.sidebar.toggle("Use Demo Data (no ephemeris)", value=False)
api_key = st.sidebar.text_input("OpenRouter API Key", type="password",
                                help="Optional. Free tier: google/gemini-2.0-flash-lite-preview-02-05:free")
if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#78350f; margin-top:2rem;">
<b>Tip:</b> Enter city as <i>City, State, Country</i> for best results.<br>
e.g. <i>Sitamarhi, Bihar, India</i> or <i>Muzaffarpur, Bihar, India</i>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# PERSISTED INPUT COMPONENT (with precise time input)
# ------------------------------------------------------------------
def birth_input_form(key_prefix: str, default_name: str):
    """Reusable birth data form with session state persistence and precise time (minute-level)."""
    ss_name = f"birth_name"
    ss_date = f"birth_date"
    ss_time = f"birth_time"
    ss_lat = f"birth_lat"
    ss_lon = f"birth_lon"
    ss_tz = f"birth_tz"
    ss_city = f"birth_city"
    ss_geo_ok = f"birth_geo_ok"

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("👤 Name", st.session_state[ss_name], key=f"{key_prefix}_name")
        st.session_state[ss_name] = name
    with c2:
        dob = st.date_input("📅 Date of Birth", st.session_state[ss_date], key=f"{key_prefix}_date")
        st.session_state[ss_date] = dob
    with c3:
        # Time input with step=60 seconds (1 minute) to allow precise minutes like 9:01, 9:03
        tob = st.time_input("🕒 Time of Birth", st.session_state[ss_time], step=60, key=f"{key_prefix}_time")
        st.session_state[ss_time] = tob

    st.markdown('<div class="card-title">📍 Birth Place</div>', unsafe_allow_html=True)
    city_col, btn_col = st.columns([4, 1])
    with city_col:
        city_name = st.text_input("City / Town (e.g., Sitamarhi, Muzaffarpur, Datia, Gwalior, Begusarai...)",
                                  st.session_state[ss_city], key=f"{key_prefix}_city",
                                  placeholder="Type city name and click Find")
        st.session_state[ss_city] = city_name
    with btn_col:
        st.write("")
        st.write("")
        find_clicked = st.button("🔍 Find", key=f"{key_prefix}_find", use_container_width=True)

    lat_key = f"{key_prefix}_lat_val"
    lon_key = f"{key_prefix}_lon_val"
    geo_ok_key = f"{key_prefix}_geo_ok"

    if find_clicked:
        with st.spinner("Locating..."):
            coords = geocode_city(city_name)
            if coords:
                st.session_state[ss_lat] = round(coords[0], 4)
                st.session_state[ss_lon] = round(coords[1], 4)
                st.session_state[ss_geo_ok] = True
                st.toast(f"✅ Found: {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                st.session_state[ss_geo_ok] = False
                st.error("❌ City not found. Please enter coordinates manually or try 'City, State, Country'.")

    tz_col, lat_col, lon_col = st.columns([2, 1, 1])
    with tz_col:
        current_tz = st.session_state[ss_tz]
        tz_index = 0
        for idx, (k, v) in enumerate(TIMEZONES.items()):
            if v == current_tz:
                tz_index = idx
                break
        tz_choice = st.selectbox("🌍 Timezone", TZ_KEYS, index=tz_index, key=f"{key_prefix}_tz")
        tz_val = TIMEZONES[tz_choice]
        if tz_val is None:
            tz_val = st.number_input("UTC Offset (+/- hrs)", -12.0, 14.0, current_tz, 0.5,
                                     key=f"{key_prefix}_tz_custom")
        st.session_state[ss_tz] = tz_val
    with lat_col:
        lat = st.number_input("Lat", -90.0, 90.0,
                            value=st.session_state[ss_lat],
                            key=lat_key)
        st.session_state[ss_lat] = lat
    with lon_col:
        lon = st.number_input("Lon", -180.0, 180.0,
                            value=st.session_state[ss_lon],
                            key=lon_key)
        st.session_state[ss_lon] = lon

    if st.session_state.get(ss_geo_ok):
        st.caption(f"✅ Coordinates locked: {lat:.4f}, {lon:.4f}")

    return name, dob, tob, lat, lon, tz_val

# ------------------------------------------------------------------
# CHART COMPUTATION HELPER
# ------------------------------------------------------------------
def get_or_compute_chart(key_prefix: str, default_name: str, force_recompute: bool = False):
    ss_chart = "computed_chart"
    ss_name = "computed_chart_name"

    if not force_recompute and st.session_state.get(ss_chart) is not None:
        return st.session_state[ss_chart], st.session_state.get(ss_name, default_name)

    name = st.session_state["birth_name"]
    date = st.session_state["birth_date"]
    time = st.session_state["birth_time"]
    lat = st.session_state["birth_lat"]
    lon = st.session_state["birth_lon"]
    tz = st.session_state["birth_tz"]

    with st.spinner("Calculating sidereal positions with Lahiri Ayanamsa..."):
        try:
            if use_demo:
                chart = generate_demo_chart()
                st.info("ℹ️ Demo mode active — install pyswisseph for live ephemeris.")
            else:
                chart = compute_chart(date.year, date.month, date.day,
                                      time.hour, time.minute, lat, lon, tz)
        except RuntimeError as e:
            st.error(f"❌ {e}")
            st.stop()

    st.session_state[ss_chart] = chart
    st.session_state[ss_name] = name
    return chart, name

# ------------------------------------------------------------------
# COMPACT NORTH INDIAN CHART — REDUCED SIZE
# ------------------------------------------------------------------
def draw_north_indian_chart(chart: ChartData, title: str):
    """Draw a compact, elegant North Indian diamond chart with smaller footprint."""
    fig, ax = plt.subplots(figsize=(7, 7))  # Reduced from 11x11
    fig.patch.set_facecolor('#faf8f5')
    ax.set_facecolor('#faf8f5')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Define diamond vertices
    diamond_verts = [(5, 10), (10, 5), (5, 0), (0, 5)]
    diamond = plt.Polygon(diamond_verts, fill=False, edgecolor='#a05a2c', linewidth=2, linestyle='-', alpha=0.7)
    ax.add_patch(diamond)
    
    # Draw the main cross lines
    ax.plot([5, 5], [10, 0], color='#a05a2c', linewidth=1, alpha=0.6)
    ax.plot([0, 10], [5, 5], color='#a05a2c', linewidth=1, alpha=0.6)
    ax.plot([2.5, 7.5], [7.5, 2.5], color='#a05a2c', linewidth=0.8, alpha=0.5, linestyle=':')
    ax.plot([2.5, 7.5], [2.5, 7.5], color='#a05a2c', linewidth=0.8, alpha=0.5, linestyle=':')
    
    # House positions (adjusted for compactness)
    houses_pos = {
        1:  (5.0, 8.2), 2:  (7.8, 7.8), 3:  (8.8, 5.0), 4:  (7.8, 2.2),
        5:  (5.0, 1.8), 6:  (2.2, 2.2), 7:  (1.2, 5.0), 8:  (2.2, 7.8),
        9:  (6.5, 6.5), 10: (6.5, 3.5), 11: (3.5, 3.5), 12: (3.5, 6.5),
    }
    
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    
    for house_num in range(1, 13):
        sign = ZODIAC[(lagna_idx + house_num - 1) % 12]
        short = ZODIAC_SHORT[(lagna_idx + house_num - 1) % 12]
        pos = houses_pos[house_num]
        ax.text(pos[0], pos[1] + 0.45, str(house_num), ha='center', va='center', fontsize=7, color='#9ca3af', fontweight='bold', alpha=0.7)
        ax.text(pos[0], pos[1], short, ha='center', va='center', fontsize=11, color='#6b2d0f', fontweight='bold')
        ax.text(pos[0], pos[1] - 0.4, SIGN_SANSKRIT[sign][:4], ha='center', va='center', fontsize=6, color='#b45309', alpha=0.8)
    
    planet_symbols = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
                      "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋"}
    planet_colors = {"Sun": "#d97706", "Moon": "#6b7280", "Mars": "#dc2626",
                     "Mercury": "#059669", "Jupiter": "#92400e", "Venus": "#db2777",
                     "Saturn": "#4b5563", "Rahu": "#7c3aed", "Ketu": "#7c3aed"}
    
    house_planets = {i: [] for i in range(1, 13)}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        house_planets[house].append(p)
    
    for house_num, planets in house_planets.items():
        if not planets:
            continue
        pos = houses_pos[house_num]
        n = len(planets)
        start_x = pos[0] - (n-1)*0.22
        for i, p in enumerate(planets):
            ax.text(start_x + i*0.44, pos[1] - 0.85, planet_symbols.get(p, p),
                    ha='center', va='center', fontsize=12, color=planet_colors.get(p, '#1f2937'), fontweight='bold')
    
    ax.text(5.0, 9.0, "LAGNA", ha='center', va='center', fontsize=8, color='#dc2626', fontweight='bold', alpha=0.9)
    ax.plot(5, 9.2, marker='^', color='#dc2626', markersize=6, alpha=0.8)
    
    ax.set_title(title, fontsize=14, color='#6b2d0f', fontweight='bold', pad=15, fontfamily='serif')
    plt.tight_layout(pad=0.5)
    return fig

# ------------------------------------------------------------------
# COMPACT CIRCULAR CHART — REDUCED SIZE
# ------------------------------------------------------------------
def draw_circular_chart(chart: ChartData, title: str):
    """Compact circular South Indian style chart."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))  # Reduced from 10x10
    fig.patch.set_facecolor('#faf8f5')
    ax.set_facecolor('#faf8f5')
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    sign_colors = ['#fff4e6', '#fef3c7', '#ffedd5', '#fef9e3', '#fff7ed', '#fefce8',
                   '#fff4e6', '#fef3c7', '#ffedd5', '#fef9e3', '#fff7ed', '#fefce8']
    for i in range(12):
        theta_start = np.radians(i*30)
        theta_end = np.radians((i+1)*30)
        ax.fill_between(np.linspace(theta_start, theta_end, 30), 0.3, 1.0,
                        color=sign_colors[i % len(sign_colors)], alpha=0.7, edgecolor='#d4af37', linewidth=0.5)
        angle = np.radians(i*30 + 15)
        ax.text(angle, 0.92, f"{ZODIAC[i]}\n{SIGN_SANSKRIT[ZODIAC[i]]}",
                ha='center', va='center', fontsize=6.5, color='#92400e', fontweight='bold')
    
    symbols = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
               "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋"}
    colors_p = {"Sun": "#d97706", "Moon": "#6b7280", "Mars": "#dc2626",
                "Mercury": "#059669", "Jupiter": "#92400e", "Venus": "#db2777",
                "Saturn": "#4b5563", "Rahu": "#7c3aed", "Ketu": "#7c3aed"}
    
    used_bins = {}
    for planet, lon in chart.planets.items():
        base = lon % 360
        bin_id = int(base / 6)
        offset = used_bins.get(bin_id, 0) * 0.06
        used_bins[bin_id] = used_bins.get(bin_id, 0) + 1
        angle = np.radians(base)
        dist = 0.55 + offset
        ax.text(angle, dist, symbols.get(planet, planet), fontsize=11,
                ha='center', va='center', color=colors_p.get(planet, '#1f2937'),
                fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
    
    asc_angle = np.radians(chart.ascendant)
    ax.plot([asc_angle, asc_angle], [0.3, 1.0], color='#dc2626', linewidth=2, linestyle='--', alpha=0.8)
    ax.text(asc_angle, 1.02, 'ASC ▲', ha='center', va='center', color='#dc2626', fontsize=8, fontweight='bold')
    
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_title(title, fontsize=14, color='#6b2d0f', fontweight='bold', pad=20, fontfamily='serif')
    plt.tight_layout(pad=0.5)
    return fig

def planet_table(chart: ChartData):
    rows = []
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
        sign, deg = longitude_to_sign(chart.planets[p])
        nak = chart.nakshatras[p]
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        rows.append({
            "Planet": p,
            "Sign": f"{sign} ({SIGN_SANSKRIT[sign]})",
            "Deg": f"{deg:.2f}°",
            "House": house,
            "Nakshatra": nak["nakshatra"],
            "Pada": nak["pada"],
            "Lord": nak["lord"],
            "Navamsa": chart.navamsa[p],
            "Dignity": chart.dignities.get(p, "")
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# SAVE CHART
# ------------------------------------------------------------------
def save_chart_ui(chart: ChartData, name: str):
    col1, col2 = st.columns(2)
    with col1:
        chart_json = json.dumps(chart.to_dict(), indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Download Chart (JSON)",
            data=chart_json,
            file_name=f"{name.replace(' ', '_')}_chart.json",
            mime="application/json",
            use_container_width=True
        )
    with col2:
        summary = f"""VEDIC CHART — {name}
{'='*50}
Birth: {chart.birth_date.strftime('%d %b %Y, %I:%M %p') if chart.birth_date else 'Unknown'}
Location: {chart.lat:.4f}°N, {chart.lon:.4f}°E

LAGNA: {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})
MOON SIGN: {chart.moon_sign} ({SIGN_SANSKRIT[chart.moon_sign]})
SUN SIGN: {chart.sun_sign} ({SIGN_SANSKRIT[chart.sun_sign]})
NAKSHATRA: {chart.nakshatras['Moon']['nakshatra']} (Pada {chart.nakshatras['Moon']['pada']})

CURRENT DASHA:
"""
        current = chart.get_current_dasha_info()
        if current:
            summary += f"MD: {current['mahadasha']} ({current['mahadasha_start']} to {current['mahadasha_end']})\n"
            summary += f"AD: {current['antardasha']} ({current['antardasha_start']} to {current['antardasha_end']})\n"
            summary += f"PD: {current['pratyantardasha']} ({current['pd_start']} to {current['pd_end']})\n"

        summary += "\nPLANETARY POSITIONS:\n"
        for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
            sign, deg = longitude_to_sign(chart.planets[p])
            summary += f"{p}: {sign} {deg:.2f}° — {chart.nakshatras[p]['nakshatra']} {chart.nakshatras[p]['pada']}\n"

        st.download_button(
            label="📄 Download Summary (TXT)",
            data=summary,
            file_name=f"{name.replace(' ', '_')}_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

# ------------------------------------------------------------------
# RULE-BASED PREDICTION DISPLAY
# ------------------------------------------------------------------
def render_fired_rules(fired_rules):
    if not fired_rules:
        st.info("No significant planetary indicators found for this topic.")
        return

    for rule in fired_rules:
        severity = rule.get("severity", "neutral")
        activation = rule.get("activation", "natal")
        css_class = severity
        score_color = "#16a34a" if rule["score"] > 0 else "#dc2626" if rule["score"] < 0 else "#6b7280"
        score_sign = "+" if rule["score"] > 0 else ""

        activation_badge = f'<span class="badge-dasha">⚡ DASHA</span>' if activation == "dasha_activated" else f'<span class="badge-natal">★ NATAL</span>'
        severity_badge = {
            "positive": '<span class="badge-positive">✓ POSITIVE</span>',
            "neutral": '<span class="badge-neutral">◈ NEUTRAL</span>',
            "caution": '<span class="badge-caution">⚠ CAUTION</span>',
            "warning": '<span class="badge-warning">✕ WARNING</span>',
        }.get(severity, '')

        st.markdown(f"""
        <div class="rule-card {css_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-weight:700; color:#1f2937; font-size:0.95rem;">{rule['title']}</span>
                <span style="font-weight:700; color:{score_color}; font-size:0.9rem;">{score_sign}{rule['score']}</span>
            </div>
            <div style="margin-bottom:0.4rem;">
                {severity_badge} {activation_badge}
            </div>
            <p style="color:#4b5563; font-size:0.88rem; line-height:1.5; margin:0;">{rule['detail']}</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------
def _muntha_interpretation(muntha: str, lagna: str) -> str:
    interpretations = {
        "Aries": "Year of new beginnings, courage, and initiative. Focus on self-development.",
        "Taurus": "Year of financial growth, stability, and material comfort. Good for investments.",
        "Gemini": "Year of communication, learning, and networking. Travel indicated.",
        "Cancer": "Year of emotional growth, family matters, and nurturing. Home improvements.",
        "Leo": "Year of recognition, creativity, and authority. Leadership opportunities.",
        "Virgo": "Year of health focus, service, and detailed work. Analytical success.",
        "Libra": "Year of relationships, partnerships, and balance. Marriage/business deals.",
        "Scorpio": "Year of transformation, research, and hidden gains. Occult interests.",
        "Sagittarius": "Year of wisdom, travel, and fortune. Higher education success.",
        "Capricorn": "Year of hard work, discipline, and career advancement. Long-term gains.",
        "Aquarius": "Year of innovation, social causes, and unconventional success. Technology.",
        "Pisces": "Year of spirituality, foreign connections, and intuition. Creative pursuits."
    }
    return interpretations.get(muntha, "Mixed results — maintain balance and adaptability.")

if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="font-size:3rem; color:#6b2d0f;">🕉️ Vedic Astrology Suite</h1>
        <p style="font-size:1.25rem; color:#78350f;">Jyotish — Ancient Wisdom, Modern Precision</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">📜</div>
            <h4>Horoscope</h4>
            <p style="font-size:0.9rem;">Sidereal chart with Lahiri Ayanamsa, Nakshatra, Navamsa, Dasha & divisionals.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">💑</div>
            <h4>Matchmaking</h4>
            <p style="font-size:0.9rem;">Full Ashtakoota (36 points) with 8 Kootas & detailed verdict.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">🔮</div>
            <h4>Predictions</h4>
            <p style="font-size:0.9rem;">Year-wise analysis for Career, Marriage, Children & Health.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">📊</div>
            <h4>Varshphal</h4>
            <p style="font-size:0.9rem;">Annual Solar Return chart (Tajaka) with Muntha analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
        <h4 class="card-title">✨ What's New in v3.1</h4>
        <ul>
            <li><b>Context-Aware Dasha Weighting:</b> Dasha rules boosted when MD planet matches topic lord</li>
            <li><b>D9/D10 Dignity Integration:</b> Navamsa & Dasamsa dignity now affects marriage/career scoring</li>
            <li><b>Full-Year Transit Averaging:</b> Jan + Jun + Dec averaged for more accurate yearly predictions</li>
            <li><b>Active Yoga Tagging:</b> Natal vs Dasha-activated rules clearly distinguished</li>
            <li><b>Redesigned North Indian Chart:</b> Proper diamond grid with correct house geometry</li>
            <li><b>Pratyantardasha Display:</b> PD level now visible in dasha card</li>
            <li><b>Persistent Birth Details:</b> Your birth data is remembered across all pages</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📂 Load Saved Chart")
    uploaded = st.file_uploader("Upload previously saved chart (JSON)", type=["json"])
    if uploaded:
        try:
            data = json.load(uploaded)
            st.session_state["loaded_chart_data"] = data
            st.success("Chart loaded! Go to Horoscope page to view.")
        except Exception as e:
            st.error(f"Error loading chart: {e}")

elif page == "📜 Horoscope":
    st.title("📜 Free Horoscope Chart")

    if "loaded_chart_data" in st.session_state:
        if st.button("📂 Use Loaded Chart", use_container_width=True):
            try:
                data = st.session_state["loaded_chart_data"]
                chart = ChartData(
                    planets=data["planets"],
                    ascendant=data["ascendant"],
                    lagna_sign=data["lagna_sign"],
                    birth_date=datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None,
                    lat=data.get("lat", 0),
                    lon=data.get("lon", 0),
                    tz=data.get("tz", 0)
                )
                st.session_state["computed_chart"] = chart
                st.session_state["computed_chart_name"] = data.get("name", "Loaded Chart")
                st.success("Chart loaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    name, date, time, lat, lon, tz = birth_input_form("chart", "Native")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✨ Generate Chart", use_container_width=True):
            with st.spinner("Calculating sidereal positions with Lahiri Ayanamsa..."):
                try:
                    if use_demo:
                        chart = generate_demo_chart()
                        st.info("ℹ️ Demo mode active — install pyswisseph for live ephemeris.")
                    else:
                        chart = compute_chart(date.year, date.month, date.day,
                                              time.hour, time.minute, lat, lon, tz)
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    st.stop()

                st.session_state["computed_chart"] = chart
                st.session_state["computed_chart_name"] = name

    with c2:
        chart_style = st.selectbox("Chart Style", ["North Indian (Diamond)", "South Indian (Circular)"])

    if st.session_state.get("computed_chart") is not None:
        chart = st.session_state["computed_chart"]
        name = st.session_state["computed_chart_name"]

        st.divider()
        save_chart_ui(chart, name)

        st.divider()
        if "North" in chart_style:
            fig = draw_north_indian_chart(chart, f"{name}'s Horoscope (D1)")
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig = draw_circular_chart(chart, f"{name}'s Horoscope (D1)")
            st.pyplot(fig)
            plt.close(fig)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🌟 Birth Summary</div>
                <p><b>Lagna:</b> {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})</p>
                <p><b>Moon Sign:</b> {chart.moon_sign} ({SIGN_SANSKRIT[chart.moon_sign]})</p>
                <p><b>Sun Sign:</b> {chart.sun_sign} ({SIGN_SANSKRIT[chart.sun_sign]})</p>
                <p><b>Nakshatra:</b> {chart.nakshatras['Moon']['nakshatra']} (Pada {chart.nakshatras['Moon']['pada']})</p>
                <p><b>Navamsa Lagna:</b> {chart.navamsa['Moon']}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            current = chart.get_current_dasha_info()
            if current:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">📖 Current Dasha</div>
                    <p><b>MD:</b> {current['mahadasha']}</p>
                    <p><small>{current['mahadasha_start']} → {current['mahadasha_end']}</small></p>
                    <p><b>AD:</b> {current['antardasha']}</p>
                    <p><small>{current['antardasha_start']} → {current['antardasha_end']}</small></p>
                    <p><b>PD:</b> {current['pratyantardasha']}</p>
                    <p><small>{current['pd_start']} → {current['pd_end']}</small></p>
                </div>
                """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📍 Birth Details</div>
                <p><b>Date:</b> {chart.birth_date.strftime('%d %b %Y') if chart.birth_date else 'N/A'}</p>
                <p><b>Time:</b> {chart.birth_date.strftime('%I:%M %p') if chart.birth_date else 'N/A'}</p>
                <p><b>Lat:</b> {chart.lat:.4f}°</p>
                <p><b>Lon:</b> {chart.lon:.4f}°</p>
                <p><b>TZ:</b> UTC{chart.tz:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("🪐 Planetary Positions")
        st.dataframe(planet_table(chart), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("📊 Divisional Charts (Varga)")
        varga_data = []
        for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
            varga_data.append({
                "Planet": p,
                "D1 (Rashi)": longitude_to_sign(chart.planets[p])[0],
                "D9 (Navamsa)": chart.navamsa[p],
                "D3 (Drekkana)": chart.drekkana[p],
                "D7 (Saptamsa)": chart.saptamsa[p],
                "D10 (Dasamsa)": chart.dasamsa[p],
                "D12 (Dwadasamsa)": chart.dwadasamsa[p]
            })
        st.dataframe(pd.DataFrame(varga_data), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("📅 Vimshottari Dasha Timeline")
        dasha_df = pd.DataFrame([
            {
                "Planet": p.planet,
                "Start": p.start_date.strftime("%d %b %Y"),
                "End": p.end_date.strftime("%d %b %Y"),
                "Years": f"{p.years:.2f}",
                "Status": "✅ Current" if p.start_date <= datetime.now() < p.end_date else ""
            }
            for p in chart.dasha_periods
        ])
        st.dataframe(dasha_df, hide_index=True, use_container_width=True)

elif page == "💑 Matchmaking":
    st.title("💑 Ashtakoota Matchmaking")
    st.caption("36-point Koota compatibility analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-title">👤 Person 1 (Groom)</div>', unsafe_allow_html=True)
        n1, d1, t1, lat1, lon1, tz1 = birth_input_form("m1", "Person 1")
    with c2:
        st.markdown('<div class="card-title">👤 Person 2 (Bride)</div>', unsafe_allow_html=True)
        n2, d2, t2, lat2, lon2, tz2 = birth_input_form("m2", "Person 2")

    if st.button("💞 Calculate Compatibility", use_container_width=True):
        with st.spinner("Matching the 8 Kootas..."):
            try:
                if use_demo:
                    chart1 = generate_demo_chart()
                    chart2 = generate_demo_chart()
                    chart2.planets = {k: (v + 55) % 360 for k, v in chart2.planets.items()}
                    chart2._compute_derived()
                else:
                    chart1 = compute_chart(d1.year, d1.month, d1.day, t1.hour, t1.minute, lat1, lon1, tz1)
                    chart2 = compute_chart(d2.year, d2.month, d2.day, t2.hour, t2.minute, lat2, lon2, tz2)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            res = calculate_ashtakoota(chart1, chart2)

            cls = f"score-{res['verdict'].lower().replace(' ', '-')}"
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h2 style="margin-bottom:0.2rem;">{res['total']} <span style="font-size:1rem; color:#78350f;">/ 36</span></h2>
                <h1 class="{cls}" style="margin-top:0;">{res['verdict']} — {res['percentage']}%</h1>
                <div style="background:#e7e5e4; border-radius:8px; height:12px; width:70%; margin:1rem auto;">
                    <div style="background:linear-gradient(90deg, #d97706, #fcd34d); width:{res['percentage']}%; height:100%; border-radius:8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            kootas = ["varna", "vashya", "tara", "yoni", "graha_maitri", "gana", "bhakoot", "nadi"]
            names = ["Varna", "Vashya", "Tara", "Yoni", "Graha Maitri", "Gana", "Bhakoot", "Nadi"]
            k_cols = st.columns(4)
            for idx, (k, label) in enumerate(zip(kootas, names)):
                with k_cols[idx % 4]:
                    score = res[k]['score']
                    mx = res[k]['max']
                    color = "#15803d" if score == mx else "#65a30d" if score >= mx*0.6 else "#ca8a04" if score > 0 else "#b91c1c"
                    st.markdown(f"""
                    <div class="metric-box" style="margin-bottom:0.8rem;">
                        <div style="font-size:0.85rem; color:#78350f; font-weight:600;">{label}</div>
                        <div style="font-size:1.4rem; font-weight:700; color:{color};">{score}/{mx}</div>
                        <div style="font-size:0.75rem; color:#57534e;">{res[k]['detail']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <div class="card-title">📖 What each Koota means</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size:0.95rem;">
                    <div><b>Varna (1):</b> Spiritual compatibility</div>
                    <div><b>Vashya (2):</b> Mutual attraction</div>
                    <div><b>Tara (3):</b> Destiny alignment</div>
                    <div><b>Yoni (4):</b> Intimacy harmony</div>
                    <div><b>Graha Maitri (5):</b> Planetary friendship</div>
                    <div><b>Gana (6):</b> Temperament match</div>
                    <div><b>Bhakoot (7):</b> Relative Moon position</div>
                    <div><b>Nadi (8):</b> Health & progeny</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            st.subheader("📋 Detailed Analysis")
            st.markdown(f"""
            <div class="card">
                <p><b>Moon Sign Compatibility:</b> {chart1.moon_sign} vs {chart2.moon_sign}</p>
                <p><b>Nakshatra Compatibility:</b> {chart1.nakshatras['Moon']['nakshatra']} vs {chart2.nakshatras['Moon']['nakshatra']}</p>
                <p><b>Gana:</b> {NAKSHATRA_GANA[chart1.nakshatras['Moon']['nakshatra']]} vs {NAKSHATRA_GANA[chart2.nakshatras['Moon']['nakshatra']]}</p>
                <p><b>Nadi:</b> {NAKSHATRA_NADI[chart1.nakshatras['Moon']['nakshatra']]} vs {NAKSHATRA_NADI[chart2.nakshatras['Moon']['nakshatra']]}</p>
                <p><b>Yoni:</b> {NAKSHATRA_YONI[chart1.nakshatras['Moon']['nakshatra']]} vs {NAKSHATRA_YONI[chart2.nakshatras['Moon']['nakshatra']]}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "🔮 Yearly Predictions":
    st.title("🔮 Yearly Predictions by Topic")
    name, date, time, lat, lon, tz = birth_input_form("pred", "Native")

    c1, c2 = st.columns([1, 2])
    with c1:
        year = st.selectbox("📅 Select Year", list(range(2024, 2036)))
    with c2:
        topic = st.segmented_control("Topic", ["Career", "Marriage", "Children", "Health", "All"], default="All")

    if st.button("🔮 Predict", use_container_width=True):
        with st.spinner("Analyzing Dasha, Transit & Varshphal..."):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            pred = get_year_prediction(chart, year)

            st.markdown(f"""
            <div class="card">
                <div class="card-title">📅 {pred['year']} — Year Summary</div>
                <p><b>Mahadasha:</b> {pred['dasha'].get('mahadasha', 'N/A')} | <b>Antardasha:</b> {pred['dasha'].get('antardasha', 'N/A')} | <b>Pratyantardasha:</b> {pred['dasha'].get('pratyantardasha', 'N/A')}</p>
                <p><b>Transit Saturn:</b> {pred.get('transit_saturn', 'N/A')} | <b>Transit Jupiter:</b> {pred.get('transit_jupiter', 'N/A')}</p>
                <p><b>Muntha:</b> {pred['varshphal'].get('muntha_sign', 'N/A')} | <b>Themes:</b> {', '.join(pred['varshphal'].get('themes', []))}</p>
                <p style="color:#b91c1c;"><b>{pred.get('sade_sati', {}).get('phase', '') if isinstance(pred.get('sade_sati'), dict) else pred.get('sade_sati', '')}</b></p>
                <p><i>{pred.get('summary', '')}</i></p>
            </div>
            """, unsafe_allow_html=True)

            topics_to_show = ["Career", "Marriage", "Children", "Health"] if topic == "All" else [topic]

            for t in topics_to_show:
                data = pred[t.lower()]
                st.markdown(f"""
                <div class="card" style="border-left: 6px solid #d97706;">
                    <div class="card-title">🔮 {t} Analysis — {data['rating']} (Score: {data['net_score']:+d})</div>
                """, unsafe_allow_html=True)

                render_fired_rules(data.get('fired_rules', []))

                st.markdown("</div>", unsafe_allow_html=True)

            if api_key:
                with st.spinner("Consulting AI Astrologer..."):
                    ctx = f"Lagna {chart.lagna_sign}, Moon {chart.moon_sign}, MD {pred['dasha'].get('mahadasha', 'N/A')}"
                    prompt = f"Detailed Vedic prediction for {topic if topic != 'All' else 'all life areas'} in {year}. Context: {ctx}."
                    try:
                        r = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                                "messages": [
                                    {"role": "system", "content": "You are an expert Vedic Astrologer."},
                                    {"role": "user", "content": prompt}
                                ]
                            }, timeout=30
                        )
                        ai = r.json()['choices'][0]['message']['content']
                        st.markdown(f"""
                        <div class="card" style="background:linear-gradient(135deg, #eff6ff, #dbeafe); border-left:6px solid #2563eb;">
                            <div class="card-title">🤖 AI Insight</div>
                            <p style="white-space:pre-wrap; color:#1e3a8a;">{ai}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"AI unavailable: {e}")

elif page == "📊 Varshphal":
    st.title("📊 Varshphal (Annual Chart)")
    st.caption("Tajaka / Solar Return Analysis with Muntha")

    name, date, time, lat, lon, tz = birth_input_form("varsh", "Native")
    year = st.selectbox("📅 Select Year for Varshphal", list(range(2024, 2036)))

    if st.button("🌟 Calculate Varshphal", use_container_width=True):
        with st.spinner("Calculating Solar Return..."):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            varsh = calculate_varshphal(chart, year)

            if not varsh:
                st.error("❌ Unable to calculate Varshphal. Birth date may be missing or invalid.")
                st.stop()

            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h2>{year} Varshphal</h2>
                <p style="font-size:1.2rem;"><b>Varshphal Date:</b> {varsh.get('varshphal_date', 'N/A')}</p>
                <p style="font-size:1.2rem;"><b>Muntha:</b> {varsh.get('muntha_sign', 'N/A')} ({varsh.get('muntha_longitude', 'N/A')}°)</p>
                <p style="font-size:1.2rem;"><b>Years Elapsed:</b> {varsh.get('years_elapsed', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <div class="card-title">📖 Annual Themes</div>
            """, unsafe_allow_html=True)
            for theme in varsh.get('themes', []):
                st.markdown(f"<p>• {theme}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            muntha_sign = varsh.get('muntha_sign', '')
            if muntha_sign:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🎯 Muntha in {muntha_sign}</div>
                    <p>Muntha lord <b>{SIGN_LORD.get(muntha_sign, 'Unknown')}</b> governs the year.</p>
                    <p>{_muntha_interpretation(muntha_sign, chart.lagna_sign)}</p>
                </div>
                """, unsafe_allow_html=True)


elif page == "❓ AI Astrologer":
    st.title("❓ Ask the AI Vedic Astrologer")
    st.caption("Powered by OpenRouter (free Gemini model)")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="card-title">📋 Chart Context</div>', unsafe_allow_html=True)
        name, date, time, lat, lon, tz = birth_input_form("ai", "Seeker")
        if st.button("📥 Load Chart Context", use_container_width=True):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
                current = chart.get_current_dasha_info()
                st.session_state["ai_ctx"] = (
                    f"Native {name}: Lagna {chart.lagna_sign}, Moon {chart.moon_sign} "
                    f"in {chart.nakshatras['Moon']['nakshatra']} pada {chart.nakshatras['Moon']['pada']}, "
                    f"MD {current.get('mahadasha', 'Unknown')}, AD {current.get('antardasha', 'Unknown')}, "
                    f"PD {current.get('pratyantardasha', 'Unknown')}."
                )
                st.success("✅ Context loaded!")
            except RuntimeError as e:
                st.error(str(e))

    with c2:
        st.markdown('<div class="card-title">💬 Your Question</div>', unsafe_allow_html=True)
        question = st.text_area("Ask anything about career, marriage, children, health, remedies...",
                                "What does my chart say about my career in 2026?", height=100)
        if st.button("🙏 Ask Astrologer", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter your OpenRouter API Key in the sidebar.")
            elif "ai_ctx" not in st.session_state:
                st.warning("⚠️ Please load chart context first (left panel).")
            else:
                with st.spinner("Consulting the stars..."):
                    try:
                        r = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                                "messages": [
                                    {"role": "system", "content": f"You are a wise Vedic Astrologer. {st.session_state['ai_ctx']}"},
                                    {"role": "user", "content": question}
                                ]
                            }, timeout=30
                        )
                        ans = r.json()['choices'][0]['message']['content']
                        st.markdown(f"""
                        <div class="card" style="background:#fffaf3;">
                            <div class="card-title">🪔 Divine Guidance</div>
                            <p style="white-space:pre-wrap; line-height:1.7;">{ans}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ API Error: {e}")

elif page == "🎲 Ram Shalaka":
    st.title("🎲 Ram Shalaka")
    st.caption("Receive divine guidance from Shri Ram Charit Manas")

    SHALAKA = [
        {"text": "Sunu siya satya aseesa hamari, pujahi mana kamana tumhari",
         "meaning": "Success is certain — your wish will be fulfilled by the grace of Lord Ram.", "type": "Positive"},
        {"text": "Prabishi nagara keeje saba kaaja, hridaya rakhi koushalapur raaja",
         "meaning": "Begin your endeavors without fear — success and protection are assured.", "type": "Positive"},
        {"text": "Hoeehai soee jo rama rachi raakhaa, ko kari taraka badhaavai saakhaa",
         "meaning": "What is destined by Lord Ram shall happen — do not worry or overthink.", "type": "Neutral"},
        {"text": "Garala sudha ripu karahi mitaee, gopada sindhu anala sitalaee",
         "meaning": "Even enemies turn into friends; the impossible becomes possible by divine grace.", "type": "Very Positive"},
        {"text": "Sakala sumangala daayaka raghunandana, sadhubara nindaaka aridata bandana",
         "meaning": "The Lord of Raghus brings all auspiciousness and destroys the pain of the noble.", "type": "Positive"},
        {"text": "Bhagati heti mori kara puja, hoyi siddhi millaahi mahatuja",
         "meaning": "Worship with devotion and faith — you shall attain perfection and great valor.", "type": "Positive"},
        {"text": "Rama charana rati mori man mahi, basahu sadaa siya sahita sadahi",
         "meaning": "Devotion to Lord Ram's feet fills the heart — dwell eternally with Siya and Ram.", "type": "Positive"},
        {"text": "Suni siya pati ke bachana suhaae, hridaya harasha gayatri guna gaae",
         "meaning": "Hearing pleasing words from the beloved husband, the heart rejoices and sings virtues.", "type": "Positive"}
    ]

    if st.button("🙏 Seek Blessing", use_container_width=True):
        verse = random.choice(SHALAKA)
        st.balloons()
        st.markdown(f"""
        <div class="card" style="text-align:center; background:linear-gradient(135deg, #fff8f0, #fef3c7); padding:2.5rem;">
            <h2 style="font-family:Cinzel; color:#78350f; font-size:1.6rem; margin-bottom:1rem;">"{verse['text']}"</h2>
            <p style="font-size:1.15rem; color:#5c2b02; font-style:italic; margin-bottom:1.5rem;">{verse['meaning']}</p>
            <span style="background:#d97706; color:white; padding:6px 18px; border-radius:20px; font-weight:600; font-size:0.9rem;">
                {verse['type']}
            </span>
        </div>
        """, unsafe_allow_html=True)
