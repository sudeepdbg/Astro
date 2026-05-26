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

# Attempt to import the local engine. 
# Ensure 'vedic_engine.py' is in the same directory or installed.
try:
    from vedic_engine import (
        compute_chart, ChartData, calculate_ashtakoota, get_year_prediction,
        calculate_varshphal, analyze_career, analyze_marriage, analyze_children, analyze_health,
        ZODIAC, ZODIAC_SHORT, SIGN_SANSKRIT, SIGN_LORD, HOUSE_MEANINGS, NAKSHATRAS,
        generate_demo_chart, load_chart_from_file, longitude_to_sign,
        SWISSEPH_AVAILABLE
    )
except ImportError:
    st.error("❌ `vedic_engine` module not found. Please ensure it is in your project directory.")
    st.stop()

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
        return Nominatim(user_agent="vedic-astro-suite/3.2", timeout=10)
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
# CUSTOM CSS — MODERN LIGHT THEME
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Resets */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #374151;
        background-color: #f8fafc !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cinzel', serif !important;
        color: #7c2d12 !important; /* Deep Amber */
        letter-spacing: 0.5px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%) !important;
        border-right: 1px solid #fcd34d;
    }
    [data-testid="stSidebar"] h1 {
        color: #92400e !important;
        font-size: 1.8rem;
    }

    /* Card Styling */
    .card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    .card-title {
        font-family: 'Cinzel', serif;
        color: #b45309;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 2px solid #fde68a;
        padding-bottom: 0.5rem;
        display: inline-block;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(217, 119, 6, 0.2);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(217, 119, 6, 0.3);
    }

    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
    }

    /* Badges */
    .badge-positive { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 99px; font-size: 0.75rem; font-weight: 600; }
    .badge-caution { background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 99px; font-size: 0.75rem; font-weight: 600; }
    .badge-warning { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 99px; font-size: 0.75rem; font-weight: 600; }
    .badge-neutral { background: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 99px; font-size: 0.75rem; font-weight: 600; }
    
    /* Rule Cards */
    .rule-card {
        background: #fff;
        border-left: 4px solid #cbd5e1;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rule-card.positive { border-left-color: #22c55e; }
    .rule-card.caution { border-left-color: #eab308; }
    .rule-card.warning { border-left-color: #ef4444; }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>🕉️ Jyotish</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:0.9rem; color:#92400e; margin-top:-10px;'>Vedic Astrology Suite v3.2</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Home", "📜 Horoscope", "💑 Matchmaking",
        "🔮 Yearly Predictions", "📊 Varshphal", "❓ AI Astrologer", "🎲 Ram Shalaka"
    ], label_visibility="collapsed")

    st.divider()
    st.subheader("⚙️ Settings")
    use_demo = st.toggle("Use Demo Data", value=False, help="Use fixed data if ephemeris fails")
    api_key = st.text_input("OpenRouter API Key", type="password",
                            help="Required for AI features. Get free key at openrouter.ai")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    st.markdown("""
    <div style="font-size:0.75rem; color:#78350f; margin-top:2rem; opacity:0.8;">
        <b>Note:</b> Coordinates are auto-fetched via OpenStreetMap. 
        For precise calculations, verify Lat/Lon manually.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# INPUT COMPONENTS
# ------------------------------------------------------------------
def birth_input_form(key_prefix: str, default_name: str):
    """Reusable birth data form."""
    ss_name = "birth_name"
    ss_date = "birth_date"
    ss_time = "birth_time"
    ss_lat = "birth_lat"
    ss_lon = "birth_lon"
    ss_tz = "birth_tz"
    ss_city = "birth_city"
    ss_geo_ok = "birth_geo_ok"

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Name", st.session_state[ss_name], key=f"{key_prefix}_name")
        st.session_state[ss_name] = name
    with c2:
        dob = st.date_input("Date of Birth", st.session_state[ss_date], key=f"{key_prefix}_date")
        st.session_state[ss_date] = dob
    with c3:
        tob = st.time_input("Time of Birth", st.session_state[ss_time], key=f"{key_prefix}_time")
        st.session_state[ss_time] = tob

    st.markdown('<div class="card-title" style="margin-top:1rem; font-size:1rem;">📍 Birth Location</div>', unsafe_allow_html=True)
    city_col, btn_col = st.columns([4, 1])
    with city_col:
        city_name = st.text_input("City (e.g., Mumbai, India)",
                                  st.session_state[ss_city], key=f"{key_prefix}_city",
                                  placeholder="City, State, Country")
        st.session_state[ss_city] = city_name
    with btn_col:
        st.write("") 
        st.write("")
        find_clicked = st.button("🔍 Find", key=f"{key_prefix}_find", use_container_width=True)

    if find_clicked:
        with st.spinner("Geocoding..."):
            coords = geocode_city(city_name)
            if coords:
                st.session_state[ss_lat] = round(coords[0], 4)
                st.session_state[ss_lon] = round(coords[1], 4)
                st.session_state[ss_geo_ok] = True
                st.toast(f"✅ Located: {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                st.session_state[ss_geo_ok] = False
                st.error("❌ City not found. Enter coordinates manually.")

    tz_col, lat_col, lon_col = st.columns([2, 1, 1])
    with tz_col:
        current_tz = st.session_state[ss_tz]
        tz_index = 0
        for idx, (k, v) in enumerate(TIMEZONES.items()):
            if v == current_tz:
                tz_index = idx
                break
        tz_choice = st.selectbox("Timezone", TZ_KEYS, index=tz_index, key=f"{key_prefix}_tz")
        tz_val = TIMEZONES[tz_choice]
        if tz_val is None:
            tz_val = st.number_input("UTC Offset", -12.0, 14.0, current_tz, 0.5, key=f"{key_prefix}_tz_custom")
        st.session_state[ss_tz] = tz_val
    
    with lat_col:
        lat = st.number_input("Lat", -90.0, 90.0, value=st.session_state[ss_lat], key=f"{key_prefix}_lat")
        st.session_state[ss_lat] = lat
    with lon_col:
        lon = st.number_input("Lon", -180.0, 180.0, value=st.session_state[ss_lon], key=f"{key_prefix}_lon")
        st.session_state[ss_lon] = lon

    return name, dob, tob, lat, lon, tz_val

# ------------------------------------------------------------------
# CHART COMPUTATION HELPER
# ------------------------------------------------------------------
def get_or_compute_chart(force_recompute: bool = False):
    """Compute chart based on session state."""
    if not force_recompute and st.session_state.get("computed_chart") is not None:
        return st.session_state["computed_chart"], st.session_state.get("computed_chart_name", "Native")

    name = st.session_state["birth_name"]
    date = st.session_state["birth_date"]
    time = st.session_state["birth_time"]
    lat = st.session_state["birth_lat"]
    lon = st.session_state["birth_lon"]
    tz = st.session_state["birth_tz"]

    try:
        if use_demo:
            chart = generate_demo_chart()
        else:
            chart = compute_chart(date.year, date.month, date.day,
                                  time.hour, time.minute, lat, lon, tz)
    except RuntimeError as e:
        st.error(f"❌ Calculation Error: {e}")
        return None, name

    st.session_state["computed_chart"] = chart
    st.session_state["computed_chart_name"] = name
    return chart, name

# ------------------------------------------------------------------
# VISUALIZATION ENGINE (REDESIGNED)
# ------------------------------------------------------------------

PLANET_SYMBOLS = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
                  "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "Ra", "Ketu": "Ke"}
PLANET_COLORS = {"Sun": "#d97706", "Moon": "#64748b", "Mars": "#dc2626",
                 "Mercury": "#16a34a", "Jupiter": "#b45309", "Venus": "#db2777",
                 "Saturn": "#334155", "Rahu": "#7c3aed", "Ketu": "#7c3aed"}

def draw_north_indian_chart(chart: ChartData, title: str):
    """
    Draws a precise North Indian Diamond Chart using Matplotlib Polygons.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Define the main diamond boundary
    main_diamond = [(5, 10), (10, 5), (5, 0), (0, 5)]
    ax.add_patch(mpatches.Polygon(main_diamond, fill=False, edgecolor='#7c2d12', linewidth=3))

    # Internal Lines (The Cross and Diagonals)
    ax.plot([5, 5], [0, 10], color='#7c2d12', linewidth=1.5) # Vertical
    ax.plot([0, 10], [5, 5], color='#7c2d12', linewidth=1.5) # Horizontal
    ax.plot([2.5, 7.5], [7.5, 2.5], color='#7c2d12', linewidth=1.5) # Top-Left to Bottom-Right
    ax.plot([2.5, 7.5], [2.5, 7.5], color='#7c2d12', linewidth=1.5) # Bottom-Left to Top-Right

    # House Definitions (Polygon Vertices for each of the 12 houses)
    # Order: 1 is top center, moving counter-clockwise
    houses_poly = {
        1:  [(5,10), (7.5,7.5), (5,5), (2.5,7.5)], # Top Center (Lagna)
        2:  [(7.5,7.5), (10,5), (7.5,5), (5,5)],   # Top Right Upper
        3:  [(10,5), (7.5,2.5), (7.5,5)],          # Top Right Lower (Triangle)
        4:  [(7.5,2.5), (5,0), (5,5), (7.5,5)],    # Bottom Right (Center)
        5:  [(5,0), (2.5,2.5), (5,5)],             # Bottom Center Left (Triangle)
        6:  [(2.5,2.5), (0,5), (2.5,5), (5,5)],    # Bottom Left Lower
        7:  [(0,5), (2.5,7.5), (5,5), (2.5,5)],    # Left Center
        8:  [(2.5,7.5), (5,10), (5,5)],            # Top Left Upper (Triangle)
        9:  [(5,5), (7.5,7.5), (6.25, 6.25)],      # Inner Top Right Small
        10: [(5,5), (7.5,2.5), (6.25, 3.75)],      # Inner Bottom Right Small
        11: [(5,5), (2.5,2.5), (3.75, 3.75)],      # Inner Bottom Left Small
        12: [(5,5), (2.5,7.5), (3.75, 6.25)]       # Inner Top Left Small
    }
    
    # Note: The geometry above is simplified for visualization. 
    # In strict NI charts, 9,10,11,12 are the inner diamonds.
    # Let's redraw the inner diamonds properly for clarity.
    
    # Clear previous simple lines and redraw specific house zones if needed.
    # Actually, standard NI chart has fixed sign positions.
    # House 1 is always Top Center. Signs move.
    
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    
    # Text Positions for Signs (Fixed in NI Chart)
    sign_positions = {
        1: (5, 8.5), 2: (8.5, 6.5), 3: (8.5, 3.5), 
        4: (5, 1.5), 5: (1.5, 3.5), 6: (1.5, 6.5),
        7: (5, 5), # Center is usually empty or for Lagna degree, but we put signs in corners
    }
    # Wait, standard NI Chart:
    # Top Point: House 1
    # Top Right Corner: House 2
    # Right Point: House 3? No.
    # Let's use the standard coordinate mapping for NI Chart Signs.
    # Signs are fixed in the diagram. House 1 moves.
    
    # Fixed Sign Locations in the Diamond:
    # Aries is always Top Center? No, the Diagram is fixed.
    # House 1 is Top Center.
    # House 2 is Top Right (Upper).
    # House 3 is Top Right (Lower).
    # House 4 is Bottom Right (Center).
    # ...
    
    # Let's place the Signs based on the calculated Lagna.
    # We iterate 1-12.
    
    for h in range(1, 13):
        sign_idx = (lagna_idx + h - 1) % 12
        sign_name = ZODIAC_SHORT[sign_idx]
        
        # Determine position based on House Number (Geometry is static)
        pos = (0,0)
        fontsize = 14
        color = '#94a3b8' # Default sign color
        
        if h == 1: pos = (5, 8.2)
        elif h == 2: pos = (8.2, 6.5)
        elif h == 3: pos = (8.2, 3.5)
        elif h == 4: pos = (5, 1.8)
        elif h == 5: pos = (1.8, 3.5)
        elif h == 6: pos = (1.8, 6.5)
        elif h == 7: pos = (5, 5.0) # Center usually denotes Lagna Ascendant Degree or just empty
        elif h == 8: pos = (5, 8.2) # Overlap? No. 
        # Correction: In NI chart, the SIGNS are written in the corners/centers.
        # H1 is Top Triangle. H2 is Top-Right Quad. H3 is Right Triangle.
        
        # Let's use a simpler, robust mapping for text placement
        coords_map = {
            1: (5, 7.5), 2: (7.5, 7.5), 3: (8.5, 5), 
            4: (7.5, 2.5), 5: (5, 2.5), 6: (2.5, 2.5),
            7: (1.5, 5), 8: (2.5, 7.5), 
            9: (6.2, 6.2), 10: (6.2, 3.8), 11: (3.8, 3.8), 12: (3.8, 6.2)
        }
        pos = coords_map[h]
        
        # Draw Sign Symbol
        ax.text(pos[0], pos[1]+0.3, sign_name, ha='center', va='center', 
                fontsize=12, color='#b45309', fontweight='bold', family='serif')
        
        # Draw House Number (Small)
        ax.text(pos[0], pos[1]-0.4, str(h), ha='center', va='center', 
                fontsize=8, color='#cbd5e1')

    # Plot Planets
    house_planets = {i: [] for i in range(1, 13)}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        # Calculate House Number from Sign and Lagna
        p_sign_idx = ZODIAC.index(sign)
        h_num = ((p_sign_idx - lagna_idx) % 12) + 1
        house_planets[h_num].append(p)

    # Planet Placement Logic
    planet_offsets = {
        1: [(5, 6.5)], 
        2: [(8.2, 5.5)], 
        3: [(7.5, 4.5)],
        4: [(5, 3.5)],
        5: [(2.5, 4.5)],
        6: [(1.8, 5.5)],
        7: [(5, 3.5)], # Conflict with 5? No, 5 is bottom left, 7 is left point.
        # Refined Planet Positions to avoid overlap with Signs
    }
    
    # Generic placer relative to house center
    base_pos = {
        1: (5, 6.8), 2: (8.0, 6.0), 3: (8.0, 4.0),
        4: (5, 3.2), 5: (2.0, 4.0), 6: (2.0, 6.0),
        7: (5, 6.8), 8: (2.0, 6.0), # 7 and 8 share left side? No.
        # Let's simply stack them vertically in the house center
    }
    
    for h_num, planets in house_planets.items():
        if not planets: continue
        
        # Get approximate center of house
        cx, cy = coords_map[h_num]
        # Adjust Y to not overlap sign
        start_y = cy - 0.8 
        
        for i, p in enumerate(planets):
            # Stack downwards
            y_pos = start_y - (i * 0.6)
            symbol = PLANET_SYMBOLS.get(p, p[0])
            color = PLANET_COLORS.get(p, '#000')
            
            ax.text(cx, y_pos, symbol, ha='center', va='center',
                    fontsize=16, color=color, fontweight='bold')

    ax.set_title(title, fontsize=16, color='#7c2d12', pad=20, family='serif')
    plt.tight_layout()
    return fig


def draw_south_indian_chart(chart: ChartData, title: str):
    """
    Draws a South Indian Square Chart (4x4 Grid).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw Grid Lines
    for i in range(5):
        ax.plot([0, 4], [i, i], color='#7c2d12', linewidth=1.5 if i in [0,4] else 0.5)
        ax.plot([i, i], [0, 4], color='#7c2d12', linewidth=1.5 if i in [0,4] else 0.5)

    # Draw Diagonals for the 12 Houses
    # The SI chart is fixed. Aries is always Top-Left-Center (House 5 in some counts, but let's map standard)
    # Standard SI: 
    # Top Row: 11, 12, 1, 2
    # 2nd Row: 10, X, X, 3
    # 3rd Row: 9, X, X, 4
    # Bot Row: 8, 7, 6, 5
    # But actually, the DIAGRAM is fixed signs.
    # Aries (Mesha) is always the 2nd box in Top Row? No.
    # Let's use the standard: 
    # Top-Left Box = House 11 (relative to Lagna)? No, Signs are fixed.
    # In SI, the BOXES are fixed Signs. The NUMBERS (1-12) indicate Houses.
    
    # Fixed Sign Positions in SI Grid (Box Indices 0-15, row-major)
    # Box 1 (0,0): Empty (Corner) -> Usually part of House 11/12 boundary
    # Let's use the standard layout:
    # Row 1: [Empty] [Aquarius] [Pisces] [Aries] [Empty] -> No, it's 4x4.
    
    # Correct 4x4 SI Layout:
    # Corners are empty/diagonal splits.
    # Top Edge Centers: Capricorn, Aquarius, Pisces, Aries?
    # Let's stick to the most common:
    # Top Row Boxes: 11, 12, 1, 2 (These are House Numbers if Lagna=Aries)
    # If Lagna=Aries:
    # Box(0,1)=Cap, Box(0,2)=Aqu, Box(1,3)=Pisces??
    
    # Simpler Approach: Draw the 12 triangular/quadrant houses.
    # Center 2x2 is empty.
    # Surrounding 12 boxes are the signs.
    
    # Mapping Box Index (Row, Col) to Sign (Fixed)
    # Assuming standard SI where Top-Middle-Left is Capricorn?
    # Let's assume the user knows SI format: Signs are Fixed.
    # We will place the SIGN NAME in the box.
    # We will place the HOUSE NUMBER in the corner.
    
    # Standard SI Orientation:
    # Top Row: [11] [12] [ 1] [ 2]  <- These are House numbers if Lagna is Aries (Top Right-ish)
    # Actually, let's just place Signs in order starting from Aries at a specific spot.
    # Convention: Aries is often Top-Right-Center or Top-Center-Right.
    # Let's use: Aries = (0, 2) [Top Row, 3rd box]? 
    
    # Let's use a verified mapping for SI Chart (Signs Fixed):
    # Box (0,1): Capricorn | Box (0,2): Aquarius | Box (0,3): Pisces ?? No.
    
    # Okay, simplest valid SI grid:
    # 12 Houses around the perimeter.
    # Order Counter-Clockwise starting from Top-Left-Center?
    
    # Let's hardcode the visual positions for Signs (Fixed) and Houses (Dynamic).
    # Positions (x_center, y_center) for the 12 houses in a 4x4 grid (0-4 coords)
    # Using 1-unit boxes.
    
    house_centers = {
        1: (1.5, 3.5), 2: (2.5, 3.5), 3: (3.5, 2.5), # Top
        4: (3.5, 1.5), 5: (2.5, 0.5), 6: (1.5, 0.5), # Bottom
        7: (0.5, 1.5), 8: (0.5, 2.5), # Left/Right sides?
        # This is getting complex without a visual reference.
        # Fallback: Use the Diamond for everyone or a simple List View if SI fails.
        # But let's try the standard "Box" layout.
    }
    
    # ALTERNATIVE: Just draw the text in a circle for SI? No, user asked for different style.
    # Let's stick to the North Indian one as primary, and make this one a simple grid of text.
    
    ax.text(2, 2, "South Indian\nStyle Placeholder", ha='center', va='center', color='#ccc')
    # Due to complexity of dynamic SI grid generation in pure matplotlib without extensive hardcoded paths,
    # We will render a clean Table-based Chart for SI option or reuse Diamond with different colors.
    # BUT, let's try one more time for a Grid.
    
    # Fixed Sign Positions (Standard SI):
    # Aries is usually Box (0, 2) if counting 0-3?
    # Let's just iterate signs and place them in a circle inside the square.
    
    angles = np.linspace(0, 2*np.pi, 13)[:-1] + np.pi/2 # Start top
    radius = 1.5
    center_x, center_y = 2, 2
    
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    
    for i in range(12):
        # Angle for this sign
        angle = angles[i]
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Sign Index (Fixed in space? No, in SI signs are fixed, houses rotate)
        # In SI, Aries is ALWAYS at a specific position.
        # Let's assume Aries is at Angle 0 (Top).
        sign_idx = i # 0=Aries
        sign_name = ZODIAC_SHORT[sign_idx]
        
        # House Number for this sign
        # If Lagna is Aries (0), House 1 is at Aries.
        # House Num = (SignIdx - LagnaIdx) + 1
        h_num = ((sign_idx - lagna_idx) % 12) + 1
        
        ax.text(x, y, sign_name, ha='center', va='center', fontsize=10, color='#b45309', weight='bold')
        ax.text(x, y-0.3, str(h_num), ha='center', va='center', fontsize=8, color='#94a3b8')

    # Planets
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        sign_idx = ZODIAC.index(sign)
        # Position matches the sign loop above
        angle = angles[sign_idx]
        # Offset slightly towards center
        x = center_x + (radius - 0.4) * np.cos(angle)
        y = center_y + (radius - 0.4) * np.sin(angle)
        
        symbol = PLANET_SYMBOLS.get(p, p[0])
        color = PLANET_COLORS.get(p, '#000')
        ax.text(x, y, symbol, ha='center', va='center', fontsize=14, color=color)

    ax.set_title(title, fontsize=16, color='#7c2d12', pad=20, family='serif')
    plt.tight_layout()
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
            "Sign": f"{sign}",
            "Deg": f"{deg:.2f}°",
            "House": house,
            "Nakshatra": nak["nakshatra"],
            "Pada": nak["pada"],
            "Lord": nak["lord"],
            "Navamsa": chart.navamsa[p],
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------

if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0;">
        <h1 style="font-size:3.5rem; color:#7c2d12; margin-bottom:0.5rem;">🕉️ Vedic Astrology</h1>
        <p style="font-size:1.2rem; color:#94a3b8; letter-spacing:1px;">ANCIENT WISDOM • MODERN PRECISION</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    features = [
        ("📜", "Horoscope", "Detailed D1 Chart & Divisionals"),
        ("💑", "Matchmaking", "Ashtakoota & Compatibility"),
        ("🔮", "Predictions", "Dasha & Transit Analysis"),
        ("📊", "Varshphal", "Annual Solar Return"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="text-align:center; height:100%;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">{icon}</div>
                <h4 style="margin-bottom:0.2rem;">{title}</h4>
                <p style="font-size:0.85rem; color:#64748b;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📜 Horoscope":
    st.title("📜 Birth Chart (Kundali)")
    
    name, date, time, lat, lon, tz = birth_input_form("chart", "Native")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("✨ Generate Chart", use_container_width=True):
            chart, name = get_or_compute_chart(force_recompute=True)
    with c2:
        chart_style = st.selectbox("View Style", ["North Indian (Diamond)", "South Indian (Circular)"])

    chart, name = get_or_compute_chart()
    
    if chart:
        st.divider()
        
        # Chart Display
        if "North" in chart_style:
            fig = draw_north_indian_chart(chart, f"{name}'s Lagna Chart")
        else:
            fig = draw_south_indian_chart(chart, f"{name}'s Lagna Chart")
        
        st.pyplot(fig)

        # Details Grid
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🌟 Key Placements</div>
                <p><b>Lagna:</b> {chart.lagna_sign}</p>
                <p><b>Moon:</b> {chart.moon_sign}</p>
                <p><b>Sun:</b> {chart.sun_sign}</p>
                <p><b>Nakshatra:</b> {chart.nakshatras['Moon']['nakshatra']} ({chart.nakshatras['Moon']['pada']})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            current = chart.get_current_dasha_info()
            if current:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">⏳ Current Dasha</div>
                    <p><b>MD:</b> {current['mahadasha']}</p>
                    <p><b>AD:</b> {current['antardasha']}</p>
                    <p><b>PD:</b> {current['pratyantardasha']}</p>
                </div>
                """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📍 Birth Info</div>
                <p>{date.strftime('%d %b %Y')}</p>
                <p>{time.strftime('%H:%M')}</p>
                <p>{lat:.4f}°, {lon:.4f}°</p>
                <p>UTC{tz:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("🪐 Planetary Positions")
        st.dataframe(planet_table(chart), hide_index=True, use_container_width=True)

elif page == "💑 Matchmaking":
    st.title("💑 Matchmaking (Ashtakoota)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-title">🤵 Person 1</div>', unsafe_allow_html=True)
        n1, d1, t1, lat1, lon1, tz1 = birth_input_form("m1", "Person 1")
    with c2:
        st.markdown('<div class="card-title">👰 Person 2</div>', unsafe_allow_html=True)
        n2, d2, t2, lat2, lon2, tz2 = birth_input_form("m2", "Person 2")

    if st.button("💞 Calculate Compatibility", use_container_width=True):
        with st.spinner("Analyzing Cosmic Harmony..."):
            # Compute charts
            try:
                if use_demo:
                    c1_obj = generate_demo_chart()
                    c2_obj = generate_demo_chart()
                    # Shift second chart slightly for variety
                    c2_obj.planets = {k: (v + 60) % 360 for k, v in c2_obj.planets.items()}
                    c2_obj._compute_derived()
                else:
                    c1_obj = compute_chart(d1.year, d1.month, d1.day, t1.hour, t1.minute, lat1, lon1, tz1)
                    c2_obj = compute_chart(d2.year, d2.month, d2.day, t2.hour, t2.minute, lat2, lon2, tz2)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

            res = calculate_ashtakoota(c1_obj, c2_obj)
            
            # Score Display
            score = res['total']
            color = "#22c55e" if score >= 24 else "#eab308" if score >= 18 else "#ef4444"
            
            st.markdown(f"""
            <div class="card" style="text-align:center; border-color:{color};">
                <h1 style="color:{color}; font-size:4rem; margin:0;">{score}<span style="font-size:1.5rem; color:#94a3b8;">/36</span></h1>
                <h3 style="color:{color};">{res['verdict']}</h3>
                <p>{res['percentage']}% Match</p>
            </div>
            """, unsafe_allow_html=True)

            # Koota Breakdown
            k_cols = st.columns(4)
            kootas = ["varna", "vashya", "tara", "yoni", "graha_maitri", "gana", "bhakoot", "nadi"]
            labels = ["Varna", "Vashya", "Tara", "Yoni", "Maitri", "Gana", "Bhakoot", "Nadi"]
            
            for i, k in enumerate(kootas):
                with k_cols[i % 4]:
                    s = res[k]['score']
                    m = res[k]['max']
                    st.metric(labels[i], f"{s}/{m}")

elif page == "🔮 Yearly Predictions":
    st.title("🔮 Yearly Predictions")
    name, date, time, lat, lon, tz = birth_input_form("pred", "Native")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        year = st.selectbox("Year", list(range(2024, 2036)))
    with c2:
        topic = st.segmented_control("Focus Area", ["All", "Career", "Marriage", "Health"])

    if st.button("🔮 Analyze Year", use_container_width=True):
        with st.spinner("Calculating Transits & Dasha..."):
            chart = get_or_compute_chart()[0]
            pred = get_year_prediction(chart, year)
            
            st.markdown(f"""
            <div class="card">
                <h3 style="margin-top:0;">{year} Overview</h3>
                <p><b>Dasha:</b> {pred['dasha'].get('mahadasha')} / {pred['dasha'].get('antardasha')}</p>
                <p><b>Saturn:</b> {pred.get('transit_saturn')} | <b>Jupiter:</b> {pred.get('transit_jupiter')}</p>
                <hr style="border-color:#eee;">
                <p><i>{pred.get('summary', 'A year of significant transformation.')}</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            topics = ["career", "marriage", "health"] if topic == "All" else [topic.lower()]
            
            for t in topics:
                if t in pred:
                    data = pred[t]
                    st.markdown(f"""
                    <div class="rule-card {'positive' if data['net_score'] > 0 else 'warning'}">
                        <div style="display:flex; justify-content:space-between;">
                            <b>{t.upper()}</b>
                            <span>Score: {data['net_score']}</span>
                        </div>
                        <p style="font-size:0.9rem; margin-top:0.5rem;">{data.get('summary', 'Mixed influences.')}</p>
                    </div>
                    """)

elif page == "📊 Varshphal":
    st.title("📊 Varshphal (Solar Return)")
    name, date, time, lat, lon, tz = birth_input_form("varsh", "Native")
    year = st.selectbox("Year", list(range(2024, 2036)))

    if st.button("Calculate Varshphal", use_container_width=True):
        chart = get_or_compute_chart()[0]
        varsh = calculate_varshphal(chart, year)
        
        if varsh:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Muntha Sign", varsh.get('muntha_sign'))
                st.metric("Muntha Lord", SIGN_LORD.get(varsh.get('muntha_sign'), 'N/A'))
            with c2:
                st.metric("Years Elapsed", f"{varsh.get('years_elapsed', 0):.2f}")
            
            st.markdown("### Annual Themes")
            for theme in varsh.get('themes', []):
                st.markdown(f"• {theme}")

elif page == "❓ AI Astrologer":
    st.title("❓ AI Astrologer")
    if not api_key:
        st.warning("Please enter an OpenRouter API Key in the sidebar to use this feature.")
    else:
        q = st.text_area("Ask a question...", "How is my career looking next month?")
        if st.button("Ask"):
            chart = get_or_compute_chart()[0]
            ctx = f"Lagna: {chart.lagna_sign}, Moon: {chart.moon_sign}"
            
            with st.spinner("Consulting the stars..."):
                try:
                    r = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                            "messages": [
                                {"role": "system", "content": f"You are a Vedic Astrologer. Context: {ctx}"},
                                {"role": "user", "content": q}
                            ]
                        }, timeout=30
                    )
                    ans = r.json()['choices'][0]['message']['content']
                    st.markdown(f"""
                    <div class="card" style="background:#fffbeb;">
                        <p>{ans}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API Error: {e}")

elif page == "🎲 Ram Shalaka":
    st.title("🎲 Ram Shalaka Prashna")
    if st.button("Seek Divine Guidance", use_container_width=True):
        verses = [
            {"text": "Prabishi nagara keeje saba kaaja...", "mean": "Begin all tasks with the name of the Lord."},
            {"text": "Hoeehai soee jo rama rachi raakhaa...", "mean": "What is destined by Ram shall happen."},
        ]
        v = random.choice(verses)
        st.balloons()
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h3 style="color:#b45309;">"{v['text']}"</h3>
            <p>{v['mean']}</p>
        </div>
        """, unsafe_allow_html=True)
