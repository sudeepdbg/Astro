import streamlit as st
import pandas as pd
import requests
import os
import random
import json
from datetime import datetime, timedelta
from io import BytesIO

# Import from your engine
# Ensure vedic_engine.py has the get_chart_svg_data function provided in the previous step
from vedic_engine import (
    compute_chart, ChartData, calculate_ashtakoota, get_year_prediction,
    calculate_varshphal, analyze_career, analyze_marriage, analyze_children, analyze_health,
    ZODIAC, ZODIAC_SHORT, SIGN_SANSKRIT, SIGN_LORD, HOUSE_MEANINGS, NAKSHATRAS,
    generate_demo_chart, load_chart_from_file, longitude_to_sign,
    SWISSEPH_AVAILABLE, get_chart_svg_data
)
from geopy.geocoders import Nominatim

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
# CUSTOM CSS (Light Theme - Paper & Gold)
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --primary: #b45309; /* Darker Gold/Bronze */
        --secondary: #d97706; /* Gold */
        --bg-light: #fffbeb; /* Very light cream */
        --card-bg: #ffffff;
        --text-dark: #451a03; /* Dark Brown */
        --text-muted: #78350f;
        --border-color: #fcd34d;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #fffdf5 !important; /* Off-white background */
        color: var(--text-dark);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cinzel', serif !important;
        color: var(--primary) !important;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fffbeb !important;
        border-right: 1px solid #fcd34d;
    }
    
    /* Cards */
    .card {
        background: var(--card-bg);
        border: 1px solid #fcebb6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(180, 83, 9, 0.08);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(180, 83, 9, 0.12);
    }

    .metric-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid var(--secondary);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #d97706 0%, #b45309 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(180, 83, 9, 0.3);
        filter: brightness(1.1);
    }

    /* Inputs */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #ffffff;
        color: #451a03;
        border: 1px solid #fcd34d;
        border-radius: 6px;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #fcd34d;
        border-radius: 8px;
    }

    /* Chart Container */
    .chart-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
        background: #ffffff;
        border-radius: 15px;
        border: 1px solid #fcd34d;
        box-shadow: inset 0 0 20px rgba(252, 211, 77, 0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fffbeb;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #b45309;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 2px solid #d97706 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# GEOCODING & UTILS
# ------------------------------------------------------------------
@st.cache_resource
def get_geolocator():
    try:
        return Nominatim(user_agent="vedic-astro-suite/3.0", timeout=10)
    except Exception:
        return None

def geocode_city(name: str):
    geo = get_geolocator()
    if not geo or not name.strip(): return None
    try:
        loc = geo.geocode(name, language="en")
        if loc: return (loc.latitude, loc.longitude)
    except Exception: pass
    return None

TIMEZONES = {
    "IST (India, UTC+5:30)": 5.5,
    "GMT / UTC (UTC+0:00)": 0.0,
    "EST (New York, UTC-5:00)": -5.0,
    "PST (Los Angeles, UTC-8:00)": -8.0,
    "CET (Berlin, UTC+1:00)": 1.0,
    "JST (Tokyo, UTC+9:00)": 9.0,
    "AEST (Sydney, UTC+10:00)": 10.0,
    "Custom Offset": None
}

# ------------------------------------------------------------------
# INPUT COMPONENT
# ------------------------------------------------------------------
def birth_input_form(key_prefix: str, default_name: str):
    st.markdown(f'<div class="card"><h4 style="margin-top:0; color:#b45309;">📍 Birth Details</h4>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("👤 Name", default_name, key=f"{key_prefix}_name")
    with c2:
        dob = st.date_input("📅 Date", datetime(1991, 4, 12), key=f"{key_prefix}_date")
    with c3:
        tob = st.time_input("🕒 Time", datetime.strptime("10:26", "%H:%M").time(), key=f"{key_prefix}_time")
    
    city_col, btn_col = st.columns([4, 1])
    with city_col:
        city_name = st.text_input("City / Town", "", key=f"{key_prefix}_city", placeholder="e.g., Sitamarhi, Bihar")
    with btn_col:
        st.write("")
        st.write("")
        find_clicked = st.button("🔍 Find", key=f"{key_prefix}_find", use_container_width=True)
        
    lat_key = f"{key_prefix}_lat_val"
    lon_key = f"{key_prefix}_lon_val"
    
    if find_clicked:
        with st.spinner("Locating..."):
            coords = geocode_city(city_name)
            if coords:
                st.session_state[lat_key] = round(coords[0], 4)
                st.session_state[lon_key] = round(coords[1], 4)
                st.toast(f"✅ Found: {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                st.error("❌ City not found. Try 'City, State, Country'.")

    tz_col, lat_col, lon_col = st.columns([2, 1, 1])
    with tz_col:
        tz_choice = st.selectbox("🌍 Timezone", list(TIMEZONES.keys()), index=0, key=f"{key_prefix}_tz")
        tz_val = TIMEZONES[tz_choice]
        if tz_val is None:
            tz_val = st.number_input("UTC Offset", -12.0, 14.0, 5.5, 0.5, key=f"{key_prefix}_tz_custom")
    with lat_col:
        lat = st.number_input("Lat", -90.0, 90.0, value=st.session_state.get(lat_key, 25.42), key=lat_key)
    with lon_col:
        lon = st.number_input("Lon", -180.0, 180.0, value=st.session_state.get(lon_key, 86.13), key=lon_key)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return name, dob, tob, lat, lon, tz_val

# ------------------------------------------------------------------
# MAIN APP LOGIC
# ------------------------------------------------------------------
page = st.sidebar.radio("Navigate", [
    "🏠 Home", "📜 Horoscope", "💑 Matchmaking",
    "🔮 Predictions", "📊 Varshphal", "❓ AI Astrologer", "🎲 Ram Shalaka"
])

use_demo = st.sidebar.toggle("Use Demo Data", value=False)
api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Optional for AI features")
if api_key: os.environ["OPENROUTER_API_KEY"] = api_key

if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0;">
        <h1 style="font-size:3.5rem; color:#b45309;">🕉️ Vedic Astrology Suite</h1>
        <p style="font-size:1.2rem; color:#78350f;">Ancient Wisdom, Modern Precision</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card" style="text-align:center;"><h3>📜 Horoscope</h3><p>Detailed Kundli with Dasha & Nakshatra</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card" style="text-align:center;"><h3>💑 Matchmaking</h3><p>Ashtakoota Compatibility Analysis</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card" style="text-align:center;"><h3>🔮 Predictions</h3><p>Yearly Forecasts & Remedies</p></div>', unsafe_allow_html=True)

    # Load Saved Chart
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
    st.title("📜 Your Horoscope")
    
    # Load Saved Chart Logic
    if "loaded_chart_data" in st.session_state:
        if st.button("📂 Use Loaded Chart"):
            data = st.session_state["loaded_chart_data"]
            chart = ChartData(
                planets=data["planets"], ascendant=data["ascendant"], lagna_sign=data["lagna_sign"],
                birth_date=datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None,
                lat=data.get("lat", 0), lon=data.get("lon", 0), tz=data.get("tz", 0)
            )
            st.session_state["computed_chart"] = chart
            st.session_state["computed_chart_name"] = data.get("name", "Loaded")
            st.success("Chart loaded!")

    name, date, time, lat, lon, tz = birth_input_form("chart", "Native")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        chart_style = st.selectbox("Chart Style", ["North Indian (Diamond)", "South Indian (Square)"])
        if st.button("✨ Generate Chart", use_container_width=True):
            with st.spinner("Calculating planetary positions..."):
                try:
                    if use_demo:
                        chart = generate_demo_chart()
                    else:
                        if not SWISSEPH_AVAILABLE:
                            st.warning("Swiss Ephemeris not installed. Using Demo Data.")
                            chart = generate_demo_chart()
                        else:
                            chart = compute_chart(date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
                    
                    st.session_state["computed_chart"] = chart
                    st.session_state["computed_chart_name"] = name
                except Exception as e:
                    st.error(f"Error: {e}")

    with c2:
        if "computed_chart" in st.session_state:
            chart = st.session_state["computed_chart"]
            name = st.session_state["computed_chart_name"]
            
            # Display Chart using SVG
            style_code = "north" if "North" in chart_style else "south"
            svg_content = get_chart_svg_data(chart, style=style_code)
            st.markdown(f'<div class="chart-container">{svg_content}</div>', unsafe_allow_html=True)

    if "computed_chart" in st.session_state:
        chart = st.session_state["computed_chart"]
        name = st.session_state["computed_chart_name"]
        
        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><b>Lagna</b><br>{chart.lagna_sign}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><b>Moon Sign</b><br>{chart.moon_sign}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><b>Nakshatra</b><br>{chart.nakshatras["Moon"]["nakshatra"]}</div>', unsafe_allow_html=True)
        with c4:
            curr_dasha = chart.get_current_dasha_info()
            dasha_str = f"{curr_dasha.get('mahadasha', 'N/A')} / {curr_dasha.get('antardasha', 'N/A')}"
            st.markdown(f'<div class="metric-card"><b>Current Dasha</b><br>{dasha_str}</div>', unsafe_allow_html=True)

        # Planetary Positions Table
        st.subheader("🪐 Planetary Positions")
        rows = []
        for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
            sign, deg = longitude_to_sign(chart.planets[p])
            rows.append({
                "Planet": p,
                "Sign": f"{sign}",
                "Degree": f"{deg:.2f}°",
                "House": chart.house_map.get(p, "-"),
                "Nakshatra": chart.nakshatras[p]["nakshatra"],
                "Pada": chart.nakshatras[p]["pada"],
                "Dignity": chart.dignities.get(p, "Neutral")
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Save Options
        st.divider()
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

elif page == "💑 Matchmaking":
    st.title("💑 Ashtakoota Matchmaking")
    c1, c2 = st.columns(2)
    with c1:
        n1, d1, t1, lat1, lon1, tz1 = birth_input_form("m1", "Person 1")
    with c2:
        n2, d2, t2, lat2, lon2, tz2 = birth_input_form("m2", "Person 2")
        
    if st.button("💞 Calculate Compatibility", use_container_width=True):
        with st.spinner("Matching stars..."):
            try:
                if use_demo:
                    chart1 = generate_demo_chart()
                    chart2 = generate_demo_chart()
                else:
                    chart1 = compute_chart(d1.year, d1.month, d1.day, t1.hour, t1.minute, lat1, lon1, tz1)
                    chart2 = compute_chart(d2.year, d2.month, d2.day, t2.hour, t2.minute, lat2, lon2, tz2)
                
                res = calculate_ashtakoota(chart1, chart2)
                
                # Verdict
                score = res['total']
                verdict = res['verdict']
                color = "#15803d" if score >= 25 else "#ca8a04" if score >= 18 else "#b91c1c"
                
                st.markdown(f"""
                <div style="text-align:center; padding: 2rem; background: #fffbeb; border-radius: 15px; border: 1px solid {color};">
                    <h1 style="color: {color}; font-size: 4rem; margin: 0;">{score} / 36</h1>
                    <h2 style="color: {color};">{verdict}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Koota Breakdown
                kootas = ["varna", "vashya", "tara", "yoni", "graha_maitri", "gana", "bhakoot", "nadi"]
                cols = st.columns(4)
                for i, k in enumerate(kootas):
                    with cols[i%4]:
                        st.markdown(f'<div class="metric-card"><b>{k.replace("_", " ").title()}</b><br>{res[k]["score"]}/{res[k]["max"]}</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "🔮 Predictions":
    st.title("🔮 Yearly Predictions")
    name, date, time, lat, lon, tz = birth_input_form("pred", "Native")
    year = st.selectbox("Select Year", list(range(2024, 2036)))
    
    if st.button("🔮 Predict", use_container_width=True):
        with st.spinner("Analyzing transits and dashas..."):
            try:
                if use_demo:
                    chart = generate_demo_chart()
                else:
                    chart = compute_chart(date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
                
                pred = get_year_prediction(chart, year)
                
                # Summary
                st.markdown(f'<div class="card"><h3>📅 {year} Overview</h3><p>{pred["overall_summary"]}</p></div>', unsafe_allow_html=True)
                
                # Tabs for detailed analysis
                tab1, tab2, tab3, tab4 = st.tabs(["Career", "Marriage", "Health", "Children"])
                
                with tab1:
                    st.markdown(f'<div class="card">{pred["career"]["narrative"]}</div>', unsafe_allow_html=True)
                with tab2:
                    st.markdown(f'<div class="card">{pred["marriage"]["narrative"]}</div>', unsafe_allow_html=True)
                with tab3:
                    st.markdown(f'<div class="card">{pred["health"]["narrative"]}</div>', unsafe_allow_html=True)
                with tab4:
                    st.markdown(f'<div class="card">{pred["children"]["narrative"]}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "📊 Varshphal":
    st.title("📊 Varshphal (Annual Return)")
    name, date, time, lat, lon, tz = birth_input_form("varsh", "Native")
    year = st.selectbox("Year", list(range(2024, 2036)))
    
    if st.button("🌟 Calculate Varshphal", use_container_width=True):
        with st.spinner("Calculating Solar Return..."):
            try:
                if use_demo: chart = generate_demo_chart()
                else: chart = compute_chart(date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
                
                varsh = calculate_varshphal(chart, year)
                
                st.markdown(f"""
                <div class="card">
                    <h3>Muntha: {varsh['muntha_sign']} (House {varsh['muntha_house']})</h3>
                    <p><b>Themes:</b> {', '.join(varsh['themes'])}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "❓ AI Astrologer":
    st.title("❓ Ask the AI Astrologer")
    if not api_key:
        st.warning("Please enter an OpenRouter API Key in the sidebar to use this feature.")
    else:
        question = st.text_area("Your Question", "What does my chart say about my career?")
        if st.button("Ask"):
            if "computed_chart" in st.session_state:
                chart = st.session_state["computed_chart"]
                ctx = f"Lagna: {chart.lagna_sign}, Moon: {chart.moon_sign}, Dasha: {chart.get_current_dasha_info().get('mahadasha', 'N/A')}"
                
                with st.spinner("Consulting the stars..."):
                    try:
                        r = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                                "messages": [
                                    {"role": "system", "content": f"You are an expert Vedic Astrologer. Context: {ctx}"},
                                    {"role": "user", "content": question}
                                ]
                            }, timeout=30
                        )
                        ans = r.json()['choices'][0]['message']['content']
                        st.markdown(f'<div class="card">{ans}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"API Error: {e}")
            else:
                st.warning("Please generate a chart first in the Horoscope tab.")

elif page == "🎲 Ram Shalaka":
    st.title("🎲 Ram Shalaka")
    st.caption("Receive divine guidance from Shri Ram Charit Manas")
    SHALAKA = [
        {"text": "Sunu siya satya aseesa hamari, pujahi mana kamana tumhari", "meaning": "Success is certain — your wish will be fulfilled by the grace of Lord Ram.", "type": "Positive"},
        {"text": "Prabishi nagara keeje saba kaaja, hridaya rakhi koushalapur raaja", "meaning": "Begin your endeavors without fear — success and protection are assured.", "type": "Positive"},
        {"text": "Hoeehai soee jo rama rachi raakhaa, ko kari taraka badhaavai saakhaa", "meaning": "What is destined by Lord Ram shall happen — do not worry or overthink.", "type": "Neutral"},
        {"text": "Garala sudha ripu karahi mitaee, gopada sindhu anala sitalaee", "meaning": "Even enemies turn into friends; the impossible becomes possible by divine grace.", "type": "Very Positive"},
        {"text": "Sakala sumangala daayaka raghunandana, sadhubara nindaaka aridata bandana", "meaning": "The Lord of Raghus brings all auspiciousness and destroys the pain of the noble.", "type": "Positive"}
    ]

    if st.button("🙏 Seek Blessing", use_container_width=True):
        verse = random.choice(SHALAKA)
        st.balloons()
        st.markdown(f"""
        <div class="card" style="text-align:center; background:linear-gradient(135deg, #fffbeb, #fef3c7); padding:2.5rem;">
            <h2 style="font-family:Cinzel; color:#78350f; font-size:1.6rem; margin-bottom:1rem;">"{verse['text']}"</h2>
            <p style="font-size:1.15rem; color:#5c2b02; font-style:italic; margin-bottom:1.5rem;">{verse['meaning']}</p>
            <span style="background:#d97706; color:white; padding:6px 18px; border-radius:20px; font-weight:600; font-size:0.9rem;">
               {verse['type']}
            </span>
        </div>
        """, unsafe_allow_html=True)
