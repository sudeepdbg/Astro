import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
import random
from datetime import datetime
from vedic_engine import (
    compute_chart, ChartData, calculate_ashtakoota, get_year_prediction,
    ZODIAC, SIGN_SANSKRIT, SIGN_LORD, HOUSE_MEANINGS, NAKSHATRAS,
    generate_demo_chart, longitude_to_sign
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
# GEOCODING (OpenStreetMap — free, no API key)
# ------------------------------------------------------------------
@st.cache_resource
def get_geolocator():
    try:
        from geopy.geocoders import Nominatim
        return Nominatim(user_agent="vedic-astro-suite/1.0", timeout=10)
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
    "EST (New York, UTC-5:00)": -5.0,
    "CST (Chicago, UTC-6:00)": -6.0,
    "MST (Denver, UTC-7:00)": -7.0,
    "PST (Los Angeles, UTC-8:00)": -8.0,
    "CET (Berlin, UTC+1:00)": 1.0,
    "JST (Tokyo, UTC+9:00)": 9.0,
    "AEST (Sydney, UTC+10:00)": 10.0,
    "Custom Offset": None
}

# ------------------------------------------------------------------
# CUSTOM CSS — Spiritual Saffron & Indigo Theme
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #3d2616;
}
h1, h2, h3, h4 {
    font-family: 'Cinzel', serif !important;
    color: #5c2b02 !important;
    letter-spacing: 0.4px;
}
.stButton>button {
    background: linear-gradient(90deg, #d97706 0%, #b45309 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 4px 14px rgba(180, 83, 9, 0.35);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(180, 83, 9, 0.45);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff8f0 0%, #fef3c7 100%) !important;
}
.card {
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(212, 175, 55, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 40px rgba(92, 43, 2, 0.06);
}
.card-title {
    font-family: 'Cinzel', serif;
    color: #92400e;
    font-size: 1.15rem;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid #fcd34d;
    padding-bottom: 0.4rem;
    display: inline-block;
}
.metric-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #fbbf24;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.score-excellent { color: #15803d; font-weight: 700; }
.score-good { color: #65a30d; font-weight: 700; }
.score-average { color: #ca8a04; font-weight: 700; }
.score-challenging { color: #b91c1c; font-weight: 700; }
hr {
    border-color: #d4af37 !important;
    opacity: 0.35;
    margin: 1.5rem 0;
}
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div, .stDateInput>div>div>input {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.markdown("<h1 style='text-align:center; font-family:Cinzel; color:#92400e;'>🕉️ Jyotish</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#78350f;'>Vedic Astrology Suite</p>", unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "🏠 Home", "📜 Horoscope", "💑 Matchmaking",
    "🔮 Yearly Predictions", "❓ AI Astrologer", "🎲 Ram Shalaka"
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
e.g. <i>Sitamarhi, Bihar, India</i>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# INPUT COMPONENT
# ------------------------------------------------------------------
def birth_input_form(key_prefix: str, default_name: str):
    """Reusable birth data form with city geocoding & timezone."""
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("👤 Name", default_name, key=f"{key_prefix}_name")
    with c2:
        dob = st.date_input("📅 Date of Birth", datetime(1995, 6, 15), key=f"{key_prefix}_date")
    with c3:
        tob = st.time_input("🕒 Time of Birth", datetime.strptime("10:30", "%H:%M").time(), key=f"{key_prefix}_time")

    st.markdown('<div class="card-title">📍 Birth Place</div>', unsafe_allow_html=True)
    city_col, btn_col = st.columns([4, 1])
    with city_col:
        city_name = st.text_input("City / Town (e.g., Sitamarhi, Muzaffarpur, Datia, Gwalior...)",
                                  "", key=f"{key_prefix}_city",
                                  placeholder="Type city name and click Find")
    with btn_col:
        st.write("")
        st.write("")
        find_clicked = st.button("🔍 Find", key=f"{key_prefix}_find", use_container_width=True)

    # Geocode on button press
    lat_key = f"{key_prefix}_lat"
    lon_key = f"{key_prefix}_lon"
    if find_clicked:
        with st.spinner("Locating..."):
            coords = geocode_city(city_name)
            if coords:
                st.session_state[lat_key] = round(coords[0], 4)
                st.session_state[lon_key] = round(coords[1], 4)
                st.session_state[f"{key_prefix}_geo_ok"] = True
                st.toast(f"Found: {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                st.session_state[f"{key_prefix}_geo_ok"] = False
                st.error("City not found. Please enter coordinates manually.")

    tz_col, lat_col, lon_col = st.columns([2, 1, 1])
    with tz_col:
        tz_choice = st.selectbox("🌍 Timezone", list(TIMEZONES.keys()),
                                 index=0, key=f"{key_prefix}_tz")
        tz_val = TIMEZONES[tz_choice]
        if tz_val is None:
            tz_val = st.number_input("UTC Offset (+/- hrs)", -12.0, 14.0, 5.5, 0.5,
                                     key=f"{key_prefix}_tz_custom")
    with lat_col:
        lat = st.number_input("Lat", -90.0, 90.0,
                            value=st.session_state.get(lat_key, 28.6139),
                            key=lat_key)
    with lon_col:
        lon = st.number_input("Lon", -180.0, 180.0,
                            value=st.session_state.get(lon_key, 77.2090),
                            key=lon_key)

    if st.session_state.get(f"{key_prefix}_geo_ok"):
        st.caption(f"✅ Coordinates locked: {lat:.4f}, {lon:.4f}")

    return name, dob, tob, lat, lon, tz_val

# ------------------------------------------------------------------
# CHART WHEEL (improved visuals)
# ------------------------------------------------------------------
def draw_chart_wheel(chart: ChartData, title: str):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    # House sectors (alternating indigo)
    colors = ['#1e293b' if i % 2 == 0 else '#0f172a' for i in range(12)]
    for i in range(12):
        theta = np.linspace(np.radians(i*30), np.radians((i+1)*30), 50)
        ax.fill_between(theta, 0.35, 1.0, color=colors[i], alpha=0.95)
        ax.plot([np.radians(i*30)]*2, [0.35, 1.0], color='#d97706', linewidth=1.0)

    # Sign labels
    for i, sign in enumerate(ZODIAC):
        angle = np.radians(i*30 + 15)
        ax.text(angle, 0.92, f"{sign}\n{SIGN_SANSKRIT[sign]}",
                ha='center', va='center', fontsize=7.5, color='#fcd34d',
                fontweight='bold', fontfamily='sans-serif')

    # Planet symbols & colors
    symbols = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
               "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋"}
    colors_p = {"Sun": "#fbbf24", "Moon": "#e2e8f0", "Mars": "#f87171",
                "Mercury": "#34d399", "Jupiter": "#fb923c", "Venus": "#f472b6",
                "Saturn": "#94a3b8", "Rahu": "#a78bfa", "Ketu": "#a78bfa"}

    # Spread planets to avoid overlap
    used_bins = {}
    for planet, lon in chart.planets.items():
        base = lon % 360
        bin_id = int(base / 6)
        offset = used_bins.get(bin_id, 0) * 0.05
        used_bins[bin_id] = used_bins.get(bin_id, 0) + 1
        angle = np.radians(base + 90)  # rotate so 0° Aries is at top
        dist = 0.58 + offset
        ax.text(angle, dist, symbols.get(planet, planet), fontsize=13,
                ha='center', va='center', color=colors_p.get(planet, '#fff'),
                fontweight='bold')

    # Ascendant marker
    asc_angle = np.radians(chart.ascendant + 90)
    ax.plot([asc_angle, asc_angle], [0.35, 1.0], color='#ef4444', linewidth=2.5, linestyle='--')
    ax.text(asc_angle, 0.97, 'ASC ▲', ha='center', va='center', color='#ef4444',
            fontsize=9, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_title(title, fontsize=15, color='#fcd34d', fontweight='bold',
                 pad=20, fontfamily='serif')
    plt.tight_layout()
    return fig

def planet_table(chart: ChartData):
    rows = []
    for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
        sign, deg = longitude_to_sign(chart.planets[p])
        nak = chart.nakshatras[p]
        rows.append({
            "Planet": p,
            "Sign": f"{sign} ({SIGN_SANSKRIT[sign]})",
            "Deg": f"{deg:.2f}°",
            "Nakshatra": nak["nakshatra"],
            "Pada": nak["pada"],
            "Lord": nak["lord"],
            "Navamsa": chart.navamsa[p]
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="font-size:3rem; color:#92400e;">🕉️ Vedic Astrology Suite</h1>
        <p style="font-size:1.25rem; color:#78350f;">Jyotish — Ancient Wisdom, Modern Precision</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">📜</div>
            <h4>Horoscope</h4>
            <p style="font-size:0.95rem;">Sidereal chart with Lahiri Ayanamsa, Nakshatra, Navamsa & Vimshottari Dasha.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">💑</div>
            <h4>Matchmaking</h4>
            <p style="font-size:0.95rem;">Full Ashtakoota (36 points) with detailed Koota breakdown & verdict.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">🔮</div>
            <h4>Predictions</h4>
            <p style="font-size:0.95rem;">Year-wise analysis for Career, Marriage, Children & Health via Dasha + Transit.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
        <h4 class="card-title">How to Use</h4>
        <ul>
            <li>Enter <b>any city or town</b> (e.g., <i>Sitamarhi, Muzaffarpur, Datia, Damoh, Pukhrayan</i>) and click <b>Find</b> to auto-fill coordinates.</li>
            <li>Select your timezone — <b>IST is default</b>; switch to <b>GMT/UTC</b> or any major zone.</li>
            <li>Generate charts, match kundalis, or ask the AI Astrologer.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "📜 Horoscope":
    st.title("📜 Free Horoscope Chart")
    name, date, time, lat, lon, tz = birth_input_form("chart", "Native")

    if st.button("✨ Generate Chart", key="gen_chart"):
        with st.spinner("Calculating sidereal positions..."):
            try:
                if use_demo:
                    chart = generate_demo_chart()
                    st.info("Demo mode active — install pyswisseph for live ephemeris.")
                else:
                    chart = compute_chart(date.year, date.month, date.day,
                                          time.hour, time.minute, lat, lon, tz)
            except Exception as e:
                st.error(f"Ephemeris error: {e}. Enable Demo Data in sidebar.")
                st.stop()

            st.session_state["chart"] = chart
            st.session_state["chart_name"] = name

    if "chart" in st.session_state:
        chart = st.session_state["chart"]
        name = st.session_state["chart_name"]

        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.pyplot(draw_chart_wheel(chart, f"{name}'s Horoscope"))
        with c2:
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

            st.markdown("""
            <div class="card">
                <div class="card-title">📖 Vimshottari Dasha</div>
            """, unsafe_allow_html=True)
            dasha_df = pd.DataFrame(chart.dasha)
            st.dataframe(dasha_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("🪐 Planetary Positions")
        st.dataframe(planet_table(chart), hide_index=True, use_container_width=True)

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
                    # Offset chart2 for variety
                    chart2.planets = {k: (v + 55) % 360 for k, v in chart2.planets.items()}
                    chart2._compute_derived()
                else:
                    chart1 = compute_chart(d1.year, d1.month, d1.day, t1.hour, t1.minute, lat1, lon1, tz1)
                    chart2 = compute_chart(d2.year, d2.month, d2.day, t2.hour, t2.minute, lat2, lon2, tz2)
            except Exception as e:
                st.error(str(e))
                st.stop()

            res = calculate_ashtakoota(chart1, chart2)

            # Verdict banner
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

            # Koota cards
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

            # Interpretation
            st.markdown("""
            <div class="card">
                <div class="card-title">📖 What each Koota means</div>
                <ul style="font-size:0.95rem; line-height:1.6;">
                    <li><b>Varna (1 pt):</b> Spiritual ego-compatibility</li>
                    <li><b>Vashya (2 pts):</b> Mutual attraction & control</li>
                    <li><b>Tara (3 pts):</b> Destiny-star alignment</li>
                    <li><b>Yoni (4 pts):</b> Sexual & intimacy harmony</li>
                    <li><b>Graha Maitri (5 pts):</b> Moon-sign lord friendship</li>
                    <li><b>Gana (6 pts):</b> Temperament (Deva / Manushya / Rakshasa)</li>
                    <li><b>Bhakoot (7 pts):</b> Relative Moon position (2/12, 6/8 checked)</li>
                    <li><b>Nadi (8 pts):</b> Health & progeny (same Nadi = 0)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "🔮 Yearly Predictions":
    st.title("🔮 Yearly Predictions by Topic")
    name, date, time, lat, lon, tz = birth_input_form("pred", "Native")

    c1, c2 = st.columns([1, 2])
    with c1:
        year = st.selectbox("📅 Select Year", list(range(2024, 2036)))
    with c2:
        topic = st.segmented_control("Topic", ["Career", "Marriage", "Children", "Health"], default="Career")

    if st.button("🔮 Predict", use_container_width=True):
        with st.spinner("Analyzing Dasha & Gochar..."):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except Exception as e:
                st.error(str(e))
                st.stop()

            pred = get_year_prediction(chart, year)

            st.markdown(f"""
            <div class="card">
                <div class="card-title">📅 {pred['year']} — Transit & Dasha Snapshot</div>
                <p><b>Mahadasha:</b> {pred['dasha']['mahadasha']} <span style="color:#92400e;">({pred['dasha']['years']} yrs)</span></p>
                <p><b>Transit Saturn:</b> {pred['transits']['Saturn']} &nbsp;|&nbsp; <b>Transit Jupiter:</b> {pred['transits']['Jupiter']}</p>
                <p style="color:#b91c1c;"><b>{pred['sade_sati']}</b></p>
            </div>
            """, unsafe_allow_html=True)

            content = pred[topic.lower()]
            st.markdown(f"""
            <div class="card" style="border-left: 6px solid #d97706;">
                <div class="card-title">🔮 {topic} Prediction</div>
                <p style="font-size:1.05rem; line-height:1.7; color:#451a03;">{content}</p>
            </div>
            """, unsafe_allow_html=True)

            if api_key:
                with st.spinner("Consulting AI Astrologer..."):
                    ctx = f"Lagna {chart.lagna_sign}, Moon {chart.moon_sign}, Dasha {pred['dasha']['mahadasha']}"
                    prompt = f"Detailed Vedic prediction for {topic} in {year}. Context: {ctx}. Base reading: {content}"
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
                st.session_state["ai_ctx"] = (
                    f"Native {name}: Lagna {chart.lagna_sign}, Moon {chart.moon_sign} "
                    f"in {chart.nakshatras['Moon']['nakshatra']} pada {chart.nakshatras['Moon']['pada']}, "
                    f"Mahadasha {chart.dasha[0]['mahadasha']}."
                )
                st.success("Context loaded!")
            except Exception as e:
                st.error(str(e))

    with c2:
        st.markdown('<div class="card-title">💬 Your Question</div>', unsafe_allow_html=True)
        question = st.text_area("Ask anything about career, marriage, children, health, remedies...",
                                "What does my chart say about my career in 2026?", height=100)
        if st.button("🙏 Ask Astrologer", use_container_width=True):
            if not api_key:
                st.error("Please enter your OpenRouter API Key in the sidebar.")
            elif "ai_ctx" not in st.session_state:
                st.warning("Please load chart context first (left panel).")
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
                        st.error(f"API Error: {e}")

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
         "meaning": "Worship with devotion and faith — you shall attain perfection and great valor.", "type": "Positive"}
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
