import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
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
# CUSTOM CSS — Spiritual Gold/Saffron Theme
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #2c1810;
}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
    font-family: 'Cinzel', serif;
    color: #b8860b;
}
.stButton>button {
    background: linear-gradient(135deg, #ff9933 0%, #b8860b 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #e68a00 0%, #8b6508 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff8f0 0%, #fff0d9 100%);
}
.card {
    background: #fffaf3;
    border: 1px solid #e6d2b5;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(139, 69, 19, 0.05);
}
.score-excellent { color: #2e7d32; font-weight: 700; }
.score-good { color: #689f38; font-weight: 700; }
.score-average { color: #f9a825; font-weight: 700; }
.score-challenging { color: #c62828; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("🕉️ Jyotish App")
page = st.sidebar.radio("Navigate", [
    "🏠 Home", "📜 Horoscope", "💑 Matchmaking", 
    "🔮 Yearly Predictions", "❓ AI Astrologer", "🎲 Ram Shalaka"
])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
use_demo = st.sidebar.checkbox("Use Demo Data (no ephemeris needed)", value=False)
api_key = st.sidebar.text_input("OpenRouter API Key (optional)", type="password", 
                                help="For AI predictions. Free tier: google/gemini-2.0-flash-lite-preview-02-05:free")
if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
CITIES = {
    "Delhi, India": (28.6139, 77.2090, 5.5),
    "Mumbai, India": (19.0760, 72.8777, 5.5),
    "Bangalore, India": (12.9716, 77.5946, 5.5),
    "Chennai, India": (13.0827, 80.2707, 5.5),
    "Kolkata, India": (22.5726, 88.3639, 5.5),
    "Hyderabad, India": (17.3850, 78.4867, 5.5),
    "Pune, India": (18.5204, 73.8567, 5.5),
    "Ahmedabad, India": (23.0225, 72.5714, 5.5),
    "New York, USA": (40.7128, -74.0060, -5.0),
    "London, UK": (51.5074, -0.1278, 0.0),
    "Tokyo, Japan": (35.6762, 139.6503, 9.0),
    "Sydney, Australia": (-33.8688, 151.2093, 10.0),
    "Dubai, UAE": (25.2048, 55.2708, 4.0),
    "Singapore": (1.3521, 103.8198, 8.0),
    "Custom": (0.0, 0.0, 0.0)
}

def get_birth_inputs(key_prefix=""):
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Name", f"Person {key_prefix}", key=f"name_{key_prefix}")
    with c2:
        date = st.date_input("Date of Birth", datetime(1995, 6, 15), key=f"date_{key_prefix}")
    with c3:
        time = st.time_input("Time of Birth", datetime.strptime("10:30", "%H:%M").time(), key=f"time_{key_prefix}")
    
    c4, c5 = st.columns(2)
    with c4:
        city = st.selectbox("Birth Place", list(CITIES.keys()), key=f"city_{key_prefix}")
    with c5:
        if city == "Custom":
            lat = st.number_input("Latitude", -90.0, 90.0, 28.6, key=f"lat_{key_prefix}")
            lon = st.number_input("Longitude", -180.0, 180.0, 77.2, key=f"lon_{key_prefix}")
            tz = st.number_input("Timezone (+/- hrs)", -12.0, 14.0, 5.5, key=f"tz_{key_prefix}")
        else:
            lat, lon, tz = CITIES[city]
            st.caption(f"Lat: {lat}, Lon: {lon}, TZ: +{tz}")
    
    return name, date, time, lat, lon, tz

def draw_chart_wheel(chart: ChartData, title: str):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#1a0f0a')
    ax.set_facecolor('#1a0f0a')

    # House sectors
    colors = ['#2d1b0e' if i % 2 == 0 else '#1f1209' for i in range(12)]
    for i in range(12):
        theta = np.linspace(np.radians(i*30), np.radians((i+1)*30), 50)
        ax.fill_between(theta, 0.3, 1.0, color=colors[i], alpha=0.9)
        ax.plot([np.radians(i*30)]*2, [0.3, 1.0], color='#b8860b', linewidth=1.2)

    # Sign labels
    for i, sign in enumerate(ZODIAC):
        angle = np.radians(i*30 + 15)
        ax.text(angle, 0.88, f"{sign}\n({SIGN_SANSKRIT[sign]})", 
                ha='center', va='center', fontsize=8, color='#daa520', fontweight='bold')

    # Planets
    symbols = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
               "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋"}
    
    # Spread planets visually if conjunct
    used_angles = {}
    for planet, lon in chart.planets.items():
        base_angle = lon % 360
        # Offset if occupied
        offset = used_angles.get(int(base_angle/5)*5, 0) * 0.04
        used_angles[int(base_angle/5)*5] = used_angles.get(int(base_angle/5)*5, 0) + 1
        angle = np.radians(base_angle + 90)  # rotate to start from top
        dist = 0.55 + offset
        ax.text(angle, dist, symbols.get(planet, planet), fontsize=14, 
                ha='center', va='center', color='#ff6b6b' if planet in ['Mars', 'Saturn'] else '#ffd700',
                fontweight='bold')

    # Ascendant
    asc_angle = np.radians(chart.ascendant + 90)
    ax.plot([asc_angle, asc_angle], [0.3, 1.0], color='#ff4500', linewidth=2.5, linestyle='--')
    ax.text(asc_angle, 0.95, 'ASC ▲', ha='center', va='center', color='#ff4500', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_title(title, fontsize=16, color='#ffd700', fontweight='bold', pad=20, fontfamily='serif')
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
    st.title("🕉️ Vedic Astrology Suite")
    st.markdown("""
    <div class="card">
    <h3>Welcome to the Jyotish Calculator</h3>
    <p>This app generates <b>free horoscope charts</b>, performs detailed <b>Ashtakoota matchmaking (36 points)</b>, 
    and answers questions on <b>Career, Marriage, Children & Health</b> using Vimshottari Dasha + Transit logic.</p>
    <ul>
        <li>📜 <b>Horoscope</b> — Planetary positions, Nakshatra, Navamsa (D9), Vimshottari Dasha</li>
        <li>💑 <b>Matchmaking</b> — 8 Kootas with detailed scoring & verdict</li>
        <li>🔮 <b>Predictions</b> — Year-wise analysis using Dasha & Gochar (transit)</li>
        <li>🤖 <b>AI Astrologer</b> — Ask anything (powered by OpenRouter free tier)</li>
        <li>🎲 <b>Ram Shalaka</b> — Divine guidance from Shri Ram Charit Manas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "📜 Horoscope":
    st.title("📜 Free Horoscope Chart")
    name, date, time, lat, lon, tz = get_birth_inputs("A")
    
    if st.button("Generate Chart", key="gen_chart"):
        with st.spinner("Calculating planetary positions with Lahiri Ayanamsa..."):
            try:
                if use_demo:
                    chart = generate_demo_chart()
                    st.info("Showing demo chart (install pyswisseph for real calculations)")
                else:
                    chart = compute_chart(date.year, date.month, date.day, 
                                          time.hour, time.minute, lat, lon, tz)
            except Exception as e:
                st.error(f"Ephemeris error: {e}. Switch on 'Use Demo Data' in sidebar.")
                st.stop()

            st.session_state["chart"] = chart
            st.session_state["chart_name"] = name

    if "chart" in st.session_state:
        chart = st.session_state["chart"]
        name = st.session_state["chart_name"]
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.pyplot(draw_chart_wheel(chart, f"{name}'s Horoscope"))
        with c2:
            st.markdown(f"""
            <div class="card">
                <h4>🌟 Birth Details</h4>
                <p><b>Lagna:</b> {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})</p>
                <p><b>Moon Sign:</b> {chart.moon_sign} ({SIGN_SANSKRIT[chart.moon_sign]})</p>
                <p><b>Sun Sign:</b> {chart.sun_sign} ({SIGN_SANSKRIT[chart.sun_sign]})</p>
                <p><b>Nakshatra:</b> {chart.nakshatras['Moon']['nakshatra']} 
                   (Pada {chart.nakshatras['Moon']['pada']})</p>
                <p><b>Navamsa Lagna:</b> {chart.navamsa['Moon']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>📖 Vimshottari Dasha</h4>
            """, unsafe_allow_html=True)
            dasha_df = pd.DataFrame(chart.dasha)
            st.dataframe(dasha_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🪐 Planetary Positions")
        st.dataframe(planet_table(chart), hide_index=True, use_container_width=True)

elif page == "💑 Matchmaking":
    st.title("💑 Ashtakoota Matchmaking (36 Points)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Person 1 (Male / Groom)")
        n1, d1, t1, lat1, lon1, tz1 = get_birth_inputs("1")
    with c2:
        st.markdown("#### Person 2 (Female / Bride)")
        n2, d2, t2, lat2, lon2, tz2 = get_birth_inputs("2")
    
    if st.button("Calculate Compatibility"):
        with st.spinner("Matching 8 Kootas..."):
            try:
                if use_demo:
                    chart1 = generate_demo_chart()
                    chart2 = generate_demo_chart()
                    # Offset chart2 slightly for demo variety
                    chart2.planets = {k: (v + 45) % 360 for k, v in chart2.planets.items()}
                    chart2._compute_derived()
                else:
                    chart1 = compute_chart(d1.year, d1.month, d1.day, t1.hour, t1.minute, lat1, lon1, tz1)
                    chart2 = compute_chart(d2.year, d2.month, d2.day, t2.hour, t2.minute, lat2, lon2, tz2)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

            result = calculate_ashtakoota(chart1, chart2)
            
            # Verdict banner
            color_class = f"score-{result['verdict'].lower().replace(' ', '-')}"
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h2>Total Score: {result['total']} / 36</h2>
                <h1 class="{color_class}">{result['verdict']} ({result['percentage']}%)</h1>
                <progress value="{result['total']}" max="36" style="width:80%; height:20px;"></progress>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed table
            rows = []
            for k in ["varna", "vashya", "tara", "yoni", "graha_maitri", "gana", "bhakoot", "nadi"]:
                rows.append({
                    "Koota": k.replace("_", " ").title(),
                    "Score": f"{result[k]['score']} / {result[k]['max']}",
                    "Detail": result[k]['detail']
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            
            # Interpretation
            st.markdown("""
            <div class="card">
                <h4>📖 Interpretation</h4>
                <ul>
                    <li><b>Varna:</b> Spiritual compatibility & ego harmony</li>
                    <li><b>Vashya:</b> Mutual control & attraction</li>
                    <li><b>Tara:</b> Destiny / birth star alignment</li>
                    <li><b>Yoni:</b> Sexual compatibility & intimacy</li>
                    <li><b>Graha Maitri:</b> Planetary friendship of Moon signs</li>
                    <li><b>Gana:</b> Temperament match (Deva / Manushya / Rakshasa)</li>
                    <li><b>Bhakoot:</b> Relative Moon sign position (2/12, 6/8 checked)</li>
                    <li><b>Nadi:</b> Health & progeny compatibility (same = 0)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "🔮 Yearly Predictions":
    st.title("🔮 Yearly Predictions by Topic")
    
    name, date, time, lat, lon, tz = get_birth_inputs("P")
    year = st.selectbox("Select Year", list(range(2024, 2036)))
    topic = st.radio("Topic", ["Career", "Marriage", "Children", "Health"], horizontal=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing Dasha & Transits..."):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
            except Exception as e:
                st.error(str(e))
                st.stop()
            
            pred = get_year_prediction(chart, year)
            
            st.markdown(f"""
            <div class="card">
                <h4>📅 Year {pred['year']} Analysis</h4>
                <p><b>Mahadasha Running:</b> {pred['dasha']['mahadasha']} 
                   ({pred['dasha']['years']} yrs balance)</p>
                <p><b>Transit Saturn:</b> {pred['transits']['Saturn']} | 
                   <b>Transit Jupiter:</b> {pred['transits']['Jupiter']}</p>
                <p>{pred['sade_sati']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            content = pred[topic.lower()]
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #b8860b;">
                <h3>🔮 {topic} Prediction</h3>
                <p style="font-size:1.1rem; line-height:1.6;">{content}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Deeper AI insight if key available
            if api_key:
                with st.spinner("Consulting AI Astrologer..."):
                    context = f"Chart: Lagna {chart.lagna_sign}, Moon {chart.moon_sign}, Dasha {pred['dasha']['mahadasha']}"
                    prompt = f"Give a detailed Vedic astrology prediction for {topic} in {year} based on this data: {context}. {content}"
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
                            },
                            timeout=30
                        )
                        ai_text = r.json()['choices'][0]['message']['content']
                        st.markdown(f"""
                        <div class="card" style="background:#f0f8ff; border-left:5px solid #4169e1;">
                            <h4>🤖 AI Insight</h4>
                            <p>{ai_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"AI unavailable: {e}")

elif page == "❓ AI Astrologer":
    st.title("❓ Ask the AI Vedic Astrologer")
    st.caption("Powered by OpenRouter (free Gemini model). Enter your birth details + question.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        name, date, time, lat, lon, tz = get_birth_inputs("AI")
        if st.button("Load Chart Context"):
            try:
                chart = generate_demo_chart() if use_demo else compute_chart(
                    date.year, date.month, date.day, time.hour, time.minute, lat, lon, tz)
                st.session_state["ai_context"] = (
                    f"Native: {name}. Lagna {chart.lagna_sign}, Moon {chart.moon_sign} "
                    f"in {chart.nakshatras['Moon']['nakshatra']} pada {chart.nakshatras['Moon']['pada']}. "
                    f"Current Mahadasha: {chart.dasha[0]['mahadasha']}."
                )
                st.success("Chart context loaded!")
            except Exception as e:
                st.error(str(e))
    
    with c2:
        question = st.text_area("Your Question", "What does my chart say about my career in 2026?", height=80)
        if st.button("Ask Astrologer"):
            if not api_key:
                st.error("Please enter your OpenRouter API Key in the sidebar.")
            elif "ai_context" not in st.session_state:
                st.warning("Please load chart context first (left column).")
            else:
                with st.spinner("Consulting the stars..."):
                    try:
                        r = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                                "messages": [
                                    {"role": "system", "content": f"You are a wise Vedic Astrologer. {st.session_state['ai_context']}"},
                                    {"role": "user", "content": question}
                                ]
                            },
                            timeout=30
                        )
                        answer = r.json()['choices'][0]['message']['content']
                        st.markdown(f"""
                        <div class="card" style="background:#fffaf3;">
                            <h4>🪔 Response</h4>
                            <p style="white-space:pre-wrap;">{answer}</p>
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
    
    if st.button("🙏 Seek Blessing", key="shalaka_btn"):
        import random
        verse = random.choice(SHALAKA)
        st.balloons()
        st.markdown(f"""
        <div class="card" style="text-align:center; background:linear-gradient(135deg, #fff8f0, #fff0d9);">
            <h2 style="color:#8b4513; font-family:serif;">"{verse['text']}"</h2>
            <p style="font-size:1.2rem; color:#5d4037;"><i>{verse['meaning']}</i></p>
            <span style="background:#ffd700; padding:4px 12px; border-radius:20px; color:#5d4037; font-weight:bold;">
                {verse['type']}
            </span>
        </div>
        """, unsafe_allow_html=True)
