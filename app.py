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
    page_title="Jyotish",
    page_icon="🕉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# MINIMAL CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.5px; color: #5c3a21; }
    .stButton>button { border-radius: 8px; font-weight: 500; }
    hr { margin: 1rem 0; border-color: #e5e7eb; }
    [data-testid="stSidebar"] { background-color: #fafaf9; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("🕉 Jyotish")
page = st.sidebar.radio("Navigate", [
    "Home", "Horoscope", "Matchmaking",
    "Predictions", "Varshphal", "AI Astrologer", "Ram Shalaka"
], label_visibility="collapsed")

with st.sidebar.expander("Settings"):
    use_demo = st.toggle("Demo mode (no ephemeris)", value=False)
    api_key = st.text_input("OpenRouter API Key", type="password")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

# ------------------------------------------------------------------
# GEOCODING
# ------------------------------------------------------------------
@st.cache_data
def geocode_city(name: str):
    try:
        from geopy.geocoders import Nominatim
        geo = Nominatim(user_agent="vedic-astro-suite/3.2", timeout=10)
        loc = geo.geocode(name, language="en")
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    return None

TIMEZONES = {
    "IST (UTC+5:30)": 5.5, "GMT (UTC+0)": 0.0, "BST (UTC+1)": 1.0,
    "EST (UTC-5)": -5.0, "CST (UTC-6)": -6.0, "MST (UTC-7)": -7.0,
    "PST (UTC-8)": -8.0, "CET (UTC+1)": 1.0, "JST (UTC+9)": 9.0,
    "AEST (UTC+10)": 10.0, "AEDT (UTC+11)": 11.0, "Custom": None
}

# ------------------------------------------------------------------
# COMPACT BIRTH FORM (isolated session keys per prefix)
# ------------------------------------------------------------------
def birth_input_form(key_prefix: str, default_name: str):
    keys = {
        "name": f"{key_prefix}_name",
        "date": f"{key_prefix}_date",
        "time": f"{key_prefix}_time",
        "lat": f"{key_prefix}_lat",
        "lon": f"{key_prefix}_lon",
        "tz": f"{key_prefix}_tz",
        "city": f"{key_prefix}_city",
    }
    defaults = {
        "name": default_name, "date": datetime(1991, 4, 12),
        "time": datetime.strptime("10:26", "%H:%M").time(),
        "lat": 25.42, "lon": 86.13, "tz": 5.5, "city": ""
    }
    for k, v in keys.items():
        if v not in st.session_state:
            st.session_state[v] = defaults[k]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Name", st.session_state[keys["name"]], key=f"{key_prefix}_w_name")
    with c2:
        dob = st.date_input("Date", st.session_state[keys["date"]], key=f"{key_prefix}_w_date")
    with c3:
        tob = st.time_input("Time", st.session_state[keys["time"]], step=60, key=f"{key_prefix}_w_time")

    for k, v in keys.items():
        if k in ("name", "date", "time"):
            st.session_state[v] = locals()[k]

    city = st.text_input("City (optional)", st.session_state[keys["city"]],
                         key=f"{key_prefix}_w_city", placeholder="City, State, Country")
    st.session_state[keys["city"]] = city

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("Find", key=f"{key_prefix}_w_find"):
            coords = geocode_city(city)
            if coords:
                st.session_state[keys["lat"]] = round(coords[0], 4)
                st.session_state[keys["lon"]] = round(coords[1], 4)
                st.toast(f"Found {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                st.error("Not found")
    with c2:
        tz_keys = list(TIMEZONES.keys())
        cur_tz = st.session_state[keys["tz"]]
        tz_idx = 0
        for i, v in enumerate(TIMEZONES.values()):
            if v == cur_tz:
                tz_idx = i
                break
        tz_choice = st.selectbox("TZ", tz_keys, index=tz_idx, key=f"{key_prefix}_w_tz")
        tz_val = TIMEZONES[tz_choice]
        if tz_val is None:
            tz_val = st.number_input("Offset", -12.0, 14.0, cur_tz, 0.5, key=f"{key_prefix}_w_tz_custom")
    with c3:
        lat = st.number_input("Lat", -90.0, 90.0, st.session_state[keys["lat"]], key=f"{key_prefix}_w_lat")
    with c4:
        lon = st.number_input("Lon", -180.0, 180.0, st.session_state[keys["lon"]], key=f"{key_prefix}_w_lon")

    st.session_state[keys["lat"]] = lat
    st.session_state[keys["lon"]] = lon
    st.session_state[keys["tz"]] = tz_val
    return name, dob, tob, lat, lon, tz_val

# ------------------------------------------------------------------
# CHART COMPUTATION
# ------------------------------------------------------------------
def do_compute(name, dob, tob, lat, lon, tz):
    with st.spinner("Computing sidereal chart (Lahiri)..."):
        try:
            if use_demo:
                chart = generate_demo_chart()
                st.info("Demo mode — install pyswisseph for live calculations.")
            else:
                chart = compute_chart(dob.year, dob.month, dob.day,
                                      tob.hour, tob.minute, lat, lon, tz)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    return chart

# ------------------------------------------------------------------
# COMPACT CHARTS
# ------------------------------------------------------------------
def draw_north_indian_chart(chart: ChartData, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    diamond = plt.Polygon([(5, 10), (10, 5), (5, 0), (0, 5)], fill=False,
                          edgecolor='#8c6b4a', linewidth=1.5)
    ax.add_patch(diamond)
    ax.plot([5, 5], [10, 0], color='#8c6b4a', linewidth=0.8, alpha=0.6)
    ax.plot([0, 10], [5, 5], color='#8c6b4a', linewidth=0.8, alpha=0.6)
    ax.plot([2.5, 7.5], [7.5, 2.5], color='#8c6b4a', linewidth=0.6, alpha=0.4, linestyle=':')
    ax.plot([2.5, 7.5], [2.5, 7.5], color='#8c6b4a', linewidth=0.6, alpha=0.4, linestyle=':')

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
        ax.text(pos[0], pos[1] + 0.45, str(house_num), ha='center', va='center',
                fontsize=6, color='#9ca3af', fontweight='bold', alpha=0.7)
        ax.text(pos[0], pos[1], short, ha='center', va='center',
                fontsize=9, color='#5c3a21', fontweight='bold')
        ax.text(pos[0], pos[1] - 0.35, SIGN_SANSKRIT[sign][:4], ha='center', va='center',
                fontsize=5, color='#a05a2c', alpha=0.8)

    symbols = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
               "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋"}
    colors = {"Sun": "#b45309", "Moon": "#4b5563", "Mars": "#dc2626",
              "Mercury": "#059669", "Jupiter": "#92400e", "Venus": "#db2777",
              "Saturn": "#374151", "Rahu": "#6d28d9", "Ketu": "#6d28d9"}

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
        start_x = pos[0] - (n-1)*0.20
        for i, p in enumerate(planets):
            ax.text(start_x + i*0.40, pos[1] - 0.75, symbols.get(p, p),
                    ha='center', va='center', fontsize=10, color=colors.get(p, '#1f2937'),
                    fontweight='bold')

    ax.text(5.0, 9.0, "▲ LAGNA", ha='center', va='center', fontsize=7,
            color='#b91c1c', fontweight='bold', alpha=0.9)
    ax.set_title(title, fontsize=11, color='#5c3a21', fontweight='bold', pad=10)
    plt.tight_layout(pad=0.3)
    return fig


def draw_circular_chart(chart: ChartData, title: str):
    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    for i in range(12):
        theta_start = np.radians(i*30)
        theta_end = np.radians((i+1)*30)
        ax.fill_between(np.linspace(theta_start, theta_end, 30), 0.3, 1.0,
                        color='#fafaf9', alpha=0.9, edgecolor='#d6d3d1', linewidth=0.5)
        angle = np.radians(i*30 + 15)
        ax.text(angle, 0.92, f"{ZODIAC_SHORT[i]}", ha='center', va='center',
                fontsize=7, color='#5c3a21', fontweight='bold')

    symbols = {"Sun": "☉", "Moon": "☽", "Mars": "♂", "Mercury": "☿",
               "Jupiter": "♃", "Venus": "♀", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋"}
    colors = {"Sun": "#b45309", "Moon": "#4b5563", "Mars": "#dc2626",
              "Mercury": "#059669", "Jupiter": "#92400e", "Venus": "#db2777",
              "Saturn": "#374151", "Rahu": "#6d28d9", "Ketu": "#6d28d9"}

    used_bins = {}
    for planet, lon in chart.planets.items():
        base = lon % 360
        bin_id = int(base / 6)
        offset = used_bins.get(bin_id, 0) * 0.06
        used_bins[bin_id] = used_bins.get(bin_id, 0) + 1
        angle = np.radians(base)
        dist = 0.55 + offset
        ax.text(angle, dist, symbols.get(planet, planet), fontsize=10,
                ha='center', va='center', color=colors.get(planet, '#1f2937'),
                fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    asc_angle = np.radians(chart.ascendant)
    ax.plot([asc_angle, asc_angle], [0.3, 1.0], color='#b91c1c', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(asc_angle, 1.02, 'ASC', ha='center', va='center', color='#b91c1c', fontsize=7, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_title(title, fontsize=11, color='#5c3a21', fontweight='bold', pad=15)
    plt.tight_layout(pad=0.3)
    return fig

# ------------------------------------------------------------------
# TABLES
# ------------------------------------------------------------------
def planet_table(chart: ChartData):
    rows = []
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    for p in chart.planets.keys():
        sign, deg = longitude_to_sign(chart.planets[p])
        nak = chart.nakshatras[p]
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        rows.append({
            "Planet": p,
            "Sign": f"{sign} ({SIGN_SANSKRIT[sign]})",
            "Deg": f"{deg:.2f}°",
            "H": house,
            "Nakshatra": nak["nakshatra"],
            "Pada": nak["pada"],
            "Navamsa": chart.navamsa[p],
            "Dignity": chart.dignities.get(p, ""),
            "Retro": "R" if chart.retrograde.get(p, False) else ""
        })
    return pd.DataFrame(rows)

def varga_table(chart: ChartData):
    rows = []
    for p in chart.planets.keys():
        rows.append({
            "Planet": p,
            "D1": longitude_to_sign(chart.planets[p])[0],
            "D9": chart.navamsa[p],
            "D3": chart.drekkana[p],
            "D7": chart.saptamsa[p],
            "D10": chart.dasamsa[p],
            "D12": chart.dwadasamsa[p]
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# SAVE / LOAD
# ------------------------------------------------------------------
def save_chart_ui(chart: ChartData, name: str):
    c1, c2 = st.columns(2)
    with c1:
        chart_json = json.dumps(chart.to_dict(), indent=2, ensure_ascii=False)
        st.download_button("Download JSON", chart_json,
                           file_name=f"{name.replace(' ', '_')}_chart.json",
                           mime="application/json", use_container_width=True)
    with c2:
        summary_lines = [
            f"VEDIC CHART — {name}",
            f"Lagna: {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})",
            f"Moon: {chart.moon_sign} in {chart.nakshatras['Moon']['nakshatra']} P{chart.nakshatras['Moon']['pada']}",
            f"Sun: {chart.sun_sign}",
            f"Atmakaraka: {chart.atmakaraka}",
            "",
            "PLANETS"
        ]
        for p in chart.planets.keys():
            s, d = longitude_to_sign(chart.planets[p])
            summary_lines.append(f"{p}: {s} {d:.2f}° — {chart.nakshatras[p]['nakshatra']}")
        st.download_button("Download TXT", "\n".join(summary_lines),
                           file_name=f"{name.replace(' ', '_')}_summary.txt",
                           mime="text/plain", use_container_width=True)

# ------------------------------------------------------------------
# RENDER PREDICTIONS (minimalist)
# ------------------------------------------------------------------
def render_fired_rules(fired_rules):
    if not fired_rules:
        st.caption("No significant indicators found.")
        return
    for rule in fired_rules:
        icon = "✓" if rule["severity"] == "positive" else "!" if rule["severity"] in ["warning", "caution"] else "•"
        activation = " [Dasha]" if rule.get("activation") == "dasha_activated" else ""
        with st.expander(f"{icon} {rule['title']} ({rule['score']:+d}){activation}", expanded=False):
            st.write(rule['detail'])
            st.caption(f"Severity: {rule['severity']} · Score: {rule['score']:+d}")

# ------------------------------------------------------------------
# HOME
# ------------------------------------------------------------------
if page == "Home":
    st.title("Vedic Astrology Suite")
    st.caption("Sidereal calculations · Lahiri Ayanamsa · Vimshottari Dasha")

    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("📜", "Horoscope", "D1, D9, Dasha & divisionals"),
        ("💑", "Matchmaking", "Ashtakoota 36-point compatibility"),
        ("🔮", "Predictions", "Year-wise topic analysis"),
        ("📊", "Varshphal", "Solar return & Muntha"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(f"**{icon} {title}**\n\n{desc}")

    st.divider()
    uploaded = st.file_uploader("Load saved chart (JSON)", type=["json"])
    if uploaded:
        try:
            data = json.load(uploaded)
            chart = ChartData(
                planets=data["planets"], ascendant=data["ascendant"],
                lagna_sign=data["lagna_sign"],
                birth_date=datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None,
                lat=data.get("lat", 0), lon=data.get("lon", 0), tz=data.get("tz", 0),
                retrograde=data.get("retrograde", {})
            )
            st.session_state["computed_chart"] = chart
            st.session_state["computed_chart_name"] = "Loaded Chart"
            st.success("Chart loaded. Go to Horoscope.")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------------------------------------------------------
# HOROSCOPE
# ------------------------------------------------------------------
elif page == "Horoscope":
    st.header("Horoscope")
    name, dob, tob, lat, lon, tz = birth_input_form("chart", "Native")

    c1, c2 = st.columns([1, 1])
    with c1:
        compute_btn = st.button("Generate Chart", use_container_width=True)
    with c2:
        chart_style = st.selectbox("Style", ["North Indian", "South Indian"], label_visibility="collapsed")

    if compute_btn:
        chart = do_compute(name, dob, tob, lat, lon, tz)
        st.session_state["computed_chart"] = chart
        st.session_state["computed_chart_name"] = name

    if st.session_state.get("computed_chart"):
        chart = st.session_state["computed_chart"]
        name = st.session_state.get("computed_chart_name", "Native")

        save_chart_ui(chart, name)
        st.divider()

        tab_chart, tab_planets, tab_varga, tab_dasha, tab_analysis = st.tabs(
            ["Chart", "Planets", "Vargas", "Dasha", "Yogas"]
        )

        with tab_chart:
            c_left, c_right = st.columns([1, 1])
            with c_left:
                if "North" in chart_style:
                    fig = draw_north_indian_chart(chart, f"{name} — D1")
                else:
                    fig = draw_circular_chart(chart, f"{name} — D1")
                st.pyplot(fig)
                plt.close(fig)

            with c_right:
                current = chart.get_current_dasha_info()
                st.markdown(f"**Lagna:** {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})")
                st.markdown(f"**Moon:** {chart.moon_sign} in *{chart.nakshatras['Moon']['nakshatra']}* P{chart.nakshatras['Moon']['pada']}")
                st.markdown(f"**Sun:** {chart.sun_sign}")
                st.markdown(f"**Atmakaraka:** {chart.atmakaraka} · **Amatyakaraka:** {chart.amatyakaraka}")
                if current:
                    st.markdown(f"**Dasha:** {current['mahadasha']} MD · {current['antardasha']} AD")
                    st.caption(f"{current['mahadasha_start']} → {current['mahadasha_end']}")

        with tab_planets:
            st.dataframe(planet_table(chart), hide_index=True, use_container_width=True)

        with tab_varga:
            st.dataframe(varga_table(chart), hide_index=True, use_container_width=True)

        with tab_dasha:
            dasha_df = pd.DataFrame([
                {
                    "Planet": p.planet,
                    "Start": p.start_date.strftime("%d %b %Y"),
                    "End": p.end_date.strftime("%d %b %Y"),
                    "Years": f"{p.years:.2f}",
                    "Now": "●" if p.start_date <= datetime.now() < p.end_date else ""
                }
                for p in chart.dasha_periods
            ])
            st.dataframe(dasha_df, hide_index=True, use_container_width=True)

        with tab_analysis:
            yogas = analyze_general_yogas(chart)
            st.metric("Natal Yogas", yogas["yoga_count"], f"Strength: {yogas['yoga_strength']}")
            render_fired_rules(yogas.get("fired_yogas", []))

# ------------------------------------------------------------------
# MATCHMAKING
# ------------------------------------------------------------------
elif page == "Matchmaking":
    st.header("Ashtakoota Matchmaking")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Person 1")
        n1, d1, t1, lat1, lon1, tz1 = birth_input_form("m1", "Person 1")
    with c2:
        st.subheader("Person 2")
        n2, d2, t2, lat2, lon2, tz2 = birth_input_form("m2", "Person 2")

    if st.button("Calculate Compatibility", use_container_width=True):
        with st.spinner("Matching 8 Kootas..."):
            if use_demo:
                chart1 = generate_demo_chart()
                chart2 = generate_demo_chart()
                chart2.planets = {k: (v + 55) % 360 for k, v in chart2.planets.items()}
                chart2._compute_derived()
            else:
                chart1 = do_compute(n1, d1, t1, lat1, lon1, tz1)
                chart2 = do_compute(n2, d2, t2, lat2, lon2, tz2)

            res = calculate_ashtakoota(chart1, chart2)

            st.metric("Compatibility Score", f"{res['total']} / 36", res['verdict'])
            st.progress(res['total'] / 36, text=f"{res['percentage']}%")

            koota_data = []
            for k in ["varna", "vashya", "tara", "yoni", "graha_maitri", "gana", "bhakoot", "nadi"]:
                koota_data.append({
                    "Koota": k.replace("_", " ").title(),
                    "Score": f"{res[k]['score']}/{res[k]['max']}",
                    "Detail": res[k]['detail']
                })
            st.dataframe(pd.DataFrame(koota_data), hide_index=True, use_container_width=True)

            if res['doshas']:
                for d in res['doshas']:
                    st.warning(d)
            else:
                st.success("No major doshas detected.")

# ------------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------------
elif page == "Predictions":
    st.header("Yearly Predictions")
    name, dob, tob, lat, lon, tz = birth_input_form("pred", "Native")

    c1, c2 = st.columns([1, 2])
    with c1:
        year = st.selectbox("Year", list(range(2024, 2036)))
    with c2:
        topic = st.selectbox("Topic", ["All", "Career", "Marriage", "Children", "Health"])

    if st.button("Predict", use_container_width=True):
        chart = do_compute(name, dob, tob, lat, lon, tz)
        pred = get_year_prediction(chart, year)

        st.markdown(f"**{year} · {pred['dasha'].get('mahadasha','?')} MD / {pred['dasha'].get('antardasha','?')} AD**")
        if pred['sade_sati'].get('active'):
            st.warning(f"Sade Sati: {pred['sade_sati']['phase']}")
        if pred['kantaka_shani']:
            st.warning("Kantaka Shani active")

        topics = ["Career", "Marriage", "Children", "Health"] if topic == "All" else [topic]
        for t in topics:
            data = pred[t.lower()]
            with st.expander(f"{t} — {data['rating']} (Score {data['net_score']:+d})", expanded=(topic!="All")):
                st.caption(data['summary'])
                render_fired_rules(data.get('fired_rules', []))

        if api_key:
            st.divider()
            q = st.text_input("Ask AI about this prediction")
            if q and st.button("Ask", key="pred_ai"):
                with st.spinner("Consulting..."):
                    ctx = f"Lagna {chart.lagna_sign}, Moon {chart.moon_sign}, MD {pred['dasha'].get('mahadasha','N/A')}, Year {year}"
                    prompt = f"Based on this Vedic astrology context ({ctx}), answer: {q}"
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
                        ans = r.json()['choices'][0]['message']['content']
                        st.info(ans)
                    except Exception as e:
                        st.warning(f"AI error: {e}")

# ------------------------------------------------------------------
# VARSHPHAL
# ------------------------------------------------------------------
elif page == "Varshphal":
    st.header("Varshphal (Solar Return)")
    name, dob, tob, lat, lon, tz = birth_input_form("varsh", "Native")
    year = st.selectbox("Year", list(range(2024, 2036)))

    if st.button("Calculate Varshphal", use_container_width=True):
        chart = do_compute(name, dob, tob, lat, lon, tz)
        varsh = calculate_varshphal(chart, year)

        if not varsh:
            st.error("Unable to calculate Varshphal.")
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Muntha", varsh.get('muntha_sign', 'N/A'))
        c2.metric("House", f"H{varsh.get('muntha_house', 'N/A')}")
        c3.metric("Varsha Lagna", varsh.get('varsha_lagna', 'N/A'))

        st.caption(f"Varshphal date: {varsh.get('varshphal_date', 'N/A')} · Muntha lord: {varsh.get('muntha_lord', 'N/A')}")

        for theme in varsh.get('themes', []):
            st.write(f"• {theme}")

        if api_key:
            st.divider()
            q = st.text_input("Ask about this Solar Return")
            if q and st.button("Consult", key="varsh_ai"):
                with st.spinner("Analyzing..."):
                    ctx = f"Varshphal {year}: Muntha in {varsh.get('muntha_sign','')} H{varsh.get('muntha_house','')}, Varsha Lagna {varsh.get('varsha_lagna','')}"
                    prompt = f"Based on this Varshphal data ({ctx}), answer: {q}"
                    try:
                        r = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                                "messages": [{"role": "user", "content": prompt}]
                            }, timeout=30
                        )
                        st.info(r.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        st.warning(f"AI error: {e}")

# ------------------------------------------------------------------
# AI ASTROLOGER
# ------------------------------------------------------------------
elif page == "AI Astrologer":
    st.header("AI Vedic Astrologer")
    c1, c2 = st.columns([1, 2])
    with c1:
        name, dob, tob, lat, lon, tz = birth_input_form("ai", "Seeker")
        if st.button("Load Chart Context", use_container_width=True):
            chart = do_compute(name, dob, tob, lat, lon, tz)
            current = chart.get_current_dasha_info()
            st.session_state["ai_ctx"] = (
                f"Native: Lagna {chart.lagna_sign}, Moon {chart.moon_sign} "
                f"in {chart.nakshatras['Moon']['nakshatra']} P{chart.nakshatras['Moon']['pada']}, "
                f"MD {current.get('mahadasha','?')}, AD {current.get('antardasha','?')}."
            )
            st.success("Context loaded.")

    with c2:
        if not api_key:
            st.warning("Enter API key in sidebar.")
        elif "ai_ctx" not in st.session_state:
            st.info("Load chart context first (left panel).")
        else:
            q = st.text_area("Your question", "What does my chart say about career in 2026?", height=80)
            if st.button("Ask Astrologer", use_container_width=True):
                with st.spinner("Consulting..."):
                    try:
                        r = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={
                                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                                "messages": [
                                    {"role": "system", "content": f"You are a wise Vedic Astrologer. {st.session_state['ai_ctx']}"},
                                    {"role": "user", "content": q}
                                ]
                            }, timeout=30
                        )
                        st.write(r.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"API error: {e}")

# ------------------------------------------------------------------
# RAM SHALAKA
# ------------------------------------------------------------------
elif page == "Ram Shalaka":
    st.header("Ram Shalaka")
    st.caption("Divine guidance from Shri Ram Charit Manas")

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

    if st.button("Seek Blessing", use_container_width=True):
        verse = random.choice(SHALAKA)
        st.session_state["last_shalaka"] = verse
        st.balloons()
        st.markdown(f"**{verse['text']}**")
        st.caption(verse['meaning'])
        st.badge(verse['type'])

    if api_key and st.session_state.get("last_shalaka"):
        st.divider()
        follow = st.text_input("Clarification or remedy regarding this shloka")
        if follow and st.button("Get Guidance", key="shalaka_ai"):
            with st.spinner("Reflecting..."):
                verse = st.session_state["last_shalaka"]
                prompt = f"Ram Shalaka said: '{verse['text']}' meaning: {verse['meaning']}. User asks: {follow}. Provide compassionate guidance."
                try:
                    r = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                            "messages": [{"role": "user", "content": prompt}]
                        }, timeout=30
                    )
                    st.write(r.json()['choices'][0]['message']['content'])
                except Exception as e:
                    st.warning(f"Error: {e}")
