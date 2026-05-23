"""
Vedic Astrology Calculation Engine
Implements: Lahiri Ayanamsa, Whole-Sign Houses, Nakshatra/Pada, Navamsa,
Vimshottari Dasha, Ashtakoota Matchmaking, Transit Analysis, Yearly Predictions
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import swisseph as swe

# ------------------------------------------------------------------
# EPHEMERIS SETUP
# ------------------------------------------------------------------
swe.set_sid_mode(swe.SIDM_LAHIRI)  # Standard Vedic ayanamsa

# ------------------------------------------------------------------
# CONSTANTS & DATA TABLES (from your reference document)
# ------------------------------------------------------------------
ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

SIGN_SANSKRIT = {
    "Aries": "Mesha", "Taurus": "Vrishabha", "Gemini": "Mithuna",
    "Cancer": "Karka", "Leo": "Simha", "Virgo": "Kanya",
    "Libra": "Tula", "Scorpio": "Vrischika", "Sagittarius": "Dhanu",
    "Capricorn": "Makara", "Aquarius": "Kumbha", "Pisces": "Meena"
}

SIGN_LORD = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury",
    "Cancer": "Moon", "Leo": "Sun", "Virgo": "Mercury",
    "Libra": "Venus", "Scorpio": "Mars", "Sagittarius": "Jupiter",
    "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

SIGN_ELEMENT = {
    "Aries": "Fire", "Taurus": "Earth", "Gemini": "Air",
    "Cancer": "Water", "Leo": "Fire", "Virgo": "Earth",
    "Libra": "Air", "Scorpio": "Water", "Sagittarius": "Fire",
    "Capricorn": "Earth", "Aquarius": "Air", "Pisces": "Water"
}

SIGN_QUALITY = {
    "Aries": "Movable", "Taurus": "Fixed", "Gemini": "Dual",
    "Cancer": "Movable", "Leo": "Fixed", "Virgo": "Dual",
    "Libra": "Movable", "Scorpio": "Fixed", "Sagittarius": "Dual",
    "Capricorn": "Movable", "Aquarius": "Fixed", "Pisces": "Dual"
}

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

# 9-lord cycle repeated 3x = 27 nakshatras
NAKSHATRA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"] * 3

NAKSHATRA_GANA = {
    "Ashwini": "Deva", "Bharani": "Manushya", "Krittika": "Rakshasa",
    "Rohini": "Manushya", "Mrigashira": "Deva", "Ardra": "Manushya",
    "Punarvasu": "Deva", "Pushya": "Deva", "Ashlesha": "Rakshasa",
    "Magha": "Rakshasa", "Purva Phalguni": "Manushya", "Uttara Phalguni": "Manushya",
    "Hasta": "Deva", "Chitra": "Rakshasa", "Swati": "Deva",
    "Vishakha": "Rakshasa", "Anuradha": "Deva", "Jyeshtha": "Rakshasa",
    "Mula": "Rakshasa", "Purva Ashadha": "Manushya", "Uttara Ashadha": "Manushya",
    "Shravana": "Deva", "Dhanishta": "Rakshasa", "Shatabhisha": "Rakshasa",
    "Purva Bhadrapada": "Manushya", "Uttara Bhadrapada": "Manushya", "Revati": "Deva"
}

NAKSHATRA_YONI = {
    "Ashwini": "Horse", "Bharani": "Elephant", "Krittika": "Sheep",
    "Rohini": "Serpent", "Mrigashira": "Serpent", "Ardra": "Dog",
    "Punarvasu": "Cat", "Pushya": "Sheep", "Ashlesha": "Cat",
    "Magha": "Rat", "Purva Phalguni": "Rat", "Uttara Phalguni": "Cow",
    "Hasta": "Buffalo", "Chitra": "Tiger", "Swati": "Buffalo",
    "Vishakha": "Tiger", "Anuradha": "Deer", "Jyeshtha": "Deer",
    "Mula": "Dog", "Purva Ashadha": "Monkey", "Uttara Ashadha": "Mongoose",
    "Shravana": "Monkey", "Dhanishta": "Lion", "Shatabhisha": "Horse",
    "Purva Bhadrapada": "Lion", "Uttara Bhadrapada": "Cow", "Revati": "Elephant"
}

NAKSHATRA_NADI = {
    "Ashwini": "Vata", "Bharani": "Pitta", "Krittika": "Kapha",
    "Rohini": "Vata", "Mrigashira": "Pitta", "Ardra": "Kapha",
    "Punarvasu": "Vata", "Pushya": "Pitta", "Ashlesha": "Kapha",
    "Magha": "Vata", "Purva Phalguni": "Pitta", "Uttara Phalguni": "Kapha",
    "Hasta": "Vata", "Chitra": "Pitta", "Swati": "Kapha",
    "Vishakha": "Vata", "Anuradha": "Pitta", "Jyeshtha": "Kapha",
    "Mula": "Vata", "Purva Ashadha": "Pitta", "Uttara Ashadha": "Kapha",
    "Shravana": "Vata", "Dhanishta": "Pitta", "Shatabhisha": "Kapha",
    "Purva Bhadrapada": "Vata", "Uttara Bhadrapada": "Pitta", "Revati": "Kapha"
}

VARNA_MAP = {"Water": "Brahmin", "Fire": "Kshatriya", "Earth": "Vaishya", "Air": "Shudra"}

VASHYA_MAP = {
    "Aries": "Quadruped", "Taurus": "Quadruped", "Gemini": "Human",
    "Cancer": "Water", "Leo": "Quadruped", "Virgo": "Human",
    "Libra": "Human", "Scorpio": "Keet", "Sagittarius": "Human",
    "Capricorn": "Quadruped", "Aquarius": "Human", "Pisces": "Water"
}

DASHA_YEARS = {
    "Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
    "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17
}
DASHA_SEQUENCE = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]

PLANET_IDS = [swe.SUN, swe.MOON, swe.MARS, swe.MERCURY, swe.JUPITER, swe.VENUS, swe.SATURN, swe.TRUE_NODE]
PLANET_NAMES = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu"]

HOUSE_MEANINGS = {
    1: "Self / Body", 2: "Wealth / Family", 3: "Courage / Siblings",
    4: "Mother / Home", 5: "Intelligence / Children", 6: "Disease / Enemies",
    7: "Marriage / Partnership", 8: "Longevity / Occult", 9: "Fortune / Dharma",
    10: "Career / Status", 11: "Gains / Friends", 12: "Loss / Liberation"
}

# ------------------------------------------------------------------
# CORE MATH
# ------------------------------------------------------------------
NAKSHATRA_SIZE = 13 + 20 / 60   # 13°20′
PADA_SIZE = 3 + 20 / 60         # 3°20′

def longitude_to_sign(longitude: float) -> Tuple[str, float]:
    idx = int(longitude // 30) % 12
    return ZODIAC[idx], longitude % 30

def get_nakshatra(longitude: float) -> Tuple[str, int, float]:
    lon = longitude % 360
    nak_idx = int(lon // NAKSHATRA_SIZE)
    rem = lon % NAKSHATRA_SIZE
    pada = int(rem // PADA_SIZE) + 1
    return NAKSHATRAS[nak_idx % 27], pada, rem

def get_navamsa(longitude: float) -> str:
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // (10 / 3))  # 3°20′ each
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12
    else:
        start = (sign_idx + 4) % 12
    return ZODIAC[(start + part) % 12]

# ------------------------------------------------------------------
# CHART DATA CLASS
# ------------------------------------------------------------------
class ChartData:
    def __init__(self, planets: Dict[str, float], ascendant: float, lagna_sign: str):
        self.planets = planets
        self.ascendant = ascendant
        self.lagna_sign = lagna_sign
        self.moon_sign = longitude_to_sign(planets["Moon"])[0]
        self.sun_sign = longitude_to_sign(planets["Sun"])[0]
        self.nakshatras = {}
        self.navamsa = {}
        self.dasha = []
        self._compute_derived()

    def _compute_derived(self):
        for p, lon in self.planets.items():
            nak, pada, rem = get_nakshatra(lon)
            self.nakshatras[p] = {
                "nakshatra": nak, "pada": pada,
                "lord": NAKSHATRA_LORDS[NAKSHATRAS.index(nak)],
                "deg_in_nakshatra": round(rem, 2)
            }
            self.navamsa[p] = get_navamsa(lon)
        # Vimshottari from Moon
        moon_lon = self.planets["Moon"]
        moon_nak = self.nakshatras["Moon"]["nakshatra"]
        self.dasha = calculate_vimshottari(moon_nak, moon_lon)

# ------------------------------------------------------------------
# VIMSHOTTARI DASHA
# ------------------------------------------------------------------
def calculate_vimshottari(moon_nakshatra: str, moon_longitude: float) -> List[Dict]:
    nak_idx = NAKSHATRAS.index(moon_nakshatra)
    lord_idx = nak_idx % 9
    start_lord = DASHA_SEQUENCE[lord_idx]

    nak_start = nak_idx * NAKSHATRA_SIZE
    degrees_covered = moon_longitude - nak_start
    remaining = NAKSHATRA_SIZE - (degrees_covered % NAKSHATRA_SIZE)
    fraction = remaining / NAKSHATRA_SIZE
    balance = fraction * DASHA_YEARS[start_lord]

    results = []
    for i in range(9):
        lord = DASHA_SEQUENCE[(lord_idx + i) % 9]
        years = balance if i == 0 else DASHA_YEARS[lord]
        results.append({
            "mahadasha": lord,
            "years": round(years, 2),
            "is_start": i == 0
        })
    return results

# ------------------------------------------------------------------
# CHART CALCULATION (SWISS EPHEMERIS)
# ------------------------------------------------------------------
def compute_chart(year, month, day, hour, minute, lat, lon, tz_offset=0.0) -> ChartData:
    # Julian Day in UT
    jd = swe.julday(year, month, day, hour + minute / 60.0 - tz_offset)

    # Whole Sign houses (Vedic standard: each house = 30°, starting at Lagna)
    # We use swe.houses_ex to get exact ascendant, then assign whole signs
    houses = swe.houses_ex(jd, lat, lon, b'W', swe.FLG_SIDEREAL)
    ascendant = houses[1][0]  # Ascendant longitude

    planets = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        planets[pname] = res[0][0]

    # Ketu opposite Rahu
    planets["Ketu"] = (planets["Rahu"] + 180.0) % 360.0

    lagna_sign, _ = longitude_to_sign(ascendant)
    return ChartData(planets, ascendant, lagna_sign)

def get_transits(year: int, month: int = 6, day: int = 15) -> Dict[str, float]:
    jd = swe.julday(year, month, day, 12.0)
    transits = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        transits[pname] = res[0][0]
    transits["Ketu"] = (transits["Rahu"] + 180.0) % 360.0
    return transits

# ------------------------------------------------------------------
# ASHTAKOOTA MATCHMAKING (36 Points)
# ------------------------------------------------------------------
def get_tara_score(n1: int, n2: int) -> int:
    diff = (n2 - n1) % 27
    tara = (diff % 9) + 1  # 1–9 cycle
    return 0 if tara in [3, 5, 7] else 3

def get_yoni_score(y1: str, y2: str) -> int:
    friendly = {
        "Horse": ["Horse", "Monkey", "Mongoose"],
        "Elephant": ["Elephant", "Sheep"],
        "Sheep": ["Sheep", "Elephant", "Monkey"],
        "Serpent": ["Serpent", "Mongoose"],
        "Dog": ["Dog", "Deer"],
        "Cat": ["Cat", "Mongoose"],
        "Rat": ["Rat", "Serpent"],
        "Cow": ["Cow", "Buffalo"],
        "Buffalo": ["Buffalo", "Cow"],
        "Tiger": ["Tiger", "Deer"],
        "Deer": ["Deer", "Tiger", "Dog"],
        "Monkey": ["Monkey", "Horse", "Sheep"],
        "Mongoose": ["Mongoose", "Serpent", "Cat"],
        "Lion": ["Lion", "Dog"],
    }
    if y1 == y2:
        return 4
    if y2 in friendly.get(y1, []):
        return 4
    return 2  # neutral default

def get_graha_maitri_score(lord1: str, lord2: str) -> int:
    friends = {
        "Sun": ["Moon", "Mars", "Jupiter"],
        "Moon": ["Sun", "Mercury"],
        "Mars": ["Sun", "Moon", "Jupiter"],
        "Mercury": ["Sun", "Venus"],
        "Jupiter": ["Sun", "Moon", "Mars"],
        "Venus": ["Mercury", "Saturn"],
        "Saturn": ["Mercury", "Venus"],
    }
    enemies = {
        "Sun": ["Venus", "Saturn"],
        "Moon": ["Rahu", "Ketu"],
        "Mars": ["Mercury"],
        "Mercury": ["Moon"],
        "Jupiter": ["Mercury", "Venus"],
        "Venus": ["Sun", "Moon"],
        "Saturn": ["Sun", "Moon", "Mars"],
    }
    if lord1 == lord2:
        return 5
    if lord2 in friends.get(lord1, []):
        return 5
    if lord2 in enemies.get(lord1, []):
        return 0
    return 3  # neutral

def get_gana_score(g1: str, g2: str) -> int:
    if g1 == g2:
        return 6
    if (g1 == "Deva" and g2 == "Manushya") or (g1 == "Manushya" and g2 == "Deva"):
        return 6
    if (g1 == "Manushya" and g2 == "Rakshasa") or (g1 == "Rakshasa" and g2 == "Manushya"):
        return 3
    return 0  # Deva-Rakshasa

def get_bhakoot_score(idx1: int, idx2: int) -> int:
    diff = (idx2 - idx1) % 12
    if diff in [2, 10, 6, 8]:  # 2/12, 6/8 bad
        return 0
    return 7

def calculate_ashtakoota(c1: ChartData, c2: ChartData) -> Dict:
    m1, m2 = c1.moon_sign, c2.moon_sign
    n1 = c1.nakshatras["Moon"]["nakshatra"]
    n2 = c2.nakshatras["Moon"]["nakshatra"]
    i1, i2 = ZODIAC.index(m1), ZODIAC.index(m2)
    ni1, ni2 = NAKSHATRAS.index(n1), NAKSHATRAS.index(n2)

    varna1 = VARNA_MAP[SIGN_ELEMENT[m1]]
    varna2 = VARNA_MAP[SIGN_ELEMENT[m2]]
    varna = 1 if varna1 == varna2 else 0

    vashya1 = VASHYA_MAP[m1]
    vashya2 = VASHYA_MAP[m2]
    vashya = 2 if (vashya1 == vashya2 or {vashya1, vashya2} == {"Human", "Water"}) else 1 if "Human" in [vashya1, vashya2] else 0

    tara = get_tara_score(ni1, ni2)

    yoni1 = NAKSHATRA_YONI[n1]
    yoni2 = NAKSHATRA_YONI[n2]
    yoni = get_yoni_score(yoni1, yoni2)

    graha = get_graha_maitri_score(SIGN_LORD[m1], SIGN_LORD[m2])

    gana1 = NAKSHATRA_GANA[n1]
    gana2 = NAKSHATRA_GANA[n2]
    gana = get_gana_score(gana1, gana2)

    bhakoot = get_bhakoot_score(i1, i2)

    nad1 = NAKSHATRA_NADI[n1]
    nad2 = NAKSHATRA_NADI[n2]
    nadi = 0 if nad1 == nad2 else 8

    total = varna + vashya + tara + yoni + graha + gana + bhakoot + nadi

    return {
        "varna": {"score": varna, "max": 1, "detail": f"{varna1} vs {varna2}"},
        "vashya": {"score": vashya, "max": 2, "detail": f"{vashya1} vs {vashya2}"},
        "tara": {"score": tara, "max": 3, "detail": f"{n1} vs {n2}"},
        "yoni": {"score": yoni, "max": 4, "detail": f"{yoni1} vs {yoni2}"},
        "graha_maitri": {"score": graha, "max": 5, "detail": f"{SIGN_LORD[m1]} vs {SIGN_LORD[m2]}"},
        "gana": {"score": gana, "max": 6, "detail": f"{gana1} vs {gana2}"},
        "bhakoot": {"score": bhakoot, "max": 7, "detail": f"{m1} vs {m2}"},
        "nadi": {"score": nadi, "max": 8, "detail": f"{nad1} vs {nad2}"},
        "total": total,
        "max_total": 36,
        "percentage": round(total / 36 * 100, 1),
        "verdict": "Excellent" if total >= 31 else "Good" if total >= 25 else "Average" if total >= 18 else "Challenging"
    }

# ------------------------------------------------------------------
# YEARLY / TOPIC PREDICTIONS (Dasha + Transit)
# ------------------------------------------------------------------
def get_year_prediction(chart: ChartData, year: int) -> Dict:
    transits = get_transits(year)
    t_saturn = longitude_to_sign(transits["Saturn"])[0]
    t_jupiter = longitude_to_sign(transits["Jupiter"])[0]
    s_idx = ZODIAC.index(t_saturn)
    j_idx = ZODIAC.index(t_jupiter)
    m_idx = ZODIAC.index(chart.moon_sign)

    # Sade Sati
    sade_sati = "No Sade Sati"
    rel = (s_idx - m_idx) % 12
    if rel in [11, 0, 1]:
        phases = {11: "Rising Phase (1st 2.5 yrs)", 0: "Peak Phase (2nd 2.5 yrs)", 1: "Setting Phase (3rd 2.5 yrs)"}
        sade_sati = f"⚠️ Sade Sati Active — {phases[rel]}"

    # Dasha running (simplified: assumes birth near start of first dasha for demo)
    # In production, compute exact date overlap
    current_dasha = chart.dasha[0] if chart.dasha else {"mahadasha": "Moon", "years": 10}

    # House lords from Lagna
    lagna_idx = ZODIAC.index(chart.lagna_sign)

    def lord_of(house_offset: int) -> str:
        return SIGN_LORD[ZODIAC[(lagna_idx + house_offset - 1) % 12]]

    # Career (10th house)
    tenth_idx = (lagna_idx + 9) % 12
    saturn_to_tenth = (s_idx - tenth_idx) % 12
    career_saturn = "Saturn influencing 10th — restructuring & long-term gains." if saturn_to_tenth in [0, 7, 10] else "Steady professional progress."

    if current_dasha["mahadasha"] in ["Jupiter", "Sun", "Saturn"]:
        career = f"{current_dasha['mahadasha']} Mahadasha: Authority, recognition & hard work rewarded. {career_saturn}"
    else:
        career = f"{current_dasha['mahadasha']} period: Build skills & network. {career_saturn}"

    # Marriage (7th house)
    seventh_idx = (lagna_idx + 6) % 12
    jupiter_to_7th = (j_idx - seventh_idx) % 12
    jup_marriage = "Jupiter blessing 7th house — excellent for commitment." if jupiter_to_7th in [0, 5, 9] else "Focus on understanding & patience."

    if current_dasha["mahadasha"] in ["Venus", "Jupiter"]:
        marriage = f"{current_dasha['mahadasha']} dasha: Favorable for marriage/relationships. {jup_marriage}"
    else:
        marriage = f"Stable relationship period. {jup_marriage}"

    # Children (5th house)
    fifth_idx = (lagna_idx + 4) % 12
    jupiter_to_5th = (j_idx - fifth_idx) % 12
    jup_child = "Jupiter aspecting 5th — highly favorable for progeny." if jupiter_to_5th in [0, 5, 9] else "Patience; divine timing at work."

    if current_dasha["mahadasha"] == "Jupiter":
        children = f"Jupiter Mahadasha: Excellent period for children. {jup_child}"
    elif chart.moon_sign in ["Aries", "Gemini", "Leo", "Libra", "Sagittarius", "Aquarius"]:
        children = f"Positive indications for progeny. {jup_child}"
    else:
        children = f"Preparation period. {jup_child}"

    # Health (1st, 6th, 8th)
    if current_dasha["mahadasha"] in ["Saturn", "Rahu", "Ketu"]:
        health = f"{current_dasha['mahadasha']} dasha: Discipline in diet & exercise required."
    else:
        health = f"Stable health under {current_dasha['mahadasha']}. Maintain routines."

    if "Sade Sati" in sade_sati:
        health += " Sade Sati calls for mental health care & chronic checkups."

    return {
        "year": year,
        "dasha": current_dasha,
        "transits": {"Saturn": t_saturn, "Jupiter": t_jupiter},
        "sade_sati": sade_sati,
        "career": career,
        "marriage": marriage,
        "children": children,
        "health": health
    }

# ------------------------------------------------------------------
# DEMO CHART (if ephemeris unavailable)
# ------------------------------------------------------------------
def generate_demo_chart() -> ChartData:
    """Returns a realistic Vedic chart for UI testing."""
    planets = {
        "Sun": 45.5, "Moon": 128.3, "Mars": 200.0, "Mercury": 50.2,
        "Jupiter": 95.0, "Venus": 70.5, "Saturn": 310.0, "Rahu": 175.0, "Ketu": 355.0
    }
    return ChartData(planets, 30.0, "Taurus")
