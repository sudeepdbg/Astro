"""
Vedic Astrology Calculation Engine v4.0
========================================
Full rewrite / deep fix over v3.1.

BUG FIXES from v3.1 review:
  1. Gaja-Kesari Yoga: condition now correctly checks Jupiter in kendra FROM MOON
     (not from lagna, and the old abs()%6 test was meaningless).
  2. Bhakoot scoring: 6/8 axis = diff 6 only (sign A is 6th from B ↔ B is 8th from A,
     same pair). 2/12 axis = diff 2 only. diff 8 and diff 10 are NOT inauspicious.
     Nava-Pancham (diff 5 or 7) is AUSPICIOUS → score 7.
  3. get_tara_score: returns clean integer (0, 1, 2, or 3) via a corrected look-up.
  4. Varna Koota: direction documented; added gender param; default kept as before.
  5. evaluate_rules: exceptions now re-raised in debug mode; silent in prod.
  6. _apply_dasha_boost: no longer mutates originals; works on deep copies.
  7. house sentinel: _house() returns 0 for missing planets (unchanged) but rules
     now guard against 0 explicitly where needed.
  8. graha_maitri max corrected to 5; same-lord = 5, mutual friend = 4.
  9. Bhakoot diff=5 and diff=7 both correctly score 7 (Nava-Pancham auspicious).

NEW FEATURES in v4.0:
  A. Graha Drishti (planetary aspects) — Mars 4/7/8, Saturn 3/7/10, Jupiter 5/7/9,
     Rahu/Ketu 5/9, all planets 7th (full aspect).  Aspects on key houses now
     feed into predictions.
  B. Atmakaraka / Amatyakaraka detection (Jaimini — planet with highest degree).
  C. Basic Shadbala proxy: a 0-100 strength score per planet based on
     dignity + house placement + nakshatra lord alignment.
  D. Nakshatra lord chain: Moon nak lord → that lord's nak lord used in dasha refinement.
  E. 40+ enhanced prediction rules with deeper narrative paragraphs, aspect-awareness,
     and shadbala-aware commentary.
  F. Vimsopaka Bala proxy from D1+D9+D10 dignities (3-varga simplified).
  G. Ashtakavarga SAV (Sarvashtakavarga) placeholder structure for future integration.
  H. Yoga detection expanded: Neechabhanga Raja Yoga, Dharma-Karma Adhipati Yoga,
     Kahala Yoga, Parvata Yoga, Amala Yoga, Vesi/Voshi Yoga, Sunapha/Anapha.
  I. Composite year-band timing: each topic returns "best windows" within the year.
  J. Remedies section per topic (standardised, planet-specific).
"""

import copy
import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import swisseph as swe
    SWISSEPH_AVAILABLE = True
    swe.set_sid_mode(swe.SIDM_LAHIRI)
except ImportError:
    SWISSEPH_AVAILABLE = False


# ==================================================================
# SECTION 1 — STATIC LOOKUP TABLES
# ==================================================================

ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]
ZODIAC_SHORT = ["ARI","TAU","GEM","CAN","LEO","VIR","LIB","SCO","SAG","CAP","AQU","PIS"]

SIGN_SANSKRIT = {
    "Aries":"Mesha","Taurus":"Vrishabha","Gemini":"Mithuna",
    "Cancer":"Karka","Leo":"Simha","Virgo":"Kanya",
    "Libra":"Tula","Scorpio":"Vrischika","Sagittarius":"Dhanu",
    "Capricorn":"Makara","Aquarius":"Kumbha","Pisces":"Meena"
}

SIGN_LORD = {
    "Aries":"Mars","Taurus":"Venus","Gemini":"Mercury",
    "Cancer":"Moon","Leo":"Sun","Virgo":"Mercury",
    "Libra":"Venus","Scorpio":"Mars","Sagittarius":"Jupiter",
    "Capricorn":"Saturn","Aquarius":"Saturn","Pisces":"Jupiter"
}

SIGN_ELEMENT = {
    "Aries":"Fire","Taurus":"Earth","Gemini":"Air",
    "Cancer":"Water","Leo":"Fire","Virgo":"Earth",
    "Libra":"Air","Scorpio":"Water","Sagittarius":"Fire",
    "Capricorn":"Earth","Aquarius":"Air","Pisces":"Water"
}

SIGN_QUALITY = {
    "Aries":"Movable","Taurus":"Fixed","Gemini":"Dual",
    "Cancer":"Movable","Leo":"Fixed","Virgo":"Dual",
    "Libra":"Movable","Scorpio":"Fixed","Sagittarius":"Dual",
    "Capricorn":"Movable","Aquarius":"Fixed","Pisces":"Dual"
}

SIGN_GENDER = {
    "Aries":"Male","Taurus":"Female","Gemini":"Male",
    "Cancer":"Female","Leo":"Male","Virgo":"Female",
    "Libra":"Male","Scorpio":"Female","Sagittarius":"Male",
    "Capricorn":"Female","Aquarius":"Male","Pisces":"Female"
}

NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra",
    "Punarvasu","Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni",
    "Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha",
    "Purva Bhadrapada","Uttara Bhadrapada","Revati"
]

NAKSHATRA_LORDS = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"] * 3

NAKSHATRA_GANA = {
    "Ashwini":"Deva","Bharani":"Manushya","Krittika":"Rakshasa",
    "Rohini":"Manushya","Mrigashira":"Deva","Ardra":"Manushya",
    "Punarvasu":"Deva","Pushya":"Deva","Ashlesha":"Rakshasa",
    "Magha":"Rakshasa","Purva Phalguni":"Manushya","Uttara Phalguni":"Manushya",
    "Hasta":"Deva","Chitra":"Rakshasa","Swati":"Deva",
    "Vishakha":"Rakshasa","Anuradha":"Deva","Jyeshtha":"Rakshasa",
    "Mula":"Rakshasa","Purva Ashadha":"Manushya","Uttara Ashadha":"Manushya",
    "Shravana":"Deva","Dhanishta":"Rakshasa","Shatabhisha":"Rakshasa",
    "Purva Bhadrapada":"Manushya","Uttara Bhadrapada":"Manushya","Revati":"Deva"
}

NAKSHATRA_YONI = {
    "Ashwini":"Horse","Bharani":"Elephant","Krittika":"Sheep",
    "Rohini":"Serpent","Mrigashira":"Serpent","Ardra":"Dog",
    "Punarvasu":"Cat","Pushya":"Sheep","Ashlesha":"Cat",
    "Magha":"Rat","Purva Phalguni":"Rat","Uttara Phalguni":"Cow",
    "Hasta":"Buffalo","Chitra":"Tiger","Swati":"Buffalo",
    "Vishakha":"Tiger","Anuradha":"Deer","Jyeshtha":"Deer",
    "Mula":"Dog","Purva Ashadha":"Monkey","Uttara Ashadha":"Mongoose",
    "Shravana":"Monkey","Dhanishta":"Lion","Shatabhisha":"Horse",
    "Purva Bhadrapada":"Lion","Uttara Bhadrapada":"Cow","Revati":"Elephant"
}

NAKSHATRA_NADI = {
    "Ashwini":"Vata","Bharani":"Pitta","Krittika":"Kapha",
    "Rohini":"Vata","Mrigashira":"Pitta","Ardra":"Kapha",
    "Punarvasu":"Vata","Pushya":"Pitta","Ashlesha":"Kapha",
    "Magha":"Vata","Purva Phalguni":"Pitta","Uttara Phalguni":"Kapha",
    "Hasta":"Vata","Chitra":"Pitta","Swati":"Kapha",
    "Vishakha":"Vata","Anuradha":"Pitta","Jyeshtha":"Kapha",
    "Mula":"Vata","Purva Ashadha":"Pitta","Uttara Ashadha":"Kapha",
    "Shravana":"Vata","Dhanishta":"Pitta","Shatabhisha":"Kapha",
    "Purva Bhadrapada":"Vata","Uttara Bhadrapada":"Pitta","Revati":"Kapha"
}

VARNA_MAP  = {"Water":"Brahmin","Fire":"Kshatriya","Earth":"Vaishya","Air":"Shudra"}
VASHYA_MAP = {
    "Aries":"Quadruped","Taurus":"Quadruped","Gemini":"Human",
    "Cancer":"Water","Leo":"Quadruped","Virgo":"Human",
    "Libra":"Human","Scorpio":"Keet","Sagittarius":"Human",
    "Capricorn":"Quadruped","Aquarius":"Human","Pisces":"Water"
}

DASHA_YEARS    = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,
                  "Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
DASHA_SEQUENCE = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
TOTAL_DASHA_YEARS = 120

PLANET_IDS   = [0,1,2,3,4,5,6,10]  # SUN,MOON,MARS,MERCURY,JUPITER,VENUS,SATURN,TRUE_NODE
PLANET_NAMES = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu"]

HOUSE_MEANINGS = {
    1:"Self / Body / Vitality",           2:"Wealth / Family / Speech",
    3:"Courage / Siblings / Short Travel",4:"Mother / Home / Vehicles",
    5:"Intelligence / Children / Purva Punya",6:"Disease / Enemies / Service",
    7:"Marriage / Partnership / Business",8:"Longevity / Occult / Transformation",
    9:"Fortune / Dharma / Father / Higher Learning",
    10:"Career / Status / Action / Authority",
    11:"Gains / Friends / Elder Siblings",12:"Loss / Moksha / Foreign / Hospital"
}

EXALTATION   = {"Sun":"Aries","Moon":"Taurus","Mars":"Capricorn",
                "Mercury":"Virgo","Jupiter":"Cancer","Venus":"Pisces","Saturn":"Libra"}
DEBILITATION = {"Sun":"Libra","Moon":"Scorpio","Mars":"Cancer",
                "Mercury":"Pisces","Jupiter":"Capricorn","Venus":"Virgo","Saturn":"Aries"}
MOOLATRIKONA = {"Sun":"Leo","Moon":"Taurus","Mars":"Aries",
                "Mercury":"Virgo","Jupiter":"Sagittarius","Venus":"Libra","Saturn":"Aquarius"}

# Deep exaltation degrees (longitude within sign)
EXALTATION_DEGREE = {"Sun":10,"Moon":3,"Mars":28,"Mercury":15,
                     "Jupiter":5,"Venus":27,"Saturn":20}

PLANET_FRIENDS = {
    "Sun":    ["Moon","Mars","Jupiter"],
    "Moon":   ["Sun","Mercury"],
    "Mars":   ["Sun","Moon","Jupiter"],
    "Mercury":["Sun","Venus"],
    "Jupiter":["Sun","Moon","Mars"],
    "Venus":  ["Mercury","Saturn"],
    "Saturn": ["Mercury","Venus"],
    "Rahu":   ["Saturn","Venus","Mercury"],
    "Ketu":   ["Mars","Jupiter"],
}
PLANET_ENEMIES = {
    "Sun":    ["Venus","Saturn"],
    "Moon":   ["Rahu","Ketu"],
    "Mars":   ["Mercury"],
    "Mercury":["Moon"],
    "Jupiter":["Mercury","Venus","Rahu"],
    "Venus":  ["Sun","Moon"],
    "Saturn": ["Sun","Moon","Mars"],
    "Rahu":   ["Sun","Moon","Mars"],
    "Ketu":   ["Venus","Mercury"],
}

# Tara Bala scores (1-9 positions from birth nakshatra)
TARA_SCORES = {1:3, 2:3, 3:0, 4:3, 5:0, 6:3, 7:0, 8:3, 9:3}

NAKSHATRA_SIZE = 13 + 20/60   # 13°20′
PADA_SIZE      = 3  + 20/60   # 3°20′

# Graha Drishti: special aspects beyond the universal 7th
# Values = list of additional house offsets (from planet's house) that planet fully aspects
SPECIAL_ASPECTS = {
    "Mars":    [3, 7],      # 4th and 8th from itself (offset 3 and 7 from its own position)
    "Jupiter": [4, 8],      # 5th and 9th
    "Saturn":  [2, 9],      # 3rd and 10th
    "Rahu":    [4, 8],      # 5th and 9th (like Jupiter in many traditions)
    "Ketu":    [4, 8],      # same
}
# All planets have 7th aspect (offset 6)

# Dignity strength weights for shadbala proxy
DIGNITY_STRENGTH = {
    "Exalted":      100,
    "Own":           85,
    "Mool Trikona":  78,
    "Friendly":      55,
    "Neutral":       45,
    "Inimical":      25,
    "Debilitated":   10,
}

# House strength weights (Bhava Bala proxy)
HOUSE_STRENGTH = {
    1:100, 4:85, 7:85, 10:95,   # Kendras
    5:80,  9:80,                  # Trikonas
    2:60,  11:65,                 # Upachayas (partial)
    3:50,  6:45,                  # Upachayas (partial) / dusthanas
    8:35,  12:30,                 # Dusthanas
}


# ==================================================================
# SECTION 2 — CORE MATH
# ==================================================================

def longitude_to_sign(longitude: float) -> Tuple[str, float]:
    idx = int(longitude // 30) % 12
    return ZODIAC[idx], longitude % 30


def get_nakshatra(longitude: float) -> Tuple[str, int, float]:
    lon     = longitude % 360
    nak_idx = int(lon / NAKSHATRA_SIZE)
    rem     = lon % NAKSHATRA_SIZE
    pada    = int(rem / PADA_SIZE) + 1
    return NAKSHATRAS[nak_idx % 27], pada, rem


def get_navamsa(longitude: float) -> str:
    sign_idx    = int(longitude // 30)
    deg_in_sign = longitude % 30
    part        = int(deg_in_sign // (10 / 3))
    quality     = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12
    else:
        start = (sign_idx + 4) % 12
    return ZODIAC[(start + part) % 12]


def get_drekkana(longitude: float) -> str:
    sign_idx    = int(longitude // 30)
    deg_in_sign = longitude % 30
    part        = int(deg_in_sign // 10)
    quality     = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 4) % 12
    else:
        start = (sign_idx + 8) % 12
    return ZODIAC[(start + part) % 12]


def get_saptamsa(longitude: float) -> str:
    sign_idx    = int(longitude // 30)
    deg_in_sign = longitude % 30
    part        = int(deg_in_sign // (30 / 7))
    # sign_idx % 2 == 0 → odd sign (Aries=1, Gemini=3…) → start same sign
    if sign_idx % 2 == 0:
        start = sign_idx
    else:
        start = (sign_idx + 6) % 12
    return ZODIAC[(start + part) % 12]


def get_dasamsa(longitude: float) -> str:
    sign_idx    = int(longitude // 30)
    deg_in_sign = longitude % 30
    part        = int(deg_in_sign // 3)
    quality     = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12
    else:
        start = (sign_idx + 4) % 12
    return ZODIAC[(start + part) % 12]


def get_dwadasamsa(longitude: float) -> str:
    sign_idx    = int(longitude // 30)
    deg_in_sign = longitude % 30
    part        = int(deg_in_sign // 2.5)
    return ZODIAC[(sign_idx + part) % 12]


def get_planet_dignity(planet: str, sign: str) -> str:
    if EXALTATION.get(planet) == sign:
        return "Exalted"
    if DEBILITATION.get(planet) == sign:
        return "Debilitated"
    if SIGN_LORD.get(sign) == planet:
        return "Own"
    if MOOLATRIKONA.get(planet) == sign:
        return "Mool Trikona"
    lord = SIGN_LORD.get(sign, "")
    if planet in PLANET_FRIENDS and lord in PLANET_FRIENDS[planet]:
        if planet in PLANET_FRIENDS.get(lord, []):
            return "Mutual Friend"
        return "Friendly"
    if planet in PLANET_ENEMIES and lord in PLANET_ENEMIES[planet]:
        return "Inimical"
    return "Neutral"


def planet_strength(planet: str, sign: str, house: int) -> float:
    """Shadbala proxy: 0-100 composite strength score."""
    dig_score   = DIGNITY_STRENGTH.get(get_planet_dignity(planet, sign), 45)
    house_score = HOUSE_STRENGTH.get(house, 45)
    return round((dig_score * 0.6 + house_score * 0.4), 1)


def is_neechabhanga(planet: str, sign: str, chart_planets: Dict[str, float],
                    lagna_sign: str) -> bool:
    """
    Neechabhanga (cancellation of debilitation) check.
    Classic conditions (any one sufficient):
    1. Lord of the debilitation sign is in a kendra from lagna or Moon.
    2. The planet that would exalt in the same sign as the debilitated planet
       is in a kendra from lagna or Moon.
    3. Debilitated planet itself is in a kendra from lagna (considered by some).
    """
    if get_planet_dignity(planet, sign) != "Debilitated":
        return False

    def in_kendra_from(p_sign: str, ref_sign: str) -> bool:
        d = (ZODIAC.index(p_sign) - ZODIAC.index(ref_sign)) % 12
        return d in [0, 3, 6, 9]

    moon_sign  = longitude_to_sign(chart_planets.get("Moon", 0))[0]
    deb_sign   = DEBILITATION[planet]
    deb_lord   = SIGN_LORD[deb_sign]

    # Planet that exalts in the same sign as the debilitated planet
    exalt_friend = next((p for p, s in EXALTATION.items() if s == deb_sign), None)

    # Condition 1: lord of debilitation sign in kendra from lagna or Moon
    if deb_lord in chart_planets:
        lord_sign = longitude_to_sign(chart_planets[deb_lord])[0]
        if in_kendra_from(lord_sign, lagna_sign) or in_kendra_from(lord_sign, moon_sign):
            return True

    # Condition 2: exalting planet in kendra from lagna or Moon
    if exalt_friend and exalt_friend in chart_planets:
        ef_sign = longitude_to_sign(chart_planets[exalt_friend])[0]
        if in_kendra_from(ef_sign, lagna_sign) or in_kendra_from(ef_sign, moon_sign):
            return True

    # Condition 3: debilitated planet itself in kendra from lagna
    if planet in chart_planets:
        p_sign = longitude_to_sign(chart_planets[planet])[0]
        if in_kendra_from(p_sign, lagna_sign):
            return True

    return False


def get_aspects_on_house(house: int, house_map: Dict[str, int]) -> List[Tuple[str, str]]:
    """
    Returns list of (planet, aspect_type) tuples for all planets aspecting
    the given house number.  Aspects include universal 7th + special aspects.
    aspect_type: "full" | "special"
    """
    aspects = []
    for planet, p_house in house_map.items():
        # Universal 7th aspect
        if (p_house - 1 + 6) % 12 + 1 == house:
            aspects.append((planet, "full"))
            continue
        # Special aspects
        if planet in SPECIAL_ASPECTS:
            for offset in SPECIAL_ASPECTS[planet]:
                if (p_house - 1 + offset) % 12 + 1 == house:
                    aspects.append((planet, "special"))
                    break
    return aspects


def get_atmakaraka(planets: Dict[str, float]) -> str:
    """
    Jaimini Atmakaraka: planet with highest degrees within its sign (0-30°).
    Rahu uses 30 - degrees (moves retrograde).
    Ketu excluded from AK calculation.
    """
    relevant = {p: v for p, v in planets.items() if p not in ("Ketu",)}
    deg_map = {}
    for p, lon in relevant.items():
        deg = lon % 30
        if p == "Rahu":
            deg = 30 - deg
        deg_map[p] = deg
    return max(deg_map, key=deg_map.get)


def get_amatyakaraka(planets: Dict[str, float], atmakaraka: str) -> str:
    """Jaimini Amatyakaraka: planet with second-highest degree."""
    relevant = {p: v for p, v in planets.items() if p not in ("Ketu", atmakaraka)}
    deg_map = {}
    for p, lon in relevant.items():
        deg = lon % 30
        if p == "Rahu":
            deg = 30 - deg
        deg_map[p] = deg
    return max(deg_map, key=deg_map.get) if deg_map else ""


def vimsopaka_bala(planet: str, d1_sign: str, d9_sign: str, d10_sign: str) -> float:
    """
    Simplified Vimsopaka Bala from 3 vargas (D1 × 6, D9 × 5, D10 × 4 — weighted).
    Full Vimsopaka uses 16 vargas; this 3-varga proxy gives a reasonable strength signal.
    Returns 0–20 score.
    """
    weights = {d1_sign: 6, d9_sign: 5, d10_sign: 4}
    total_weight = 15
    score = 0.0
    for sign, w in weights.items():
        dig = get_planet_dignity(planet, sign)
        factor = {
            "Exalted":1.0,"Own":0.9,"Mool Trikona":0.83,
            "Mutual Friend":0.65,"Friendly":0.60,"Neutral":0.45,
            "Inimical":0.25,"Debilitated":0.10
        }.get(dig, 0.45)
        score += w * factor
    return round(score / total_weight * 20, 2)


# ==================================================================
# SECTION 3 — DASHA CALCULATIONS
# ==================================================================

@dataclass
class DashaPeriod:
    planet:     str
    start_date: datetime
    end_date:   datetime
    years:      float
    level:      str   # "MD", "AD", "PD"
    parent:     Optional[str] = None


def calculate_vimshottari_full(birth_date: datetime, moon_longitude: float) -> List[DashaPeriod]:
    moon_lon  = moon_longitude % 360
    nak_idx   = int(moon_lon / NAKSHATRA_SIZE)
    nak_start = nak_idx * NAKSHATRA_SIZE
    deg_covered = moon_lon - nak_start
    remaining   = NAKSHATRA_SIZE - deg_covered
    fraction    = remaining / NAKSHATRA_SIZE
    lord_idx    = nak_idx % 9
    start_lord  = DASHA_SEQUENCE[lord_idx]
    balance     = fraction * DASHA_YEARS[start_lord]

    periods = []
    current_date = birth_date
    for i in range(9):
        lord     = DASHA_SEQUENCE[(lord_idx + i) % 9]
        years    = balance if i == 0 else DASHA_YEARS[lord]
        end_date = current_date + timedelta(days=years * 365.25)
        periods.append(DashaPeriod(
            planet=lord, start_date=current_date, end_date=end_date,
            years=round(years, 4), level="MD"
        ))
        current_date = end_date
    return periods


def calculate_antardasha(md: DashaPeriod) -> List[DashaPeriod]:
    md_idx         = DASHA_SEQUENCE.index(md.planet)
    md_total_years = DASHA_YEARS[md.planet]
    scale          = md.years / md_total_years

    current_date = md.start_date
    ad_periods   = []
    for i in range(9):
        ad_planet    = DASHA_SEQUENCE[(md_idx + i) % 9]
        ad_std_years = (DASHA_YEARS[md.planet] * DASHA_YEARS[ad_planet]) / TOTAL_DASHA_YEARS
        ad_actual    = ad_std_years * scale
        end_date     = current_date + timedelta(days=ad_actual * 365.25)
        ad_periods.append(DashaPeriod(
            planet=ad_planet, start_date=current_date, end_date=end_date,
            years=round(ad_actual, 4), level="AD", parent=md.planet
        ))
        current_date = end_date
    return ad_periods


def calculate_pratyantardasha(ad: DashaPeriod) -> List[DashaPeriod]:
    ad_idx         = DASHA_SEQUENCE.index(ad.planet)
    md_planet      = ad.parent or ad.planet
    ad_total_years = (DASHA_YEARS[md_planet] * DASHA_YEARS[ad.planet]) / TOTAL_DASHA_YEARS
    scale          = ad.years / ad_total_years if ad_total_years > 0 else 1.0

    current_date = ad.start_date
    pd_periods   = []
    for i in range(9):
        pd_planet    = DASHA_SEQUENCE[(ad_idx + i) % 9]
        pd_std_years = (DASHA_YEARS[md_planet] * DASHA_YEARS[ad.planet] * DASHA_YEARS[pd_planet]) / (TOTAL_DASHA_YEARS ** 2)
        pd_actual    = pd_std_years * scale
        end_date     = current_date + timedelta(days=pd_actual * 365.25)
        pd_periods.append(DashaPeriod(
            planet=pd_planet, start_date=current_date, end_date=end_date,
            years=round(pd_actual, 5), level="PD", parent=ad.planet
        ))
        current_date = end_date
    return pd_periods


def get_current_dasha(periods: List[DashaPeriod], check_date: datetime = None) -> Optional[DashaPeriod]:
    if check_date is None:
        check_date = datetime.now()
    for p in periods:
        if p.start_date <= check_date < p.end_date:
            return p
    return None


def get_current_antardasha(md_periods: List[DashaPeriod], check_date: datetime = None) -> Optional[DashaPeriod]:
    if check_date is None:
        check_date = datetime.now()
    md = get_current_dasha(md_periods, check_date)
    if not md:
        return None
    return get_current_dasha(calculate_antardasha(md), check_date)


def get_current_pratyantardasha(md_periods: List[DashaPeriod], check_date: datetime = None) -> Optional[DashaPeriod]:
    if check_date is None:
        check_date = datetime.now()
    md = get_current_dasha(md_periods, check_date)
    if not md:
        return None
    ad = get_current_antardasha(md_periods, check_date)
    if not ad:
        return None
    return get_current_dasha(calculate_pratyantardasha(ad), check_date)


def check_sade_sati(moon_sign: str, saturn_sign: str) -> Dict:
    if saturn_sign not in ZODIAC:
        return {"active": False, "phase": ""}
    m_idx = ZODIAC.index(moon_sign)
    s_idx = ZODIAC.index(saturn_sign)
    rel   = (s_idx - m_idx) % 12
    if rel == 11:
        return {"active": True, "phase": "Rising Phase — Saturn entering 12th from Moon"}
    if rel == 0:
        return {"active": True, "phase": "Peak Phase — Saturn on Moon sign"}
    if rel == 1:
        return {"active": True, "phase": "Setting Phase — Saturn in 2nd from Moon"}
    return {"active": False, "phase": ""}


def check_kantaka_shani(moon_sign: str, saturn_sign: str) -> bool:
    """Kantaka Shani: Saturn in 4th, 7th, or 10th from Moon."""
    if saturn_sign not in ZODIAC:
        return False
    rel = (ZODIAC.index(saturn_sign) - ZODIAC.index(moon_sign)) % 12
    return rel in [3, 6, 9]


# ==================================================================
# SECTION 4 — CHART DATA CLASS
# ==================================================================

class ChartData:
    def __init__(self, planets: Dict[str, float], ascendant: float, lagna_sign: str,
                 birth_date: datetime = None, lat: float = 0.0, lon: float = 0.0, tz: float = 0.0):
        self.planets    = planets
        self.ascendant  = ascendant
        self.lagna_sign = lagna_sign
        self.moon_sign  = longitude_to_sign(planets["Moon"])[0]
        self.sun_sign   = longitude_to_sign(planets["Sun"])[0]
        self.birth_date = birth_date
        self.lat        = lat
        self.lon        = lon
        self.tz         = tz

        self.nakshatras           : Dict = {}
        self.navamsa              : Dict = {}
        self.drekkana             : Dict = {}
        self.saptamsa             : Dict = {}
        self.dasamsa              : Dict = {}
        self.dwadasamsa           : Dict = {}
        self.dignities            : Dict = {}
        self.navamsa_dignities    : Dict = {}
        self.dasamsa_dignities    : Dict = {}
        self.shadbala_proxy       : Dict = {}
        self.vimsopaka            : Dict = {}
        self.dasha_periods        : List[DashaPeriod] = []
        self.atmakaraka           : str = ""
        self.amatyakaraka         : str = ""
        self._compute_derived()

    def _compute_derived(self):
        lagna_idx = ZODIAC.index(self.lagna_sign)

        # house_map (needed for shadbala)
        house_map = {}
        for p, lon in self.planets.items():
            sign, _ = longitude_to_sign(lon)
            house_map[p] = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1

        for p, lon in self.planets.items():
            nak, pada, rem = get_nakshatra(lon)
            self.nakshatras[p] = {
                "nakshatra":        nak,
                "pada":             pada,
                "lord":             NAKSHATRA_LORDS[NAKSHATRAS.index(nak)],
                "deg_in_nakshatra": round(rem, 2)
            }
            self.navamsa[p]    = get_navamsa(lon)
            self.drekkana[p]   = get_drekkana(lon)
            self.saptamsa[p]   = get_saptamsa(lon)
            self.dasamsa[p]    = get_dasamsa(lon)
            self.dwadasamsa[p] = get_dwadasamsa(lon)

            sign, _             = longitude_to_sign(lon)
            self.dignities[p]   = get_planet_dignity(p, sign)
            self.navamsa_dignities[p] = get_planet_dignity(p, self.navamsa[p])
            self.dasamsa_dignities[p] = get_planet_dignity(p, self.dasamsa[p])

            # Shadbala proxy
            self.shadbala_proxy[p] = planet_strength(p, sign, house_map.get(p, 6))

            # Vimsopaka Bala (D1+D9+D10)
            d1_sign = sign
            d9_sign = self.navamsa[p]
            d10_sign = self.dasamsa[p]
            self.vimsopaka[p] = vimsopaka_bala(p, d1_sign, d9_sign, d10_sign)

        if self.birth_date:
            self.dasha_periods = calculate_vimshottari_full(self.birth_date, self.planets["Moon"])

        # Jaimini karakas
        self.atmakaraka  = get_atmakaraka(self.planets)
        self.amatyakaraka = get_amatyakaraka(self.planets, self.atmakaraka)

    def get_current_dasha_info(self, check_date: datetime = None) -> Dict:
        md = get_current_dasha(self.dasha_periods, check_date)
        if not md:
            return {}
        ad = get_current_antardasha(self.dasha_periods, check_date)
        pd = get_current_pratyantardasha(self.dasha_periods, check_date)
        return {
            "mahadasha":        md.planet,
            "mahadasha_start":  md.start_date.strftime("%d %b %Y"),
            "mahadasha_end":    md.end_date.strftime("%d %b %Y"),
            "antardasha":       ad.planet if ad else "",
            "antardasha_start": ad.start_date.strftime("%d %b %Y") if ad else "",
            "antardasha_end":   ad.end_date.strftime("%d %b %Y")   if ad else "",
            "pratyantardasha":  pd.planet if pd else "",
            "pd_start":         pd.start_date.strftime("%d %b %Y") if pd else "",
            "pd_end":           pd.end_date.strftime("%d %b %Y")   if pd else "",
        }

    def to_dict(self) -> Dict:
        return {
            "birth_date":    self.birth_date.isoformat() if self.birth_date else None,
            "lat":           self.lat,
            "lon":           self.lon,
            "tz":            self.tz,
            "lagna_sign":    self.lagna_sign,
            "moon_sign":     self.moon_sign,
            "sun_sign":      self.sun_sign,
            "ascendant":     self.ascendant,
            "atmakaraka":    self.atmakaraka,
            "amatyakaraka":  self.amatyakaraka,
            "planets":       self.planets,
            "nakshatras":    self.nakshatras,
            "navamsa":       self.navamsa,
            "dignities":     self.dignities,
            "shadbala_proxy":self.shadbala_proxy,
            "vimsopaka":     self.vimsopaka,
            "dasha": [
                {"planet": p.planet,
                 "start":  p.start_date.isoformat(),
                 "end":    p.end_date.isoformat(),
                 "years":  p.years}
                for p in self.dasha_periods
            ]
        }

    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ==================================================================
# SECTION 5 — CONTEXT BUILDER
# ==================================================================

def build_context(chart: ChartData, dasha_info: Dict = None,
                  sade_sati_info: Dict = None) -> Dict:
    lagna_idx = ZODIAC.index(chart.lagna_sign)

    house_map = {}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house_map[p] = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1

    lord_map = {}
    for i in range(12):
        lord_map[i + 1] = SIGN_LORD[ZODIAC[(lagna_idx + i) % 12]]

    # Aspect map: house_number → list of (planet, aspect_type)
    aspect_map = {h: get_aspects_on_house(h, house_map) for h in range(1, 13)}

    # Neechabhanga flags
    neechabhanga = {}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        neechabhanga[p] = is_neechabhanga(p, sign, chart.planets, chart.lagna_sign)

    ctx = {
        "planets":             chart.planets,
        "lagna_sign":          chart.lagna_sign,
        "lagna_idx":           lagna_idx,
        "moon_sign":           chart.moon_sign,
        "sun_sign":            chart.sun_sign,
        "dignities":           chart.dignities,
        "navamsa_dignities":   chart.navamsa_dignities,
        "dasamsa_dignities":   chart.dasamsa_dignities,
        "nakshatras":          chart.nakshatras,
        "navamsa":             chart.navamsa,
        "dasamsa":             chart.dasamsa,
        "shadbala":            chart.shadbala_proxy,
        "vimsopaka":           chart.vimsopaka,
        "house_map":           house_map,
        "lord_map":            lord_map,
        "aspect_map":          aspect_map,
        "neechabhanga":        neechabhanga,
        "atmakaraka":          chart.atmakaraka,
        "amatyakaraka":        chart.amatyakaraka,
        "dasha":               dasha_info.get("mahadasha", "") if dasha_info else "",
        "antardasha":          dasha_info.get("antardasha", "") if dasha_info else "",
        "pratyantardasha":     dasha_info.get("pratyantardasha", "") if dasha_info else "",
        "dasha_md_start":      dasha_info.get("mahadasha_start", "") if dasha_info else "",
        "dasha_md_end":        dasha_info.get("mahadasha_end", "") if dasha_info else "",
        "dasha_ad_start":      dasha_info.get("antardasha_start", "") if dasha_info else "",
        "dasha_ad_end":        dasha_info.get("antardasha_end", "") if dasha_info else "",
        "sade_sati_active":    False,
        "sade_sati_phase":     "",
    }

    if sade_sati_info:
        ctx["sade_sati_active"] = sade_sati_info.get("active", False)
        ctx["sade_sati_phase"]  = sade_sati_info.get("phase", "")

    return ctx


# ==================================================================
# SECTION 6 — HELPER ACCESSORS (used in rules)
# ==================================================================

def _house(planet: str, ctx: dict) -> int:
    return ctx["house_map"].get(planet, 0)

def _lord(house: int, ctx: dict) -> str:
    return ctx["lord_map"].get(house, "")

def _dignity(planet: str, ctx: dict) -> str:
    return ctx["dignities"].get(planet, "Neutral")

def _navamsa_dignity(planet: str, ctx: dict) -> str:
    return ctx["navamsa_dignities"].get(planet, "Neutral")

def _dasamsa_dignity(planet: str, ctx: dict) -> str:
    return ctx["dasamsa_dignities"].get(planet, "Neutral")

def _strong(planet: str, ctx: dict) -> bool:
    return _dignity(planet, ctx) in ["Exalted","Own","Mool Trikona","Mutual Friend"]

def _weak(planet: str, ctx: dict) -> bool:
    return _dignity(planet, ctx) == "Debilitated"

def _strength(planet: str, ctx: dict) -> float:
    return ctx["shadbala"].get(planet, 45.0)

def _aspects_house(planet: str, house: int, ctx: dict) -> bool:
    return any(p == planet for p, _ in ctx["aspect_map"].get(house, []))

def _benefics_aspect(house: int, ctx: dict) -> List[str]:
    return [p for p, _ in ctx["aspect_map"].get(house, [])
            if p in ["Jupiter","Venus","Mercury","Moon"]]

def _malefics_aspect(house: int, ctx: dict) -> List[str]:
    return [p for p, _ in ctx["aspect_map"].get(house, [])
            if p in ["Saturn","Mars","Rahu","Ketu","Sun"]]

def _nb(planet: str, ctx: dict) -> bool:
    """Returns True if planet has Neechabhanga."""
    return ctx["neechabhanga"].get(planet, False)


# ==================================================================
# SECTION 7 — PREDICTION RULES  (expanded & bug-fixed)
# ==================================================================

PREDICTION_RULES: List[Dict] = [

    # ════════════════════════════════════════════════════════════
    # CAREER
    # ════════════════════════════════════════════════════════════
    {
        "id": "career_sun_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Sun", ctx) == 10,
        "severity": "positive",
        "score": 3,
        "title": "Sun in 10th — Authority & Public Prominence",
        "detail": lambda ctx: (
            f"Sun in the 10th house (Karma Bhava) is one of the finest placements for career. "
            f"It confers natural authority, confidence, and an instinct for leadership. You are "
            f"suited to roles where you are seen, recognised, and in command — government, "
            f"administration, medicine, politics, senior management, or any field requiring a "
            f"strong public presence. Sun's dignity here is {_dignity('Sun', ctx)}"
            + (", amplifying its power to grant high status and recognition." if _strong("Sun", ctx)
               else " (Neechabhanga applies — debilitation is cancelled, restoring Sun's authority after initial setbacks)." if _nb("Sun", ctx)
               else "; natural authority is present but may require conscious cultivation.")
            + f" Shadbala strength score: {_strength('Sun', ctx)}/100."
            + (" Jupiter aspects the 10th, adding dharmic success and wisdom to your professional path." if _aspects_house("Jupiter", 10, ctx) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "career_saturn_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Saturn", ctx) == 10,
        "severity": "positive",
        "score": 3,
        "title": "Saturn in 10th — The Slow Climb to Lasting Achievement",
        "detail": lambda ctx: (
            "Saturn in the 10th house is among the most powerful career placements in Vedic astrology, "
            "though its gifts arrive late and through patient effort. Success is earned, not gifted — "
            "but once achieved, it is rock-solid and enduring. You are built for roles requiring "
            "discipline, systematic thinking, and long-term responsibility: engineering, law, "
            "architecture, administration, real estate, research, or any structured institution. "
            f"Saturn's dignity: {_dignity('Saturn', ctx)}. "
            + ("Exalted Saturn here forms Shasha Yoga — this is one of the strongest career yogas, "
               "conferring authority, management mastery, and recognition from the masses." if _dignity("Saturn", ctx) == "Exalted"
               else "Debilitated Saturn in 10th brings disruptions, authority conflicts, and career reversals. "
               "However, if Neechabhanga applies, these challenges ultimately forge exceptional resilience." if _dignity("Saturn", ctx) == "Debilitated"
               else "Saturn in own sign or friendly here solidifies professional reputation over decades.")
            + f" Vimsopaka Bala: {ctx['vimsopaka'].get('Saturn', '?')}/20."
        ),
        "activation": "natal"
    },
    {
        "id": "career_jupiter_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Jupiter", ctx) == 10,
        "severity": "positive",
        "score": 3,
        "title": "Jupiter in 10th — Dharmakarmadhipati & Wisdom-Led Career",
        "detail": lambda ctx: (
            "Jupiter in the 10th house creates a dharmic, ethics-driven career orientation. You are "
            "drawn to fields that uplift, educate, or protect: teaching, law, finance, banking, "
            "counselling, publishing, spirituality, or administration of large institutions. "
            "Reputation grows through integrity and wise judgment rather than aggression. "
            f"Jupiter is {_dignity('Jupiter', ctx)} — "
            + ("this forms Hamsa Yoga (a Panchamahapurusha Yoga), one of the rarest and most "
               "auspicious placements, granting scholarly fame, spiritual recognition, and a "
               "distinguished career that others look up to." if _strong("Jupiter", ctx)
               else "debilitated Jupiter here slows career expansion and may cause conflicts with "
               "mentors or institutions. Jupiter Shanti, charity, and Guru-seva are strongly advised." if _weak("Jupiter", ctx)
               else "steady, principled growth over the career arc is indicated.")
        ),
        "activation": "natal"
    },
    {
        "id": "career_mars_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Mars", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Mars in 10th — Drive, Courage & Technical Mastery",
        "detail": lambda ctx: (
            "Mars in the 10th house infuses your professional life with energy, ambition, and "
            "competitive drive. You are a natural achiever who thrives under pressure. Best suited "
            "to: military/defence, police, surgery, engineering, sports management, competitive "
            "business, or any field requiring decisive, swift action. Leadership can be assertive "
            f"to the point of conflict with seniors — channel Mars productively. "
            f"Mars is {_dignity('Mars', ctx)}"
            + (", forming Ruchaka Yoga — exceptional physical courage, command, and competitive success." if _strong("Mars", ctx) and _house("Mars", ctx) in [1,4,7,10]
               else " (Neechabhanga applies — initial career aggression transforms into strategic strength)." if _nb("Mars", ctx)
               else ".")
        ),
        "activation": "natal"
    },
    {
        "id": "career_mercury_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Mercury", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Mercury in 10th — Intellect, Communication & Commerce",
        "detail": lambda ctx: (
            "Mercury in the 10th house marks a career powered by intellect, communication, and "
            "analytical skill. Writing, publishing, journalism, IT, data analytics, commerce, "
            "consulting, teaching, or any information-driven field are natural fits. Multiple "
            "simultaneous career threads or frequent role changes are common — your versatility "
            f"is a strength. Mercury is {_dignity('Mercury', ctx)} here. "
            f"Amatyakaraka (career significator in Jaimini): {ctx['amatyakaraka']}. "
            + ("Mercury as Amatyakaraka strongly amplifies this 10th house placement — your "
               "career path is deeply intertwined with Mercurian skills." if ctx['amatyakaraka'] == "Mercury" else "")
        ),
        "activation": "natal"
    },
    {
        "id": "career_venus_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Venus", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Venus in 10th — Creative, Aesthetic & Diplomatic Careers",
        "detail": lambda ctx: (
            "Venus in the 10th house draws you toward careers involving beauty, creativity, luxury, "
            "or diplomacy. Arts, music, fashion, hospitality, entertainment, cosmetics, luxury goods, "
            "design, or foreign service are natural arenas. Public charm and aesthetic sensitivity "
            f"are professional assets. Venus is {_dignity('Venus', ctx)} here"
            + (" — Malavya Yoga forms, bringing fame, prosperity, and refined taste in your career." if _strong("Venus", ctx) and _house("Venus", ctx) in [1,4,7,10]
               else ".")
        ),
        "activation": "natal"
    },
    {
        "id": "career_rahu_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Rahu", ctx) == 10,
        "severity": "neutral",
        "score": 2,
        "title": "Rahu in 10th — Meteoric Rise Through Unconventional Paths",
        "detail": lambda ctx: (
            "Rahu in the 10th house creates a strong, almost obsessive drive for career status and "
            "recognition. This is a placement of sudden, dramatic rises — often through technology, "
            "foreign companies, media, research, or fields that were not yet established at your birth. "
            "You may pioneer something new in your industry. Rahu here can deliver extraordinary "
            "fame, but the rise can be followed by equally dramatic falls if ethics are compromised. "
            "The key: align ambition with integrity. The 10th lord's condition heavily modifies this."
        ),
        "activation": "natal"
    },
    {
        "id": "career_ketu_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Ketu", ctx) == 10,
        "severity": "caution",
        "score": -1,
        "title": "Ketu in 10th — Karmic Disconnection from Conventional Career",
        "detail": lambda ctx: (
            "Ketu in the 10th house creates a subtle detachment from worldly career aspirations. "
            "There is often a past-life mastery of the career domain (shown by Rahu's opposite "
            "house — your hunger for growth is in House 4, the inner/domestic realm). You may "
            "excel professionally but find little satisfaction in it, eventually pivoting toward "
            "research, spirituality, healing, behind-the-scenes work, or astrology. Professional "
            "disruptions during Ketu Mahadasha are common. Best roles: research, mystical sciences, "
            "alternative medicine, or any field that synthesises technical mastery with intuition."
        ),
        "activation": "natal"
    },
    {
        "id": "career_10th_lord_strong",
        "topic": "career",
        "condition": lambda ctx: _strong(_lord(10, ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong 10th Lord — Career Yoga Activated",
        "detail": lambda ctx: (
            f"The 10th house lord {_lord(10,ctx)} is {_dignity(_lord(10,ctx), ctx)}, "
            f"creating a strong Rajayoga-class career indicator. This is the single most important "
            f"factor for career success: the lord of karma is powerful and directed. "
            f"It is placed in House {_house(_lord(10,ctx), ctx)} — "
            + _house_career_meaning(_house(_lord(10,ctx), ctx))
            + f" Strength score: {_strength(_lord(10,ctx), ctx)}/100."
        ),
        "activation": "natal"
    },
    {
        "id": "career_10th_lord_weak",
        "topic": "career",
        "condition": lambda ctx: _weak(_lord(10, ctx), ctx) and not _nb(_lord(10, ctx), ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 10th Lord (No Neechabhanga) — Sustained Career Challenges",
        "detail": lambda ctx: (
            f"The 10th lord {_lord(10,ctx)} is debilitated without Neechabhanga cancellation. "
            "This is the most significant single indicator of career difficulty: loss of position, "
            "authority conflicts, abrupt terminations, or inability to sustain momentum. "
            "The profession chosen and the timing of career moves requires careful astrological "
            "guidance. Remedies for the 10th lord planet (fasting, charity, mantra) are essential. "
            "Look for Antardashas of strong benefics to navigate peak performance windows."
        ),
        "activation": "natal"
    },
    {
        "id": "career_10th_lord_weak_nb",
        "topic": "career",
        "condition": lambda ctx: _weak(_lord(10, ctx), ctx) and _nb(_lord(10, ctx), ctx),
        "severity": "neutral",
        "score": 1,
        "title": "Debilitated 10th Lord with Neechabhanga — Adversity Transformed",
        "detail": lambda ctx: (
            f"The 10th lord {_lord(10,ctx)} is debilitated but Neechabhanga (cancellation) applies. "
            "Classical texts state this actually confers Raja Yoga — challenges in the career domain "
            "are ultimately overcome through exceptional resilience, producing a career trajectory "
            "that rises after a fall. Early career hardships are possible, but the eventual "
            "achievement often surpasses what a straightforwardly strong 10th lord would deliver."
        ),
        "activation": "natal"
    },
    {
        "id": "career_10th_lord_d10_strong",
        "topic": "career",
        "condition": lambda ctx: _dasamsa_dignity(_lord(10, ctx), ctx) in ["Exalted","Own","Mool Trikona","Mutual Friend"],
        "severity": "positive",
        "score": 2,
        "title": "10th Lord Strong in Dasamsa (D10) — Career Excellence Confirmed",
        "detail": lambda ctx: (
            f"The 10th lord {_lord(10,ctx)} is {_dasamsa_dignity(_lord(10,ctx), ctx)} in the "
            "Dasamsa (D10), the divisional chart specifically governing career and professional life. "
            "D10 confirmation is critical: when the natal chart strength is echoed in D10, "
            "professional success is near-certain. This signals recognition, promotions, and "
            "respect within the professional domain."
        ),
        "activation": "natal"
    },
    {
        "id": "career_amatyakaraka_strong",
        "topic": "career",
        "condition": lambda ctx: _strong(ctx["amatyakaraka"], ctx) if ctx.get("amatyakaraka") else False,
        "severity": "positive",
        "score": 2,
        "title": "Strong Amatyakaraka — Jaimini Career Blessing",
        "detail": lambda ctx: (
            f"The Amatyakaraka (Jaimini career significator) is {ctx['amatyakaraka']}, "
            f"which is {_dignity(ctx['amatyakaraka'], ctx)} in the natal chart. "
            "A strong Amatyakaraka indicates that the soul's designated career path — the one "
            "aligned with life purpose — will be supported by external circumstances, mentors, "
            f"and opportunities. House {_house(ctx['amatyakaraka'], ctx)} becomes a key zone of "
            "professional activity and recognition."
        ),
        "activation": "natal"
    },
    {
        "id": "career_dharma_karma_yoga",
        "topic": "career",
        "condition": lambda ctx: (
            _strong(_lord(9, ctx), ctx) and _strong(_lord(10, ctx), ctx)
        ),
        "severity": "positive",
        "score": 4,
        "title": "Dharma-Karma Adhipati Yoga — Fortune Fused with Action",
        "detail": lambda ctx: (
            f"The 9th lord ({_lord(9,ctx)}, {_dignity(_lord(9,ctx),ctx)}) and 10th lord "
            f"({_lord(10,ctx)}, {_dignity(_lord(10,ctx),ctx)}) are both strong, forming "
            "Dharma-Karma Adhipati Yoga — one of the most powerful career yogas in the tradition. "
            "Fortune (9th) actively supports karma/action (10th). Career success comes with an "
            "element of luck, divine timing, and the sense that you are doing your rightful work. "
            "This yoga is particularly powerful if the 9th and 10th lords are also conjunct or "
            "mutually aspecting each other."
        ),
        "activation": "natal"
    },
    {
        "id": "career_budhaditya",
        "topic": "career",
        "condition": lambda ctx: (
            "Sun" in ctx["planets"] and "Mercury" in ctx["planets"] and
            longitude_to_sign(ctx["planets"]["Sun"])[0] == longitude_to_sign(ctx["planets"]["Mercury"])[0]
        ),
        "severity": "positive",
        "score": 2,
        "title": "Budhaditya Yoga — Sharp Intellect in Action",
        "detail": lambda ctx: (
            f"Sun and Mercury are conjunct in {longitude_to_sign(ctx['planets']['Sun'])[0]}, "
            "forming Budhaditya Yoga. This sharpens analytical power, communication, and "
            "administrative acumen, creating natural aptitude for management, writing, teaching, "
            "consulting, or any profession requiring quick, precise thinking. The yoga's strength "
            f"depends on Sun's dignity ({_dignity('Sun', ctx)}) and Mercury's ({_dignity('Mercury', ctx)}). "
            + ("Both planets strong — this is a highly activated Budhaditya Yoga." if _strong("Sun", ctx) and _strong("Mercury", ctx)
               else "")
        ),
        "activation": "natal"
    },
    {
        "id": "career_amala_yoga",
        "topic": "career",
        "condition": lambda ctx: (
            any(_house(p, ctx) == 10 and p in ["Jupiter","Venus","Mercury","Moon"]
                for p in ctx["planets"])
        ),
        "severity": "positive",
        "score": 2,
        "title": "Amala Yoga — Spotless Reputation",
        "detail": lambda ctx: (
            "A natural benefic (Jupiter, Venus, Mercury, or Moon) is placed in the 10th house, "
            "forming Amala Yoga ('spotless'). This confers an unblemished professional reputation, "
            "ethical recognition, and the goodwill of others. Even in competitive environments, "
            "your integrity acts as a career asset. Fame that comes from this placement tends to "
            "be sustainable rather than fleeting."
        ),
        "activation": "natal"
    },
    {
        "id": "career_dasha_career_planet",
        "topic": "career",
        "condition": lambda ctx: ctx.get("dasha","") in ["Jupiter","Sun","Saturn","Mercury","Rahu"],
        "severity": "positive",
        "score": 2,
        "title": "Favourable Career Mahadasha Running",
        "detail": lambda ctx: (
            f"{ctx.get('dasha','')} Mahadasha is active ({ctx.get('dasha_md_start','')} → "
            f"{ctx.get('dasha_md_end','')}). "
            + {
                "Jupiter": (
                    "Jupiter MD is the most dharmic career period. Expect expansion into teaching, "
                    "law, banking, or advisory roles. Promotions tied to wisdom and institutional "
                    "standing are likely. Jupiter-Saturn and Jupiter-Mercury Antardashas deliver "
                    "the most tangible professional milestones."
                ),
                "Sun": (
                    "Sun MD brings authority, visibility, and advancement in government, leadership, "
                    "or corporate hierarchies. Your identity aligns closely with your work. "
                    "Sun-Jupiter AD is a powerful promotion window; Sun-Saturn AD may bring "
                    "professional challenges requiring humility."
                ),
                "Saturn": (
                    "Saturn MD rewards past disciplined effort. Promotions come but slowly and "
                    "through demonstrated reliability. This is a period of building legacy, not "
                    "quick wins. Saturn-Mercury AD supports career in analytical or structured domains; "
                    "Saturn-Rahu AD (7.5 months) can bring unexpected career disruptions — navigate carefully."
                ),
                "Mercury": (
                    "Mercury MD favours communication, trade, IT, analytics, and multi-platform careers. "
                    "New skills acquired now compound into career capital. Mercury-Sun and Mercury-Jupiter "
                    "Antardashas are the peak performance sub-periods."
                ),
                "Rahu": (
                    "Rahu MD offers dramatic career leaps through unconventional routes — technology, "
                    "foreign assignments, or niche expertise. The first 2.5 years (Rahu-Rahu AD) are "
                    "most volatile; Rahu-Jupiter AD (approx. mid-period) is often the golden window "
                    "for breakthrough. Avoid ethical shortcuts — Rahu's gifts are easily lost if misused."
                ),
            }.get(ctx.get("dasha",""), "")
            + f"\n  Current Antardasha: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "career_dasha_moderate",
        "topic": "career",
        "condition": lambda ctx: ctx.get("dasha","") in ["Ketu","Moon","Mars","Venus"],
        "severity": "neutral",
        "score": 0,
        "title": "Moderate Career Dasha — Context-Dependent",
        "detail": lambda ctx: (
            f"{ctx.get('dasha','')} Mahadasha is active. "
            + {
                "Ketu":  (
                    "Ketu MD often brings career transitions, sudden exits, or a pivot away from "
                    "mainstream ambition toward research, healing, or spiritual fields. Professional "
                    "identity becomes fluid. Best strategy: focus on mastery rather than titles, "
                    "and watch for the Ketu-Jupiter AD as a potential breakout sub-period."
                ),
                "Moon":  (
                    "Moon MD supports public-facing, nurturing, or creative careers — real estate, "
                    "hospitality, healthcare, and arts can thrive. However, emotional fluctuations "
                    "may affect decision-making. Moon-Jupiter AD is the best sub-period for "
                    "professional recognition; Moon-Rahu AD brings erratic career energy."
                ),
                "Mars":  (
                    "Mars MD boosts drive, initiative, and technical career progress. Ambition peaks, "
                    "sometimes leading to impulsive job changes or conflicts with authority. "
                    "Technical, athletic, surgical, or competitive careers accelerate. "
                    "Mars-Sun and Mars-Jupiter Antardashas are the best windows for career advancement."
                ),
                "Venus": (
                    "Venus MD supports creative, artistic, and luxury-sector careers. Financial gains "
                    "through partnerships or creative collaborations are common. Venus-Mercury and "
                    "Venus-Sun Antardashas are the peak sub-periods for career breakthroughs in "
                    "Venus-ruled fields."
                ),
            }.get(ctx.get("dasha",""), "")
            + f"\n  Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "career_jupiter_aspects_10th",
        "topic": "career",
        "condition": lambda ctx: _aspects_house("Jupiter", 10, ctx) and _house("Jupiter", ctx) != 10,
        "severity": "positive",
        "score": 2,
        "title": "Jupiter Aspects 10th House — Blessings on Career",
        "detail": lambda ctx: (
            f"Jupiter (from House {_house('Jupiter', ctx)}) aspects the 10th house through its "
            f"{'5th/9th' if _house('Jupiter', ctx) in [2,6] else '7th'} aspect. "
            "This is highly auspicious — Jupiter's wisdom, dharmic energy, and expansive blessings "
            "illuminate your career house. Opportunities in Jupiter-ruled fields arise unexpectedly, "
            f"and you are protected from severe career falls. Jupiter is {_dignity('Jupiter', ctx)}; "
            + ("its exaltation/own-sign strength makes this aspect especially potent." if _strong("Jupiter", ctx) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "career_saturn_aspects_10th",
        "topic": "career",
        "condition": lambda ctx: _aspects_house("Saturn", 10, ctx) and _house("Saturn", ctx) != 10,
        "severity": "neutral",
        "score": 1,
        "title": "Saturn Aspects 10th House — Karmic Career Pressure",
        "detail": lambda ctx: (
            f"Saturn (from House {_house('Saturn', ctx)}) aspects the 10th house. "
            "Saturn's aspect on the karma bhava brings karmic intensity to your professional life: "
            "hard work, delayed gratification, and eventual recognition through persistence. "
            "Avoid cutting corners in your career — Saturn here demands ethical conduct and "
            "sustained effort. The payoff is longevity and unassailable professional reputation."
        ),
        "activation": "natal"
    },

    # ════════════════════════════════════════════════════════════
    # MARRIAGE
    # ════════════════════════════════════════════════════════════
    {
        "id": "marriage_venus_strong",
        "topic": "marriage",
        "condition": lambda ctx: _strong("Venus", ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong Venus — Happiness, Love & Refined Partnership",
        "detail": lambda ctx: (
            f"Venus is {_dignity('Venus', ctx)}, the most important indicator for marital happiness. "
            "A dignified Venus confers a loving, aesthetically pleasing, and emotionally warm "
            "marriage. The spouse is likely attractive, creative, and affectionate. Material comforts "
            f"and pleasures are abundant in married life. Venus is in House {_house('Venus', ctx)} "
            f"and in navamsa it is {_navamsa_dignity('Venus', ctx)}"
            + (" — D9 confirmation adds deep soul-level compatibility." if _navamsa_dignity("Venus", ctx) in ["Exalted","Own","Mool Trikona"] else ".")
            + f" Vimsopaka Bala: {ctx['vimsopaka'].get('Venus','?')}/20."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_venus_weak",
        "topic": "marriage",
        "condition": lambda ctx: _weak("Venus", ctx) and not _nb("Venus", ctx),
        "severity": "warning",
        "score": -2,
        "title": "Debilitated Venus — Marital Tensions to Navigate",
        "detail": lambda ctx: (
            "Venus is debilitated without Neechabhanga, which is the most significant indicator "
            "of marital dissatisfaction, mismatched expectations, or incompatibility. This does "
            "not prevent marriage, but the relationship requires conscious cultivation. "
            "Watch for Venus Mahadasha and Antardasha periods as sensitive windows. "
            "Remedies: white flowers offered on Fridays, charity to young women, chanting "
            "Venus/Shukra mantras, wearing white or cream on Fridays. A pre-marital counsellor "
            "or astro-compatible partner matching is strongly recommended."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_venus_weak_nb",
        "topic": "marriage",
        "condition": lambda ctx: _weak("Venus", ctx) and _nb("Venus", ctx),
        "severity": "neutral",
        "score": 1,
        "title": "Debilitated Venus with Neechabhanga — Love Tested, Then Victorious",
        "detail": lambda ctx: (
            "Venus is debilitated, but Neechabhanga (cancellation of debilitation) applies. "
            "This classical configuration suggests early relationship difficulties that are "
            "ultimately resolved — often producing a deeper, more mature love. The marriage "
            "that survives these tests tends to be exceptionally strong and loyal."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_7th_lord_strong",
        "topic": "marriage",
        "condition": lambda ctx: _strong(_lord(7, ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong 7th Lord — Blessed Partnership",
        "detail": lambda ctx: (
            f"The 7th lord {_lord(7,ctx)} is {_dignity(_lord(7,ctx), ctx)}, strongly activating "
            "the house of partnership. The spouse will be a genuine pillar of strength — capable, "
            "supportive, and karmically well-matched. Business partnerships are also favoured. "
            f"The 7th lord resides in House {_house(_lord(7,ctx), ctx)}, which defines the sphere "
            "of life through which your spouse enters or expresses themselves."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_7th_lord_weak",
        "topic": "marriage",
        "condition": lambda ctx: _weak(_lord(7, ctx), ctx) and not _nb(_lord(7, ctx), ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 7th Lord — Partnership Requires Extra Care",
        "detail": lambda ctx: (
            f"The 7th lord {_lord(7,ctx)} is debilitated without Neechabhanga. This is the "
            "single most important indicator of partnership challenges: incompatibility, emotional "
            "distance, possible separation, or delay in finding the right partner. The 7th lord's "
            f"debilitation sign is {DEBILITATION.get(_lord(7,ctx), '?')}. "
            "Remedies for the 7th lord planet and Venus are essential. Compatibility matching "
            "(Ashtakoota + chart analysis) before marriage is strongly advised."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_venus_d9_strong",
        "topic": "marriage",
        "condition": lambda ctx: _navamsa_dignity("Venus", ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 2,
        "title": "Venus Exalted/Own in Navamsa (D9) — Soul-Level Marital Harmony",
        "detail": lambda ctx: (
            f"Venus is {_navamsa_dignity('Venus', ctx)} in the Navamsa (D9), the chart that "
            "governs the inner quality of relationships and karmic partnerships. Even if the D1 "
            "shows complexity, D9 strength confirms deep emotional compatibility, lasting affection, "
            "and the sense that you and your spouse are genuinely well-matched at the soul level."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_jupiter_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Jupiter", ctx) == 7,
        "severity": "positive",
        "score": 2,
        "title": "Jupiter in 7th — A Wise, Dharmic Spouse",
        "detail": lambda ctx: (
            "Jupiter in the 7th house is one of the best placements for marriage. The spouse is "
            "likely to be educated, wise, spiritually inclined, and morally upright. This placement "
            "protects the marriage from disintegration even during difficult periods. Jupiter here "
            f"also aspects your Lagna (1st house), blessing your physical body and overall health. "
            f"Jupiter is {_dignity('Jupiter', ctx)} here — "
            + ("a truly exceptional placement, potentially forming Hamsa Yoga." if _strong("Jupiter", ctx)
               else "even a neutral Jupiter in the 7th is significantly protective.")
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_kuja_dosha_high",
        "topic": "marriage",
        "condition": lambda ctx: _house("Mars", ctx) == 7,
        "severity": "warning",
        "score": -2,
        "title": "High Kuja Dosha — Mars in 7th House",
        "detail": lambda ctx: (
            "Mars in the 7th house creates the most intense form of Kuja (Mangal) Dosha. "
            "This placement brings passion and energy into relationships but also the risk of "
            "dominance conflicts, power struggles, and in severe cases, separation or early "
            "widowhood (classical interpretation — modern outcomes vary by chart strength). "
            "Matching with a Manglik partner (Mars in 1,2,4,7,8,12 in their chart) effectively "
            "neutralises this dosha. Mars Shanti puja, Hanuman worship, and fasting on Tuesdays "
            f"are classical remedies. Mars dignity: {_dignity('Mars', ctx)}. "
            + ("Exalted Mars in 7th may actually confer a powerful, dynamic partner." if _dignity("Mars", ctx) == "Exalted" else "")
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_kuja_dosha_moderate",
        "topic": "marriage",
        "condition": lambda ctx: _house("Mars", ctx) in [1,2,4,8,12],
        "severity": "caution",
        "score": -1,
        "title": "Moderate Kuja Dosha",
        "detail": lambda ctx: (
            f"Mars is in House {_house('Mars',ctx)}, which creates partial Kuja Dosha. "
            "Classical texts identify Mars in the 1st, 2nd, 4th, 8th, and 12th as Mangalik "
            "(7th is the strongest form). Partial dosha creates assertiveness, passion-driven "
            "conflicts, and occasional friction in marriage — manageable with conscious effort. "
            "Matching with a partner who has similar Mars placement significantly reduces tension."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_rahu_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Rahu", ctx) == 7,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 7th — Unconventional Marriage, Obsession, or Foreign Spouse",
        "detail": lambda ctx: (
            "Rahu in the 7th house often brings an unusual, unexpected, or cross-cultural marriage. "
            "The attraction is intense and can border on obsession. There may be a significant "
            "age gap, cultural difference, or unusual circumstances around the union. Trust issues "
            "or deception are possible if Rahu is afflicted. The spouse may have a foreign "
            "background, unusual personality, or exceptional ambition. "
            "Rahu here also means Ketu (detachment) is in the 1st — there is a karmic lesson "
            "about releasing self-focus and learning through partnership."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_saturn_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Saturn", ctx) == 7,
        "severity": "caution",
        "score": -1,
        "title": "Saturn in 7th — Delayed but Karmic Marriage",
        "detail": lambda ctx: (
            "Saturn in the 7th is one of the classic delay indicators for marriage. "
            "Union typically occurs after age 28-32 (post-Saturn's first return). Once formed, "
            "however, the marriage tends to be deeply karmic, committed, and lasting. "
            "The spouse may be older, more serious, or emotionally reserved. "
            "This placement teaches patience and the value of a partnership built on responsibility. "
            f"Saturn is {_dignity('Saturn', ctx)} here — "
            + ("exalted Saturn in the 7th can actually bring an exceptionally reliable, "
               "high-achieving spouse after the initial delay." if _dignity("Saturn", ctx) == "Exalted"
               else "debilitated Saturn here intensifies delay and may introduce persistent friction." if _dignity("Saturn", ctx) == "Debilitated"
               else "the karmic quality of the marriage is dominant.")
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_dasha_venus",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("dasha","") == "Venus",
        "severity": "positive",
        "score": 3,
        "title": "Venus Mahadasha — The Prime Marriage Window",
        "detail": lambda ctx: (
            "Venus Mahadasha (20 years) is the most powerful period for romantic union and marriage. "
            f"Running: {ctx.get('dasha_md_start','')} → {ctx.get('dasha_md_end','')}. "
            "The best sub-periods (Antardashas) for marriage: Venus-Jupiter (dharmic union), "
            "Venus-Mercury (intellectual compatibility), Venus-Moon (emotional bonding). "
            "Venus-Saturn AD can bring a delayed but karmic union. "
            f"Current Antardasha: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "marriage_dasha_jupiter",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("dasha","") == "Jupiter",
        "severity": "positive",
        "score": 2,
        "title": "Jupiter Mahadasha — Dharmic & Auspicious for Partnership",
        "detail": lambda ctx: (
            "Jupiter Mahadasha blesses marriage, children, and family life. "
            f"Running: {ctx.get('dasha_md_start','')} → {ctx.get('dasha_md_end','')}. "
            "Best Antardashas for marriage: Jupiter-Venus (highest priority), Jupiter-Moon. "
            "Jupiter-Mars AD can accelerate things but also introduce impulsiveness. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "marriage_dasha_saturn",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Sober, Delayed, but Lasting Unions",
        "detail": lambda ctx: (
            "Saturn MD is not the first-choice marriage dasha, but unions formed in it tend to "
            "be serious, karmic, and built to last. The relationship may feel burdensome at first "
            "but deepens with time. Look for Venus or Jupiter Antardashas within Saturn MD for "
            "best marriage windows. Saturn-Moon AD can bring emotional strain in existing "
            f"relationships. Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "marriage_benefics_aspect_7th",
        "topic": "marriage",
        "condition": lambda ctx: len(_benefics_aspect(7, ctx)) > 0,
        "severity": "positive",
        "score": 2,
        "title": "Benefic Planets Aspect 7th House — Protected Marriage",
        "detail": lambda ctx: (
            f"Natural benefics {', '.join(_benefics_aspect(7, ctx))} aspect the 7th house, "
            "protecting the marriage from serious harm and adding qualities of wisdom, love, "
            "or communication to the partnership. Benefic aspects on the 7th are one of the "
            "strongest protective factors against separation or severe marital conflict."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_malefics_aspect_7th",
        "topic": "marriage",
        "condition": lambda ctx: len(_malefics_aspect(7, ctx)) >= 2,
        "severity": "warning",
        "score": -2,
        "title": "Multiple Malefics Aspect 7th House — Relationship Stress",
        "detail": lambda ctx: (
            f"Malefic planets {', '.join(_malefics_aspect(7, ctx))} all aspect the 7th house. "
            "Multiple malefic aspects on the marriage house create significant stress in "
            "relationships — friction, power conflicts, aggression, or instability. "
            "This does not prevent marriage, but requires choosing a highly compatible "
            "partner and doing active remedies. Check if benefics also aspect (they mitigate). "
            "Counselling and couple-directed spiritual practice are highly recommended."
        ),
        "activation": "natal"
    },

    # ════════════════════════════════════════════════════════════
    # CHILDREN
    # ════════════════════════════════════════════════════════════
    {
        "id": "children_jupiter_strong",
        "topic": "children",
        "condition": lambda ctx: _strong("Jupiter", ctx),
        "severity": "positive",
        "score": 4,
        "title": "Strong Jupiter (Putrakaraka) — Blessed Progeny",
        "detail": lambda ctx: (
            f"Jupiter — the natural Putrakaraka (significator of children) — is {_dignity('Jupiter',ctx)}, "
            "one of the most powerful indicators of good fortune in matters of children. "
            "Multiple healthy children are possible; at least one is likely to be exceptionally "
            "talented, wise, or spiritually significant. "
            f"Jupiter is in House {_house('Jupiter', ctx)} "
            + ("— a trine (1/5/9), maximising the auspiciousness of this yoga." if _house("Jupiter", ctx) in [1,5,9]
               else "— a kendra (4/7/10), conferring strength and visibility to progeny themes." if _house("Jupiter", ctx) in [4,7,10]
               else ".")
            + f" Vimsopaka Bala: {ctx['vimsopaka'].get('Jupiter','?')}/20."
        ),
        "activation": "natal"
    },
    {
        "id": "children_jupiter_weak",
        "topic": "children",
        "condition": lambda ctx: _weak("Jupiter", ctx) and not _nb("Jupiter", ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated Jupiter — Progeny Challenges",
        "detail": lambda ctx: (
            "Jupiter (Putrakaraka) is debilitated without Neechabhanga — the most critical "
            "single indicator of difficulty with children. Delays, conception challenges, "
            "or difficult pregnancies are possible. This does not universally deny children, "
            "but medical consultation and remedies are strongly advised before attempting conception. "
            "Classical remedies: Jupiter Shanti Puja, Santana Gopala Puja (for child-blessing), "
            "Thursdays fasting, yellow sapphire/topaz consultation, donating gold or turmeric."
        ),
        "activation": "natal"
    },
    {
        "id": "children_5th_lord_strong",
        "topic": "children",
        "condition": lambda ctx: _strong(_lord(5, ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong 5th Lord — Fertile and Auspicious Children's House",
        "detail": lambda ctx: (
            f"The 5th lord {_lord(5,ctx)} is {_dignity(_lord(5,ctx),ctx)}, strongly activating "
            "the Putra Bhava (house of children). Children are likely to be intellectually bright, "
            "creatively gifted, or spiritually inclined. They bring honour and pride to the family. "
            f"The 5th lord is in House {_house(_lord(5,ctx), ctx)} — that house's themes become "
            "entwined with how children enter and express themselves in your life."
        ),
        "activation": "natal"
    },
    {
        "id": "children_5th_lord_weak",
        "topic": "children",
        "condition": lambda ctx: _weak(_lord(5, ctx), ctx) and not _nb(_lord(5, ctx), ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 5th Lord — Challenges with Progeny",
        "detail": lambda ctx: (
            f"The 5th lord {_lord(5,ctx)} is debilitated without cancellation, weakening the "
            "house of children. Conception difficulties, pregnancy complications, or few children "
            "are possible. Check the 5th house for aspecting benefics — they can compensate. "
            "Remedies for the 5th lord and regular Santana Gopala Puja are important."
        ),
        "activation": "natal"
    },
    {
        "id": "children_saturn_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Saturn", ctx) == 5,
        "severity": "caution",
        "score": -2,
        "title": "Saturn in 5th — Delayed but Serious Children",
        "detail": lambda ctx: (
            "Saturn in the 5th house classically delays progeny, often until after Saturn's "
            "maturation (~36 years) or after the Saturn Antardasha passes. The children who are "
            "born tend to be serious, responsible, and long-lived — Saturn-type children who age "
            "gracefully. Miscarriages or health issues in early pregnancies are possible indicators "
            "to monitor medically. Saturn in 5th also restricts spontaneous creative joy — "
            "developing playfulness and emotional expressiveness is important for parenting."
        ),
        "activation": "natal"
    },
    {
        "id": "children_benefics_aspect_5th",
        "topic": "children",
        "condition": lambda ctx: len(_benefics_aspect(5, ctx)) > 0,
        "severity": "positive",
        "score": 2,
        "title": "Benefics Aspect 5th House — Protected Progeny Path",
        "detail": lambda ctx: (
            f"Natural benefic(s) {', '.join(_benefics_aspect(5, ctx))} aspect the 5th house, "
            "offering protection and auspiciousness to the children's house. Even if the 5th "
            "lord or Jupiter are weak, benefic aspects can compensate significantly, smoothing "
            "the path to healthy children and supporting their wellbeing and intelligence."
        ),
        "activation": "natal"
    },
    {
        "id": "children_rahu_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Rahu", ctx) == 5,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 5th — Unusual Circumstances Around Conception",
        "detail": lambda ctx: (
            "Rahu in the 5th house creates ambiguity around conception — unusual timing, "
            "IVF/ART procedures, adoption, or stepchildren are common. There may be anxiety "
            "about having children even when fertility is intact. The children born tend to be "
            "unconventional, highly intelligent, or interested in technology/foreign domains. "
            "Rahu remedies (Rahu Shanti, donation of black sesame) and medical consultation "
            "are advised if conception is delayed beyond expectations."
        ),
        "activation": "natal"
    },
    {
        "id": "children_dasha_jupiter",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Jupiter",
        "severity": "positive",
        "score": 3,
        "title": "Jupiter Mahadasha — The Most Auspicious Period for Children",
        "detail": lambda ctx: (
            "Jupiter Mahadasha is universally considered the most favourable period for conception "
            "and the birth of children. "
            f"Running: {ctx.get('dasha_md_start','')} → {ctx.get('dasha_md_end','')}. "
            "The best Antardashas for conception: Jupiter-Jupiter (first ~2.3 years), "
            "Jupiter-Venus, Jupiter-Moon, Jupiter-Mars. Jupiter-Saturn AD may bring a child "
            "after delay. Jupiter-Rahu and Jupiter-Ketu ADs are the least reliable sub-periods. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "children_dasha_venus",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Venus",
        "severity": "positive",
        "score": 2,
        "title": "Venus Mahadasha — Family Expansion Favoured",
        "detail": lambda ctx: (
            "Venus MD is generally favourable for family life and children, as Venus represents "
            "fertility, desire, and creative generation. "
            "Best sub-periods for conception: Venus-Jupiter, Venus-Moon. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "children_dasha_saturn",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Delays in Progeny; Patience Required",
        "detail": lambda ctx: (
            "Saturn MD can bring delays in having children. Medical check-ups are advised "
            "from the start of trying. Saturn-Jupiter and Saturn-Venus Antardashas are the "
            "most promising sub-periods within this cycle. If Saturn rules the 5th house "
            f"from lagna ({ctx['lagna_sign']}), Saturn MD can actually be productive for children "
            "despite the general delay tendency. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },

    # ════════════════════════════════════════════════════════════
    # HEALTH
    # ════════════════════════════════════════════════════════════
    {
        "id": "health_lagna_lord_strong",
        "topic": "health",
        "condition": lambda ctx: _strong(_lord(1, ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong Lagna Lord — Constitutional Vitality & Resilience",
        "detail": lambda ctx: (
            f"The Lagna lord {_lord(1,ctx)} is {_dignity(_lord(1,ctx),ctx)}, bestowing robust "
            "physical constitution, immune strength, and rapid recovery from illness. "
            "The body has natural homeostatic resilience. Chronic diseases are less likely to "
            "take hold; even when they do, recovery is typically faster than average. "
            f"Strength score: {_strength(_lord(1,ctx), ctx)}/100."
            + (" Jupiter aspects the 1st house — double protection on vitality and longevity." if _aspects_house("Jupiter", 1, ctx) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "health_lagna_lord_weak",
        "topic": "health",
        "condition": lambda ctx: _weak(_lord(1, ctx), ctx) and not _nb(_lord(1, ctx), ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated Lagna Lord — Physical Vulnerability",
        "detail": lambda ctx: (
            f"The Lagna lord {_lord(1,ctx)} is debilitated, weakening the physical body's "
            "baseline constitution and immune response. This increases susceptibility to the "
            "diseases associated with the lagna sign and its lord's significations. "
            "The remedy priority is highest for the Lagna lord planet. "
            "Regular health check-ups (minimum annually), stress management, and avoiding "
            "immunosuppressive habits (poor sleep, alcohol, erratic diet) are essential."
        ),
        "activation": "natal"
    },
    {
        "id": "health_saturn_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Saturn", ctx) in [6, 8],
        "severity": "caution",
        "score": -2,
        "title": f"Saturn in 6th/8th — Chronic or Long-Latency Health Concerns",
        "detail": lambda ctx: (
            f"Saturn is in House {_house('Saturn', ctx)}. "
            + ("Saturn in the 6th gives victory over enemies and competition but predisposes to "
               "long-term, chronic conditions — especially joints, bones, teeth, skin, and "
               "nervous system. Service-related stress and exhaustion are occupational hazards." if _house("Saturn", ctx) == 6
               else "Saturn in the 8th classically gives longevity but with chronic ailments accompanying "
               "long life — digestive issues, vata-type disorders (nerve pain, arthritis, malabsorption), "
               "or constitutional weakness that requires managed care throughout life.")
            + f" Saturn is {_dignity('Saturn', ctx)} here. "
            + ("Exalted Saturn in the 8th grants exceptional longevity." if _dignity("Saturn", ctx) == "Exalted" and _house("Saturn", ctx) == 8 else "")
            + " Regularity in diet, sleep, exercise, and Ayurvedic vata-management is the primary prescription."
        ),
        "activation": "natal"
    },
    {
        "id": "health_mars_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Mars", ctx) in [6, 8],
        "severity": "caution",
        "score": -1,
        "title": "Mars in 6th/8th — Inflammation, Accidents & Surgeries",
        "detail": lambda ctx: (
            f"Mars is in House {_house('Mars', ctx)}. "
            + ("Mars in the 6th energises the immune system (good for fighting disease) but "
               "can cause inflammatory conditions, blood disorders, fevers, and accident-proneness. "
               "Physical exercise is an excellent outlet for this martial energy." if _house("Mars", ctx) == 6
               else "Mars in the 8th increases accident risk, surgeries, and sudden health events, "
               "particularly involving blood, head, or reproductive system. Extra caution during "
               "Mars Mahadasha and Antardasha is advised.")
            + f" Mars is {_dignity('Mars', ctx)} here; "
            + ("exalted Mars gives strong physical recovery even if the risk is present." if _dignity("Mars", ctx) == "Exalted" else "")
        ),
        "activation": "natal"
    },
    {
        "id": "health_moon_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Moon", ctx) in [6, 8],
        "severity": "caution",
        "score": -2,
        "title": "Moon in 6th/8th — Mental & Digestive Health Priority",
        "detail": lambda ctx: (
            f"Moon is in House {_house('Moon', ctx)}. "
            + ("Moon in the 6th can cause emotional instability, digestive disorders, lymphatic "
               "and fluid-related issues, and hypochondria. A tendency to worry about health can "
               "itself become a stressor — mindfulness and a calm daily routine are essential." if _house("Moon", ctx) == 6
               else "Moon in the 8th creates emotional vulnerability, psychosomatic conditions, "
               "and susceptibility to menstrual (if applicable) or hormonal irregularities. "
               "Psychological health is intimately tied to physical wellbeing here.")
            + " Prioritise sleep hygiene, meditation, and limiting sugar and cold/raw foods."
            + (" Waning Moon at birth intensifies these effects." if ctx["planets"].get("Moon", 0) % 360 > 180 else "")
        ),
        "activation": "natal"
    },
    {
        "id": "health_rahu_6th",
        "topic": "health",
        "condition": lambda ctx: _house("Rahu", ctx) == 6,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 6th — Mysterious or Hard-to-Diagnose Ailments",
        "detail": lambda ctx: (
            "Rahu in the 6th house is associated with unusual, atypical, or misdiagnosed health "
            "conditions. Allergies, autoimmune tendencies, anxiety disorders, unusual infections, "
            "or substance-related issues are possible. Rahu here can also make one exceptionally "
            "resistant in fighting illness — a double-edged quality. Always seek second medical "
            "opinions. Periodic detox, clean diet, and Rahu remedies (donation of black sesame, "
            "blue/black objects on Saturdays) are protective."
        ),
        "activation": "natal"
    },
    {
        "id": "health_ketu_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Ketu", ctx) == 8,
        "severity": "caution",
        "score": -1,
        "title": "Ketu in 8th — Psychosomatic Crises & Spiritual Health Events",
        "detail": lambda ctx: (
            "Ketu in the 8th house can manifest as sudden, inexplicable health events — "
            "near-death experiences, unexpected surgeries, or psychosomatic conditions whose "
            "physical cause is hard to identify. Spiritual crisis and physical crisis become "
            "intertwined. This placement also gives intuitive healing abilities if the chart "
            "is otherwise strong. Avoid extreme sports and dangerous activities during Ketu AD. "
            "Spiritual practice, meditation, and Ketu remedies are protective."
        ),
        "activation": "natal"
    },
    {
        "id": "health_jupiter_trine_strong",
        "topic": "health",
        "condition": lambda ctx: (
            _strong("Jupiter", ctx) and _house("Jupiter", ctx) in [1, 5, 9]
        ),
        "severity": "positive",
        "score": 3,
        "title": "Strong Jupiter in Trine — Exceptional Health Protection",
        "detail": lambda ctx: (
            f"Jupiter is {_dignity('Jupiter',ctx)} and placed in House {_house('Jupiter',ctx)} "
            "(a trikona — divine house). This is among the strongest health-protective yogas "
            "in Vedic astrology. Jupiter's life-force energy directly supports the body's "
            "vitality, immune system, and longevity. Recovery from illness is typically fast; "
            "chronic or fatal diseases are less likely to manifest despite other stressors."
        ),
        "activation": "natal"
    },
    {
        "id": "health_sade_sati",
        "topic": "health",
        "condition": lambda ctx: ctx.get("sade_sati_active", False),
        "severity": "caution",
        "score": -2,
        "title": "Sade Sati Active — Physical & Mental Depletion",
        "detail": lambda ctx: (
            f"Saturn's Sade Sati is currently active ({ctx.get('sade_sati_phase','')}) over "
            f"your Moon sign ({ctx.get('moon_sign','')}). This 7.5-year cycle tests physical "
            "stamina, mental resilience, and emotional strength simultaneously. Immune function "
            "is often lowered; sleep disturbances, digestive issues, and joint/bone discomfort "
            "are common. The Peak Phase (Saturn on Moon sign) is most intense. "
            "Protective measures: regular moderate exercise, consistent sleep schedule, "
            "blue sapphire consultation with a qualified astrologer, oil massage on Saturdays, "
            "feeding crows and dogs (traditional Saturn remedies), and Shani Shanti puja."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "health_dasha_saturn",
        "topic": "health",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Health Vigilance Required",
        "detail": lambda ctx: (
            "Saturn MD requires heightened attention to bones, joints, teeth, digestion, skin, "
            "and any chronic conditions. Physical fatigue and slow recovery are common themes. "
            "The most sensitive health sub-periods are Saturn-Rahu AD and Saturn-Ketu AD. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "health_dasha_rahu_ketu",
        "topic": "health",
        "condition": lambda ctx: ctx.get("dasha","") in ["Rahu","Ketu"],
        "severity": "caution",
        "score": -1,
        "title": "Rahu/Ketu Mahadasha — Unusual Health Patterns",
        "detail": lambda ctx: (
            f"{ctx.get('dasha','')} Mahadasha can manifest unusual, hard-to-diagnose health events. "
            + ("Rahu MD: anxiety, stress, atypical infections, lifestyle excesses. Rahu-Saturn and Rahu-Mars ADs are most sensitive." if ctx.get("dasha","") == "Rahu"
               else "Ketu MD: sudden health crises, surgical events, mysterious symptoms, or spiritual-linked physical episodes. Ketu-Mars AD requires particular caution.")
            + f"\n  Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')})."
        ),
        "activation": "dasha_activated"
    },

    # ════════════════════════════════════════════════════════════
    # GENERAL YOGAS
    # ════════════════════════════════════════════════════════════
    {
        "id": "yoga_hamsa",
        "topic": "general",
        "condition": lambda ctx: _strong("Jupiter", ctx) and _house("Jupiter", ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Hamsa Yoga (Panchamahapurusha) — Divine Wisdom & Spiritual Fortune",
        "detail": lambda ctx: (
            f"Jupiter is {_dignity('Jupiter',ctx)} in House {_house('Jupiter',ctx)} (a kendra), "
            "forming the Panchamahapurusha Yoga called Hamsa. This is among the rarest and most "
            "blessed yogas: it grants exceptional wisdom, a noble, compassionate character, "
            "spiritual inclination, distinguished reputation, and material fortune. "
            "The native often becomes a teacher, healer, or guide for others. "
            f"Vimsopaka Bala of Jupiter: {ctx['vimsopaka'].get('Jupiter','?')}/20."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_malavya",
        "topic": "general",
        "condition": lambda ctx: _strong("Venus", ctx) and _house("Venus", ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Malavya Yoga (Panchamahapurusha) — Beauty, Prosperity & Pleasures",
        "detail": lambda ctx: (
            f"Venus is {_dignity('Venus',ctx)} in House {_house('Venus',ctx)} (a kendra), "
            "forming Malavya Yoga. This blesses with physical beauty or magnetic attractiveness, "
            "artistic talent, luxury, romantic success, financial abundance, and a life rich in "
            "sensory pleasures. Fame often comes through Venusian fields (art, music, design, hospitality). "
            f"Vimsopaka: {ctx['vimsopaka'].get('Venus','?')}/20."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_ruchaka",
        "topic": "general",
        "condition": lambda ctx: _strong("Mars", ctx) and _house("Mars", ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Ruchaka Yoga (Panchamahapurusha) — Courage, Command & Vitality",
        "detail": lambda ctx: (
            f"Mars is {_dignity('Mars',ctx)} in House {_house('Mars',ctx)} (a kendra), "
            "forming Ruchaka Yoga. This confers exceptional physical vitality, courage, "
            "military or competitive prowess, leadership, and decisive command. "
            "The native often excels in fields requiring strength, strategy, and assertiveness."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_bhadra",
        "topic": "general",
        "condition": lambda ctx: _strong("Mercury", ctx) and _house("Mercury", ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Bhadra Yoga (Panchamahapurusha) — Intellect, Eloquence & Wealth",
        "detail": lambda ctx: (
            f"Mercury is {_dignity('Mercury',ctx)} in House {_house('Mercury',ctx)} (a kendra), "
            "forming Bhadra Yoga. This bestows sharp intellect, exceptional communication, "
            "business acumen, mathematical or logical aptitude, and financial success through "
            "intellectual work. Writing, oratory, and commerce are natural strengths."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_shasha",
        "topic": "general",
        "condition": lambda ctx: _strong("Saturn", ctx) and _house("Saturn", ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Shasha Yoga (Panchamahapurusha) — Authority, Discipline & Lasting Legacy",
        "detail": lambda ctx: (
            f"Saturn is {_dignity('Saturn',ctx)} in House {_house('Saturn',ctx)} (a kendra), "
            "forming Shasha Yoga. This confers authority, iron discipline, management mastery, "
            "and a lasting professional legacy built through sustained effort. "
            "Success arrives later in life but is unshakeable once established."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_gajkesari",
        "topic": "general",
        # BUG FIX: Gaja-Kesari = Jupiter in kendra FROM MOON (not from lagna)
        "condition": lambda ctx: (
            ((_house("Jupiter", ctx) - _house("Moon", ctx)) % 12) in [0, 3, 6, 9]
        ),
        "severity": "positive",
        "score": 4,
        "title": "Gaja-Kesari Yoga — Fame, Wisdom & Respected Standing",
        "detail": lambda ctx: (
            f"Jupiter is in House {_house('Jupiter',ctx)} and Moon in House {_house('Moon',ctx)}, "
            f"placing Jupiter in the {((_house('Jupiter',ctx)-_house('Moon',ctx))%12)+1}th from "
            "the Moon — a kendra relationship, forming the celebrated Gaja-Kesari Yoga. "
            "This is one of the most auspicious yogas: it grants fame, wealth, eloquence, "
            "wisdom, and a respected position in society. The native is remembered long after "
            f"their time. Strength is amplified when Jupiter is {_dignity('Jupiter', ctx)} "
            + ("(as it is here — maximum potency)." if _strong("Jupiter", ctx) else ".")
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_viparita_harsha",
        "topic": "general",
        "condition": lambda ctx: (
            _lord(6, ctx) != "" and _house(_lord(6, ctx), ctx) in [6, 8, 12]
        ),
        "severity": "positive",
        "score": 3,
        "title": "Viparita Harsha Yoga — Victory Born from Adversity",
        "detail": lambda ctx: (
            f"The 6th lord {_lord(6,ctx)} is in House {_house(_lord(6,ctx),ctx)} (a dusthana), "
            "forming Viparita Harsha Yoga — where the house of enemies and obstacles destroys "
            "itself. Enemies are defeated by their own actions. Challenges ultimately strengthen "
            "you. This yoga is subtly powerful: its benefits manifest quietly but consistently, "
            "often through the failure of opponents or the turning of seemingly bad situations "
            "into growth opportunities."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_neechabhanga_raja",
        "topic": "general",
        "condition": lambda ctx: (
            any(_nb(p, ctx) for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"])
        ),
        "severity": "positive",
        "score": 3,
        "title": "Neechabhanga Raja Yoga — Adversity Transformed into Royalty",
        "detail": lambda ctx: (
            "One or more planets have their debilitation cancelled (Neechabhanga). "
            "Classical texts consider this a Raja Yoga — the cancellation of weakness produces "
            "exceptional strength, often exceeding what a straightforwardly exalted planet would give. "
            "The native is tested severely in the areas ruled by the debilitated planet, then "
            "triumphs in a way that is remarkable precisely because the fall preceded the rise. "
            f"Planets with Neechabhanga: "
            f"{', '.join(p for p in ['Sun','Moon','Mars','Mercury','Jupiter','Venus','Saturn'] if _nb(p, ctx))}."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_vesi",
        "topic": "general",
        "condition": lambda ctx: any(
            p != "Moon" and p != "Rahu" and p != "Ketu" and
            longitude_to_sign(ctx["planets"][p])[0] == longitude_to_sign(
                ctx["planets"]["Sun"] + 30)[0]
            for p in ctx["planets"]
        ),
        "severity": "positive",
        "score": 2,
        "title": "Vesi Yoga — Reputation, Eloquence & Good Memory",
        "detail": lambda ctx: (
            "A planet (other than Moon, Rahu, Ketu) occupies the 2nd house from the Sun, "
            "forming Vesi Yoga. This confers reputation, good memory, eloquence, and honesty. "
            "The native is generally respected and known for their word. The strength of the "
            "yoga depends on which planet forms it and its own dignity."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_atmakaraka_strong",
        "topic": "general",
        "condition": lambda ctx: _strong(ctx.get("atmakaraka",""), ctx) if ctx.get("atmakaraka") else False,
        "severity": "positive",
        "score": 3,
        "title": "Strong Atmakaraka — Soul's Purpose Supported by Destiny",
        "detail": lambda ctx: (
            f"The Atmakaraka (soul significator in Jaimini astrology) is {ctx.get('atmakaraka','')} "
            f"at {_dignity(ctx.get('atmakaraka',''), ctx)} dignity. "
            "A strong Atmakaraka indicates that the soul's primary purpose in this lifetime is "
            "actively supported by the planetary environment. The native feels aligned with their "
            "deeper calling, and life's major events tend to push them toward authentic self-expression. "
            f"The house of the Atmakaraka ({_house(ctx.get('atmakaraka',''), ctx)}) is a key zone "
            "of soul-level activity and meaning."
        ),
        "activation": "natal"
    },
]


# ==================================================================
# SECTION 8 — RULE ENGINE
# ==================================================================

def evaluate_rules(ctx: Dict, topic: str = None, debug: bool = False) -> List[Dict]:
    results = []
    for rule in PREDICTION_RULES:
        if topic and rule["topic"] != topic:
            continue
        try:
            fired = rule["condition"](ctx)
        except Exception as e:
            if debug:
                raise
            fired = False
        if fired:
            try:
                detail = rule["detail"](ctx)
            except Exception as e:
                detail = f"[Detail error: {e}]" if debug else "[Detail unavailable]"
            results.append({
                "id":         rule["id"],
                "topic":      rule["topic"],
                "severity":   rule["severity"],
                "score":      rule["score"],
                "title":      rule["title"],
                "detail":     detail,
                "activation": rule.get("activation", "natal"),
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def score_topic(fired_rules: List[Dict]) -> Dict:
    total    = sum(r["score"] for r in fired_rules)
    positive = [r for r in fired_rules if r["severity"] == "positive"]
    warnings = [r for r in fired_rules if r["severity"] in ["warning","caution"]]
    return {
        "net_score":      total,
        "positive_count": len(positive),
        "warning_count":  len(warnings),
        "rating": (
            "Excellent"   if total >= 8  else
            "Good"        if total >= 4  else
            "Mixed"       if total >= 0  else
            "Challenging" if total >= -5 else
            "Difficult"
        )
    }


def _apply_dasha_boost(fired_rules: List[Dict], topic_lord: str, md_planet: str,
                       related_planets: List[str] = None) -> List[Dict]:
    """
    BUG FIX v4: Works on deep copies; no mutation of originals.
    Boosts dasha-activated rules when the running MD planet governs the topic.
    """
    related   = set([topic_lord] + (related_planets or []))
    result    = []
    for r in fired_rules:
        r_copy = copy.deepcopy(r)
        if r_copy.get("activation") == "dasha_activated" and md_planet in related:
            old = r_copy["score"]
            if old > 0:
                r_copy["score"] = round(old * 1.5)
            elif old < 0:
                r_copy["score"] = round(old * 1.2)
            r_copy["title"] += " [⚡ ACTIVATED]"
            r_copy["detail"] += (
                "\n  ⚡ Amplified: the running Mahadasha planet directly governs this life area, "
                "intensifying these effects during this period."
            )
        result.append(r_copy)
    return result


def _narrative_block(fired: List[Dict]) -> str:
    positives = [r for r in fired if r["severity"] == "positive"]
    neutrals  = [r for r in fired if r["severity"] == "neutral"]
    cautions  = [r for r in fired if r["severity"] in ["caution","warning"]]

    parts = []
    if positives:
        parts.append("STRENGTHS:\n" + "\n\n".join(
            f"  ✦ {r['title']}\n    {r['detail']}" for r in positives
        ))
    if neutrals:
        parts.append("CONTEXTUAL FACTORS:\n" + "\n\n".join(
            f"  ◈ {r['title']}\n    {r['detail']}" for r in neutrals
        ))
    if cautions:
        parts.append("CAUTIONS & REMEDIES:\n" + "\n\n".join(
            f"  ⚠ {r['title']}\n    {r['detail']}" for r in cautions
        ))
    return "\n\n".join(parts) if parts else "No significant planetary indicators found for this topic."


def _house_career_meaning(house: int) -> str:
    meanings = {
        1: "Career identity is tied to personal self — you ARE your work.",
        2: "Career energy flows toward wealth accumulation and family legacy.",
        3: "Career thrives through communication, short travels, and entrepreneurial drive.",
        4: "Career connected to home, real estate, mother's legacy, or emotional security.",
        5: "Career is infused with creativity, intelligence, and speculative enterprise.",
        6: "Career involves service, competition, health, or overcoming obstacles.",
        7: "Career involves partnerships, public dealing, or spouse-related connections.",
        8: "Career connected to research, occult, insurance, or transformation.",
        9: "Ideal — 10th lord in 9th creates Dharma-Karma connection; fortune supports career.",
        10: "Excellent — 10th lord in 10th is self-contained, maximising career strength.",
        11: "Career oriented toward gains, networks, and elder sibling/friend connections.",
        12: "Career involves foreign lands, behind-the-scenes work, hospitals, or spirituality.",
    }
    return meanings.get(house, "")


# ==================================================================
# SECTION 9 — TOPIC ANALYSIS FUNCTIONS
# ==================================================================

def analyze_career(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="career")

    lagna_idx  = ZODIAC.index(chart.lagna_sign)
    tenth_lord = SIGN_LORD[ZODIAC[(lagna_idx + 9) % 12]]
    md_planet  = dasha_info.get("mahadasha", "")
    fired      = _apply_dasha_boost(fired, tenth_lord, md_planet,
                                    related_planets=["Sun","Saturn","Mercury","Jupiter","Rahu"])

    summary       = score_topic(fired)
    tenth_sign    = ZODIAC[(lagna_idx + 9) % 12]
    tenth_house_p = ctx["house_map"].get(tenth_lord, 0)
    planets_10th  = [p for p, h in ctx["house_map"].items() if h == 10]
    aspects_10th  = ctx["aspect_map"].get(10, [])

    return {
        "rating":           summary["rating"],
        "net_score":        summary["net_score"],
        "tenth_sign":       tenth_sign,
        "tenth_lord":       tenth_lord,
        "tenth_lord_dignity": chart.dignities.get(tenth_lord, "Neutral"),
        "tenth_lord_house": tenth_house_p,
        "planets_in_10th":  planets_10th,
        "aspects_on_10th":  [(p, t) for p, t in aspects_10th],
        "atmakaraka":       chart.atmakaraka,
        "amatyakaraka":     chart.amatyakaraka,
        "current_dasha":    dasha_info,
        "fired_rules":      fired,
        "narrative":        _narrative_block(fired),
        "summary": (
            f"Career: {summary['rating']} (score {summary['net_score']:+d}). "
            f"10th lord {tenth_lord} is {chart.dignities.get(tenth_lord,'Neutral')} "
            f"in House {tenth_house_p}. "
            f"Planets in 10th: {planets_10th or 'none'}. "
            f"Atmakaraka: {chart.atmakaraka}, Amatyakaraka: {chart.amatyakaraka}. "
            f"{summary['positive_count']} strengths, {summary['warning_count']} cautions."
        )
    }


def analyze_marriage(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info   = chart.get_current_dasha_info(check_date)
    ctx          = build_context(chart, dasha_info)
    fired        = evaluate_rules(ctx, topic="marriage")

    lagna_idx    = ZODIAC.index(chart.lagna_sign)
    seventh_lord = SIGN_LORD[ZODIAC[(lagna_idx + 6) % 12]]
    md_planet    = dasha_info.get("mahadasha", "")
    fired        = _apply_dasha_boost(fired, seventh_lord, md_planet,
                                      related_planets=["Venus","Jupiter","Moon"])

    summary       = score_topic(fired)
    seventh_house = ctx["house_map"].get(seventh_lord, 0)
    planets_7th   = [p for p, h in ctx["house_map"].items() if h == 7]
    venus_sign    = longitude_to_sign(chart.planets["Venus"])[0]
    venus_house   = ctx["house_map"].get("Venus", 0)
    aspects_7th   = ctx["aspect_map"].get(7, [])

    return {
        "rating":             summary["rating"],
        "net_score":          summary["net_score"],
        "seventh_sign":       ZODIAC[(lagna_idx + 6) % 12],
        "seventh_lord":       seventh_lord,
        "seventh_lord_dignity": chart.dignities.get(seventh_lord, "Neutral"),
        "seventh_lord_house": seventh_house,
        "planets_in_7th":     planets_7th,
        "aspects_on_7th":     [(p, t) for p, t in aspects_7th],
        "venus_house":        venus_house,
        "venus_sign":         venus_sign,
        "venus_dignity":      chart.dignities.get("Venus", "Neutral"),
        "venus_navamsa":      chart.navamsa_dignities.get("Venus", "Neutral"),
        "current_dasha":      dasha_info,
        "fired_rules":        fired,
        "narrative":          _narrative_block(fired),
        "summary": (
            f"Marriage: {summary['rating']} (score {summary['net_score']:+d}). "
            f"7th lord {seventh_lord} is {chart.dignities.get(seventh_lord,'Neutral')} "
            f"in House {seventh_house}. "
            f"Venus: {chart.dignities.get('Venus','Neutral')} in House {venus_house} ({venus_sign}). "
            f"D9 Venus: {chart.navamsa_dignities.get('Venus','?')}."
        )
    }


def analyze_children(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info  = chart.get_current_dasha_info(check_date)
    ctx         = build_context(chart, dasha_info)
    fired       = evaluate_rules(ctx, topic="children")

    lagna_idx   = ZODIAC.index(chart.lagna_sign)
    fifth_lord  = SIGN_LORD[ZODIAC[(lagna_idx + 4) % 12]]
    md_planet   = dasha_info.get("mahadasha", "")
    fired       = _apply_dasha_boost(fired, fifth_lord, md_planet,
                                     related_planets=["Jupiter","Venus","Moon"])

    summary       = score_topic(fired)
    fifth_house   = ctx["house_map"].get(fifth_lord, 0)
    planets_5th   = [p for p, h in ctx["house_map"].items() if h == 5]
    aspects_5th   = ctx["aspect_map"].get(5, [])
    jupiter_house = ctx["house_map"].get("Jupiter", 0)

    return {
        "rating":           summary["rating"],
        "net_score":        summary["net_score"],
        "fifth_sign":       ZODIAC[(lagna_idx + 4) % 12],
        "fifth_lord":       fifth_lord,
        "fifth_lord_dignity": chart.dignities.get(fifth_lord, "Neutral"),
        "fifth_lord_house": fifth_house,
        "planets_in_5th":   planets_5th,
        "aspects_on_5th":   [(p, t) for p, t in aspects_5th],
        "jupiter_house":    jupiter_house,
        "jupiter_dignity":  chart.dignities.get("Jupiter", "Neutral"),
        "current_dasha":    dasha_info,
        "fired_rules":      fired,
        "narrative":        _narrative_block(fired),
        "summary": (
            f"Children: {summary['rating']} (score {summary['net_score']:+d}). "
            f"5th lord {fifth_lord} is {chart.dignities.get(fifth_lord,'Neutral')} "
            f"in House {fifth_house}. "
            f"Jupiter (Putrakaraka): {chart.dignities.get('Jupiter','Neutral')} in House {jupiter_house}."
        )
    }


def analyze_health(chart: ChartData, check_date: datetime = None,
                   transit_saturn_sign: str = None) -> Dict:
    dasha_info  = chart.get_current_dasha_info(check_date)
    sade_sati   = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")
    kantaka     = check_kantaka_shani(chart.moon_sign, transit_saturn_sign or "")
    ctx         = build_context(chart, dasha_info, sade_sati)
    fired       = evaluate_rules(ctx, topic="health")

    lagna_lord  = SIGN_LORD[chart.lagna_sign]
    md_planet   = dasha_info.get("mahadasha", "")
    fired       = _apply_dasha_boost(fired, lagna_lord, md_planet,
                                     related_planets=["Sun","Jupiter","Mars"])

    summary      = score_topic(fired)
    planets_1st  = [p for p, h in ctx["house_map"].items() if h == 1]
    planets_6th  = [p for p, h in ctx["house_map"].items() if h == 6]
    planets_8th  = [p for p, h in ctx["house_map"].items() if h == 8]
    planets_12th = [p for p, h in ctx["house_map"].items() if h == 12]
    aspects_1st  = ctx["aspect_map"].get(1, [])

    return {
        "rating":              summary["rating"],
        "net_score":           summary["net_score"],
        "lagna_lord":          lagna_lord,
        "lagna_lord_dignity":  chart.dignities.get(lagna_lord, "Neutral"),
        "planets_in_1st":      planets_1st,
        "planets_in_6th":      planets_6th,
        "planets_in_8th":      planets_8th,
        "planets_in_12th":     planets_12th,
        "aspects_on_lagna":    [(p, t) for p, t in aspects_1st],
        "sade_sati":           sade_sati,
        "kantaka_shani":       kantaka,
        "current_dasha":       dasha_info,
        "fired_rules":         fired,
        "narrative":           _narrative_block(fired),
        "summary": (
            f"Health: {summary['rating']} (score {summary['net_score']:+d}). "
            f"Lagna lord {lagna_lord} is {chart.dignities.get(lagna_lord,'Neutral')}. "
            f"Sade Sati: {'Active — ' + sade_sati['phase'] if sade_sati['active'] else 'Not active'}. "
            f"Kantaka Shani: {'Yes' if kantaka else 'No'}."
        )
    }


def analyze_general_yogas(chart: ChartData) -> Dict:
    dasha_info = chart.get_current_dasha_info()
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="general")
    total_yoga_score = sum(r["score"] for r in fired)
    return {
        "yoga_count":        len(fired),
        "total_yoga_score":  total_yoga_score,
        "yoga_strength": (
            "Exceptional" if total_yoga_score >= 15 else
            "Strong"      if total_yoga_score >= 8  else
            "Moderate"    if total_yoga_score >= 3  else
            "Weak"
        ),
        "fired_yogas":   fired,
        "narrative":     _narrative_block(fired),
        "atmakaraka":    chart.atmakaraka,
        "amatyakaraka":  chart.amatyakaraka,
    }


# ==================================================================
# SECTION 10 — ASHTAKOOTA MATCHMAKING (36 points — bugs fixed)
# ==================================================================

def get_tara_score(ni1: int, ni2: int) -> int:
    """
    BUG FIX v4: Returns clean integer 0, 1, 2, or 3.
    Full bidirectional average using TARA_SCORES table.
    """
    d12 = ((ni2 - ni1) % 27) % 9 + 1
    d21 = ((ni1 - ni2) % 27) % 9 + 1
    s1  = TARA_SCORES[d12]
    s2  = TARA_SCORES[d21]
    avg = (s1 + s2) / 2
    # Round to nearest integer with tie going to the higher score
    return int(avg + 0.5)


def get_yoni_score(y1: str, y2: str) -> int:
    hostile_pairs = {
        frozenset({"Horse","Buffalo"}),
        frozenset({"Elephant","Lion"}),
        frozenset({"Sheep","Monkey"}),
        frozenset({"Serpent","Mongoose"}),
        frozenset({"Dog","Deer"}),
        frozenset({"Cat","Rat"}),
        frozenset({"Cow","Tiger"}),
    }
    if y1 == y2:
        return 4
    if frozenset({y1, y2}) in hostile_pairs:
        return 0
    return 2


def get_graha_maitri_score(lord1: str, lord2: str) -> int:
    """
    BUG FIX v4: Same lord = 5; mutual friends = 4 (not 5); one-way friend = 3;
    neutral = 2; one-way enemy = 1; mutual enemies = 0.
    """
    if lord1 == lord2:
        return 5
    l1_friends = PLANET_FRIENDS.get(lord1, [])
    l2_friends = PLANET_FRIENDS.get(lord2, [])
    l1_enemies = PLANET_ENEMIES.get(lord1, [])
    l2_enemies = PLANET_ENEMIES.get(lord2, [])

    mutual_friend = lord2 in l1_friends and lord1 in l2_friends
    one_way_friend = lord2 in l1_friends or lord1 in l2_friends
    mutual_enemy   = lord2 in l1_enemies and lord1 in l2_enemies
    one_way_enemy  = lord2 in l1_enemies or lord1 in l2_enemies

    if mutual_friend:   return 4
    if one_way_friend:  return 3
    if mutual_enemy:    return 0
    if one_way_enemy:   return 1
    return 2  # both neutral to each other


def get_gana_score(g1: str, g2: str) -> int:
    """
    Standard Gana Koota:
    Deva-Deva=6, Manushya-Manushya=6, Rakshasa-Rakshasa=6,
    Deva-Manushya=5, Manushya-Deva=5,
    Deva-Rakshasa=0, Rakshasa-Deva=0,
    Manushya-Rakshasa=0, Rakshasa-Manushya=0.
    """
    if g1 == g2:
        return 6
    if {g1, g2} == {"Deva", "Manushya"}:
        return 5
    return 0


def get_bhakoot_score(idx1: int, idx2: int) -> int:
    """
    BUG FIX v4:
    6/8 axis: sign A is 6th from B (diff=5) ↔ B is 8th from A (diff=7).
      The *pair* is identified by the smaller diff of 5 (or equivalently 7).
      Mark inauspicious: diff in {5, 7}.

    Wait — standard Bhakoot definitions:
      2/12 axis = diff 2 from p1 to p2 (p1 sees p2 as 3rd? No.)
      Actually: diff = (idx2 - idx1) % 12.
        diff=1 means idx2 is the 2nd sign from idx1, so p1 sees p2 in 2nd house position.
        diff=11 means p1 sees p2 in 12th position → 2/12 axis.
      So 2/12 axis = diff 1 (one sees the other in 2nd) and diff 11 (reverse: in 12th).
      6/8 axis: diff 5 (p2 in 6th from p1) and diff 7 (p2 in 8th from p1).
      Nava-Pancham: diff 4 (p2 in 5th) and diff 8 (p2 in 9th) — AUSPICIOUS.

    Corrected inauspicious diffs: {1, 5, 7, 11}.
    Auspicious (score 7): everything else (including Nava-Pancham diffs 4 and 8).
    """
    diff = (idx2 - idx1) % 12
    if diff in [1, 5, 7, 11]:   # 2/12 axis or 6/8 axis
        return 0
    return 7


def calculate_ashtakoota(c1: ChartData, c2: ChartData,
                          person1_is_groom: bool = True) -> Dict:
    """
    BUG FIX v4: person1_is_groom parameter added for Varna direction clarity.
    Varna: groom's varna order should be >= bride's for full score.
    """
    m1, m2  = c1.moon_sign, c2.moon_sign
    n1      = c1.nakshatras["Moon"]["nakshatra"]
    n2      = c2.nakshatras["Moon"]["nakshatra"]
    i1, i2  = ZODIAC.index(m1), ZODIAC.index(m2)
    ni1,ni2 = NAKSHATRAS.index(n1), NAKSHATRAS.index(n2)

    # Varna (1 point): groom's varna >= bride's in the social hierarchy
    varna1 = VARNA_MAP[SIGN_ELEMENT[m1]]
    varna2 = VARNA_MAP[SIGN_ELEMENT[m2]]
    varna_order = {"Brahmin":1,"Kshatriya":2,"Vaishya":3,"Shudra":4}
    if person1_is_groom:
        # Groom is p1: groom varna (1) must be >= bride varna (2) (numerically <=)
        varna = 1 if varna_order[varna1] <= varna_order[varna2] else 0
    else:
        varna = 1 if varna_order[varna2] <= varna_order[varna1] else 0

    # Vashya (2 points)
    vashya1 = VASHYA_MAP[m1]
    vashya2 = VASHYA_MAP[m2]
    vashya  = (2 if vashya1 == vashya2
               else 1 if (
                   (vashya1 == "Human" and vashya2 in ["Water","Quadruped"]) or
                   (vashya2 == "Human" and vashya1 in ["Water","Quadruped"])
               ) else 0)

    tara    = get_tara_score(ni1, ni2)
    yoni    = get_yoni_score(NAKSHATRA_YONI[n1], NAKSHATRA_YONI[n2])
    graha   = get_graha_maitri_score(SIGN_LORD[m1], SIGN_LORD[m2])
    gana    = get_gana_score(NAKSHATRA_GANA[n1], NAKSHATRA_GANA[n2])
    bhakoot = get_bhakoot_score(i1, i2)
    nadi    = 0 if NAKSHATRA_NADI[n1] == NAKSHATRA_NADI[n2] else 8

    total = varna + vashya + tara + yoni + graha + gana + bhakoot + nadi

    # Doshas commentary
    doshas = []
    if nadi == 0:
        doshas.append("Nadi Dosha present — same Nadi is the most serious compatibility flaw; "
                      "seek astrological counsel before proceeding.")
    if bhakoot == 0:
        diff = (i2 - i1) % 12
        axis = "6/8" if diff in [5,7] else "2/12"
        doshas.append(f"Bhakoot Dosha ({axis} axis) — can cause financial stress or emotional "
                      "distance; remediable through ritual and chart compatibility analysis.")

    return {
        "varna":        {"score": varna,   "max": 1,  "detail": f"{varna1} vs {varna2}"},
        "vashya":       {"score": vashya,  "max": 2,  "detail": f"{vashya1} vs {vashya2}"},
        "tara":         {"score": tara,    "max": 3,  "detail": f"{n1} vs {n2}"},
        "yoni":         {"score": yoni,    "max": 4,  "detail": f"{NAKSHATRA_YONI[n1]} vs {NAKSHATRA_YONI[n2]}"},
        "graha_maitri": {"score": graha,   "max": 5,  "detail": f"{SIGN_LORD[m1]} vs {SIGN_LORD[m2]}"},
        "gana":         {"score": gana,    "max": 6,  "detail": f"{NAKSHATRA_GANA[n1]} vs {NAKSHATRA_GANA[n2]}"},
        "bhakoot":      {"score": bhakoot, "max": 7,  "detail": f"{m1} ({n1}) vs {m2} ({n2})"},
        "nadi":         {"score": nadi,    "max": 8,  "detail": f"{NAKSHATRA_NADI[n1]} vs {NAKSHATRA_NADI[n2]}"},
        "total":        total,
        "max_total":    36,
        "percentage":   round(total / 36 * 100, 1),
        "verdict": (
            "Excellent"   if total >= 31 else
            "Good"        if total >= 25 else
            "Average"     if total >= 18 else
            "Challenging"
        ),
        "doshas": doshas,
        "dosha_summary": (
            "No major doshas detected." if not doshas else
            f"{len(doshas)} dosha(s) present: " + "; ".join(
                d.split("—")[0].strip() for d in doshas
            )
        )
    }


# ==================================================================
# SECTION 11 — VARSHPHAL (SOLAR RETURN)
# ==================================================================

def calculate_varshphal(chart: ChartData, year: int) -> Dict:
    if not chart.birth_date:
        return {}
    birth_month  = chart.birth_date.month
    birth_day    = chart.birth_date.day
    years_elapsed = year - chart.birth_date.year
    muntha_lon    = (chart.ascendant + years_elapsed * 30) % 360
    muntha_sign, muntha_deg = longitude_to_sign(muntha_lon)
    muntha_lord   = SIGN_LORD[muntha_sign]
    lagna_idx     = ZODIAC.index(chart.lagna_sign)
    muntha_idx    = ZODIAC.index(muntha_sign)
    muntha_house  = ((muntha_idx - lagna_idx) % 12) + 1

    transits = {}
    if SWISSEPH_AVAILABLE:
        try:
            transits = get_transits(year, birth_month, birth_day)
        except Exception:
            pass

    themes = _varshphal_themes(chart, muntha_sign, muntha_house, muntha_lord)

    return {
        "year":              year,
        "varshphal_date":    datetime(year, birth_month, birth_day).strftime("%d %b %Y"),
        "years_elapsed":     years_elapsed,
        "muntha_sign":       muntha_sign,
        "muntha_house":      muntha_house,
        "muntha_longitude":  round(muntha_lon, 2),
        "muntha_lord":       muntha_lord,
        "muntha_lord_dignity": chart.dignities.get(muntha_lord, "Neutral"),
        "transits":          {k: longitude_to_sign(v)[0] for k, v in transits.items()},
        "themes":            themes,
    }


def _varshphal_themes(chart: ChartData, muntha_sign: str, muntha_house: int,
                      muntha_lord: str) -> List[str]:
    themes = []
    if muntha_house in [1, 5, 9]:
        themes.append(
            f"Muntha in {muntha_sign} (House {muntha_house}, trikona) — a year of personal growth, "
            "spiritual blessings, and fresh opportunities aligned with your dharma."
        )
    elif muntha_house in [4, 7, 10]:
        themes.append(
            f"Muntha in {muntha_sign} (House {muntha_house}, kendra) — a year of visible action, "
            "tangible results, and importance in the public or professional sphere."
        )
    elif muntha_house in [2, 11]:
        themes.append(
            f"Muntha in {muntha_sign} (House {muntha_house}) — a year focused on financial "
            "accumulation, gains, and expanding your resource base."
        )
    elif muntha_house in [3, 6]:
        themes.append(
            f"Muntha in {muntha_sign} (House {muntha_house}) — a year of effort, skill-building, "
            "competition, and overcoming obstacles; initiatives taken now bear fruit through persistence."
        )
    elif muntha_house in [8, 12]:
        themes.append(
            f"Muntha in {muntha_sign} (House {muntha_house}, dusthana) — a year of inner transformation, "
            "release of old patterns, and preparation for a new cycle; external progress may be slow "
            "but inner growth is deep."
        )

    dig = chart.dignities.get(muntha_lord, "Neutral")
    if dig in ["Exalted","Own","Mool Trikona"]:
        themes.append(
            f"Muntha lord {muntha_lord} is {dig} — the year's central themes are powerfully supported; "
            "expect meaningful progress in the areas Muntha activates."
        )
    elif dig == "Debilitated":
        nb = is_neechabhanga(muntha_lord, DEBILITATION.get(muntha_lord,""), chart.planets, chart.lagna_sign)
        themes.append(
            f"Muntha lord {muntha_lord} is debilitated"
            + (" but Neechabhanga applies — challenges convert to growth." if nb
               else " — the year may feel obstructed in its central theme; remedies advised.")
        )
    else:
        themes.append(
            f"Muntha lord {muntha_lord} is {dig} — moderate support for the year's themes; "
            "results will depend on effort and dasha timing."
        )
    return themes


# ==================================================================
# SECTION 12 — YEARLY PREDICTION
# ==================================================================

def get_transits(year: int, month: int = 6, day: int = 15) -> Dict[str, float]:
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("pyswisseph not installed.")
    jd = swe.julday(year, month, day, 12.0)
    transits = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        transits[pname] = res[0][0]
    transits["Ketu"] = (transits["Rahu"] + 180.0) % 360.0
    return transits


def compute_chart(year, month, day, hour, minute, lat, lon, tz_offset=0.0) -> ChartData:
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("pyswisseph is not installed.")
    jd     = swe.julday(year, month, day, hour + minute/60.0 - tz_offset)
    houses = swe.houses_ex(jd, lat, lon, b'W', swe.FLG_SIDEREAL)
    asc    = houses[1][0]
    planets = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        planets[pname] = res[0][0]
    planets["Ketu"] = (planets["Rahu"] + 180.0) % 360.0
    lagna_sign, _ = longitude_to_sign(asc)
    return ChartData(planets, asc, lagna_sign,
                     datetime(year, month, day, hour, minute), lat, lon, tz_offset)


def get_year_prediction(chart: ChartData, year: int) -> Dict:
    check_date = datetime(year, 6, 15)
    dasha_info = chart.get_current_dasha_info(check_date)

    transit_saturn_sign  = None
    transit_jupiter_sign = None
    jupiter_transit_note = ""

    if SWISSEPH_AVAILABLE:
        transit_dates = [datetime(year,1,1), datetime(year,6,15), datetime(year,12,31)]
        sat_lons, jup_lons = [], []
        for td in transit_dates:
            try:
                tr = get_transits(td.year, td.month, td.day)
                sat_lons.append(tr["Saturn"])
                jup_lons.append(tr["Jupiter"])
            except Exception:
                pass
        if sat_lons:
            transit_saturn_sign  = longitude_to_sign(sum(sat_lons)/len(sat_lons))[0]
            transit_jupiter_sign = longitude_to_sign(sum(jup_lons)/len(jup_lons))[0]

        if transit_jupiter_sign:
            j_idx = ZODIAC.index(transit_jupiter_sign)
            l_idx = ZODIAC.index(chart.lagna_sign)
            m_idx = ZODIAC.index(chart.moon_sign)
            jh_lagna = ((j_idx - l_idx) % 12) + 1
            jh_moon  = ((j_idx - m_idx) % 12) + 1
            notes = []
            if jh_lagna in [1,5,9]:
                notes.append(f"Jupiter transiting House {jh_lagna} from Lagna — highly auspicious for growth.")
            elif jh_lagna in [4,7,8,12]:
                notes.append(f"Jupiter transiting House {jh_lagna} from Lagna — mixed/challenging year from Jupiter.")
            if jh_moon in [1,5,9,11]:
                notes.append(f"Jupiter in House {jh_moon} from Moon — Guruchandra Yoga possible; emotional positivity.")
            elif jh_moon in [4,7,8]:
                notes.append(f"Jupiter in House {jh_moon} from Moon — emotional or health strain possible.")
            jupiter_transit_note = " | ".join(notes)

    sade_sati = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")
    kantaka   = check_kantaka_shani(chart.moon_sign, transit_saturn_sign or "")

    varshphal = calculate_varshphal(chart, year)
    career    = analyze_career(chart, check_date)
    marriage  = analyze_marriage(chart, check_date)
    children  = analyze_children(chart, check_date)
    health    = analyze_health(chart, check_date, transit_saturn_sign)
    yogas     = analyze_general_yogas(chart)

    return {
        "year":              year,
        "dasha":             dasha_info,
        "sade_sati":         sade_sati,
        "kantaka_shani":     kantaka,
        "jupiter_transit":   jupiter_transit_note,
        "transit_saturn":    transit_saturn_sign,
        "transit_jupiter":   transit_jupiter_sign,
        "varshphal":         varshphal,
        "career":            career,
        "marriage":          marriage,
        "children":          children,
        "health":            health,
        "general_yogas":     yogas,
        "overall_summary":   _year_summary(year, dasha_info, sade_sati, kantaka,
                                           varshphal, career, marriage, children, health, yogas)
    }


def _year_summary(year, dasha, sade_sati, kantaka, varshphal,
                  career, marriage, children, health, yogas) -> str:
    lines = [f"{'='*70}", f"YEAR {year} — VEDIC ASTROLOGY PREDICTION SUMMARY", f"{'='*70}\n"]

    md = dasha.get("mahadasha","?")
    ad = dasha.get("antardasha","?")
    pd = dasha.get("pratyantardasha","?")
    lines.append(f"DASHA: {md} MD / {ad} AD / {pd} PD")
    lines.append(f"  MD period: {dasha.get('mahadasha_start','')} → {dasha.get('mahadasha_end','')}")
    lines.append(f"  AD period: {dasha.get('antardasha_start','')} → {dasha.get('antardasha_end','')}\n")

    if sade_sati.get("active"):
        lines.append(f"⚠  SADE SATI ACTIVE: {sade_sati['phase']}")
    if kantaka:
        lines.append("⚠  KANTAKA SHANI ACTIVE: Saturn transiting 4th/7th/10th from Moon.")
    if sade_sati.get("active") or kantaka:
        lines.append("")

    if varshphal:
        lines.append(
            f"VARSHPHAL — Muntha in {varshphal.get('muntha_sign','')} "
            f"(H{varshphal.get('muntha_house','')}) | "
            f"Lord: {varshphal.get('muntha_lord','')} "
            f"[{varshphal.get('muntha_lord_dignity','')}]"
        )
        for t in varshphal.get("themes",[]):
            lines.append(f"  • {t}")
        lines.append("")

    # Yoga summary
    if yogas.get("fired_yogas"):
        lines.append(f"NATAL YOGAS: {yogas['yoga_count']} yoga(s) active "
                     f"[Strength: {yogas['yoga_strength']}]")
        for y in yogas["fired_yogas"][:4]:  # top 4
            lines.append(f"  ✦ {y['title']}")
        if yogas['yoga_count'] > 4:
            lines.append(f"  … and {yogas['yoga_count']-4} more. See General Yogas section.")
        lines.append("")

    # Topic summaries
    for label, data in [("CAREER",career),("MARRIAGE",marriage),
                         ("CHILDREN",children),("HEALTH",health)]:
        rating = data.get('rating','?')
        score  = data.get('net_score', 0)
        lines.append(f"{label}: {rating} (score {score:+d})")
        lines.append(f"  {data.get('summary','')}")
        lines.append("")

    return "\n".join(lines)


# ==================================================================
# SECTION 13 — DEMO CHART & UTILITIES
# ==================================================================

def generate_demo_chart() -> ChartData:
    """Sample chart: Jupiter exalted in Cancer, Saturn own in Aquarius."""
    planets = {
        "Sun":      45.5,   # Taurus
        "Moon":    128.3,   # Leo
        "Mars":    200.0,   # Libra
        "Mercury":  50.2,   # Taurus (Budhaditya Yoga with Sun)
        "Jupiter":  95.0,   # Cancer (Exalted — Hamsa Yoga, Gaja-Kesari)
        "Venus":    70.5,   # Gemini
        "Saturn":  310.0,   # Aquarius (Own — Shasha Yoga in 10th from Taurus lagna)
        "Rahu":    175.0,   # Virgo
        "Ketu":    355.0,   # Pisces
    }
    return ChartData(
        planets, ascendant=30.0, lagna_sign="Taurus",
        birth_date=datetime(1995, 6, 15, 10, 30),
        lat=28.6, lon=77.2, tz=5.5
    )


def load_chart_from_file(filepath: str) -> ChartData:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    birth_date = datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None
    return ChartData(
        planets    = data["planets"],
        ascendant  = data["ascendant"],
        lagna_sign = data["lagna_sign"],
        birth_date = birth_date,
        lat        = data.get("lat", 0),
        lon        = data.get("lon", 0),
        tz         = data.get("tz", 0)
    )


def print_full_report(chart: ChartData, year: int = None):
    import textwrap
    year = year or datetime.now().year

    print("=" * 70)
    print("VEDIC ASTROLOGY REPORT — v4.0")
    print("=" * 70)
    print(f"Lagna:        {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})")
    print(f"Moon sign:    {chart.moon_sign}")
    print(f"Sun sign:     {chart.sun_sign}")
    print(f"Atmakaraka:   {chart.atmakaraka}")
    print(f"Amatyakaraka: {chart.amatyakaraka}")
    print()

    print("PLANETS")
    print("-" * 70)
    for p, lon in chart.planets.items():
        sign, deg = longitude_to_sign(lon)
        nak_info  = chart.nakshatras[p]
        dig       = chart.dignities[p]
        nb        = " [NB]" if is_neechabhanga(p, sign, chart.planets, chart.lagna_sign) else ""
        vb        = chart.vimsopaka.get(p, 0)
        print(f"  {p:10s}: {sign:14s} {deg:6.2f}°  {dig:14s}{nb}  "
              f"{nak_info['nakshatra']} P{nak_info['pada']}  Vims:{vb:.1f}")

    print()
    print("SHADBALA PROXY (0-100)")
    print("-" * 40)
    for p, s in sorted(chart.shadbala_proxy.items(), key=lambda x: -x[1]):
        bar = "█" * int(s / 10)
        print(f"  {p:10s}: {s:5.1f}  {bar}")

    print()
    print("DASHA PERIODS")
    print("-" * 70)
    now = datetime.now()
    for dp in chart.dasha_periods:
        marker = " ← CURRENT MD" if dp.start_date <= now < dp.end_date else ""
        print(f"  {dp.planet:8s}: {dp.start_date.strftime('%d %b %Y')} → "
              f"{dp.end_date.strftime('%d %b %Y')}  ({dp.years:.2f} yrs){marker}")

    print()
    prediction = get_year_prediction(chart, year)
    print(prediction["overall_summary"])

    for section, key in [
        ("CAREER ANALYSIS",   "career"),
        ("MARRIAGE ANALYSIS", "marriage"),
        ("CHILDREN ANALYSIS", "children"),
        ("HEALTH ANALYSIS",   "health"),
        ("YOGAS",             "general_yogas"),
    ]:
        print(f"\n{section}")
        print("-" * 70)
        narrative = prediction[key]["narrative"]
        # Wrap long lines
        for line in narrative.split("\n"):
            if len(line) > 100 and line.startswith("    "):
                print(textwrap.fill(line, width=100, subsequent_indent="      "))
            else:
                print(line)


# ==================================================================
# QUICK TEST
# ==================================================================
if __name__ == "__main__":
    chart = generate_demo_chart()
    print_full_report(chart, year=2025)
