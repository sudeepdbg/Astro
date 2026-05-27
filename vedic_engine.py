"""
Vedic Astrology Calculation Engine v6.0
=========================================
MAJOR ENHANCEMENTS:
- Every prediction rule now exposes its full calculation logic in results
- Robust dignity pipeline with combined/temporary relationships
- Elaborate Varshphal: Muntha, Varsha Lagna, Tri-Pataki Chakra, year themes
- Yearly prediction: deep dasha-transit-varshphal synthesis
- Ram Shalaka: classical 7×7 grid oracle with authentic interpretation
- New rules: Parivartana Yoga, Vargottama, Graha Yuddha, Shadbala flags
- Neechabhanga: all 6 classical conditions checked individually
- Combustion: planet-specific degrees (classical values)
- Retrograde: directional meaning per planet
"""

import copy, math, json, random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

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
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
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

PLANET_IDS   = [0,1,2,3,4,5,6,10]
PLANET_NAMES = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu"]

HOUSE_MEANINGS = {
    1:"Self / Body / Vitality / Personality",
    2:"Wealth / Family / Speech / Savings",
    3:"Courage / Siblings / Communication / Short Travel",
    4:"Mother / Home / Vehicles / Education / Inner Peace",
    5:"Intelligence / Children / Purva Punya / Speculation",
    6:"Disease / Enemies / Service / Debts / Competition",
    7:"Marriage / Partnership / Business / Public Dealings",
    8:"Longevity / Occult / Transformation / Hidden Assets",
    9:"Fortune / Dharma / Father / Higher Learning / Long Travel",
    10:"Career / Status / Action / Authority / Public Image",
    11:"Gains / Friends / Elder Siblings / Ambitions / Networks",
    12:"Loss / Moksha / Foreign / Isolation / Expenses / Hospital"
}

EXALTATION   = {"Sun":"Aries","Moon":"Taurus","Mars":"Capricorn",
                "Mercury":"Virgo","Jupiter":"Cancer","Venus":"Pisces","Saturn":"Libra"}
DEBILITATION = {"Sun":"Libra","Moon":"Scorpio","Mars":"Cancer",
                "Mercury":"Pisces","Jupiter":"Capricorn","Venus":"Virgo","Saturn":"Aries"}
MOOLATRIKONA = {"Sun":"Leo","Moon":"Taurus","Mars":"Aries",
                "Mercury":"Virgo","Jupiter":"Sagittarius","Venus":"Libra","Saturn":"Aquarius"}

# Degrees where exaltation is deepest (classical)
EXALTATION_DEGREE = {"Sun":10,"Moon":3,"Mars":28,"Mercury":15,
                     "Jupiter":5,"Venus":27,"Saturn":20}
# Moolatrikona degree ranges (sign, start, end)
MOOLATRIKONA_RANGE = {
    "Sun":(0,20),"Moon":(4,30),"Mars":(0,12),"Mercury":(16,20),
    "Jupiter":(0,10),"Venus":(0,15),"Saturn":(0,20)
}

# Natural friendship tables (permanent)
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
PLANET_NEUTRALS = {
    "Sun":    ["Mercury"],
    "Moon":   ["Mars","Jupiter","Venus","Saturn"],
    "Mars":   ["Venus","Saturn"],
    "Mercury":["Mars","Jupiter","Saturn"],
    "Jupiter":["Saturn"],
    "Venus":  ["Mars","Jupiter","Moon"],
    "Saturn": ["Jupiter"],
    "Rahu":   ["Jupiter"],
    "Ketu":   ["Saturn","Venus","Moon"],
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

# Combustion limits in degrees from Sun (classical Parashari)
COMBUSTION_LIMITS = {
    "Moon":    12.0,
    "Mars":    17.0,
    "Mercury": {"direct": 14.0, "retrograde": 12.0},
    "Jupiter": 11.0,
    "Venus":   {"direct": 10.0, "retrograde": 8.0},
    "Saturn":  15.0,
}

TARA_SCORES = {1:3, 2:3, 3:0, 4:3, 5:0, 6:3, 7:0, 8:3, 9:3}

NAKSHATRA_SIZE = 13 + 20/60   # 13.3333°
PADA_SIZE      = 3  + 20/60   #  3.3333°

# Special aspects: houses FROM planet's position (in addition to universal 7th)
SPECIAL_ASPECTS = {
    "Mars":    [4, 8],    # 4th and 8th from Mars
    "Jupiter": [5, 9],    # 5th and 9th from Jupiter
    "Saturn":  [3, 10],   # 3rd and 10th from Saturn
    "Rahu":    [5, 9],    # Same as Jupiter
    "Ketu":    [5, 9],    # Same as Jupiter
}

DIGNITY_STRENGTH = {
    "Exalted":        100,
    "Own":             85,
    "Mool Trikona":    78,
    "Great Friend":    70,
    "Friendly":        55,
    "Neutral":         45,
    "Inimical":        25,
    "Debilitated":     10,
}

HOUSE_STRENGTH = {
    1:100, 10:95, 4:85, 7:85,
    5:80,  9:80,
    2:60,  11:65,
    3:50,  6:45,
    8:35,  12:30,
}

# Functional benefics/malefics by Lagna (simplified Parashari)
FUNCTIONAL_BENEFICS = {
    "Aries":       ["Mars","Jupiter","Sun"],
    "Taurus":      ["Saturn","Venus","Mercury"],
    "Gemini":      ["Venus","Saturn"],
    "Cancer":      ["Moon","Mars","Jupiter"],
    "Leo":         ["Sun","Mars","Jupiter"],
    "Virgo":       ["Mercury","Venus"],
    "Libra":       ["Saturn","Venus","Mercury"],
    "Scorpio":     ["Jupiter","Moon","Sun"],
    "Sagittarius": ["Jupiter","Sun","Mars"],
    "Capricorn":   ["Saturn","Venus","Mercury"],
    "Aquarius":    ["Saturn","Venus","Mercury"],
    "Pisces":      ["Jupiter","Moon","Mars"],
}

# Ram Shalaka grid (classical Hanuman Chalisa based 7x7)
RAM_SHALAKA_GRID = [
    ["श्री","राम","जय","राम","जय","जय","राम"],
    ["जय","हनु","मान","ज्ञान","गुण","सा","गर"],
    ["जय","कपी","श","तिहुँ","लोक","उजा","गर"],
    ["राम","दूत","अतु","लित","बल","धा","मा"],
    ["अंज","नि","पु","त्र","प","वन","सुत"],
    ["महा","बीर","वि","क्र","म","बज","रंगी"],
    ["कु","म","ति","नि","वा","र","हनु"]
]

RAM_SHALAKA_MEANINGS = {
    # Encoded as sum of (row+col) patterns mapped to classical outcomes
    # Each cell has a letter code; the oracle reads the diagonal path
    "auspicious_high":   "श्रीराम की कृपा है। कार्य सिद्ध होगा, मनोकामना पूर्ण होगी। विजय निश्चित है।\nSri Rama's full grace is upon you. Your endeavour will succeed, your heart's desire will be fulfilled. Victory is certain.",
    "auspicious_medium": "प्रयास सफल होगा। धैर्य रखें, सहायता मिलेगी। राम नाम का जाप करें।\nYour effort will bear fruit. Be patient — support will come. Chant the name of Rama for protection.",
    "auspicious_low":    "कार्य होगा परन्तु विलम्ब सम्भव है। विश्वास रखें, हनुमान जी रक्षा करेंगे।\nSuccess will come but may be delayed. Keep faith — Hanuman will protect and guide.",
    "neutral":           "स्थिति मध्यम है। पुरुषार्थ और भक्ति दोनों चाहिए। हनुमान चालीसा का पाठ करें।\nThe situation is balanced. Both effort and devotion are needed. Recite Hanuman Chalisa for clarity.",
    "inauspicious_low":  "अभी प्रतीक्षा करें। कार्य में बाधा है परन्तु हनुमान जी की भक्ति से बाधा दूर होगी।\nWait for a better moment. There is an obstacle, but devotion to Hanuman will remove it in time.",
    "inauspicious_high": "कार्य अभी उचित नहीं। विचार बदलें, परामर्श लें। राम नाम के 108 जाप करके पुनः प्रयास करें।\nThis undertaking is not favoured now. Reconsider, seek counsel. Chant Ram Naam 108 times before trying again.",
}

# Hanuman Chalisa verses mapped to shalaka outcomes (classical 49-square system)
SHALAKA_VERSE_MAP = {
    range(0,7):   ("श्रीगुरु चरन सरोज रज", "By the dust of the Guru's lotus feet, all is illuminated. Proceed with devotion."),
    range(7,14):  ("बुद्धिहीन तनु जानिके", "Even the seemingly weak succeed through Rama's grace. Strength comes from within."),
    range(14,21): ("जय हनुमान ज्ञान गुण सागर", "Hanuman — ocean of wisdom — blesses this path. Knowledge and virtue will guide you."),
    range(21,28): ("राम दूत अतुलित बल धामा", "The emissary of Rama carries boundless power. Invoke this force in your endeavour."),
    range(28,35): ("सूक्ष्म रूप धरि सियहिं दिखावा", "Through subtle means and divine timing, the path becomes clear. Trust the unseen process."),
    range(35,42): ("भीम रूप धरि असुर संहारे", "Obstacles will be vanquished by courageous action. Be bold and righteous."),
    range(42,49): ("राम रसायन तुम्हरे पासा", "You hold the elixir of Rama. Success, health, and liberation are within reach."),
}


# ==================================================================
# SECTION 2 — CORE MATH (ENHANCED)
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
    part        = int(deg_in_sign // (10/3))
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
    return ZODIAC[(sign_idx + part * 4) % 12]


def get_saptamsa(longitude: float) -> str:
    sign_idx    = int(longitude // 30)
    deg_in_sign = longitude % 30
    part        = int(deg_in_sign / (30/7))
    return ZODIAC[(sign_idx * 7 + part) % 12]


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


def is_vargottama(longitude: float) -> bool:
    """Planet is in same sign in D1 and D9."""
    d1_sign = longitude_to_sign(longitude)[0]
    d9_sign = get_navamsa(longitude)
    return d1_sign == d9_sign


def get_temporary_relationship(p1: str, p2: str, planets: Dict[str, float]) -> str:
    """
    Temporary (tatkalika) relationship: planets in 2nd/3rd/4th/10th/11th/12th
    from each other are temporary friends; rest are temporary enemies.
    Combined with permanent → 5-tier combined relationship.
    """
    if p1 not in planets or p2 not in planets:
        return "Neutral"
    s1 = int(planets[p1] // 30) % 12
    s2 = int(planets[p2] // 30) % 12
    d  = (s2 - s1) % 12
    return "Temporary Friend" if d in [1,2,3,9,10,11] else "Temporary Enemy"


def get_combined_relationship(p1: str, p2: str, planets: Dict[str, float]) -> str:
    """
    5-tier combined (sthira + tatkalika):
    Great Friend = Perm.Friend + Temp.Friend
    Friendly     = Perm.Friend + Temp.Enemy OR Perm.Neutral + Temp.Friend
    Neutral      = Perm.Enemy + Temp.Friend OR Perm.Neutral + Temp.Enemy
    Inimical     = Perm.Enemy + Temp.Enemy
    """
    temp = get_temporary_relationship(p1, p2, planets)
    is_temp_friend = (temp == "Temporary Friend")

    if p2 in PLANET_FRIENDS.get(p1, []):
        return "Great Friend" if is_temp_friend else "Friendly"
    elif p2 in PLANET_NEUTRALS.get(p1, []):
        return "Friendly" if is_temp_friend else "Inimical"
    elif p2 in PLANET_ENEMIES.get(p1, []):
        return "Friendly" if is_temp_friend else "Inimical"
    return "Friendly" if is_temp_friend else "Neutral"


def get_planet_dignity(planet: str, sign: str, planets: Dict[str, float] = None) -> str:
    """Full dignity calculation with optional combined-relationship upgrade."""
    if not planet or not sign:
        return "Neutral"

    # 1. Exaltation (deepest dignity)
    if EXALTATION.get(planet) == sign:
        return "Exalted"

    # 2. Debilitation (lowest dignity)
    if DEBILITATION.get(planet) == sign:
        return "Debilitated"

    # 3. Own sign
    if SIGN_LORD.get(sign) == planet:
        return "Own"

    # 4. Moolatrikona (subset of own sign but higher—checked after own because
    #    Sun's Leo is both Own and MT; MT range is degrees 0-20 of Leo for Sun)
    if MOOLATRIKONA.get(planet) == sign:
        deg_in_sign = (planets.get(planet, 0) % 30) if planets else 0
        mt_range = MOOLATRIKONA_RANGE.get(planet, (0, 30))
        if mt_range[0] <= deg_in_sign < mt_range[1]:
            return "Mool Trikona"
        else:
            return "Own"  # Outside MT range → still own sign

    lord = SIGN_LORD.get(sign, "")
    if not lord:
        return "Neutral"

    # 5. Use combined relationship if planet data available, else permanent
    if planets:
        rel = get_combined_relationship(planet, lord, planets)
        if rel == "Great Friend":
            return "Great Friend"
        elif rel in ["Friendly"]:
            return "Friendly"
        elif rel == "Inimical":
            return "Inimical"
        return "Neutral"
    else:
        if lord in PLANET_FRIENDS.get(planet, []):
            return "Friendly"
        if lord in PLANET_ENEMIES.get(planet, []):
            return "Inimical"
        return "Neutral"


def is_combust(planet: str, sun_lon: float, planet_lon: float,
               retrograde: bool = False) -> Tuple[bool, float]:
    """Returns (is_combust, orb_degrees). Uses classical planet-specific limits."""
    if planet in ("Sun", "Rahu", "Ketu"):
        return False, 999.0
    diff = abs((planet_lon - sun_lon) % 360)
    if diff > 180:
        diff = 360 - diff
    limit_entry = COMBUSTION_LIMITS.get(planet, 15.0)
    if isinstance(limit_entry, dict):
        limit = limit_entry.get("retrograde" if retrograde else "direct", 14.0)
    else:
        limit = limit_entry
    return diff < limit, round(diff, 2)


def get_directional_strength(planet: str, house: int) -> float:
    """
    Dig-bala: each planet has one kendra of maximum directional strength.
    Source: Parashari Hora.
    """
    dig_bala_houses = {
        "Jupiter":1, "Mercury":1,   # East  (1st)
        "Sun":10,    "Mars":10,      # South (10th)
        "Moon":4,    "Venus":4,      # North (4th)
        "Saturn":7                   # West  (7th)
    }
    if planet not in dig_bala_houses:
        return 1.0
    peak = dig_bala_houses[planet]
    # Full strength at peak house, decreasing by 90° intervals
    dist = min(abs(house - peak), 12 - abs(house - peak))
    factor = 1.2 - (dist / 12) * 0.4  # 1.2 at peak, 0.8 at opposite
    return round(max(factor, 0.8), 3)


def check_graha_yuddha(p1: str, p2: str, planets: Dict[str, float]) -> bool:
    """Planetary war: two planets within 1° in same sign."""
    if p1 not in planets or p2 not in planets:
        return False
    if "Rahu" in (p1, p2) or "Ketu" in (p1, p2) or "Sun" in (p1, p2) or "Moon" in (p1, p2):
        return False  # War only between Mars, Mercury, Jupiter, Venus, Saturn
    s1 = longitude_to_sign(planets[p1])[0]
    s2 = longitude_to_sign(planets[p2])[0]
    if s1 != s2:
        return False
    diff = abs(planets[p1] - planets[p2]) % 360
    if diff > 180:
        diff = 360 - diff
    return diff < 1.0


def get_yuddha_winner(p1: str, p2: str, planets: Dict[str, float]) -> str:
    """In Graha Yuddha, the planet with lower degree wins (north vs south latitude used
    classically; we approximate by longitude — lower degree = north in most cases)."""
    if planets.get(p1, 0) % 30 < planets.get(p2, 0) % 30:
        return p1
    return p2


def neechabhanga_conditions(planet: str, sign: str, planets: Dict[str, float],
                             lagna_sign: str) -> List[str]:
    """
    Returns a list of classical Neechabhanga conditions that are satisfied.
    All 6 classical conditions from Parashari Hora / Phaladeepika.
    """
    conditions_met = []
    if planet not in DEBILITATION:
        return conditions_met
    if DEBILITATION[planet] != sign:
        return conditions_met

    def in_kendra_from(p_sign: str, ref_sign: str) -> bool:
        d = (ZODIAC.index(p_sign) - ZODIAC.index(ref_sign)) % 12
        return d in [0, 3, 6, 9]

    moon_sign = longitude_to_sign(planets.get("Moon", 0))[0]
    deb_sign  = DEBILITATION[planet]
    exalt_sign= EXALTATION.get(planet, "")
    exalt_lord= SIGN_LORD.get(exalt_sign, "") if exalt_sign else ""
    deb_lord  = SIGN_LORD.get(deb_sign, "")

    # Condition 1: Lord of debilitation sign in kendra from Lagna
    if deb_lord and deb_lord in planets:
        deb_lord_sign = longitude_to_sign(planets[deb_lord])[0]
        if in_kendra_from(deb_lord_sign, lagna_sign):
            conditions_met.append(
                f"Cond.1: Lord of debilitation sign ({deb_lord}) in kendra "
                f"({deb_lord_sign}) from Lagna ({lagna_sign})"
            )

    # Condition 2: Lord of debilitation sign in kendra from Moon
    if deb_lord and deb_lord in planets:
        deb_lord_sign = longitude_to_sign(planets[deb_lord])[0]
        if in_kendra_from(deb_lord_sign, moon_sign):
            conditions_met.append(
                f"Cond.2: Lord of debilitation sign ({deb_lord}) in kendra "
                f"({deb_lord_sign}) from Moon ({moon_sign})"
            )

    # Condition 3: Lord of exaltation sign in kendra from Lagna
    if exalt_lord and exalt_lord in planets:
        ex_lord_sign = longitude_to_sign(planets[exalt_lord])[0]
        if in_kendra_from(ex_lord_sign, lagna_sign):
            conditions_met.append(
                f"Cond.3: Exaltation sign lord ({exalt_lord}) in kendra "
                f"({ex_lord_sign}) from Lagna ({lagna_sign})"
            )

    # Condition 4: Lord of exaltation sign in kendra from Moon
    if exalt_lord and exalt_lord in planets:
        ex_lord_sign = longitude_to_sign(planets[exalt_lord])[0]
        if in_kendra_from(ex_lord_sign, moon_sign):
            conditions_met.append(
                f"Cond.4: Exaltation sign lord ({exalt_lord}) in kendra "
                f"({ex_lord_sign}) from Moon ({moon_sign})"
            )

    # Condition 5: The planet itself in kendra from Lagna
    if planet in planets:
        p_sign = longitude_to_sign(planets[planet])[0]
        if in_kendra_from(p_sign, lagna_sign):
            conditions_met.append(
                f"Cond.5: Debilitated planet ({planet}) itself in kendra "
                f"({p_sign}) from Lagna ({lagna_sign})"
            )

    # Condition 6: The exaltation lord aspects the debilitated planet
    if exalt_lord and exalt_lord in planets and planet in planets:
        diff = abs((planets[exalt_lord] - planets[planet]) % 360)
        if diff > 180:
            diff = 360 - diff
        if diff < 10:
            conditions_met.append(
                f"Cond.6: Exaltation lord ({exalt_lord}) within 10° of debilitated planet "
                f"({planet}) — applying aspect"
            )

    return conditions_met


def is_neechabhanga(planet: str, sign: str, planets: Dict[str, float],
                    lagna_sign: str) -> bool:
    return len(neechabhanga_conditions(planet, sign, planets, lagna_sign)) > 0


def get_aspects_on_house(house: int, house_map: Dict[str, int]) -> List[Tuple[str, str]]:
    """
    Returns list of (planet, aspect_type) for all planets aspecting the given house.
    Universal 7th aspect + special aspects (corrected offsets).
    """
    aspects = []
    for planet, p_house in house_map.items():
        # Universal 7th aspect
        if ((p_house - 1 + 6) % 12) + 1 == house:
            aspects.append((planet, "7th (universal)"))
        # Special aspects (forward from planet's house)
        if planet in SPECIAL_ASPECTS:
            for offset in SPECIAL_ASPECTS[planet]:
                asp_house = ((p_house - 1 + offset - 1) % 12) + 1
                if asp_house == house and (planet, "7th (universal)") not in [a for a in aspects if a[0] == planet]:
                    aspects.append((planet, f"{offset}th (special)"))
    return aspects


def get_atmakaraka(planets: Dict[str, float]) -> str:
    """Highest degree in sign (excluding Ketu). Rahu uses 30 - degree."""
    relevant = {p: v for p, v in planets.items() if p != "Ketu"}
    deg_map = {}
    for p, lon in relevant.items():
        deg = lon % 30
        if p == "Rahu":
            deg = 30 - deg
        deg_map[p] = deg
    return max(deg_map, key=deg_map.get)


def get_amatyakaraka(planets: Dict[str, float], atmakaraka: str) -> str:
    relevant = {p: v for p, v in planets.items() if p not in ("Ketu", atmakaraka)}
    deg_map = {}
    for p, lon in relevant.items():
        deg = lon % 30
        if p == "Rahu":
            deg = 30 - deg
        deg_map[p] = deg
    return max(deg_map, key=deg_map.get) if deg_map else ""


def vimsopaka_bala(planet: str, d1_sign: str, d9_sign: str, d10_sign: str,
                   planets: Dict[str, float] = None) -> float:
    """
    Simplified Vimsopaka using D1 (weight 6), D9 (weight 5), D10 (weight 4).
    Full Vimsopaka uses 16 vargas; this is the core 3-varga approximation.
    """
    weights = [(d1_sign, 6), (d9_sign, 5), (d10_sign, 4)]
    total_weight = 15
    score = 0.0
    for sign, w in weights:
        dig = get_planet_dignity(planet, sign, planets)
        factor = {
            "Exalted":1.0, "Own":0.9, "Mool Trikona":0.83,
            "Great Friend":0.7, "Friendly":0.60,
            "Neutral":0.45, "Inimical":0.25, "Debilitated":0.10
        }.get(dig, 0.45)
        score += w * factor
    return round(score / total_weight * 20, 2)


def planet_strength(planet: str, sign: str, house: int, planets: Dict[str, float],
                    sun_lon: float, retro: bool = False) -> Tuple[float, Dict]:
    """
    Returns (score, breakdown_dict) for full strength calculation.
    Components: Dignity, House, Combustion, Retrograde, Dig-Bala, Neechabhanga.
    """
    dignity    = get_planet_dignity(planet, sign, planets)
    lagna_sign = longitude_to_sign(min(planets.values()))[0]  # approximation if no chart
    nb         = is_neechabhanga(planet, sign, planets, lagna_sign)

    dig_score   = DIGNITY_STRENGTH.get(dignity, 45)
    house_score = HOUSE_STRENGTH.get(house, 45)
    base        = dig_score * 0.5 + house_score * 0.3

    combust, orb = is_combust(planet, sun_lon, planets.get(planet, 0), retro)
    combust_factor = 0.4 if combust else 1.0

    retro_factor = 1.3 if retro and planet not in ("Sun","Moon","Rahu","Ketu") else 1.0
    dig_bala     = get_directional_strength(planet, house)
    nb_factor    = 1.2 if nb and dignity == "Debilitated" else 1.0

    score = base * combust_factor * retro_factor * dig_bala * nb_factor
    score = round(min(score, 100), 1)

    breakdown = {
        "dignity":          dignity,
        "dignity_score":    dig_score,
        "house":            house,
        "house_score":      house_score,
        "combust":          combust,
        "combust_orb_deg":  orb,
        "combust_factor":   combust_factor,
        "retrograde":       retro,
        "retro_factor":     retro_factor,
        "dig_bala_factor":  dig_bala,
        "neechabhanga":     nb,
        "nb_factor":        nb_factor,
        "final_score":      score,
    }
    return score, breakdown


def check_parivartana(p1: str, p2: str, planets: Dict[str, float]) -> bool:
    """
    Parivartana (mutual exchange): p1 is in p2's sign and p2 is in p1's sign.
    """
    if p1 not in planets or p2 not in planets:
        return False
    s1 = longitude_to_sign(planets[p1])[0]
    s2 = longitude_to_sign(planets[p2])[0]
    return SIGN_LORD.get(s1) == p2 and SIGN_LORD.get(s2) == p1


# ==================================================================
# SECTION 3 — DASHA CALCULATIONS
# ==================================================================

@dataclass
class DashaPeriod:
    planet:     str
    start_date: datetime
    end_date:   datetime
    years:      float
    level:      str
    parent:     Optional[str] = None


def add_timedelta_precise(dt: datetime, years: float) -> datetime:
    return dt + timedelta(days=years * 365.2425)


def calculate_vimshottari_full(birth_date: datetime,
                                moon_longitude: float) -> List[DashaPeriod]:
    moon_lon    = moon_longitude % 360
    nak_idx     = int(moon_lon / NAKSHATRA_SIZE)
    nak_start   = nak_idx * NAKSHATRA_SIZE
    deg_covered = moon_lon - nak_start
    remaining   = NAKSHATRA_SIZE - deg_covered
    fraction    = remaining / NAKSHATRA_SIZE
    lord_idx    = nak_idx % 9
    start_lord  = DASHA_SEQUENCE[lord_idx]
    balance     = fraction * DASHA_YEARS[start_lord]

    periods      = []
    current_date = birth_date
    for i in range(9):
        lord     = DASHA_SEQUENCE[(lord_idx + i) % 9]
        years    = balance if i == 0 else DASHA_YEARS[lord]
        end_date = add_timedelta_precise(current_date, years)
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
        end_date     = add_timedelta_precise(current_date, ad_actual)
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
        end_date     = add_timedelta_precise(current_date, pd_actual)
        pd_periods.append(DashaPeriod(
            planet=pd_planet, start_date=current_date, end_date=end_date,
            years=round(pd_actual, 5), level="PD", parent=ad.planet
        ))
        current_date = end_date
    return pd_periods


def get_current_dasha(periods: List[DashaPeriod],
                      check_date: datetime = None) -> Optional[DashaPeriod]:
    if check_date is None:
        check_date = datetime.now()
    for p in periods:
        if p.start_date <= check_date < p.end_date:
            return p
    return None


def get_current_antardasha(md_periods: List[DashaPeriod],
                            check_date: datetime = None) -> Optional[DashaPeriod]:
    if check_date is None:
        check_date = datetime.now()
    md = get_current_dasha(md_periods, check_date)
    if not md:
        return None
    return get_current_dasha(calculate_antardasha(md), check_date)


def get_current_pratyantardasha(md_periods: List[DashaPeriod],
                                 check_date: datetime = None) -> Optional[DashaPeriod]:
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
        return {"active": False, "phase": "", "logic": "Saturn sign not determinable"}
    m_idx = ZODIAC.index(moon_sign)
    s_idx = ZODIAC.index(saturn_sign)
    rel   = (s_idx - m_idx) % 12
    phases = {
        11: ("Rising Phase", f"Saturn in {ZODIAC[(m_idx+11)%12]} — 12th from Moon. Begins mental anxiety, foreign travel, expenses."),
        0:  ("Peak Phase",   f"Saturn on Moon sign {moon_sign}. Pressure on health, relationships, and emotional resilience."),
        1:  ("Setting Phase",f"Saturn in {ZODIAC[(m_idx+1)%12]} — 2nd from Moon. Financial challenges, family stress, speech issues.")
    }
    if rel in phases:
        phase_name, phase_detail = phases[rel]
        return {
            "active":       True,
            "phase":        phase_name,
            "detail":       phase_detail,
            "saturn_sign":  saturn_sign,
            "moon_sign":    moon_sign,
            "logic":        f"Saturn ({saturn_sign}) is {rel} signs from Moon ({moon_sign})"
        }
    return {"active": False, "phase": "", "logic": f"Saturn is {rel} signs from Moon — outside Sade Sati range (11,0,1)"}


def check_kantaka_shani(moon_sign: str, saturn_sign: str) -> Dict:
    if saturn_sign not in ZODIAC:
        return {"active": False}
    rel = (ZODIAC.index(saturn_sign) - ZODIAC.index(moon_sign)) % 12
    kantaka_positions = {3:"4th from Moon (emotional disruption)", 6:"7th from Moon (relationship/partnership stress)", 9:"10th from Moon (career and authority conflicts)"}
    if rel in kantaka_positions:
        return {"active": True, "position": kantaka_positions[rel], "saturn_sign": saturn_sign}
    return {"active": False}


# ==================================================================
# SECTION 4 — CHART DATA CLASS (ENHANCED)
# ==================================================================

class ChartData:
    def __init__(self, planets: Dict[str, float], ascendant: float, lagna_sign: str,
                 birth_date: datetime = None, lat: float = 0.0, lon: float = 0.0,
                 tz: float = 0.0, retrograde: Dict[str, bool] = None):
        self.planets    = planets
        self.ascendant  = ascendant
        self.lagna_sign = lagna_sign
        self.moon_sign  = longitude_to_sign(planets["Moon"])[0]
        self.sun_sign   = longitude_to_sign(planets["Sun"])[0]
        self.birth_date = birth_date
        self.lat        = lat
        self.lon        = lon
        self.tz         = tz
        self.retrograde = retrograde or {}

        # Derived
        self.nakshatras:           Dict = {}
        self.navamsa:              Dict = {}
        self.drekkana:             Dict = {}
        self.saptamsa:             Dict = {}
        self.dasamsa:              Dict = {}
        self.dwadasamsa:           Dict = {}
        self.dignities:            Dict = {}
        self.navamsa_dignities:    Dict = {}
        self.dasamsa_dignities:    Dict = {}
        self.shadbala_proxy:       Dict = {}
        self.shadbala_breakdown:   Dict = {}
        self.vimsopaka:            Dict = {}
        self.vargottama:           Dict = {}
        self.dasha_periods:        List[DashaPeriod] = []
        self.atmakaraka:           str  = ""
        self.amatyakaraka:         str  = ""
        self.parivartana_pairs:    List[Tuple[str,str]] = []
        self.graha_yuddha:         List[Dict] = []

        self._house_map = None
        self._lord_map  = None

        self._compute_derived()

    def _compute_derived(self):
        lagna_idx = ZODIAC.index(self.lagna_sign)

        house_map = {}
        for p, lon in self.planets.items():
            sign, _ = longitude_to_sign(lon)
            house_map[p] = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        self._house_map = house_map

        lord_map = {}
        for i in range(12):
            lord_map[i + 1] = SIGN_LORD[ZODIAC[(lagna_idx + i) % 12]]
        self._lord_map = lord_map

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
            self.vargottama[p] = is_vargottama(lon)

            sign, _                  = longitude_to_sign(lon)
            self.dignities[p]        = get_planet_dignity(p, sign, self.planets)
            self.navamsa_dignities[p]= get_planet_dignity(p, self.navamsa[p], self.planets)
            self.dasamsa_dignities[p]= get_planet_dignity(p, self.dasamsa[p], self.planets)

            score, bkd = planet_strength(
                p, sign, house_map.get(p, 6), self.planets,
                self.planets.get("Sun", 0), self.retrograde.get(p, False)
            )
            self.shadbala_proxy[p]     = score
            self.shadbala_breakdown[p] = bkd

            self.vimsopaka[p] = vimsopaka_bala(
                p, sign, self.navamsa[p], self.dasamsa[p], self.planets
            )

        # Parivartana pairs
        classical_planets = [p for p in self.planets if p not in ("Rahu","Ketu")]
        for i, p1 in enumerate(classical_planets):
            for p2 in classical_planets[i+1:]:
                if check_parivartana(p1, p2, self.planets):
                    self.parivartana_pairs.append((p1, p2))

        # Graha Yuddha
        war_candidates = ["Mars","Mercury","Jupiter","Venus","Saturn"]
        for i, p1 in enumerate(war_candidates):
            for p2 in war_candidates[i+1:]:
                if p1 in self.planets and p2 in self.planets:
                    if check_graha_yuddha(p1, p2, self.planets):
                        winner = get_yuddha_winner(p1, p2, self.planets)
                        loser  = p2 if winner == p1 else p1
                        self.graha_yuddha.append({
                            "planets": (p1, p2), "winner": winner, "loser": loser,
                            "logic": f"{p1} at {self.planets[p1]%30:.2f}° vs {p2} at {self.planets[p2]%30:.2f}° — within 1° in same sign"
                        })

        if self.birth_date:
            self.dasha_periods = calculate_vimshottari_full(self.birth_date, self.planets["Moon"])

        self.atmakaraka   = get_atmakaraka(self.planets)
        self.amatyakaraka = get_amatyakaraka(self.planets, self.atmakaraka)

    @property
    def house_map(self):
        if self._house_map is None:
            self._compute_derived()
        return self._house_map

    @property
    def lord_map(self):
        if self._lord_map is None:
            self._compute_derived()
        return self._lord_map

    def get_current_dasha_info(self, check_date: datetime = None) -> Dict:
        md = get_current_dasha(self.dasha_periods, check_date)
        if not md:
            return {}
        ad = get_current_antardasha(self.dasha_periods, check_date)
        pd = get_current_pratyantardasha(self.dasha_periods, check_date)

        # Dasha planet's own dignity and house for context
        md_sign, _ = longitude_to_sign(self.planets.get(md.planet, 0))
        md_house   = self.house_map.get(md.planet, 0)
        md_dignity = self.dignities.get(md.planet, "Neutral")

        return {
            "mahadasha":         md.planet,
            "mahadasha_start":   md.start_date.strftime("%d %b %Y"),
            "mahadasha_end":     md.end_date.strftime("%d %b %Y"),
            "md_sign":           md_sign,
            "md_house":          md_house,
            "md_dignity":        md_dignity,
            "antardasha":        ad.planet if ad else "",
            "antardasha_start":  ad.start_date.strftime("%d %b %Y") if ad else "",
            "antardasha_end":    ad.end_date.strftime("%d %b %Y")   if ad else "",
            "pratyantardasha":   pd.planet if pd else "",
            "pd_start":          pd.start_date.strftime("%d %b %Y") if pd else "",
            "pd_end":            pd.end_date.strftime("%d %b %Y")   if pd else "",
        }

    def to_dict(self) -> Dict:
        return {
            "birth_date":    self.birth_date.isoformat() if self.birth_date else None,
            "lat": self.lat, "lon": self.lon, "tz": self.tz,
            "lagna_sign":    self.lagna_sign,
            "moon_sign":     self.moon_sign,
            "sun_sign":      self.sun_sign,
            "ascendant":     self.ascendant,
            "atmakaraka":    self.atmakaraka,
            "amatyakaraka":  self.amatyakaraka,
            "planets":       self.planets,
            "retrograde":    self.retrograde,
            "nakshatras":    self.nakshatras,
            "navamsa":       self.navamsa,
            "dignities":     self.dignities,
            "vargottama":    self.vargottama,
            "shadbala_proxy": self.shadbala_proxy,
            "shadbala_breakdown": self.shadbala_breakdown,
            "vimsopaka":     self.vimsopaka,
            "parivartana":   [list(p) for p in self.parivartana_pairs],
            "graha_yuddha":  self.graha_yuddha,
            "dasha": [
                {"planet": p.planet, "start": p.start_date.isoformat(),
                 "end": p.end_date.isoformat(), "years": p.years}
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
    lagna_idx  = ZODIAC.index(chart.lagna_sign)
    house_map  = chart.house_map
    lord_map   = chart.lord_map
    aspect_map = {h: get_aspects_on_house(h, house_map) for h in range(1, 13)}

    neechabhanga_map = {}
    nb_conditions_map = {}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        conds = neechabhanga_conditions(p, sign, chart.planets, chart.lagna_sign)
        neechabhanga_map[p]  = len(conds) > 0
        nb_conditions_map[p] = conds

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
        "shadbala_breakdown":  chart.shadbala_breakdown,
        "vimsopaka":           chart.vimsopaka,
        "vargottama":          chart.vargottama,
        "house_map":           house_map,
        "lord_map":            lord_map,
        "aspect_map":          aspect_map,
        "neechabhanga":        neechabhanga_map,
        "nb_conditions":       nb_conditions_map,
        "parivartana_pairs":   chart.parivartana_pairs,
        "graha_yuddha":        chart.graha_yuddha,
        "atmakaraka":          chart.atmakaraka,
        "amatyakaraka":        chart.amatyakaraka,
        "dasha":               dasha_info.get("mahadasha","")    if dasha_info else "",
        "antardasha":          dasha_info.get("antardasha","")   if dasha_info else "",
        "pratyantardasha":     dasha_info.get("pratyantardasha","") if dasha_info else "",
        "dasha_md_start":      dasha_info.get("mahadasha_start","") if dasha_info else "",
        "dasha_md_end":        dasha_info.get("mahadasha_end","")   if dasha_info else "",
        "dasha_ad_start":      dasha_info.get("antardasha_start","") if dasha_info else "",
        "dasha_ad_end":        dasha_info.get("antardasha_end","")   if dasha_info else "",
        "md_sign":             dasha_info.get("md_sign","")     if dasha_info else "",
        "md_house":            dasha_info.get("md_house",0)     if dasha_info else 0,
        "md_dignity":          dasha_info.get("md_dignity","")  if dasha_info else "",
        "sade_sati_active":    False,
        "sade_sati_phase":     "",
        "sade_sati_detail":    "",
    }

    if sade_sati_info:
        ctx["sade_sati_active"] = sade_sati_info.get("active", False)
        ctx["sade_sati_phase"]  = sade_sati_info.get("phase", "")
        ctx["sade_sati_detail"] = sade_sati_info.get("detail", "")

    return ctx


# ==================================================================
# SECTION 6 — HELPER ACCESSORS
# ==================================================================

def _house(planet: str, ctx: dict) -> int:
    return ctx["house_map"].get(planet, 0)

def _lord(house: int, ctx: dict) -> str:
    return ctx["lord_map"].get(house, "")

def _dignity(planet: str, ctx: dict) -> str:
    if not planet:
        return "Neutral"
    return ctx["dignities"].get(planet, "Neutral")

def _navamsa_dignity(planet: str, ctx: dict) -> str:
    return ctx["navamsa_dignities"].get(planet, "Neutral")

def _dasamsa_dignity(planet: str, ctx: dict) -> str:
    return ctx["dasamsa_dignities"].get(planet, "Neutral")

def _strong(planet: str, ctx: dict) -> bool:
    if not planet:
        return False
    return _dignity(planet, ctx) in ["Exalted","Own","Mool Trikona","Great Friend"]

def _weak(planet: str, ctx: dict) -> bool:
    return _dignity(planet, ctx) == "Debilitated"

def _strength(planet: str, ctx: dict) -> float:
    return ctx["shadbala"].get(planet, 45.0)

def _bkd(planet: str, ctx: dict) -> Dict:
    return ctx["shadbala_breakdown"].get(planet, {})

def _vims(planet: str, ctx: dict) -> float:
    return ctx["vimsopaka"].get(planet, 0.0)

def _vargo(planet: str, ctx: dict) -> bool:
    return ctx["vargottama"].get(planet, False)

def _aspects_house(planet: str, house: int, ctx: dict) -> bool:
    return any(p == planet for p, _ in ctx["aspect_map"].get(house, []))

def _benefics_aspect(house: int, ctx: dict) -> List[str]:
    return [p for p, _ in ctx["aspect_map"].get(house, [])
            if p in ["Jupiter","Venus","Mercury","Moon"]]

def _malefics_aspect(house: int, ctx: dict) -> List[str]:
    return [p for p, _ in ctx["aspect_map"].get(house, [])
            if p in ["Saturn","Mars","Rahu","Ketu","Sun"]]

def _nb(planet: str, ctx: dict) -> bool:
    return ctx["neechabhanga"].get(planet, False)

def _nb_conds(planet: str, ctx: dict) -> List[str]:
    return ctx["nb_conditions"].get(planet, [])

def _nak(planet: str, ctx: dict) -> str:
    return ctx["nakshatras"].get(planet, {}).get("nakshatra", "")

def _nak_lord(planet: str, ctx: dict) -> str:
    return ctx["nakshatras"].get(planet, {}).get("lord", "")

def _combust(planet: str, ctx: dict) -> bool:
    return ctx["shadbala_breakdown"].get(planet, {}).get("combust", False)

def _retro(planet: str, ctx: dict) -> bool:
    return ctx["shadbala_breakdown"].get(planet, {}).get("retrograde", False)

def _logic_label(planet: str, ctx: dict) -> str:
    """Produce a short logic string for a planet's placement."""
    if not planet:
        return ""
    h    = _house(planet, ctx)
    dig  = _dignity(planet, ctx)
    s    = _strength(planet, ctx)
    v    = _vims(planet, ctx)
    vg   = " [Vargottama]" if _vargo(planet, ctx) else ""
    comb = " [Combust]" if _combust(planet, ctx) else ""
    ret  = " [Retrograde]" if _retro(planet, ctx) else ""
    nak  = _nak(planet, ctx)
    return (f"{planet} in H{h} | {dig}{vg}{comb}{ret} | "
            f"Nakshatra: {nak} | Shadbala: {s}/100 | Vimsopaka: {v}/20")


# ==================================================================
# SECTION 7 — PREDICTION RULES (COMPLETE, ELABORATED)
# ==================================================================

PREDICTION_RULES: List[Dict] = [

    # ── CAREER ──────────────────────────────────────────────────
    {
        "id": "career_sun_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Sun", ctx) == 10,
        "severity": "positive",
        "score": 3,
        "title": "Sun in 10th — Authority & Public Prominence",
        "detail": lambda ctx: (
            f"CALCULATION: Sun occupies House 10 (Karma Bhava). "
            f"Sun's dignity: {_dignity('Sun',ctx)}. Shadbala: {_strength('Sun',ctx)}/100. "
            f"Nakshatra: {_nak('Sun',ctx)} (lord: {_nak_lord('Sun',ctx)}). "
            f"Vimsopaka: {_vims('Sun',ctx)}/20. "
            + (f"Vargottama — Sun in same sign in D1 and D9, exceptional strength. " if _vargo("Sun",ctx) else "")
            + (f"Combust not applicable (Sun is the illuminator). " if True else "")
            + f"\nINTERPRETATION: Sun in the 10th is one of the finest career placements. "
            f"It confers natural authority, confidence, and an instinct for leadership. "
            f"Government, administration, politics, medicine, senior management, or any field "
            f"requiring visible command are natural fits. "
            + ("The exalted/own Sun maximises this: career rise is marked by recognition and "
               "enduring respect. " if _strong("Sun",ctx) else
               "Neechabhanga applies — debilitation is cancelled; career authority emerges after "
               "initial adversity. " if _nb("Sun",ctx) and _weak("Sun",ctx) else
               "Sun in neutral/friendly dignity — authority is present but must be consciously built. ")
            + (f"\nNB conditions: {'; '.join(_nb_conds('Sun',ctx))}" if _nb("Sun",ctx) else "")
            + ("\nJupiter aspects the 10th — dharmic success and institutional recognition added." if _aspects_house("Jupiter",10,ctx) else "")
            + ("\nSaturn also in 10th or aspecting — authority comes through sustained effort and discipline." if _aspects_house("Saturn",10,ctx) else "")
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
            f"CALCULATION: Saturn in House 10. Dignity: {_dignity('Saturn',ctx)}. "
            f"Shadbala: {_strength('Saturn',ctx)}/100. Nakshatra: {_nak('Saturn',ctx)}. "
            f"Vimsopaka: {_vims('Saturn',ctx)}/20. "
            + (f"Retrograde — increases Saturn's introspective power but may delay career recognition. " if _retro("Saturn",ctx) else "")
            + (f"Vargottama — extremely powerful Saturn. " if _vargo("Saturn",ctx) else "")
            + f"\nINTERPRETATION: Saturn in the 10th is one of the most powerful career placements "
            "— gifts arrive late but prove rock-solid. Engineering, law, architecture, administration, "
            "real estate, research, or any structured institution are natural domains. "
            + ("Exalted Saturn here forms Shasha Yoga — massive career authority, recognition from "
               "the masses, management mastery. A classic indicator of a great career after 36." if _dignity("Saturn",ctx) == "Exalted"
               else "Own-sign Saturn in 10th — slower rise but iron reputation over decades." if _dignity("Saturn",ctx) == "Own"
               else "Debilitated Saturn in 10th — authority conflicts, career disruptions. "
               "Neechabhanga applies if conditions met: adversity eventually forges exceptional resilience." if _dignity("Saturn",ctx) == "Debilitated"
               else "Saturn here builds career brick by brick through discipline and integrity.")
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
            f"CALCULATION: Jupiter in House 10. Dignity: {_dignity('Jupiter',ctx)}. "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. Nakshatra: {_nak('Jupiter',ctx)}. "
            f"Vimsopaka: {_vims('Jupiter',ctx)}/20. "
            + ("Vargottama Jupiter — wisdom extraordinarily amplified. " if _vargo("Jupiter",ctx) else "")
            + f"\nINTERPRETATION: Jupiter in the 10th creates a dharmic, ethics-driven career. "
            "Teaching, law, finance, banking, counselling, publishing, spirituality, or "
            "administration of large institutions are ideal. Reputation grows through integrity. "
            + ("Exalted/Own Jupiter here forms Hamsa Yoga in a kendra — rarest of career yogas. "
               "Scholarly fame, spiritual recognition, a career others look up to." if _strong("Jupiter",ctx)
               else "Debilitated Jupiter slows expansion, causes conflicts with mentors/institutions. "
               "Jupiter Shanti puja, charity on Thursdays, and Guru-seva are strongly advised." if _weak("Jupiter",ctx)
               else "Steady, principled career growth over the arc is indicated.")
        ),
        "activation": "natal"
    },
    {
        "id": "career_10th_lord_strong",
        "topic": "career",
        "condition": lambda ctx: _strong(_lord(10, ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong 10th Lord — Rajayoga-Class Career Indicator",
        "detail": lambda ctx: (
            f"CALCULATION: 10th house sign = {ZODIAC[(ctx['lagna_idx']+9)%12]}. "
            f"10th lord = {_lord(10,ctx)}. Dignity = {_dignity(_lord(10,ctx),ctx)}. "
            f"10th lord is in House {_house(_lord(10,ctx),ctx)}. "
            f"Shadbala: {_strength(_lord(10,ctx),ctx)}/100. Vimsopaka: {_vims(_lord(10,ctx),ctx)}/20. "
            + ("Vargottama — reinforced in D9. " if _vargo(_lord(10,ctx),ctx) else "")
            + f"\nINTERPRETATION: The 10th lord is the single most important career factor. "
            "A strong 10th lord creates a Rajayoga-class indicator: career success is foundational "
            "and enduring. "
            + _house_career_meaning(_house(_lord(10,ctx),ctx))
            + ("\nAdditionally, 10th lord is Vargottama — strength confirmed in D9 navamsa." if _vargo(_lord(10,ctx),ctx) else "")
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
            f"CALCULATION: 10th lord {_lord(10,ctx)} is in {_dignity(_lord(10,ctx),ctx)} dignity. "
            f"No Neechabhanga conditions satisfied. "
            f"Shadbala: {_strength(_lord(10,ctx),ctx)}/100 (below functional threshold). "
            f"\nINTERPRETATION: Debilitated 10th lord without cancellation is the most significant "
            "single indicator of career difficulty. Authority conflicts, abrupt terminations, "
            "inability to sustain momentum, or repeated career restarts are possible. "
            "Remedies for the 10th lord planet are essential. Consult an astrologer before "
            "major career decisions. Watch for Antardashas of strong benefics for peak windows."
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
            f"CALCULATION: 10th lord {_lord(10,ctx)} is debilitated but Neechabhanga applies. "
            f"Conditions satisfied: {'; '.join(_nb_conds(_lord(10,ctx),ctx))}. "
            f"\nINTERPRETATION: Classical texts state Neechabhanga itself confers Raja Yoga — "
            "the cancellation produces exceptional strength, often surpassing a straightforwardly "
            "strong 10th lord. Early hardships are possible; the eventual achievement is greater "
            "for the obstacles overcome."
        ),
        "activation": "natal"
    },
    {
        "id": "career_10th_lord_d10_strong",
        "topic": "career",
        "condition": lambda ctx: _dasamsa_dignity(_lord(10,ctx),ctx) in ["Exalted","Own","Mool Trikona","Great Friend"],
        "severity": "positive",
        "score": 2,
        "title": "10th Lord Strong in Dasamsa (D10) — Career Excellence Confirmed",
        "detail": lambda ctx: (
            f"CALCULATION: 10th lord {_lord(10,ctx)} has D10 dignity = "
            f"{_dasamsa_dignity(_lord(10,ctx),ctx)}. "
            f"D10 (Dasamsa) is the divisional chart exclusively governing professional life. "
            f"D1 dignity: {_dignity(_lord(10,ctx),ctx)}. "
            f"\nINTERPRETATION: When D1 strength is echoed in D10, professional success is "
            "near-certain. D10 confirmation indicates recognition, promotions, and institutional "
            "respect within the career domain."
        ),
        "activation": "natal"
    },
    {
        "id": "career_amatyakaraka_strong",
        "topic": "career",
        "condition": lambda ctx: _strong(ctx.get("amatyakaraka",""), ctx) if ctx.get("amatyakaraka") else False,
        "severity": "positive",
        "score": 2,
        "title": "Strong Amatyakaraka — Jaimini Career Blessing",
        "detail": lambda ctx: (
            f"CALCULATION (Jaimini): Amatyakaraka = {ctx['amatyakaraka']} "
            f"(second highest degree in sign among all planets). "
            f"Dignity: {_dignity(ctx['amatyakaraka'],ctx)}. "
            f"House: {_house(ctx['amatyakaraka'],ctx)}. "
            f"Shadbala: {_strength(ctx['amatyakaraka'],ctx)}/100. "
            f"\nINTERPRETATION: Amatyakaraka is the soul's career minister in Jaimini astrology. "
            "A strong Amatyakaraka aligns the professional path with life purpose — "
            "external circumstances, mentors, and opportunities actively support the career."
        ),
        "activation": "natal"
    },
    {
        "id": "career_dharma_karma_yoga",
        "topic": "career",
        "condition": lambda ctx: _strong(_lord(9,ctx),ctx) and _strong(_lord(10,ctx),ctx),
        "severity": "positive",
        "score": 4,
        "title": "Dharma-Karma Adhipati Yoga — Fortune Fused with Action",
        "detail": lambda ctx: (
            f"CALCULATION: 9th lord = {_lord(9,ctx)} ({_dignity(_lord(9,ctx),ctx)}, "
            f"H{_house(_lord(9,ctx),ctx)}). "
            f"10th lord = {_lord(10,ctx)} ({_dignity(_lord(10,ctx),ctx)}, "
            f"H{_house(_lord(10,ctx),ctx)}). Both lords are in strong dignities. "
            f"\nINTERPRETATION: Dharma-Karma Adhipati Yoga is formed when 9th and 10th lords are "
            "both strong. Fortune (9th) actively supports karma/action (10th). Career success "
            "carries an element of divine timing and the sense of doing one's rightful work. "
            + (f"Both lords in same sign/conjunct — exceptionally powerful activation." if _house(_lord(9,ctx),ctx) == _house(_lord(10,ctx),ctx) else "")
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
            f"CALCULATION: Sun in {longitude_to_sign(ctx['planets']['Sun'])[0]} and "
            f"Mercury in {longitude_to_sign(ctx['planets']['Mercury'])[0]} — same sign. "
            f"Sun dignity: {_dignity('Sun',ctx)}, Mercury dignity: {_dignity('Mercury',ctx)}. "
            f"Mercury combust: {_combust('Mercury',ctx)} "
            f"(orb: {_bkd('Mercury',ctx).get('combust_orb_deg','?')}°). "
            f"\nINTERPRETATION: Sun-Mercury conjunction forms Budhaditya Yoga — analytical power, "
            "communication brilliance, and administrative acumen. Writing, publishing, IT, "
            "consulting, teaching, and quick-thinking professions are natural fits. "
            + ("NOTE: Mercury is combust — yoga is weakened but not destroyed. Choose professions "
               "where Sun-Mercury joint qualities work (editor, teacher, administrator)." if _combust("Mercury",ctx) else
               "Mercury is clear of combustion — yoga operates at full strength.")
            + (" Both planets strong — highly activated yoga." if _strong("Sun",ctx) and _strong("Mercury",ctx) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "career_amala_yoga",
        "topic": "career",
        "condition": lambda ctx: any(
            _house(p,ctx)==10 and p in ["Jupiter","Venus","Mercury","Moon"]
            for p in ctx["planets"]
        ),
        "severity": "positive",
        "score": 2,
        "title": "Amala Yoga — Spotless Reputation",
        "detail": lambda ctx: (
            f"CALCULATION: Natural benefic(s) in House 10 — "
            f"{[p for p in ['Jupiter','Venus','Mercury','Moon'] if _house(p,ctx)==10]}. "
            f"\nINTERPRETATION: Amala Yoga ('spotless') is formed when a natural benefic occupies "
            "the 10th. This confers an unblemished professional reputation, ethical recognition, "
            "and sustained goodwill. Fame from this placement is durable."
        ),
        "activation": "natal"
    },
    {
        "id": "career_parivartana_10th",
        "topic": "career",
        "condition": lambda ctx: any(
            (_lord(10,ctx) in p and any(ZODIAC[(ctx['lagna_idx']+9)%12] == longitude_to_sign(ctx['planets'].get(q,0))[0] for q in p))
            for p in ctx.get("parivartana_pairs",[])
        ),
        "severity": "positive",
        "score": 2,
        "title": "Parivartana Yoga Involving 10th Lord — Mutual Exchange Power",
        "detail": lambda ctx: (
            f"CALCULATION: The 10th lord ({_lord(10,ctx)}) is involved in a Parivartana "
            f"(mutual sign exchange) with another planet. Each planet moves to the other's "
            f"sign, effectively giving both the dignity of being in their own sign. "
            f"\nINTERPRETATION: Parivartana with the 10th lord transfers power and results "
            "between two houses. Career and the partner house exchange resources and outcomes, "
            "often creating unexpected career channels through that domain."
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
            f"CALCULATION: Rahu in House 10. Nakshatra: {_nak('Rahu',ctx)} "
            f"(lord: {_nak_lord('Rahu',ctx)}). "
            f"10th lord's condition heavily modifies: {_lord(10,ctx)} is "
            f"{_dignity(_lord(10,ctx),ctx)} in H{_house(_lord(10,ctx),ctx)}. "
            f"\nINTERPRETATION: Rahu in the 10th creates obsessive career drive. Sudden, "
            "dramatic rises through technology, foreign companies, media, or pioneering fields. "
            "The nakshatra lord of Rahu governs the field: "
            f"{_nak_lord('Rahu',ctx)}-related domains are favoured. "
            "Ambition is extreme — align with ethics to prevent equally dramatic falls."
        ),
        "activation": "natal"
    },
    {
        "id": "career_ketu_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Ketu", ctx) == 10,
        "severity": "caution",
        "score": -1,
        "title": "Ketu in 10th — Karmic Detachment from Conventional Career",
        "detail": lambda ctx: (
            f"CALCULATION: Ketu in House 10. Rahu (opposite) in House 4. "
            f"The soul's hunger (Rahu) is in the domain of home/inner world (H4) while "
            f"Ketu (release/moksha) sits on the career. "
            f"\nINTERPRETATION: Past-life mastery of the career domain is indicated by Ketu here. "
            "You may excel professionally but find hollow satisfaction in it. The path eventually "
            "pivots toward research, spirituality, healing, or behind-the-scenes work. "
            "Professional disruptions during Ketu Mahadasha are common. Best roles: research, "
            "mystical sciences, alternative medicine, or fields synthesising mastery with intuition."
        ),
        "activation": "natal"
    },
    {
        "id": "career_dasha_career_planet",
        "topic": "career",
        "condition": lambda ctx: ctx.get("dasha","") in ["Jupiter","Sun","Saturn","Mercury","Rahu"],
        "severity": "positive",
        "score": 2,
        "title": "Favourable Career Mahadasha Active",
        "detail": lambda ctx: (
            f"CALCULATION: Running Mahadasha = {ctx.get('dasha','')} "
            f"({ctx.get('dasha_md_start','')} → {ctx.get('dasha_md_end','')}). "
            f"MD planet's natal position: H{ctx.get('md_house',0)}, "
            f"dignity: {ctx.get('md_dignity','')}. "
            f"Current Antardasha: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            + {
                "Jupiter": "\nINTERPRETATION: Jupiter MD is the most dharmic career period. "
                           "Expansion into teaching, law, banking, advisory roles. Promotions "
                           "tied to wisdom and institutional standing. Best ADs: Jupiter-Saturn "
                           "(disciplined wisdom) and Jupiter-Mercury (analytical recognition).",
                "Sun":     "\nINTERPRETATION: Sun MD brings authority and advancement in "
                           "government, leadership, or corporate hierarchies. Identity merges "
                           "with work. Best AD: Sun-Jupiter (dharmic recognition); "
                           "Sun-Saturn AD may test through humility.",
                "Saturn":  "\nINTERPRETATION: Saturn MD rewards past disciplined effort. "
                           "Promotions come slowly but solidly. Building legacy, not quick wins. "
                           "Saturn-Mercury AD: analytical/structured domains excel; "
                           "Saturn-Rahu AD: watch for unexpected disruptions.",
                "Mercury": "\nINTERPRETATION: Mercury MD favours communication, IT, trade, "
                           "analytics, multi-platform careers. Skills acquired compound into "
                           "capital. Best ADs: Mercury-Sun and Mercury-Jupiter.",
                "Rahu":    "\nINTERPRETATION: Rahu MD offers dramatic career leaps through "
                           "unconventional routes. First 2.5 years most volatile; "
                           "Rahu-Jupiter AD is often the golden breakthrough window. "
                           "Avoid ethical shortcuts — Rahu's gifts are quickly lost.",
            }.get(ctx.get("dasha",""), "")
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "career_jupiter_aspects_10th",
        "topic": "career",
        "condition": lambda ctx: _aspects_house("Jupiter",10,ctx) and _house("Jupiter",ctx) != 10,
        "severity": "positive",
        "score": 2,
        "title": "Jupiter Aspects 10th House — Dharmic Blessings on Career",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter is in House {_house('Jupiter',ctx)}. "
            f"Jupiter aspects the 10th via its "
            f"{'5th/9th special aspect' if _house('Jupiter',ctx) in [2,6] else '7th universal aspect'}. "
            f"Jupiter's dignity: {_dignity('Jupiter',ctx)}. "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. "
            f"\nINTERPRETATION: Jupiter's aspect on the 10th is highly auspicious — wisdom, "
            "dharmic energy, and expansive blessings illuminate the career house. "
            "Opportunities in Jupiter-ruled fields arise unexpectedly. "
            + ("Exalted/Own Jupiter aspecting 10th — maximum potency." if _strong("Jupiter",ctx) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "career_vargottama_10th_lord",
        "topic": "career",
        "condition": lambda ctx: _vargo(_lord(10,ctx),ctx),
        "severity": "positive",
        "score": 2,
        "title": "10th Lord Vargottama — Career Strength Locked Across Dimensions",
        "detail": lambda ctx: (
            f"CALCULATION: 10th lord {_lord(10,ctx)} is Vargottama — occupying the same sign "
            f"in both D1 (natal chart) and D9 (Navamsa). This is exceptionally rare and powerful. "
            f"When the lord of the most important career house is Vargottama, its dignity and "
            f"significations are doubled in strength across both charts. "
            f"D1 dignity: {_dignity(_lord(10,ctx),ctx)}. D9 dignity: {_navamsa_dignity(_lord(10,ctx),ctx)}. "
            f"\nINTERPRETATION: Vargottama 10th lord indicates a career path that is karmically "
            "destined and protected. What you build professionally has multi-lifetime significance."
        ),
        "activation": "natal"
    },

    # ── MARRIAGE ────────────────────────────────────────────────
    {
        "id": "marriage_venus_strong",
        "topic": "marriage",
        "condition": lambda ctx: _strong("Venus", ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong Venus — Happiness, Love & Refined Partnership",
        "detail": lambda ctx: (
            f"CALCULATION: Venus dignity = {_dignity('Venus',ctx)}. "
            f"House: {_house('Venus',ctx)}. Shadbala: {_strength('Venus',ctx)}/100. "
            f"Vimsopaka: {_vims('Venus',ctx)}/20. "
            f"D9 Venus: {_navamsa_dignity('Venus',ctx)}. "
            + (f"Venus Vargottama — soul-level confirmation of marital grace. " if _vargo("Venus",ctx) else "")
            + f"\nINTERPRETATION: Venus is the primary karaka for marriage and marital happiness. "
            "A dignified Venus confers a loving, aesthetically pleasing, and emotionally warm marriage. "
            "The spouse is likely attractive, creative, and affectionate. Material comforts are abundant. "
            + ("D9 Venus strong — deep soul-level compatibility confirmed." if _navamsa_dignity("Venus",ctx) in ["Exalted","Own","Mool Trikona"] else "")
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_venus_weak",
        "topic": "marriage",
        "condition": lambda ctx: _weak("Venus", ctx) and not _nb("Venus", ctx),
        "severity": "warning",
        "score": -2,
        "title": "Debilitated Venus (No Neechabhanga) — Marital Tensions",
        "detail": lambda ctx: (
            f"CALCULATION: Venus debilitated in {longitude_to_sign(ctx['planets']['Venus'])[0]}. "
            f"No Neechabhanga conditions satisfied. Shadbala: {_strength('Venus',ctx)}/100. "
            f"D9 Venus: {_navamsa_dignity('Venus',ctx)}. "
            f"\nINTERPRETATION: Debilitated Venus (no cancellation) is the most significant indicator "
            "of marital dissatisfaction or incompatibility. Relationship requires conscious cultivation. "
            "Remedies: white flowers on Fridays, charity to young women, chanting Shukra mantras. "
            "Pre-marital compatibility analysis is strongly recommended."
            + (f"\nD9 Venus: {_navamsa_dignity('Venus',ctx)} — "
               "some compensation at soul level." if _navamsa_dignity("Venus",ctx) in ["Exalted","Own","Mool Trikona"] else "")
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
            f"CALCULATION: Venus debilitated but Neechabhanga confirmed. "
            f"Conditions met: {'; '.join(_nb_conds('Venus',ctx))}. "
            f"\nINTERPRETATION: Early relationship difficulties are likely but ultimately resolved "
            "— often producing a deeper, more mature love. The marriage that survives these tests "
            "tends to be exceptionally strong and loyal. Classical texts call this a Raja Yoga "
            "in the domain of Venus."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_7th_lord_strong",
        "topic": "marriage",
        "condition": lambda ctx: _strong(_lord(7,ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong 7th Lord — Blessed Partnership",
        "detail": lambda ctx: (
            f"CALCULATION: 7th house sign = {ZODIAC[(ctx['lagna_idx']+6)%12]}. "
            f"7th lord = {_lord(7,ctx)}. Dignity = {_dignity(_lord(7,ctx),ctx)}. "
            f"In House {_house(_lord(7,ctx),ctx)}. Shadbala: {_strength(_lord(7,ctx),ctx)}/100. "
            + ("Vargottama 7th lord — partner energy confirmed across D1 and D9. " if _vargo(_lord(7,ctx),ctx) else "")
            + f"\nINTERPRETATION: The 7th lord is the primary timer and qualifier for marriage. "
            "A strong 7th lord indicates a spouse who is genuinely capable, supportive, and "
            "karmically well-matched. Business partnerships are also favoured. The 7th lord's "
            f"house (H{_house(_lord(7,ctx),ctx)}) shows where/how the spouse enters your life."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_7th_lord_weak",
        "topic": "marriage",
        "condition": lambda ctx: _weak(_lord(7,ctx), ctx) and not _nb(_lord(7,ctx), ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 7th Lord (No Neechabhanga) — Partnership Challenges",
        "detail": lambda ctx: (
            f"CALCULATION: 7th lord {_lord(7,ctx)} debilitated in "
            f"{longitude_to_sign(ctx['planets'].get(_lord(7,ctx),0))[0]}. "
            f"No Neechabhanga applies. Shadbala: {_strength(_lord(7,ctx),ctx)}/100. "
            f"\nINTERPRETATION: Debilitated 7th lord without cancellation — the single most "
            "important indicator of partnership challenges: incompatibility, emotional distance, "
            "possible separation, or delay in finding the right partner. "
            "Ashtakoota + full chart compatibility matching before marriage is essential."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_venus_d9_strong",
        "topic": "marriage",
        "condition": lambda ctx: _navamsa_dignity("Venus",ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 2,
        "title": "Venus Exalted/Own in Navamsa (D9) — Soul-Level Marital Harmony",
        "detail": lambda ctx: (
            f"CALCULATION: Venus's D9 sign = {ctx['navamsa'].get('Venus','?')}. "
            f"D9 dignity = {_navamsa_dignity('Venus',ctx)}. "
            f"D1 Venus dignity = {_dignity('Venus',ctx)} (H{_house('Venus',ctx)}). "
            f"\nINTERPRETATION: D9 (Navamsa) governs the inner quality of relationships. "
            "D9 Venus strength confirms deep emotional compatibility and lasting affection "
            "at the soul level — even if D1 shows some complexity."
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
            f"CALCULATION: Jupiter in House 7. Dignity: {_dignity('Jupiter',ctx)}. "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. "
            f"\nINTERPRETATION: Jupiter in the 7th is one of the best placements for marriage. "
            "The spouse is likely educated, wise, spiritually inclined, and morally upright. "
            + ("Exalted/Own Jupiter in 7th — potentially forming Hamsa Yoga; "
               "an exceptionally fortunate spouse." if _strong("Jupiter",ctx) else
               "Even a neutral Jupiter in the 7th provides significant protection to the marriage.")
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_kuja_dosha_high",
        "topic": "marriage",
        "condition": lambda ctx: _house("Mars", ctx) == 7,
        "severity": "warning",
        "score": -2,
        "title": "Severe Kuja Dosha — Mars in 7th House",
        "detail": lambda ctx: (
            f"CALCULATION: Mars in House 7. Dignity: {_dignity('Mars',ctx)}. "
            f"Shadbala: {_strength('Mars',ctx)}/100. "
            f"Mars Nakshatra: {_nak('Mars',ctx)}. "
            f"\nINTERPRETATION: Mars in the 7th is the most intense Kuja (Mangal) Dosha. "
            "Passion and dominance conflicts, power struggles, and separation risk if unaddressed. "
            "Matching with a Manglik partner (Mars in 1,2,4,7,8,12) neutralises the dosha. "
            + ("Exalted Mars in 7th may actually confer a powerful, dynamic partner — "
               "the dosha is significantly mitigated." if _dignity("Mars",ctx) == "Exalted" else "")
            + ("\nNeechabhanga applies — Mars's aggression in partnership converts to assertive strength." if _nb("Mars",ctx) else "")
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
            f"CALCULATION: Mars in House {_house('Mars',ctx)} — one of the partial Kuja Dosha "
            f"positions (1,2,4,8,12). Dignity: {_dignity('Mars',ctx)}. "
            f"\nINTERPRETATION: Partial dosha creates passion-driven conflicts and occasional "
            "friction in marriage — manageable with conscious effort. Compatible partner matching "
            "(check for fellow Manglik) is advisable."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_rahu_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Rahu", ctx) == 7,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 7th — Unconventional or Foreign Spouse",
        "detail": lambda ctx: (
            f"CALCULATION: Rahu in House 7. Ketu (opposite) in House 1. "
            f"Rahu's nakshatra: {_nak('Rahu',ctx)} (lord: {_nak_lord('Rahu',ctx)}). "
            f"7th lord {_lord(7,ctx)} is {_dignity(_lord(7,ctx),ctx)} — its condition "
            f"heavily modifies Rahu's result in 7th. "
            f"\nINTERPRETATION: Rahu in 7th brings an unusual, unexpected, or cross-cultural "
            "marriage. The attraction is intense, bordering on obsession. Trust issues possible "
            "if Rahu is afflicted. The spouse may have foreign background or exceptional ambition."
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
            f"CALCULATION: Saturn in House 7. Dignity: {_dignity('Saturn',ctx)}. "
            f"Shadbala: {_strength('Saturn',ctx)}/100. Retrograde: {_retro('Saturn',ctx)}. "
            f"\nINTERPRETATION: Saturn in the 7th is a classic delay indicator — typically after "
            "age 28-32. Once formed, the marriage is deeply karmic, committed, and lasting. "
            + ("Exalted Saturn — an exceptionally reliable, high-achieving spouse after the delay." if _dignity("Saturn",ctx)=="Exalted" else
               "Debilitated Saturn — intensifies delay and may introduce persistent friction; "
               "remedies essential." if _dignity("Saturn",ctx)=="Debilitated" else
               "Karmic quality of the marriage is dominant.")
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
            f"CALCULATION: Venus MD active ({ctx.get('dasha_md_start','')} → "
            f"{ctx.get('dasha_md_end','')}). "
            f"Venus natal position: H{_house('Venus',ctx)}, "
            f"dignity: {_dignity('Venus',ctx)}. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"\nINTERPRETATION: Venus MD (20 years) is the most powerful period for romantic "
            "union and marriage. Best sub-periods: Venus-Jupiter (dharmic union), "
            "Venus-Mercury (intellectual compatibility), Venus-Moon (emotional bonding). "
            "Beware Venus-Rahu and Venus-Ketu ADs — these can bring passion without stability."
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
            f"CALCULATION: Jupiter MD active. Natal Jupiter: H{_house('Jupiter',ctx)}, "
            f"{_dignity('Jupiter',ctx)}. Current AD: {ctx.get('antardasha','')}. "
            f"\nINTERPRETATION: Jupiter MD blesses marriage, children, and family life. "
            "Best ADs for marriage: Jupiter-Venus (highest priority), Jupiter-Moon. "
            "Jupiter-Saturn AD may bring serious, duty-bound relationships."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "marriage_benefics_aspect_7th",
        "topic": "marriage",
        "condition": lambda ctx: len(_benefics_aspect(7,ctx)) > 0,
        "severity": "positive",
        "score": 2,
        "title": "Benefic Planets Aspect 7th House — Protected Marriage",
        "detail": lambda ctx: (
            f"CALCULATION: Benefics aspecting 7th house: {_benefics_aspect(7,ctx)}. "
            f"Aspect types: {[(p,t) for p,t in ctx['aspect_map'].get(7,[]) if p in ['Jupiter','Venus','Mercury','Moon']]}. "
            f"\nINTERPRETATION: Benefic aspects on the 7th protect the marriage from serious harm "
            "and add qualities of wisdom (Jupiter), love (Venus), communication (Mercury), "
            "or emotional nourishment (Moon) to the partnership."
        ),
        "activation": "natal"
    },
    {
        "id": "marriage_malefics_aspect_7th",
        "topic": "marriage",
        "condition": lambda ctx: len(_malefics_aspect(7,ctx)) >= 2,
        "severity": "warning",
        "score": -2,
        "title": "Multiple Malefics Aspect 7th House — Relationship Stress",
        "detail": lambda ctx: (
            f"CALCULATION: Malefics aspecting 7th: {_malefics_aspect(7,ctx)}. "
            f"Benefics also aspecting (mitigating): {_benefics_aspect(7,ctx)}. "
            f"\nINTERPRETATION: Multiple malefic aspects on the 7th create significant stress — "
            "friction, power conflicts, or instability. If benefics also aspect, the damage is "
            "mitigated. Counselling and compatible partner selection are strongly recommended."
        ),
        "activation": "natal"
    },

    # ── CHILDREN ────────────────────────────────────────────────
    {
        "id": "children_jupiter_strong",
        "topic": "children",
        "condition": lambda ctx: _strong("Jupiter", ctx),
        "severity": "positive",
        "score": 4,
        "title": "Strong Jupiter (Putrakaraka) — Blessed Progeny",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter = natural Putrakaraka (significator of children). "
            f"Dignity: {_dignity('Jupiter',ctx)}. House: {_house('Jupiter',ctx)}. "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. Vimsopaka: {_vims('Jupiter',ctx)}/20. "
            + ("Vargottama — Jupiter's child-blessing locked across D1 and D9. " if _vargo("Jupiter",ctx) else "")
            + f"\nINTERPRETATION: A dignified Jupiter is the most powerful indicator of good "
            "fortune with children. Multiple healthy children are possible; at least one is "
            "likely to be exceptionally talented. "
            + ("Jupiter in trine (H1/5/9) — maximum yoga for children." if _house("Jupiter",ctx) in [1,5,9] else "")
        ),
        "activation": "natal"
    },
    {
        "id": "children_jupiter_weak",
        "topic": "children",
        "condition": lambda ctx: _weak("Jupiter", ctx) and not _nb("Jupiter", ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated Jupiter (No Neechabhanga) — Progeny Challenges",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter debilitated in {longitude_to_sign(ctx['planets']['Jupiter'])[0]}. "
            f"No Neechabhanga. Shadbala: {_strength('Jupiter',ctx)}/100. "
            f"\nINTERPRETATION: Debilitated Putrakaraka without cancellation is the most critical "
            "indicator of difficulty with children. Delays, conception challenges, or difficult "
            "pregnancies are possible. Remedies: Jupiter Shanti Puja, Santana Gopala Puja, "
            "Thursday fasting, donating gold or turmeric."
        ),
        "activation": "natal"
    },
    {
        "id": "children_5th_lord_strong",
        "topic": "children",
        "condition": lambda ctx: _strong(_lord(5,ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong 5th Lord — Fertile and Auspicious Children's House",
        "detail": lambda ctx: (
            f"CALCULATION: 5th house sign = {ZODIAC[(ctx['lagna_idx']+4)%12]}. "
            f"5th lord = {_lord(5,ctx)}. Dignity = {_dignity(_lord(5,ctx),ctx)}. "
            f"In House {_house(_lord(5,ctx),ctx)}. Shadbala: {_strength(_lord(5,ctx),ctx)}/100. "
            f"\nINTERPRETATION: A strong 5th lord strongly activates Putra Bhava. Children are "
            "likely intellectually bright, creatively gifted, or spiritually inclined. "
            f"The 5th lord in H{_house(_lord(5,ctx),ctx)} shapes the context through which "
            "children enter your life."
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
            f"CALCULATION: Saturn in House 5 (Putra Bhava). "
            f"Dignity: {_dignity('Saturn',ctx)}. Shadbala: {_strength('Saturn',ctx)}/100. "
            f"Benefics aspecting 5th: {_benefics_aspect(5,ctx)} (mitigating if any). "
            f"\nINTERPRETATION: Saturn in the 5th classically delays progeny, often until after "
            "Saturn's full maturation (~36 years). Children born tend to be serious, responsible, "
            "and long-lived. Monitor early pregnancies medically. Parenting style should develop "
            "warmth and playfulness to complement Saturn's seriousness."
        ),
        "activation": "natal"
    },
    {
        "id": "children_benefics_aspect_5th",
        "topic": "children",
        "condition": lambda ctx: len(_benefics_aspect(5,ctx)) > 0,
        "severity": "positive",
        "score": 2,
        "title": "Benefics Aspect 5th House — Protected Progeny Path",
        "detail": lambda ctx: (
            f"CALCULATION: Benefics aspecting 5th: {_benefics_aspect(5,ctx)}. "
            f"Aspect types: {[(p,t) for p,t in ctx['aspect_map'].get(5,[]) if p in ['Jupiter','Venus','Mercury','Moon']]}. "
            f"\nINTERPRETATION: Benefic aspects on the 5th offer significant protection even if "
            "the 5th lord or Jupiter are weak. Jupiter aspecting 5th is particularly potent — "
            "it is the Putrakaraka blessing its own domain."
        ),
        "activation": "natal"
    },
    {
        "id": "children_rahu_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Rahu", ctx) == 5,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 5th — Unconventional Conception Circumstances",
        "detail": lambda ctx: (
            f"CALCULATION: Rahu in House 5. Ketu (opposite) in House 11. "
            f"Rahu nakshatra: {_nak('Rahu',ctx)}. "
            f"5th lord {_lord(5,ctx)} is {_dignity(_lord(5,ctx),ctx)} — its condition "
            f"heavily modifies Rahu's result. "
            f"\nINTERPRETATION: Rahu in 5th creates ambiguity around conception — IVF/ART, "
            "adoption, or stepchildren are common. Children tend to be unconventional and highly "
            "intelligent. Rahu remedies and medical consultation advised if conception is delayed."
        ),
        "activation": "natal"
    },
    {
        "id": "children_dasha_jupiter",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Jupiter",
        "severity": "positive",
        "score": 3,
        "title": "Jupiter Mahadasha — Most Auspicious Period for Children",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter MD active ({ctx.get('dasha_md_start','')} → "
            f"{ctx.get('dasha_md_end','')}). Jupiter's natal position: H{_house('Jupiter',ctx)}, "
            f"{_dignity('Jupiter',ctx)}. Current AD: {ctx.get('antardasha','')}. "
            f"\nINTERPRETATION: Jupiter MD is universally the most favourable period for "
            "conception. Best ADs: Jupiter-Jupiter, Jupiter-Venus, Jupiter-Moon, Jupiter-Mars. "
            "The year of conception is likely when Jupiter transits Houses 1, 5, 9, or 11 "
            "from the natal Moon."
        ),
        "activation": "dasha_activated"
    },

    # ── HEALTH ──────────────────────────────────────────────────
    {
        "id": "health_lagna_lord_strong",
        "topic": "health",
        "condition": lambda ctx: _strong(_lord(1,ctx), ctx),
        "severity": "positive",
        "score": 3,
        "title": "Strong Lagna Lord — Constitutional Vitality & Resilience",
        "detail": lambda ctx: (
            f"CALCULATION: Lagna = {ctx['lagna_sign']}. Lagna lord = {_lord(1,ctx)}. "
            f"Dignity: {_dignity(_lord(1,ctx),ctx)}. House: {_house(_lord(1,ctx),ctx)}. "
            f"Shadbala: {_strength(_lord(1,ctx),ctx)}/100. Vimsopaka: {_vims(_lord(1,ctx),ctx)}/20. "
            + ("Combust — vitality is reduced despite strong dignity; Sun remedies advised." if _combust(_lord(1,ctx),ctx) else "")
            + f"\nINTERPRETATION: The Lagna lord is the primary indicator of physical constitution "
            "and immunity. A strong Lagna lord bestows robust vitality, rapid recovery from "
            "illness, and a resilient constitution throughout life. "
            + ("Jupiter aspects Lagna — double vitality protection." if _aspects_house("Jupiter",1,ctx) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "health_lagna_lord_weak",
        "topic": "health",
        "condition": lambda ctx: _weak(_lord(1,ctx), ctx) and not _nb(_lord(1,ctx), ctx),
        "severity": "warning",
        "score": -3,
        "title": "Debilitated Lagna Lord — Physical Vulnerability",
        "detail": lambda ctx: (
            f"CALCULATION: Lagna lord {_lord(1,ctx)} debilitated. "
            f"No Neechabhanga. Shadbala: {_strength(_lord(1,ctx),ctx)}/100. "
            f"\nINTERPRETATION: Debilitated Lagna lord (no cancellation) weakens baseline "
            "constitution and immune response. Regular health check-ups, stress management, "
            "and remedies for the Lagna lord planet are essential throughout life."
        ),
        "activation": "natal"
    },
    {
        "id": "health_saturn_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Saturn", ctx) in [6,8],
        "severity": "caution",
        "score": -2,
        "title": "Saturn in 6th/8th — Chronic or Long-Latency Health Concerns",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn in House {_house('Saturn',ctx)}. "
            f"Dignity: {_dignity('Saturn',ctx)}. Shadbala: {_strength('Saturn',ctx)}/100. "
            f"Retrograde: {_retro('Saturn',ctx)} "
            f"({'increases Saturn\'s depth here' if _retro('Saturn',ctx) else 'direct'}). "
            + (f"Neechabhanga applies: {'; '.join(_nb_conds('Saturn',ctx))}. " if _nb("Saturn",ctx) else "")
            + f"\nINTERPRETATION: "
            + ("Saturn in 6th: chronic conditions — joints, bones, teeth, skin, nervous system. "
               "Service-related stress is an occupational hazard." if _house("Saturn",ctx)==6
               else "Saturn in 8th: longevity but with chronic ailments — digestive issues, "
               "vata disorders (nerve pain, arthritis, malabsorption). "
               "Exalted Saturn in 8th grants exceptional longevity." if _dignity("Saturn",ctx)=="Exalted" else "")
            + " Regularity in diet, sleep, oil massage on Saturdays, and vata management are essential."
        ),
        "activation": "natal"
    },
    {
        "id": "health_mars_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Mars", ctx) in [6,8],
        "severity": "caution",
        "score": -1,
        "title": "Mars in 6th/8th — Inflammation, Accidents & Surgeries",
        "detail": lambda ctx: (
            f"CALCULATION: Mars in House {_house('Mars',ctx)}. "
            f"Dignity: {_dignity('Mars',ctx)}. Shadbala: {_strength('Mars',ctx)}/100. "
            + ("Neechabhanga applies. " if _nb("Mars",ctx) else "")
            + f"\nINTERPRETATION: "
            + ("Mars in 6th: energises immunity but causes inflammatory conditions, "
               "blood disorders, fevers, and accident-proneness." if _house("Mars",ctx)==6
               else "Mars in 8th: accident risk, surgeries, sudden health events involving "
               "blood, head, or reproductive system. Avoid extreme sports, especially during "
               "Mars Mahadasha.")
        ),
        "activation": "natal"
    },
    {
        "id": "health_moon_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Moon", ctx) in [6,8],
        "severity": "caution",
        "score": -2,
        "title": "Moon in 6th/8th — Mental & Digestive Health Priority",
        "detail": lambda ctx: (
            f"CALCULATION: Moon in House {_house('Moon',ctx)}. "
            f"Dignity: {_dignity('Moon',ctx)} (in {ctx['moon_sign']}). "
            f"Shadbala: {_strength('Moon',ctx)}/100. "
            f"\nINTERPRETATION: "
            + ("Moon in 6th: emotional instability, digestive disorders, lymphatic issues. "
               "Mindfulness and calm daily routine are essential." if _house("Moon",ctx)==6
               else "Moon in 8th: emotional vulnerability, psychosomatic conditions, "
               "hormonal irregularities. Psychological health is intimately tied to physical wellbeing.")
            + " Prioritise sleep hygiene, meditation, and limiting cold/raw foods."
        ),
        "activation": "natal"
    },
    {
        "id": "health_jupiter_trine_strong",
        "topic": "health",
        "condition": lambda ctx: _strong("Jupiter",ctx) and _house("Jupiter",ctx) in [1,5,9],
        "severity": "positive",
        "score": 3,
        "title": "Strong Jupiter in Trikona — Exceptional Health Protection",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter {_dignity('Jupiter',ctx)} in House {_house('Jupiter',ctx)} "
            f"(a trikona: 1/5/9). Shadbala: {_strength('Jupiter',ctx)}/100. "
            f"Vimsopaka: {_vims('Jupiter',ctx)}/20. "
            f"\nINTERPRETATION: This is one of the strongest health-protective yogas. Jupiter's "
            "life-force directly supports vitality, immune system, and longevity from a trikona "
            "position. The native often has remarkable recuperative power."
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
            f"CALCULATION: {ctx.get('sade_sati_detail','')}. "
            f"Phase: {ctx.get('sade_sati_phase','')}. "
            f"\nINTERPRETATION: Sade Sati lowers immunity, disturbs sleep, and creates digestive "
            "and joint discomfort. "
            + f"Phase details: {ctx.get('sade_sati_detail','')}. "
            + "Protective measures: moderate exercise, consistent sleep, oil massage on Saturdays, "
            "Shani Shanti puja, and chanting Shani mantras."
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
            f"CALCULATION: Saturn MD active. Saturn's natal position: "
            f"H{_house('Saturn',ctx)}, {_dignity('Saturn',ctx)}. "
            f"Current AD: {ctx.get('antardasha','')}. "
            f"\nINTERPRETATION: Saturn MD heightens attention needed for bones, joints, teeth, "
            "digestion, skin. Most sensitive sub-periods: Saturn-Rahu AD and Saturn-Ketu AD. "
            "Saturn-Jupiter AD is often the healthiest sub-period within Saturn MD."
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
            f"CALCULATION: {ctx.get('dasha','')} MD active. "
            f"Natal position: H{_house(ctx.get('dasha','Rahu'),ctx)}, "
            f"nakshatra: {_nak(ctx.get('dasha','Rahu'),ctx)}. "
            f"\nINTERPRETATION: "
            + ("Rahu MD: anxiety, stress, atypical infections, lifestyle excesses, "
               "hard-to-diagnose conditions." if ctx.get("dasha","")=="Rahu"
               else "Ketu MD: sudden health crises, surgical events, mysterious symptoms, "
               "spiritual health events.")
            + " Most critical sub-period: the AD of Saturn (Rahu-Saturn or Ketu-Saturn)."
        ),
        "activation": "dasha_activated"
    },

    # ── GENERAL YOGAS ────────────────────────────────────────────
    {
        "id": "yoga_hamsa",
        "topic": "general",
        "condition": lambda ctx: _strong("Jupiter",ctx) and _house("Jupiter",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Hamsa Yoga (Panchamahapurusha) — Divine Wisdom & Spiritual Fortune",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter is {_dignity('Jupiter',ctx)} in House {_house('Jupiter',ctx)} "
            f"(a kendra). For Hamsa Yoga: Jupiter must be in own/exalted sign in a kendra. "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. Vimsopaka: {_vims('Jupiter',ctx)}/20. "
            + ("Vargottama Jupiter — supreme strength." if _vargo("Jupiter",ctx) else "")
            + f"\nINTERPRETATION: Hamsa Yoga confers exceptional wisdom, noble character, "
            "spiritual inclination, and distinguished reputation. The native often becomes "
            "a teacher, healer, judge, or spiritual guide. Physical grace and nobility of "
            "conduct are characteristic."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_malavya",
        "topic": "general",
        "condition": lambda ctx: _strong("Venus",ctx) and _house("Venus",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Malavya Yoga (Panchamahapurusha) — Beauty, Prosperity & Pleasures",
        "detail": lambda ctx: (
            f"CALCULATION: Venus is {_dignity('Venus',ctx)} in House {_house('Venus',ctx)} "
            f"(a kendra). Shadbala: {_strength('Venus',ctx)}/100. Vimsopaka: {_vims('Venus',ctx)}/20. "
            + ("Vargottama Venus." if _vargo("Venus",ctx) else "")
            + f"\nINTERPRETATION: Malavya Yoga confers physical beauty, artistic talent, "
            "luxury, romantic success, and financial abundance. A life of refined pleasures, "
            "artistic recognition, and magnetic personal charm."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_ruchaka",
        "topic": "general",
        "condition": lambda ctx: _strong("Mars",ctx) and _house("Mars",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Ruchaka Yoga (Panchamahapurusha) — Courage, Command & Vitality",
        "detail": lambda ctx: (
            f"CALCULATION: Mars is {_dignity('Mars',ctx)} in House {_house('Mars',ctx)} (a kendra). "
            f"Shadbala: {_strength('Mars',ctx)}/100. "
            f"\nINTERPRETATION: Ruchaka Yoga confers exceptional physical vitality, courage, "
            "and competitive prowess. Military, athletic, surgical, or leadership roles are natural. "
            "The native commands respect through strength and boldness."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_bhadra",
        "topic": "general",
        "condition": lambda ctx: _strong("Mercury",ctx) and _house("Mercury",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Bhadra Yoga (Panchamahapurusha) — Intellect, Eloquence & Wealth",
        "detail": lambda ctx: (
            f"CALCULATION: Mercury is {_dignity('Mercury',ctx)} in House {_house('Mercury',ctx)} "
            f"(a kendra). Combust: {_combust('Mercury',ctx)}. Shadbala: {_strength('Mercury',ctx)}/100. "
            + ("NOTE: Mercury is combust — yoga is partially weakened." if _combust("Mercury",ctx) else "")
            + f"\nINTERPRETATION: Bhadra Yoga confers sharp intellect, exceptional communication, "
            "and business acumen. The native thrives in trade, writing, mathematics, technology, "
            "or any field requiring rapid information processing."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_shasha",
        "topic": "general",
        "condition": lambda ctx: _strong("Saturn",ctx) and _house("Saturn",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Shasha Yoga (Panchamahapurusha) — Authority, Discipline & Lasting Legacy",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn is {_dignity('Saturn',ctx)} in House {_house('Saturn',ctx)} "
            f"(a kendra). Retrograde: {_retro('Saturn',ctx)}. Shadbala: {_strength('Saturn',ctx)}/100. "
            + ("Retrograde Saturn in kendra — especially powerful for discipline and longevity." if _retro("Saturn",ctx) else "")
            + f"\nINTERPRETATION: Shasha Yoga confers iron discipline, authority over masses, "
            "and a lasting professional legacy. Mines, land, real estate, or large institutions "
            "figure prominently. The native rises late but achieves enduring prominence."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_gajkesari",
        "topic": "general",
        "condition": lambda ctx: (
            ((_house("Jupiter",ctx) - _house("Moon",ctx)) % 12) in [0,3,6,9]
        ),
        "severity": "positive",
        "score": 4,
        "title": "Gaja-Kesari Yoga — Fame, Wisdom & Respected Standing",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter in House {_house('Jupiter',ctx)}, "
            f"Moon in House {_house('Moon',ctx)}. "
            f"Separation = {(_house('Jupiter',ctx)-_house('Moon',ctx))%12} houses "
            f"(kendra = 0,3,6,9 ✓). "
            f"Jupiter dignity: {_dignity('Jupiter',ctx)}. Moon dignity: {_dignity('Moon',ctx)}. "
            f"\nINTERPRETATION: Gaja-Kesari Yoga — formed when Jupiter is in kendra from Moon. "
            "Confers fame, wealth, eloquence, wisdom, and respected position in society. "
            "Potency depends on both planets' dignity. "
            + ("Maximum potency — both planets strong." if _strong("Jupiter",ctx) and _strong("Moon",ctx) else
               "Good potency — at least one planet strong." if _strong("Jupiter",ctx) or _strong("Moon",ctx) else
               "Moderate potency — both planets in average dignity.")
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_parivartana",
        "topic": "general",
        "condition": lambda ctx: len(ctx.get("parivartana_pairs",[])) > 0,
        "severity": "positive",
        "score": 3,
        "title": "Parivartana Yoga — Mutual Exchange of Power",
        "detail": lambda ctx: (
            f"CALCULATION: Parivartana pairs found: {ctx.get('parivartana_pairs',[])}. "
            f"Logic: Planet A is in Planet B's sign, AND Planet B is in Planet A's sign. "
            f"Each planet effectively acquires own-sign status through the exchange. "
            f"\nINTERPRETATION: Parivartana creates a powerful link between two houses. The "
            "results of both houses intermingle and support each other. This is particularly "
            "powerful when involving kendras (1,4,7,10) and trikonas (1,5,9). "
            + ("Involves trikona/kendra lords — Rajayoga-class." if any(
                h in [1,4,5,7,9,10] for pair in ctx.get("parivartana_pairs",[])
                for p in pair
                for h in [ctx["house_map"].get(p,0)]
            ) else "")
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_vargottama",
        "topic": "general",
        "condition": lambda ctx: any(ctx.get("vargottama",{}).values()),
        "severity": "positive",
        "score": 2,
        "title": "Vargottama Planet(s) — Strength Locked Across D1 and D9",
        "detail": lambda ctx: (
            f"CALCULATION: Vargottama planets = "
            f"{[p for p,v in ctx.get('vargottama',{}).items() if v]}. "
            f"A planet is Vargottama when it occupies the same sign in D1 (natal) and D9 (Navamsa). "
            f"This means its natural qualities are reinforced across the primary and harmonic chart. "
            f"\nINTERPRETATION: Vargottama planets have a stabilising, deepening quality. They "
            "deliver their significations more fully and with greater consistency. A Vargottama "
            "planet often shows karmic destiny attached to its domain."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_graha_yuddha",
        "topic": "general",
        "condition": lambda ctx: len(ctx.get("graha_yuddha",[])) > 0,
        "severity": "caution",
        "score": -1,
        "title": "Graha Yuddha (Planetary War) — One Planet Eclipsed",
        "detail": lambda ctx: (
            f"CALCULATION: Planetary war detected. "
            f"{[g['winner'] + " wins over " + g['loser'] for g in ctx.get('graha_yuddha',[])]}. "
            f"Logic: {[g['logic'] for g in ctx.get('graha_yuddha',[])]}. "
            f"In Graha Yuddha, the losing planet's significations are weakened for the lifetime. "
            f"\nINTERPRETATION: The loser in a planetary war loses signification strength. "
            "Examine which houses and karakas the losing planet rules to understand the "
            "life domain most affected. The winning planet's domains are correspondingly amplified."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_neechabhanga_raja",
        "topic": "general",
        "condition": lambda ctx: any(
            _nb(p,ctx) for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
        ),
        "severity": "positive",
        "score": 3,
        "title": "Neechabhanga Raja Yoga — Adversity Transformed into Royalty",
        "detail": lambda ctx: (
            f"CALCULATION: Planets with Neechabhanga: "
            f"{[p for p in ['Sun','Moon','Mars','Mercury','Jupiter','Venus','Saturn'] if _nb(p,ctx)]}. "
            + "\n".join([
                f"  {p}: {'; '.join(_nb_conds(p,ctx))}"
                for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
                if _nb(p,ctx)
            ])
            + f"\nINTERPRETATION: Classical texts consider Neechabhanga itself a Raja Yoga — "
            "the cancellation produces exceptional strength, often exceeding what a "
            "straightforwardly exalted planet gives. The native rises after adversity "
            "to achieve lasting success in the domain of the cancelled debilitation."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_viparita_harsha",
        "topic": "general",
        "condition": lambda ctx: _lord(6,ctx) != "" and _house(_lord(6,ctx),ctx) in [6,8,12],
        "severity": "positive",
        "score": 3,
        "title": "Viparita Harsha Yoga — Victory Born from Adversity",
        "detail": lambda ctx: (
            f"CALCULATION: 6th lord = {_lord(6,ctx)}, placed in House {_house(_lord(6,ctx),ctx)} "
            f"(a dusthana: 6,8,12). Dignity: {_dignity(_lord(6,ctx),ctx)}. "
            f"\nINTERPRETATION: Viparita Harsha Yoga — enemies are defeated by their own actions. "
            "The malefic energy of the 6th lord is contained in a dusthana, weakening it. "
            "Challenges become stepping stones; adversities ultimately strengthen. The native "
            "excels in competitive environments and court/legal battles."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_gajkesari_strong",
        "topic": "general",
        "condition": lambda ctx: _strong("Jupiter",ctx) and _strong("Moon",ctx) and
                                  ((_house("Jupiter",ctx)-_house("Moon",ctx))%12) in [0,3,6,9],
        "severity": "positive",
        "score": 4,
        "title": "Strong Gaja-Kesari Yoga — Maximum Potency Fame & Wisdom",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter {_dignity('Jupiter',ctx)} (H{_house('Jupiter',ctx)}) and "
            f"Moon {_dignity('Moon',ctx)} (H{_house('Moon',ctx)}) — both strong AND in kendra. "
            f"\nINTERPRETATION: This is the maximum expression of Gaja-Kesari. Both the planet "
            "of wisdom and the planet of mind/public are fully empowered. Fame, prosperity, and "
            "community leadership of an exceptional order are indicated. The native becomes "
            "a landmark personality in their field."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_kahala",
        "topic": "general",
        "condition": lambda ctx: (
            _strong(ctx["lord_map"].get(4,""), ctx) and
            _strong(ctx["lord_map"].get(9,""), ctx)
        ),
        "severity": "positive",
        "score": 3,
        "title": "Kahala Yoga — Courage, Authority & Commanding Presence",
        "detail": lambda ctx: (
            f"CALCULATION: 4th lord = {_lord(4,ctx)} ({_dignity(_lord(4,ctx),ctx)}, "
            f"H{_house(_lord(4,ctx),ctx)}). "
            f"9th lord = {_lord(9,ctx)} ({_dignity(_lord(9,ctx),ctx)}, "
            f"H{_house(_lord(9,ctx),ctx)}). Both are strong. "
            f"\nINTERPRETATION: Kahala Yoga combines emotional roots (4th: home, vehicle, mother) "
            "and fortune (9th: dharma, luck) with full strength. Boldness, leadership, a "
            "commanding presence, and the ability to marshal resources toward ambitious goals "
            "are hallmarks. The native often leads institutions or commands armies."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_atmakaraka_strong",
        "topic": "general",
        "condition": lambda ctx: _strong(ctx.get("atmakaraka",""),ctx) if ctx.get("atmakaraka") else False,
        "severity": "positive",
        "score": 3,
        "title": "Strong Atmakaraka — Soul's Purpose Actively Supported",
        "detail": lambda ctx: (
            f"CALCULATION: Atmakaraka = {ctx.get('atmakaraka','')} "
            f"(highest degree in sign among all planets, excluding Ketu). "
            f"Dignity: {_dignity(ctx.get('atmakaraka',''),ctx)}. "
            f"House: {_house(ctx.get('atmakaraka',''),ctx)}. "
            f"Shadbala: {_strength(ctx.get('atmakaraka',''),ctx)}/100. "
            f"\nINTERPRETATION: A strong Atmakaraka indicates the soul's primary purpose is "
            "actively supported by destiny. House "
            f"{_house(ctx.get('atmakaraka',''),ctx)} becomes a key zone of soul-level meaning "
            "and fulfillment. Events in this domain carry the deepest personal significance."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_sunapha",
        "topic": "general",
        "condition": lambda ctx: any(
            p not in ("Sun","Rahu","Ketu") and
            ctx["house_map"].get(p,0) == (ctx["house_map"].get("Moon",1) % 12) + 1
            for p in ctx["planets"] if p != "Moon"
        ),
        "severity": "positive",
        "score": 2,
        "title": "Sunapha Yoga — Wealth & Self-Made Prosperity",
        "detail": lambda ctx: (
            f"CALCULATION: Planet(s) in 2nd house from Moon "
            f"(Moon in H{_house('Moon',ctx)}, so 2nd from Moon = "
            f"H{(_house('Moon',ctx)%12)+1}): "
            f"{[p for p in ctx['planets'] if p not in ('Sun','Rahu','Ketu','Moon') and ctx['house_map'].get(p,0)==(_house('Moon',ctx)%12)+1]}. "
            f"\nINTERPRETATION: Sunapha Yoga confers self-made wealth, intelligence, and the "
            "ability to attract resources through one's own effort. The native is respected "
            "and admired in their community. Financial independence is characteristic."
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
            fired = False
            if debug:
                print(f"Rule {rule['id']} condition error: {e}")
        if fired:
            try:
                detail = rule["detail"](ctx)
            except Exception as e:
                detail = f"[Detail unavailable: {e}]"
            results.append({
                "id":         rule["id"],
                "topic":      rule["topic"],
                "severity":   rule["severity"],
                "score":      rule["score"],
                "title":      rule["title"],
                "detail":     detail,
                "activation": rule.get("activation","natal"),
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
            "Exceptional" if total >= 12 else
            "Excellent"   if total >= 8  else
            "Good"        if total >= 4  else
            "Mixed"       if total >= 0  else
            "Challenging" if total >= -5 else
            "Difficult"
        )
    }


def _apply_dasha_boost(fired_rules, topic_lord, md_planet, related_planets=None):
    related  = set([topic_lord] + (related_planets or []))
    result   = []
    boosted  = set()
    for r in fired_rules:
        rc = copy.deepcopy(r)
        if rc.get("activation") == "dasha_activated" and md_planet in related and rc["id"] not in boosted:
            old = rc["score"]
            rc["score"] = round(old * 1.5) if old > 0 else round(old * 1.2)
            rc["title"] += " [⚡ DASHA ACTIVATED]"
            rc["detail"] += "\n  ⚡ Amplified: the running Mahadasha planet directly governs this life area."
            boosted.add(rc["id"])
        result.append(rc)
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
    return "\n\n".join(parts) if parts else "No significant indicators found for this topic."


def _house_career_meaning(house: int) -> str:
    meanings = {
        1:  "In H1 (Lagna) — career identity merges with self; you ARE your work. Strong visibility.",
        2:  "In H2 — career energy channels into wealth and family legacy. Finance/resource roles.",
        3:  "In H3 — career thrives through communication, media, entrepreneurship, or siblings.",
        4:  "In H4 — career connected to home, real estate, psychology, or emotional domains.",
        5:  "In H5 — career infused with creativity, intelligence, or speculative enterprise.",
        6:  "In H6 — career involves service, health, law, competition, or overcoming enemies.",
        7:  "In H7 — career through partnerships, public dealing, or foreign connections.",
        8:  "In H8 — research, occult, insurance, or transformation-related professions.",
        9:  "In H9 (ideal) — 10th lord in 9th creates Dharma-Karma connection; fortune supports career.",
        10: "In H10 (best) — 10th lord in 10th: self-contained, maximising career power.",
        11: "In H11 — career oriented toward networks, gains, and elder sibling/friend connections.",
        12: "In H12 — career in foreign lands, hospitals, ashrams, or behind-the-scenes roles.",
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
    md_planet  = dasha_info.get("mahadasha","")
    fired      = _apply_dasha_boost(fired, tenth_lord, md_planet,
                                    related_planets=["Sun","Saturn","Mercury","Jupiter","Rahu"])

    summary      = score_topic(fired)
    tenth_sign   = ZODIAC[(lagna_idx + 9) % 12]
    planets_10th = [p for p, h in ctx["house_map"].items() if h == 10]
    aspects_10th = ctx["aspect_map"].get(10, [])

    return {
        "rating":             summary["rating"],
        "net_score":          summary["net_score"],
        "tenth_sign":         tenth_sign,
        "tenth_lord":         tenth_lord,
        "tenth_lord_dignity": chart.dignities.get(tenth_lord,"Neutral"),
        "tenth_lord_house":   ctx["house_map"].get(tenth_lord, 0),
        "planets_in_10th":    planets_10th,
        "aspects_on_10th":    aspects_10th,
        "atmakaraka":         chart.atmakaraka,
        "amatyakaraka":       chart.amatyakaraka,
        "current_dasha":      dasha_info,
        "fired_rules":        fired,
        "narrative":          _narrative_block(fired),
        "summary": (
            f"Career: {summary['rating']} (score {summary['net_score']:+d}). "
            f"10th lord {tenth_lord} is {chart.dignities.get(tenth_lord,'Neutral')} "
            f"in House {ctx['house_map'].get(tenth_lord,0)}. "
            f"Planets in 10th: {planets_10th or 'none'}. "
            f"{summary['positive_count']} strengths, {summary['warning_count']} cautions."
        )
    }


def analyze_marriage(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info  = chart.get_current_dasha_info(check_date)
    ctx         = build_context(chart, dasha_info)
    fired       = evaluate_rules(ctx, topic="marriage")

    lagna_idx   = ZODIAC.index(chart.lagna_sign)
    seventh_lord= SIGN_LORD[ZODIAC[(lagna_idx + 6) % 12]]
    md_planet   = dasha_info.get("mahadasha","")
    fired       = _apply_dasha_boost(fired, seventh_lord, md_planet,
                                     related_planets=["Venus","Jupiter","Moon"])

    summary = score_topic(fired)
    venus_sign = longitude_to_sign(chart.planets["Venus"])[0]

    return {
        "rating":               summary["rating"],
        "net_score":            summary["net_score"],
        "seventh_sign":         ZODIAC[(lagna_idx + 6) % 12],
        "seventh_lord":         seventh_lord,
        "seventh_lord_dignity": chart.dignities.get(seventh_lord,"Neutral"),
        "seventh_lord_house":   ctx["house_map"].get(seventh_lord, 0),
        "planets_in_7th":       [p for p,h in ctx["house_map"].items() if h==7],
        "aspects_on_7th":       ctx["aspect_map"].get(7,[]),
        "venus_house":          ctx["house_map"].get("Venus",0),
        "venus_sign":           venus_sign,
        "venus_dignity":        chart.dignities.get("Venus","Neutral"),
        "venus_navamsa":        chart.navamsa_dignities.get("Venus","Neutral"),
        "current_dasha":        dasha_info,
        "fired_rules":          fired,
        "narrative":            _narrative_block(fired),
        "summary": (
            f"Marriage: {summary['rating']} (score {summary['net_score']:+d}). "
            f"7th lord {seventh_lord} is {chart.dignities.get(seventh_lord,'Neutral')}. "
            f"Venus: {chart.dignities.get('Venus','Neutral')} ({venus_sign}). "
            f"D9 Venus: {chart.navamsa_dignities.get('Venus','?')}."
        )
    }


def analyze_children(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="children")

    lagna_idx  = ZODIAC.index(chart.lagna_sign)
    fifth_lord = SIGN_LORD[ZODIAC[(lagna_idx + 4) % 12]]
    md_planet  = dasha_info.get("mahadasha","")
    fired      = _apply_dasha_boost(fired, fifth_lord, md_planet,
                                    related_planets=["Jupiter","Venus","Moon"])

    summary = score_topic(fired)
    return {
        "rating":            summary["rating"],
        "net_score":         summary["net_score"],
        "fifth_sign":        ZODIAC[(lagna_idx + 4) % 12],
        "fifth_lord":        fifth_lord,
        "fifth_lord_dignity":chart.dignities.get(fifth_lord,"Neutral"),
        "fifth_lord_house":  ctx["house_map"].get(fifth_lord,0),
        "planets_in_5th":    [p for p,h in ctx["house_map"].items() if h==5],
        "aspects_on_5th":    ctx["aspect_map"].get(5,[]),
        "jupiter_house":     ctx["house_map"].get("Jupiter",0),
        "jupiter_dignity":   chart.dignities.get("Jupiter","Neutral"),
        "current_dasha":     dasha_info,
        "fired_rules":       fired,
        "narrative":         _narrative_block(fired),
        "summary": (
            f"Children: {summary['rating']} (score {summary['net_score']:+d}). "
            f"5th lord {fifth_lord} is {chart.dignities.get(fifth_lord,'Neutral')}. "
            f"Jupiter (Putrakaraka): {chart.dignities.get('Jupiter','Neutral')} "
            f"in H{ctx['house_map'].get('Jupiter',0)}."
        )
    }


def analyze_health(chart: ChartData, check_date: datetime = None,
                   transit_saturn_sign: str = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    sade_sati  = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")
    kantaka    = check_kantaka_shani(chart.moon_sign, transit_saturn_sign or "")
    ctx        = build_context(chart, dasha_info, sade_sati)
    fired      = evaluate_rules(ctx, topic="health")

    lagna_lord = SIGN_LORD[chart.lagna_sign]
    md_planet  = dasha_info.get("mahadasha","")
    fired      = _apply_dasha_boost(fired, lagna_lord, md_planet,
                                    related_planets=["Sun","Jupiter","Mars"])

    summary = score_topic(fired)
    return {
        "rating":             summary["rating"],
        "net_score":          summary["net_score"],
        "lagna_lord":         lagna_lord,
        "lagna_lord_dignity": chart.dignities.get(lagna_lord,"Neutral"),
        "planets_in_1st":     [p for p,h in chart.house_map.items() if h==1],
        "planets_in_6th":     [p for p,h in chart.house_map.items() if h==6],
        "planets_in_8th":     [p for p,h in chart.house_map.items() if h==8],
        "planets_in_12th":    [p for p,h in chart.house_map.items() if h==12],
        "sade_sati":          sade_sati,
        "kantaka_shani":      kantaka,
        "current_dasha":      dasha_info,
        "fired_rules":        fired,
        "narrative":          _narrative_block(fired),
        "summary": (
            f"Health: {summary['rating']} (score {summary['net_score']:+d}). "
            f"Lagna lord {lagna_lord} is {chart.dignities.get(lagna_lord,'Neutral')}. "
            f"Sade Sati: {'Active — ' + sade_sati['phase'] if sade_sati['active'] else 'Not active'}. "
            f"Kantaka Shani: {'Yes — ' + kantaka.get('position','') if kantaka.get('active') else 'No'}."
        )
    }


def analyze_general_yogas(chart: ChartData) -> Dict:
    dasha_info = chart.get_current_dasha_info()
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="general")
    total_yoga_score = sum(r["score"] for r in fired)
    return {
        "yoga_count":       len(fired),
        "total_yoga_score": total_yoga_score,
        "yoga_strength": (
            "Exceptional" if total_yoga_score >= 20 else
            "Strong"      if total_yoga_score >= 12 else
            "Moderate"    if total_yoga_score >= 5  else
            "Weak"
        ),
        "fired_yogas":  fired,
        "narrative":    _narrative_block(fired),
        "atmakaraka":   chart.atmakaraka,
        "amatyakaraka": chart.amatyakaraka,
    }


# ==================================================================
# SECTION 10 — ASHTAKOOTA MATCHMAKING
# ==================================================================

def get_tara_score(ni1: int, ni2: int) -> int:
    d12 = ((ni2 - ni1) % 27) % 9 + 1
    d21 = ((ni1 - ni2) % 27) % 9 + 1
    return math.floor((TARA_SCORES[d12] + TARA_SCORES[d21]) / 2)


def get_yoni_score(y1: str, y2: str) -> int:
    hostile_pairs = {
        frozenset({"Horse","Buffalo"}), frozenset({"Elephant","Lion"}),
        frozenset({"Sheep","Monkey"}),  frozenset({"Serpent","Mongoose"}),
        frozenset({"Dog","Deer"}),      frozenset({"Cat","Rat"}),
        frozenset({"Cow","Tiger"}),
    }
    if y1 == y2:
        return 4
    if frozenset({y1, y2}) in hostile_pairs:
        return 0
    return 2


def get_graha_maitri_score(lord1: str, lord2: str) -> int:
    if lord1 == lord2:
        return 5
    l1_friends = PLANET_FRIENDS.get(lord1, [])
    l2_friends = PLANET_FRIENDS.get(lord2, [])
    l1_enemies = PLANET_ENEMIES.get(lord1, [])
    l2_enemies = PLANET_ENEMIES.get(lord2, [])
    if lord2 in l1_friends and lord1 in l2_friends:
        return 4
    if lord2 in l1_friends or lord1 in l2_friends:
        return 3
    if lord2 in l1_enemies and lord1 in l2_enemies:
        return 0
    if lord2 in l1_enemies or lord1 in l2_enemies:
        return 1
    return 2


def get_gana_score(g1: str, g2: str) -> int:
    if g1 == g2:
        return 6
    if {g1, g2} == {"Deva", "Manushya"}:
        return 5
    return 0


def get_bhakoot_score(idx1: int, idx2: int) -> int:
    diff = (idx2 - idx1) % 12
    if diff in [1, 5, 7, 11]:
        return 0
    return 7


def calculate_ashtakoota(c1: ChartData, c2: ChartData,
                          person1_is_groom: bool = True) -> Dict:
    m1, m2  = c1.moon_sign, c2.moon_sign
    n1 = c1.nakshatras["Moon"]["nakshatra"]
    n2 = c2.nakshatras["Moon"]["nakshatra"]
    i1, i2  = ZODIAC.index(m1), ZODIAC.index(m2)
    ni1,ni2 = NAKSHATRAS.index(n1), NAKSHATRAS.index(n2)

    varna1 = VARNA_MAP[SIGN_ELEMENT[m1]]
    varna2 = VARNA_MAP[SIGN_ELEMENT[m2]]
    varna_order = {"Brahmin":1,"Kshatriya":2,"Vaishya":3,"Shudra":4}
    varna = 1 if (varna_order[varna1] <= varna_order[varna2] if person1_is_groom
                  else varna_order[varna2] <= varna_order[varna1]) else 0

    vashya1 = VASHYA_MAP[m1]
    vashya2 = VASHYA_MAP[m2]
    vashya  = (2 if vashya1 == vashya2
               else 1 if (
                   (vashya1=="Human" and vashya2 in ["Water","Quadruped"]) or
                   (vashya2=="Human" and vashya1 in ["Water","Quadruped"])
               ) else 0)

    tara    = get_tara_score(ni1, ni2)
    yoni    = get_yoni_score(NAKSHATRA_YONI[n1], NAKSHATRA_YONI[n2])
    graha   = get_graha_maitri_score(SIGN_LORD[m1], SIGN_LORD[m2])
    gana    = get_gana_score(NAKSHATRA_GANA[n1], NAKSHATRA_GANA[n2])
    bhakoot = get_bhakoot_score(i1, i2)
    nadi    = 0 if NAKSHATRA_NADI[n1] == NAKSHATRA_NADI[n2] else 8
    total   = varna + vashya + tara + yoni + graha + gana + bhakoot + nadi

    doshas = []
    if nadi == 0:
        doshas.append(f"Nadi Dosha — both have {NAKSHATRA_NADI[n1]} Nadi. Most serious "
                      "compatibility flaw; can cause health issues in offspring and marital discord.")
    if bhakoot == 0:
        diff = (i2 - i1) % 12
        axis = "6/8" if diff in [5,7] else "2/12"
        doshas.append(f"Bhakoot Dosha ({axis} axis) — financial stress or emotional distance; "
                      "remediable through ritual and full chart analysis.")

    return {
        "varna":        {"score":varna,   "max":1,  "detail":f"{varna1} vs {varna2}"},
        "vashya":       {"score":vashya,  "max":2,  "detail":f"{vashya1} vs {vashya2}"},
        "tara":         {"score":tara,    "max":3,  "detail":f"{n1} vs {n2}"},
        "yoni":         {"score":yoni,    "max":4,  "detail":f"{NAKSHATRA_YONI[n1]} vs {NAKSHATRA_YONI[n2]}"},
        "graha_maitri": {"score":graha,   "max":5,  "detail":f"{SIGN_LORD[m1]} vs {SIGN_LORD[m2]}"},
        "gana":         {"score":gana,    "max":6,  "detail":f"{NAKSHATRA_GANA[n1]} vs {NAKSHATRA_GANA[n2]}"},
        "bhakoot":      {"score":bhakoot, "max":7,  "detail":f"{m1} ({n1}) vs {m2} ({n2})"},
        "nadi":         {"score":nadi,    "max":8,  "detail":f"{NAKSHATRA_NADI[n1]} vs {NAKSHATRA_NADI[n2]}"},
        "total": total, "max_total":36,
        "percentage":   round(total/36*100, 1),
        "verdict": (
            "Excellent"   if total >= 31 else
            "Good"        if total >= 25 else
            "Average"     if total >= 18 else
            "Challenging"
        ),
        "doshas": doshas,
        "dosha_summary": (
            "No major doshas detected." if not doshas else
            f"{len(doshas)} dosha(s): " + "; ".join(d.split("—")[0].strip() for d in doshas)
        )
    }


# ==================================================================
# SECTION 11 — VARSHPHAL (SOLAR RETURN) — ELABORATED
# ==================================================================

def calculate_varshphal(chart: ChartData, year: int,
                         transit_planets: Dict[str, float] = None) -> Dict:
    """
    Full Varshphal calculation:
    - Muntha position and lord
    - Varsha Lagna (Solar Return Ascendant)
    - Year lord (Varshesha)
    - Tri-Pataki Chakra positions
    - Detailed themes for the year
    """
    if not chart.birth_date:
        return {}

    years_elapsed = year - chart.birth_date.year

    # Muntha: moves 1 sign per year from birth Lagna
    muntha_lon  = (chart.ascendant + years_elapsed * 30) % 360
    muntha_sign, muntha_deg = longitude_to_sign(muntha_lon)
    muntha_lord = SIGN_LORD[muntha_sign]
    lagna_idx   = ZODIAC.index(chart.lagna_sign)
    muntha_idx  = ZODIAC.index(muntha_sign)
    muntha_house= ((muntha_idx - lagna_idx) % 12) + 1

    # Muntha lord dignity and house (using natal chart for lord's placement)
    muntha_lord_dignity = chart.dignities.get(muntha_lord, "Neutral")
    muntha_lord_house   = chart.house_map.get(muntha_lord, 0)

    # Tri-Pataki Chakra: three key muntha points
    # Udaya (rising) = Muntha itself, Madhya (peak) = 4 signs ahead, Asta (setting) = 7 signs ahead
    tri_pataki = {
        "udaya_muntha":  muntha_sign,
        "madhya_muntha": ZODIAC[(muntha_idx + 3) % 12],
        "asta_muntha":   ZODIAC[(muntha_idx + 6) % 12],
    }

    # Varsha Lagna (Solar Return Ascendant) — approximate if no ephemeris
    # We use the natal lagna progressed by year fraction
    varsha_lagna_sign = ZODIAC[(lagna_idx + years_elapsed % 12) % 12]
    varsha_lagna_lord = SIGN_LORD[varsha_lagna_sign]
    varsha_lagna_lord_dignity = chart.dignities.get(varsha_lagna_lord, "Neutral")

    # Varshesha (Year Lord) — the planet that lords the day of the Solar Return
    # Approximation: lord of the weekday of the solar return date
    # Solar return ≈ birth month/day in the given year
    try:
        sr_date = datetime(year, chart.birth_date.month, chart.birth_date.day)
        weekday = sr_date.weekday()  # 0=Mon, 6=Sun
        day_lords = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]
        varshesha = day_lords[weekday]
    except Exception:
        varshesha = "Sun"

    varshesha_dignity = chart.dignities.get(varshesha, "Neutral")
    varshesha_house   = chart.house_map.get(varshesha, 0)

    # Build detailed themes
    themes = _varshphal_themes_v2(
        chart, muntha_sign, muntha_house, muntha_lord,
        muntha_lord_dignity, muntha_lord_house,
        varsha_lagna_sign, varsha_lagna_lord, varsha_lagna_lord_dignity,
        varshesha, varshesha_dignity, varshesha_house,
        tri_pataki, transit_planets
    )

    return {
        "year":                   year,
        "years_elapsed":          years_elapsed,
        "muntha_sign":            muntha_sign,
        "muntha_degree":          round(muntha_deg, 2),
        "muntha_house":           muntha_house,
        "muntha_lord":            muntha_lord,
        "muntha_lord_dignity":    muntha_lord_dignity,
        "muntha_lord_house":      muntha_lord_house,
        "tri_pataki":             tri_pataki,
        "varsha_lagna_sign":      varsha_lagna_sign,
        "varsha_lagna_lord":      varsha_lagna_lord,
        "varsha_lagna_lord_dignity": varsha_lagna_lord_dignity,
        "varshesha":              varshesha,
        "varshesha_dignity":      varshesha_dignity,
        "varshesha_house":        varshesha_house,
        "themes":                 themes,
    }


def _varshphal_themes_v2(
    chart, muntha_sign, muntha_house, muntha_lord,
    muntha_lord_dignity, muntha_lord_house,
    varsha_lagna_sign, varsha_lagna_lord, varsha_lagna_lord_dignity,
    varshesha, varshesha_dignity, varshesha_house,
    tri_pataki, transit_planets
) -> List[Dict]:
    """
    Generate richly detailed, logically explained Varshphal themes.
    Each theme includes: calculation basis, classical rule, and interpretation.
    """
    themes = []

    # --- Theme 1: Muntha ---
    muntha_nature = {
        1:  ("Auspicious", "Muntha in 1st — direct on Lagna. A year of major personal initiative, new identity, and fresh beginnings. You stand at the centre of events."),
        2:  ("Auspicious", "Muntha in 2nd — focus on wealth, speech, and family. Financial accumulation efforts pay off. Family events are prominent."),
        3:  ("Moderate",   "Muntha in 3rd — courage, communication, and sibling matters dominate. Short travels, new skills, media work, or writing projects."),
        4:  ("Auspicious", "Muntha in 4th (kendra) — home, property, mother, vehicle, and inner peace are highlighted. Real estate decisions are favoured."),
        5:  ("Auspicious", "Muntha in 5th — creativity, children, love, and speculative ventures come to the fore. Educational breakthroughs likely."),
        6:  ("Challenging","Muntha in 6th — a year of service, health focus, and competition. Overcoming enemies and debt. Discipline in health is paramount."),
        7:  ("Auspicious", "Muntha in 7th (kendra) — partnership, marriage, and public dealings dominate. Business ventures and relational events peak."),
        8:  ("Challenging","Muntha in 8th — transformation, hidden matters, sudden changes. Research, occult, or inheritance events. Health monitoring essential."),
        9:  ("Auspicious", "Muntha in 9th (trikona) — fortune, dharma, long travel, father, and higher learning are blessed. A year of spiritual growth and good luck."),
        10: ("Auspicious", "Muntha in 10th (kendra) — career, public image, and authority are at peak focus. Major professional milestones are likely."),
        11: ("Auspicious", "Muntha in 11th — gains, social networks, ambitions fulfilled. Elder siblings and friends play a key role."),
        12: ("Challenging","Muntha in 12th — expenses, isolation, foreign connections, or inner retreat. Spiritual practice deeply rewarding."),
    }
    nature, desc = muntha_nature.get(muntha_house, ("Moderate",""))
    themes.append({
        "category":  "Muntha (Annual Ascendant Marker)",
        "nature":    nature,
        "calculation": (
            f"Muntha moves 1 sign per year from the natal Lagna. "
            f"Birth Lagna: {chart.lagna_sign} (sign {ZODIAC.index(chart.lagna_sign)+1}). "
            f"Years elapsed: {chart.birth_date.year} → {chart.birth_date.year + (chart.birth_date.year - chart.birth_date.year)} "
            f"({len([1 for _ in range(1)])} year cycles). "
            f"Muntha = sign {(ZODIAC.index(chart.lagna_sign) + (datetime.now().year - chart.birth_date.year)) % 12 + 1} = {muntha_sign}, "
            f"at {round(muntha_house, 1)}° of that sign, falling in House {muntha_house} of the natal chart."
        ),
        "classical_rule": "Muntha is to Varshphal what the Lagna is to the natal chart. Its house position determines the primary domain of the year's events.",
        "interpretation": desc,
        "modifier": (
            f"Muntha lord {muntha_lord} is {muntha_lord_dignity} in House {muntha_lord_house}. "
            + ("A strong Muntha lord powerfully supports the year's themes and resolves challenges quickly." if muntha_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "A debilitated Muntha lord weakens the year's results; remedies for this planet are essential this year." if muntha_lord_dignity == "Debilitated"
               else "A moderate Muntha lord delivers mixed results — sustained effort is required.")
        )
    })

    # --- Theme 2: Tri-Pataki Chakra ---
    themes.append({
        "category":  "Tri-Pataki Chakra (Three-Phase Muntha Wheel)",
        "nature":    "Neutral",
        "calculation": (
            f"Tri-Pataki divides the year into three 4-month phases. "
            f"Phase 1 (Udaya/Rising, months 1-4): Muntha in {tri_pataki['udaya_muntha']}. "
            f"Phase 2 (Madhya/Peak, months 5-8): Muntha progresses to 4th sign = {tri_pataki['madhya_muntha']}. "
            f"Phase 3 (Asta/Setting, months 9-12): Muntha at 7th sign = {tri_pataki['asta_muntha']}."
        ),
        "classical_rule": "The Tri-Pataki Chakra is used to time events within the Solar Return year, showing which domains become active in each trimester.",
        "interpretation": (
            f"Rising Phase ({tri_pataki['udaya_muntha']}): initial themes and first-quarter events. "
            f"Peak Phase ({tri_pataki['madhya_muntha']}): intensification, mid-year turning points. "
            f"Setting Phase ({tri_pataki['asta_muntha']}): consolidation, closure, and preparation for the next cycle."
        ),
        "modifier": f"Lord of {tri_pataki['udaya_muntha']}: {SIGN_LORD[tri_pataki['udaya_muntha']]} | Lord of {tri_pataki['madhya_muntha']}: {SIGN_LORD[tri_pataki['madhya_muntha']]} | Lord of {tri_pataki['asta_muntha']}: {SIGN_LORD[tri_pataki['asta_muntha']]}"
    })

    # --- Theme 3: Varsha Lagna ---
    themes.append({
        "category":  "Varsha Lagna (Solar Return Ascendant)",
        "nature":    "Auspicious" if varsha_lagna_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"] else "Moderate",
        "calculation": (
            f"Varsha Lagna = {varsha_lagna_sign} (Lagna progressed by annual cycle). "
            f"Lord: {varsha_lagna_lord}. Lord's natal dignity: {varsha_lagna_lord_dignity}. "
            f"Lord's natal house: {chart.house_map.get(varsha_lagna_lord, 0)}."
        ),
        "classical_rule": "The Varsha Lagna (Solar Return Ascendant) sets the overall tone and body-mind-personality energy for the entire year. Its lord is the primary significator of the year's overall experience.",
        "interpretation": (
            f"The year is coloured by {varsha_lagna_sign}'s energy ({SIGN_ELEMENT[varsha_lagna_sign]} element, "
            f"{SIGN_QUALITY[varsha_lagna_sign]} quality). "
            f"Lord {varsha_lagna_lord} is {varsha_lagna_lord_dignity} — "
            + ("the year's overall energy is powerfully supported; personal vitality and opportunities flow freely." if varsha_lagna_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "the year's energy is somewhat depleted; discipline and remedies recommended for the Varsha Lagna lord." if varsha_lagna_lord_dignity == "Debilitated"
               else "moderate support; sustained effort unlocks the year's potential.")
        ),
        "modifier": ""
    })

    # --- Theme 4: Varshesha (Year Lord) ---
    themes.append({
        "category":  "Varshesha (Ruler of the Year)",
        "nature":    "Auspicious" if varshesha_dignity in ["Exalted","Own","Mool Trikona","Great Friend"] else "Challenging" if varshesha_dignity == "Debilitated" else "Moderate",
        "calculation": (
            f"Varshesha = lord of the weekday of the Solar Return = {varshesha}. "
            f"Natal dignity: {varshesha_dignity}. Natal house: {varshesha_house}. "
            f"Shadbala: {chart.shadbala_proxy.get(varshesha,0)}/100."
        ),
        "classical_rule": "The Varshesha (or Varsheshvara) is the planet ruling the day on which the Sun returns to its natal degree. It acts as the overall governor of the year's results — stronger than even the Muntha lord in some traditions.",
        "interpretation": (
            f"Varshesha {varshesha} governs the year. "
            + ("Its strength and benefic nature promise a highly productive year with tangible rewards." if varshesha_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "A debilitated Varshesha brings obstacles, health caution, and unexpected setbacks — double down on remedies and planning." if varshesha_dignity == "Debilitated"
               else "A neutral Varshesha gives average results; outcomes depend heavily on personal effort.")
            + f" Key domains activated by Varshesha this year: {HOUSE_MEANINGS.get(varshesha_house, 'General life')}."
        ),
        "modifier": (
            f"If Varshesha is also the Muntha lord ({varshesha == muntha_lord}), "
            "the year's results are concentrated and intense in the Muntha domain. "
            if varshesha == muntha_lord else ""
        )
    })

    # --- Theme 5: Muntha in dusthana warning ---
    if muntha_house in [6, 8, 12]:
        themes.append({
            "category":  "Dusthana Muntha — Year of Transformation & Challenge",
            "nature":    "Challenging",
            "calculation": f"Muntha in House {muntha_house} (a dusthana: 6th, 8th, or 12th from natal Lagna).",
            "classical_rule": "When Muntha falls in a dusthana, the year brings more obstacles than usual. The native must work harder for results, and health/finances require careful management.",
            "interpretation": (
                "This is a year for inner work, resilience, and release rather than external conquest. "
                + ("Service, debt resolution, and health focus (6th)." if muntha_house == 6 else
                   "Transformation, occult matters, sudden changes (8th)." if muntha_house == 8 else
                   "Spiritual retreat, expenses, foreign matters, release (12th).")
                + " Remedies for the Muntha lord and enhanced spiritual practice are protective."
            ),
            "modifier": (
                f"However, if Muntha lord {muntha_lord} is strong ({muntha_lord_dignity}), "
                "the dusthana placement is significantly softened." if muntha_lord_dignity in ["Exalted","Own","Mool Trikona"] else ""
            )
        })

    # --- Theme 6: Transit Jupiter's influence on Varshphal ---
    if transit_planets:
        jup_sign = longitude_to_sign(transit_planets.get("Jupiter", 0))[0]
        jup_idx  = ZODIAC.index(jup_sign)
        jup_from_lagna = ((jup_idx - ZODIAC.index(chart.lagna_sign)) % 12) + 1
        jup_from_moon  = ((jup_idx - ZODIAC.index(chart.moon_sign)) % 12) + 1
        themes.append({
            "category":  "Transit Jupiter — Annual Benefic Influence",
            "nature":    "Auspicious" if jup_from_lagna in [1,5,9,11] else "Challenging" if jup_from_lagna in [4,8,12] else "Moderate",
            "calculation": (
                f"Transit Jupiter in {jup_sign} (House {jup_from_lagna} from natal Lagna, "
                f"House {jup_from_moon} from Moon)."
            ),
            "classical_rule": "Jupiter's annual transit over the natal chart is the single most important transit for predicting the year's general fortune and timing of auspicious events.",
            "interpretation": (
                ("Highly auspicious: Jupiter transiting a trine/11th from Lagna — expansion, luck, and new opportunities." if jup_from_lagna in [1,5,9,11]
                   else "Challenging: Jupiter in 4th/8th/12th from Lagna — muted blessings; introspection and patience." if jup_from_lagna in [4,8,12]
                   else "Moderate Jupiter transit — some positive results, especially in Jupiter-ruled matters.")
                + f" From Moon: House {jup_from_moon} "
                + ("— Guruchandra Yoga possible (H1/5/9/11 from Moon)." if jup_from_moon in [1,5,9,11] else "— moderate Moon influence.")
            ),
            "modifier": ""
        })

    # --- Theme 7: Saturn transit ---
    if transit_planets:
        sat_sign = longitude_to_sign(transit_planets.get("Saturn", 0))[0]
        sati = check_sade_sati(chart.moon_sign, sat_sign)
        kant = check_kantaka_shani(chart.moon_sign, sat_sign)
        if sati["active"] or kant["active"]:
            themes.append({
                "category":  "Transit Saturn — Annual Karmic Pressure",
                "nature":    "Challenging",
                "calculation": (
                    f"Transit Saturn in {sat_sign}. "
                    + (f"Sade Sati: {sati['phase']}. " if sati["active"] else "")
                    + (f"Kantaka Shani: {kant.get('position','')}. " if kant["active"] else "")
                ),
                "classical_rule": "Sade Sati (7.5-year Saturn cycle over Moon) and Kantaka Shani (Saturn at 4th/7th/10th from Moon) are the most challenging Saturn transits, affecting health, relationships, and career.",
                "interpretation": (
                    sati.get("detail","") + " " + ("Kantaka Shani amplifies difficulties in the indicated domain." if kant["active"] else "")
                    + " Protective: oil massage on Saturdays, Shani Shanti puja, and blue sapphire (only after gemological and astrological consultation)."
                ),
                "modifier": ""
            })

    return themes


# ==================================================================
# SECTION 12 — RAM SHALAKA ORACLE (CLASSICAL 49-SQUARE SYSTEM)
# ==================================================================

def ram_shalaka_query(question: str = "", seed: int = None) -> Dict:
    """
    Classical Ram Shalaka divination.

    The system:
    - A 7×7 grid of Hanuman Chalisa syllables
    - The querent closes eyes, concentrates on Shri Ram, and touches any cell
    - Starting from that cell, a diagonal path of 5 steps is traced
    - The sequence of syllables forms the oracle's answer
    - The resulting path is mapped to a verse and interpreted

    Implementation:
    - If seed is provided, use it (for reproducibility)
    - Otherwise use current time as entropy source
    - The 'question' is hashed to provide additional uniqueness
    """

    if seed is None:
        import time
        question_hash = sum(ord(c) for c in question) if question else 0
        seed = int(time.time() * 1000) % 997 + question_hash % 49

    random.seed(seed)
    start_row = random.randint(0, 6)
    start_col = random.randint(0, 6)

    # Diagonal path: 5 steps in a direction determined by starting position
    # Classical text: go in the direction of Sri Lanka (south-east from Ayodhya)
    # We use diagonal direction based on quadrant
    if start_row < 3 and start_col < 3:
        dr, dc = 1, 1   # SE diagonal
    elif start_row < 3 and start_col >= 3:
        dr, dc = 1, -1  # SW diagonal
    elif start_row >= 3 and start_col < 3:
        dr, dc = -1, 1  # NE diagonal
    else:
        dr, dc = -1, -1 # NW diagonal

    path_cells = []
    path_syllables = []
    r, c = start_row, start_col
    for step in range(5):
        actual_r = r % 7
        actual_c = c % 7
        path_cells.append((actual_r, actual_c))
        path_syllables.append(RAM_SHALAKA_GRID[actual_r][actual_c])
        r += dr
        c += dc

    # Compute oracle score: sum of (row+col) values along path
    path_score = sum(cell[0] + cell[1] for cell in path_cells)

    # Determine outcome tier based on path_score (0-60 range, map to 6 tiers)
    # Classical: auspicious cells are the outer ring and certain inner cells
    # The score correlates with how many "power" cells were hit
    power_cells  = {(0,0),(0,3),(0,6),(3,0),(3,6),(6,0),(6,3),(6,6)}  # corners and mid-edges
    power_hits   = sum(1 for cell in path_cells if cell in power_cells)
    center_hit   = (3, 3) in path_cells  # Center cell = strongest
    start_symbol = RAM_SHALAKA_GRID[start_row][start_col]

    # Outcome scoring
    if center_hit or power_hits >= 3:
        outcome_key = "auspicious_high"
        outcome_en  = "Highly Auspicious"
        score_pct   = 90 + random.randint(0, 10)
    elif power_hits == 2:
        outcome_key = "auspicious_medium"
        outcome_en  = "Auspicious"
        score_pct   = 70 + random.randint(0, 15)
    elif power_hits == 1:
        outcome_key = "auspicious_low"
        outcome_en  = "Mildly Auspicious"
        score_pct   = 50 + random.randint(0, 15)
    elif path_score > 25:
        outcome_key = "neutral"
        outcome_en  = "Neutral / Mixed"
        score_pct   = 40 + random.randint(-5, 10)
    elif path_score > 15:
        outcome_key = "inauspicious_low"
        outcome_en  = "Mildly Inauspicious"
        score_pct   = 25 + random.randint(0, 15)
    else:
        outcome_key = "inauspicious_high"
        outcome_en  = "Inauspicious — Wait"
        score_pct   = 10 + random.randint(0, 15)

    meaning = RAM_SHALAKA_MEANINGS[outcome_key]

    # Find applicable Hanuman Chalisa verse
    verse_hindi, verse_en = "श्रीगुरु चरन सरोज रज", "By the Guru's grace, proceed with faith."
    for r_range, (h, e) in SHALAKA_VERSE_MAP.items():
        if path_score in r_range:
            verse_hindi, verse_en = h, e
            break

    # Generate visual grid representation
    grid_display = []
    path_set     = set(map(tuple, path_cells))
    for row_idx in range(7):
        row_str = ""
        for col_idx in range(7):
            cell = RAM_SHALAKA_GRID[row_idx][col_idx]
            if (row_idx, col_idx) == (start_row, start_col):
                row_str += f"[{cell}★] "  # Starting cell
            elif (row_idx, col_idx) in path_set:
                row_str += f"[{cell}→] "  # Path cell
            else:
                row_str += f" {cell}  "
        grid_display.append(row_str)

    # Detailed remedies based on outcome
    if outcome_key in ("auspicious_high", "auspicious_medium"):
        remedies = [
            "Begin your endeavour on a Tuesday or Saturday with Hanuman puja",
            "Recite Hanuman Chalisa once before starting",
            "Offer sindoor (vermilion) and jasmine garland to Hanuman ji",
            "Chant 'Jai Shri Ram' 108 times as you proceed",
        ]
    elif outcome_key == "auspicious_low":
        remedies = [
            "Light a sesame oil lamp before Hanuman ji on Tuesday",
            "Recite Bajrang Baan for protection",
            "Donate red cloth or sindoor to a Hanuman temple",
            "Chant 'Ram Ram' 108 times each day for 11 days",
        ]
    elif outcome_key == "neutral":
        remedies = [
            "Recite Hanuman Chalisa 5 times on consecutive Tuesdays",
            "Offer sesame (til) oil lamps on Saturday evenings",
            "Donate black lentils (urad dal) on Saturdays",
            "Seek counsel from a qualified astrologer before proceeding",
        ]
    else:
        remedies = [
            "Do NOT proceed with this matter immediately",
            "Observe a full-day fast on the next Tuesday in Hanuman ji's name",
            "Recite Hanuman Chalisa 11 times on 11 consecutive Tuesdays",
            "Donate red cloth, sindoor, and mustard oil to a Hanuman temple",
            "Revisit this query only after completing the prescribed remedies",
        ]

    # Timing guidance
    timing = {
        "auspicious_high":    "Proceed immediately or within 3 days. Tuesday is ideal.",
        "auspicious_medium":  "Proceed within the week, preferably on a Tuesday or Thursday.",
        "auspicious_low":     "Proceed after completing one Hanuman Chalisa recitation. Wait for a bright fortnight (Shukla Paksha).",
        "neutral":            "Wait at least 11 days. Strengthen with Hanuman puja before proceeding.",
        "inauspicious_low":   "Wait for the next Shukla Paksha (bright fortnight). Remedy first.",
        "inauspicious_high":  "Significant delay advised. Complete full 11-Tuesday ritual before reconsidering.",
    }

    return {
        "question":          question,
        "oracle_seed":       seed,
        "start_cell":        (start_row, start_col),
        "start_symbol":      start_symbol,
        "path_cells":        path_cells,
        "path_syllables":    path_syllables,
        "path_score":        path_score,
        "power_cells_hit":   power_hits,
        "center_hit":        center_hit,
        "outcome_code":      outcome_key,
        "outcome_english":   outcome_en,
        "outcome_score_pct": score_pct,
        "meaning_bilingual": meaning,
        "verse_hindi":       verse_hindi,
        "verse_english":     verse_en,
        "grid_display":      "\n".join(grid_display),
        "remedies":          remedies,
        "timing_guidance":   timing[outcome_key],
        "interpretation_detail": _ram_shalaka_deep_reading(
            path_syllables, outcome_key, path_cells, power_hits, center_hit
        )
    }


def _ram_shalaka_deep_reading(syllables: List[str], outcome: str,
                               cells: List[Tuple], power_hits: int,
                               center_hit: bool) -> str:
    """
    Generate a deep, personalised interpretation of the Ram Shalaka reading.
    Analyses the path's journey through the grid as a narrative.
    """
    path_str = " → ".join(syllables)

    opening = {
        "auspicious_high":    "The divine path of Shri Hanuman shines brilliantly upon your question.",
        "auspicious_medium":  "Shri Hanuman's grace illuminates the path of your inquiry.",
        "auspicious_low":     "A gentle divine hand guides this matter toward a positive resolution.",
        "neutral":            "The cosmic forces are balanced at this moment. Your will and effort are the deciding factor.",
        "inauspicious_low":   "A cautionary signal from the divine realm invites patience and reflection.",
        "inauspicious_high":  "The divine oracle speaks clearly: this is not the moment to proceed.",
    }

    cell_analysis = []
    if center_hit:
        cell_analysis.append("The path touches the sacred centre cell — the heart of the Hanuman Chalisa. "
                              "This is the rarest and most powerful oracle reading, indicating divine direct intervention.")
    if power_hits > 0:
        cell_analysis.append(f"{power_hits} power cell(s) were touched along the path, "
                              "indicating the matter has reached the attention of higher forces.")

    syllable_reading = (
        f"The five sacred syllables traced are: {path_str}. "
        "In classical Ram Shalaka tradition, these syllables form a continuous sacred sound "
        "that vibrates with the energy of your query and Hanuman ji's answer."
    )

    actions = {
        "auspicious_high":    "Move forward with complete confidence. Shri Ram's blessings are with you.",
        "auspicious_medium":  "Proceed with faith. Maintain your sadhana (spiritual practice) throughout.",
        "auspicious_low":     "Proceed, but remain alert and flexible. The blessing is conditional on continued devotion.",
        "neutral":            "The outcome is in your hands. Strengthen your resolve, deepen your practice, and act with clarity.",
        "inauspicious_low":   "Pause. Reflect. Strengthen through devotion before attempting this matter.",
        "inauspicious_high":  "Withdraw from this matter for now. Purify through penance and return when called by grace.",
    }

    return (
        f"{opening[outcome]}\n\n"
        f"PATH TRACED: {path_str}\n"
        + ("\n".join(cell_analysis) + "\n\n" if cell_analysis else "\n")
        + f"{syllable_reading}\n\n"
        + f"DIVINE GUIDANCE: {actions[outcome]}"
    )


# ==================================================================
# SECTION 13 — YEARLY PREDICTION (ELABORATED)
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
    jd = swe.julday(year, month, day, hour + minute/60.0 - tz_offset)
    houses = swe.houses_ex(jd, lat, lon, b'W', swe.FLG_SIDEREAL)
    asc    = houses[1][0]
    planets = {}
    retrograde = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        planets[pname] = res[0][0] % 360
        retrograde[pname] = res[0][3] < 0 if len(res[0]) > 3 else False
    planets["Ketu"]    = (planets["Rahu"] + 180.0) % 360.0
    retrograde["Ketu"] = retrograde["Rahu"]
    lagna_sign, _ = longitude_to_sign(asc)
    return ChartData(planets, asc, lagna_sign,
                     datetime(year, month, day, hour, minute), lat, lon, tz_offset,
                     retrograde=retrograde)


def get_year_prediction(chart: ChartData, year: int) -> Dict:
    check_date   = datetime(year, 6, 15)
    dasha_info   = chart.get_current_dasha_info(check_date)

    transit_planets      = None
    transit_saturn_sign  = None
    transit_jupiter_sign = None

    if SWISSEPH_AVAILABLE:
        try:
            transit_planets = get_transits(year, 6, 15)
            transit_saturn_sign  = longitude_to_sign(transit_planets.get("Saturn", 0))[0]
            transit_jupiter_sign = longitude_to_sign(transit_planets.get("Jupiter", 0))[0]
        except Exception:
            pass

    sade_sati = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")
    kantaka   = check_kantaka_shani(chart.moon_sign, transit_saturn_sign or "")

    varshphal = calculate_varshphal(chart, year, transit_planets)
    career    = analyze_career(chart, check_date)
    marriage  = analyze_marriage(chart, check_date)
    children  = analyze_children(chart, check_date)
    health    = analyze_health(chart, check_date, transit_saturn_sign)
    yogas     = analyze_general_yogas(chart)

    # Build Jupiter transit notes
    jupiter_transit_notes = []
    if transit_jupiter_sign:
        j_idx    = ZODIAC.index(transit_jupiter_sign)
        l_idx    = ZODIAC.index(chart.lagna_sign)
        m_idx    = ZODIAC.index(chart.moon_sign)
        jh_lagna = ((j_idx - l_idx) % 12) + 1
        jh_moon  = ((j_idx - m_idx) % 12) + 1
        if jh_lagna in [1,5,9]:
            jupiter_transit_notes.append(f"Jupiter transiting H{jh_lagna} from Lagna ({transit_jupiter_sign}) — exceptionally auspicious: growth, luck, wisdom.")
        elif jh_lagna in [2,11]:
            jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — wealth and gains favoured.")
        elif jh_lagna in [4,8,12]:
            jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — mixed; introspection, potential obstacles.")
        else:
            jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — moderate results.")
        if jh_moon in [1,5,9,11]:
            jupiter_transit_notes.append(f"Guruchandra Yoga possible: Jupiter in H{jh_moon} from Moon — emotional expansion and social recognition.")

    return {
        "year":               year,
        "dasha":              dasha_info,
        "sade_sati":          sade_sati,
        "kantaka_shani":      kantaka,
        "jupiter_transit":    " | ".join(jupiter_transit_notes),
        "transit_saturn":     transit_saturn_sign,
        "transit_jupiter":    transit_jupiter_sign,
        "transit_planets":    {p: longitude_to_sign(v)[0] for p,v in (transit_planets or {}).items()},
        "varshphal":          varshphal,
        "career":             career,
        "marriage":           marriage,
        "children":           children,
        "health":             health,
        "general_yogas":      yogas,
        "overall_summary":    _year_summary_v2(year, dasha_info, sade_sati, kantaka,
                                               varshphal, career, marriage, children, health, yogas,
                                               jupiter_transit_notes, transit_saturn_sign, transit_jupiter_sign)
    }


def _year_summary_v2(year, dasha, sade_sati, kantaka, varshphal,
                      career, marriage, children, health, yogas,
                      jupiter_notes, transit_saturn, transit_jupiter) -> str:
    lines = [
        "=" * 72,
        f"VEDIC ASTROLOGY YEAR PREDICTION — {year}  (Engine v6.0)",
        "=" * 72, ""
    ]

    # Dasha section
    md  = dasha.get("mahadasha","?")
    ad  = dasha.get("antardasha","?")
    pd  = dasha.get("pratyantardasha","?")
    lines += [
        "▶ DASHA OPERATING PERIOD",
        f"  Mahadasha:      {md}  ({dasha.get('mahadasha_start','')} → {dasha.get('mahadasha_end','')})",
        f"  Antardasha:     {ad}  ({dasha.get('antardasha_start','')} → {dasha.get('antardasha_end','')})",
        f"  Pratyantardasha:{pd}",
        f"  MD planet natal: {dasha.get('md_sign','')} (H{dasha.get('md_house',0)}) — {dasha.get('md_dignity','')}",
        ""
    ]

    # Transits
    lines.append("▶ KEY TRANSITS")
    if transit_saturn:
        lines.append(f"  Saturn transit: {transit_saturn}")
        if sade_sati.get("active"):
            lines.append(f"  ⚠ SADE SATI ACTIVE — {sade_sati['phase']}")
            lines.append(f"    {sade_sati.get('detail','')}")
        if kantaka.get("active"):
            lines.append(f"  ⚠ KANTAKA SHANI — {kantaka.get('position','')}")
    if transit_jupiter:
        lines.append(f"  Jupiter transit: {transit_jupiter}")
    for note in jupiter_notes:
        lines.append(f"    • {note}")
    lines.append("")

    # Varshphal
    if varshphal:
        lines.append("▶ VARSHPHAL (SOLAR RETURN ANALYSIS)")
        lines.append(f"  Muntha: {varshphal.get('muntha_sign','')} (House {varshphal.get('muntha_house','')}) | Lord: {varshphal.get('muntha_lord','')} [{varshphal.get('muntha_lord_dignity','')}]")
        lines.append(f"  Varsha Lagna: {varshphal.get('varsha_lagna_sign','')} | Lord: {varshphal.get('varsha_lagna_lord','')} [{varshphal.get('varsha_lagna_lord_dignity','')}]")
        lines.append(f"  Varshesha (Year Ruler): {varshphal.get('varshesha','')} [{varshphal.get('varshesha_dignity','')}] in H{varshphal.get('varshesha_house',0)}")
        tp = varshphal.get("tri_pataki",{})
        lines.append(f"  Tri-Pataki: Rising={tp.get('udaya_muntha','')} | Peak={tp.get('madhya_muntha','')} | Setting={tp.get('asta_muntha','')}")
        lines.append("")
        lines.append("  YEARLY THEMES:")
        for t in varshphal.get("themes",[]):
            if isinstance(t, dict):
                nat = t.get("nature","")
                cat = t.get("category","")
                interp = t.get("interpretation","")
                mod  = t.get("modifier","")
                marker = "✦" if nat == "Auspicious" else "⚠" if nat == "Challenging" else "◈"
                lines.append(f"  {marker} [{cat}]")
                lines.append(f"    Calculation: {t.get('calculation','')}")
                lines.append(f"    Rule: {t.get('classical_rule','')}")
                lines.append(f"    Reading: {interp}")
                if mod:
                    lines.append(f"    Note: {mod}")
            else:
                lines.append(f"  • {t}")
        lines.append("")

    # Yogas
    if yogas.get("fired_yogas"):
        lines.append(f"▶ ACTIVE YOGAS ({yogas['yoga_count']} total — Strength: {yogas['yoga_strength']})")
        for y in yogas["fired_yogas"][:6]:
            lines.append(f"  ✦ {y['title']} (score: {y['score']:+d})")
        if yogas['yoga_count'] > 6:
            lines.append(f"  … and {yogas['yoga_count']-6} more.")
        lines.append("")

    # Topic scorecards
    lines.append("▶ LIFE DOMAIN SCORECARDS")
    for label, data in [("CAREER",career),("MARRIAGE",marriage),("CHILDREN",children),("HEALTH",health)]:
        rating = data.get("rating","?")
        score  = data.get("net_score",0)
        bar    = "█" * max(0, score + 10)
        lines.append(f"  {label:10s}: {rating:12s} (score {score:+3d})  {bar}")
    lines.append("")

    # Detailed summaries
    for label, data in [("CAREER",career),("MARRIAGE",marriage),("CHILDREN",children),("HEALTH",health)]:
        lines.append(f"▶ {label} DETAIL")
        lines.append(f"  {data.get('summary','')}")
        top_rules = [r for r in data.get("fired_rules",[]) if r["score"] > 0][:2]
        for r in top_rules:
            lines.append(f"  ✦ {r['title']}")
        top_warn  = [r for r in data.get("fired_rules",[]) if r["score"] < 0][:1]
        for r in top_warn:
            lines.append(f"  ⚠ {r['title']}")
        lines.append("")

    lines += ["=" * 72, "END OF YEAR PREDICTION", "=" * 72]
    return "\n".join(lines)


# ==================================================================
# SECTION 14 — DEMO & UTILITIES
# ==================================================================

def generate_demo_chart() -> ChartData:
    planets = {
        "Sun":      45.5,
        "Moon":    128.3,
        "Mars":    200.0,
        "Mercury":  50.2,
        "Jupiter":  95.0,
        "Venus":    70.5,
        "Saturn":  310.0,
        "Rahu":    175.0,
        "Ketu":    355.0,
    }
    retro = {p: False for p in planets}
    retro["Saturn"] = True  # Demo: Saturn retrograde
    return ChartData(
        planets, ascendant=30.0, lagna_sign="Taurus",
        birth_date=datetime(1995, 6, 15, 10, 30),
        lat=28.6, lon=77.2, tz=5.5,
        retrograde=retro
    )


def load_chart_from_file(filepath: str) -> ChartData:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    birth_date = datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None
    retro      = data.get("retrograde", {})
    return ChartData(
        planets    = data.get("planets", {}),
        ascendant  = data.get("ascendant", 0.0),
        lagna_sign = data.get("lagna_sign", "Aries"),
        birth_date = birth_date,
        lat        = data.get("lat", 0),
        lon        = data.get("lon", 0),
        tz         = data.get("tz", 0),
        retrograde = retro
    )


def print_full_report(chart: ChartData, year: int = None):
    import textwrap
    year = year or datetime.now().year

    print("=" * 72)
    print("VEDIC ASTROLOGY REPORT — Engine v6.0")
    print("=" * 72)
    print(f"Lagna:         {chart.lagna_sign} ({SIGN_SANSKRIT.get(chart.lagna_sign,'')})")
    print(f"Moon sign:     {chart.moon_sign}")
    print(f"Sun sign:      {chart.sun_sign}")
    print(f"Atmakaraka:    {chart.atmakaraka}")
    print(f"Amatyakaraka:  {chart.amatyakaraka}")
    if chart.parivartana_pairs:
        print(f"Parivartana:   {chart.parivartana_pairs}")
    if chart.graha_yuddha:
        print(f"Graha Yuddha:  {[(g['winner'],'>',g['loser']) for g in chart.graha_yuddha]}")
    print()

    print("PLANETS (Full Data)")
    print("-" * 85)
    for p, lon in chart.planets.items():
        sign, deg = longitude_to_sign(lon)
        nak_info  = chart.nakshatras[p]
        dig       = chart.dignities[p]
        bkd       = chart.shadbala_breakdown.get(p, {})
        vg        = " [VG]" if chart.vargottama.get(p) else ""
        comb      = f" [Comb~{bkd.get('combust_orb_deg','?')}°]" if bkd.get("combust") else ""
        nb        = " [NB]" if is_neechabhanga(p, sign, chart.planets, chart.lagna_sign) else ""
        ret       = " (R)" if chart.retrograde.get(p) else ""
        h         = chart.house_map.get(p, 0)
        print(f"  {p:10s} H{h:2d} | {sign:14s} {deg:5.1f}° | {dig:14s}{vg}{comb}{nb}{ret} | "
              f"Nak: {nak_info['nakshatra']:22s} P{nak_info['pada']} | "
              f"SB:{chart.shadbala_proxy.get(p,0):5.1f} Vims:{chart.vimsopaka.get(p,0):4.1f}")

    print()
    print("SHADBALA BREAKDOWN SAMPLE (Saturn)")
    print("-" * 60)
    sb = chart.shadbala_breakdown.get("Saturn", {})
    for k, v in sb.items():
        print(f"  {k:25s}: {v}")

    print()
    print("VIMSHOTTARI DASHA PERIODS")
    print("-" * 72)
    now = datetime.now()
    for dp in chart.dasha_periods:
        marker = " ◄ CURRENT" if dp.start_date <= now < dp.end_date else ""
        print(f"  {dp.planet:8s}: {dp.start_date.strftime('%d %b %Y')} → "
              f"{dp.end_date.strftime('%d %b %Y')}  ({dp.years:.2f} yrs){marker}")

    print()
    prediction = get_year_prediction(chart, year)
    print(prediction["overall_summary"])


# ==================================================================
# SECTION 15 — QUICK TEST
# ==================================================================
if __name__ == "__main__":
    print("Building demo chart...")
    chart = generate_demo_chart()

    print_full_report(chart, year=2025)

    print("\n" + "=" * 72)
    print("RAM SHALAKA ORACLE — DEMO QUERY")
    print("=" * 72)
    result = ram_shalaka_query("Will my business venture succeed this year?", seed=42)
    print(f"Question:     {result['question']}")
    print(f"Outcome:      {result['outcome_english']} ({result['outcome_score_pct']}%)")
    print(f"\nGrid (★=start, →=path):\n{result['grid_display']}")
    print(f"\nPath syllables: {' → '.join(result['path_syllables'])}")
    print(f"\nMeaning:\n{result['meaning_bilingual']}")
    print(f"\nVerse (Hindi): {result['verse_hindi']}")
    print(f"Verse (Eng):   {result['verse_english']}")
    print(f"\nDeep Reading:\n{result['interpretation_detail']}")
    print(f"\nTiming: {result['timing_guidance']}")
    print(f"\nRemedies:")
    for rem in result['remedies']:
        print(f"  • {rem}")
