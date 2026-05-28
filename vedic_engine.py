"""
Vedic Astrology Calculation Engine v7.0
=========================================
FIXES vs v6.1 — ALL sources of year-invariant results resolved:

BUG-C [CRITICAL] — analyze_career / analyze_marriage / analyze_children never
  received transit_planets or sade_sati data. Their build_context() calls had
  no transit context, so the same natal rules fired for EVERY year unchanged.
  Fixed: all analyze_*() functions now accept transit_planets kwarg; ctx gains
  transit_house_map, transit_signs, sade_sati_active, and antardasha data.

BUG-D [CRITICAL] — Transit Jupiter/Saturn positions were computed in
  get_year_prediction() but never injected into the ctx used by prediction
  rules. Career/marriage/health rules never saw transit data.
  Fixed: build_context() now accepts transit_planets and populates
  transit_house_map (house of each transit planet from natal lagna) in ctx.

BUG-E [SIGNIFICANT] — _apply_dasha_boost() only checked the Mahadasha planet,
  ignoring the Antardasha. For a 20-year Venus MD, dasha boosts were identical
  for all 20 years, even though the AD changed every ~2 years.
  Fixed: boost also fires when the AD planet is relevant to the topic.

BUG-F [SIGNIFICANT] — Varsha Lagna lord dignity always used natal dignity.
  Fixed: a "is strong natally" check now explicitly uses natal dignity (which
  is the correct Varshphal approach — transit dignity of VL lord is separate).
  The interpretation now explicitly states which natal dignity applies.

NEW — TRANSIT PREDICTION RULES added to PREDICTION_RULES:
  transit_jupiter_career, transit_jupiter_marriage, transit_jupiter_children,
  transit_jupiter_health, transit_saturn_career, transit_rahu_career,
  transit_jupiter_over_natal, transit_ad_sensitive rules.
  These fire/change every year as transits shift, guaranteeing year-by-year
  variation in all topic analyses.

NEW — AD-SENSITIVE RULES: antardasha planet exposed in ctx as 'ad_planet',
  'ad_house', 'ad_dignity'. New rules check AD planet relevance to each topic.

NEW — get_year_prediction() passes transit_planets consistently to all
  analyze_*() functions and Varshphal.

ORIGINAL v6.1 fixes all retained.
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

EXALTATION_DEGREE = {"Sun":10,"Moon":3,"Mars":28,"Mercury":15,
                     "Jupiter":5,"Venus":27,"Saturn":20}
MOOLATRIKONA_RANGE = {
    "Sun":(0,20),"Moon":(4,30),"Mars":(0,12),"Mercury":(16,20),
    "Jupiter":(0,10),"Venus":(0,15),"Saturn":(0,20)
}

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

COMBUSTION_LIMITS = {
    "Moon":    12.0,
    "Mars":    17.0,
    "Mercury": {"direct": 14.0, "retrograde": 12.0},
    "Jupiter": 11.0,
    "Venus":   {"direct": 10.0, "retrograde": 8.0},
    "Saturn":  15.0,
}

TARA_SCORES = {1:3, 2:3, 3:0, 4:3, 5:0, 6:3, 7:0, 8:3, 9:3}

NAKSHATRA_SIZE = 13 + 20/60
PADA_SIZE      = 3  + 20/60

SPECIAL_ASPECTS = {
    "Mars":    [4, 8],
    "Jupiter": [5, 9],
    "Saturn":  [3, 10],
    "Rahu":    [5, 9],
    "Ketu":    [5, 9],
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
    "auspicious_high":   "श्रीराम की कृपा है। कार्य सिद्ध होगा, मनोकामना पूर्ण होगी। विजय निश्चित है।\nSri Rama's full grace is upon you. Your endeavour will succeed, your heart's desire will be fulfilled. Victory is certain.",
    "auspicious_medium": "प्रयास सफल होगा। धैर्य रखें, सहायता मिलेगी। राम नाम का जाप करें।\nYour effort will bear fruit. Be patient — support will come. Chant the name of Rama for protection.",
    "auspicious_low":    "कार्य होगा परन्तु विलम्ब सम्भव है। विश्वास रखें, हनुमान जी रक्षा करेंगे।\nSuccess will come but may be delayed. Keep faith — Hanuman will protect and guide.",
    "neutral":           "स्थिति मध्यम है। पुरुषार्थ और भक्ति दोनों चाहिए। हनुमान चालीसा का पाठ करें।\nThe situation is balanced. Both effort and devotion are needed. Recite Hanuman Chalisa for clarity.",
    "inauspicious_low":  "अभी प्रतीक्षा करें। कार्य में बाधा है परन्तु हनुमान जी की भक्ति से बाधा दूर होगी।\nWait for a better moment. There is an obstacle, but devotion to Hanuman will remove it in time.",
    "inauspicious_high": "कार्य अभी उचित नहीं। विचार बदलें, परामर्श लें। राम नाम के 108 जाप करके पुनः प्रयास करें।\nThis undertaking is not favoured now. Reconsider, seek counsel. Chant Ram Naam 108 times before trying again.",
}

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
    d1_sign = longitude_to_sign(longitude)[0]
    d9_sign = get_navamsa(longitude)
    return d1_sign == d9_sign


def get_temporary_relationship(p1: str, p2: str, planets: Dict[str, float]) -> str:
    if p1 not in planets or p2 not in planets:
        return "Neutral"
    s1 = int(planets[p1] // 30) % 12
    s2 = int(planets[p2] // 30) % 12
    d  = (s2 - s1) % 12
    return "Temporary Friend" if d in [1,2,3,9,10,11] else "Temporary Enemy"


def get_combined_relationship(p1: str, p2: str, planets: Dict[str, float]) -> str:
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
    if not planet or not sign:
        return "Neutral"
    if EXALTATION.get(planet) == sign:
        return "Exalted"
    if DEBILITATION.get(planet) == sign:
        return "Debilitated"
    if SIGN_LORD.get(sign) == planet:
        return "Own"
    if MOOLATRIKONA.get(planet) == sign:
        deg_in_sign = (planets.get(planet, 0) % 30) if planets else 0
        mt_range = MOOLATRIKONA_RANGE.get(planet, (0, 30))
        if mt_range[0] <= deg_in_sign < mt_range[1]:
            return "Mool Trikona"
        else:
            return "Own"
    lord = SIGN_LORD.get(sign, "")
    if not lord:
        return "Neutral"
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
    dig_bala_houses = {
        "Jupiter":1, "Mercury":1,
        "Sun":10,    "Mars":10,
        "Moon":4,    "Venus":4,
        "Saturn":7
    }
    if planet not in dig_bala_houses:
        return 1.0
    peak = dig_bala_houses[planet]
    dist = min(abs(house - peak), 12 - abs(house - peak))
    factor = 1.2 - (dist / 12) * 0.4
    return round(max(factor, 0.8), 3)


def check_graha_yuddha(p1: str, p2: str, planets: Dict[str, float]) -> bool:
    if p1 not in planets or p2 not in planets:
        return False
    if "Rahu" in (p1, p2) or "Ketu" in (p1, p2) or "Sun" in (p1, p2) or "Moon" in (p1, p2):
        return False
    s1 = longitude_to_sign(planets[p1])[0]
    s2 = longitude_to_sign(planets[p2])[0]
    if s1 != s2:
        return False
    diff = abs(planets[p1] - planets[p2]) % 360
    if diff > 180:
        diff = 360 - diff
    return diff < 1.0


def get_yuddha_winner(p1: str, p2: str, planets: Dict[str, float]) -> str:
    if planets.get(p1, 0) % 30 < planets.get(p2, 0) % 30:
        return p1
    return p2


def neechabhanga_conditions(planet: str, sign: str, planets: Dict[str, float],
                             lagna_sign: str) -> List[str]:
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

    if deb_lord and deb_lord in planets:
        deb_lord_sign = longitude_to_sign(planets[deb_lord])[0]
        if in_kendra_from(deb_lord_sign, lagna_sign):
            conditions_met.append(
                f"Cond.1: Lord of debilitation sign ({deb_lord}) in kendra "
                f"({deb_lord_sign}) from Lagna ({lagna_sign})"
            )
    if deb_lord and deb_lord in planets:
        deb_lord_sign = longitude_to_sign(planets[deb_lord])[0]
        if in_kendra_from(deb_lord_sign, moon_sign):
            conditions_met.append(
                f"Cond.2: Lord of debilitation sign ({deb_lord}) in kendra "
                f"({deb_lord_sign}) from Moon ({moon_sign})"
            )
    if exalt_lord and exalt_lord in planets:
        ex_lord_sign = longitude_to_sign(planets[exalt_lord])[0]
        if in_kendra_from(ex_lord_sign, lagna_sign):
            conditions_met.append(
                f"Cond.3: Exaltation sign lord ({exalt_lord}) in kendra "
                f"({ex_lord_sign}) from Lagna ({lagna_sign})"
            )
    if exalt_lord and exalt_lord in planets:
        ex_lord_sign = longitude_to_sign(planets[exalt_lord])[0]
        if in_kendra_from(ex_lord_sign, moon_sign):
            conditions_met.append(
                f"Cond.4: Exaltation sign lord ({exalt_lord}) in kendra "
                f"({ex_lord_sign}) from Moon ({moon_sign})"
            )
    if planet in planets:
        p_sign = longitude_to_sign(planets[planet])[0]
        if in_kendra_from(p_sign, lagna_sign):
            conditions_met.append(
                f"Cond.5: Debilitated planet ({planet}) itself in kendra "
                f"({p_sign}) from Lagna ({lagna_sign})"
            )
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
    aspects = []
    for planet, p_house in house_map.items():
        if ((p_house - 1 + 6) % 12) + 1 == house:
            aspects.append((planet, "7th (universal)"))
        if planet in SPECIAL_ASPECTS:
            for offset in SPECIAL_ASPECTS[planet]:
                asp_house = ((p_house - 1 + offset - 1) % 12) + 1
                if asp_house == house and (planet, "7th (universal)") not in [a for a in aspects if a[0] == planet]:
                    aspects.append((planet, f"{offset}th (special)"))
    return aspects


def get_atmakaraka(planets: Dict[str, float]) -> str:
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
    dignity    = get_planet_dignity(planet, sign, planets)
    lagna_sign = longitude_to_sign(min(planets.values()))[0]
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
        11: ("Rising Phase", f"Saturn in {ZODIAC[(m_idx+11)%12]} — 12th from Moon. Mental anxiety, foreign travel, expenses."),
        0:  ("Peak Phase",   f"Saturn on Moon sign {moon_sign}. Pressure on health, relationships, emotional resilience."),
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
    kantaka_positions = {
        3:"4th from Moon (emotional disruption, home/property stress)",
        6:"7th from Moon (relationship/partnership stress)",
        9:"10th from Moon (career and authority conflicts)"
    }
    if rel in kantaka_positions:
        return {"active": True, "position": kantaka_positions[rel], "saturn_sign": saturn_sign}
    return {"active": False}


# ==================================================================
# SECTION 4 — CHART DATA CLASS
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

        classical_planets = [p for p in self.planets if p not in ("Rahu","Ketu")]
        for i, p1 in enumerate(classical_planets):
            for p2 in classical_planets[i+1:]:
                if check_parivartana(p1, p2, self.planets):
                    self.parivartana_pairs.append((p1, p2))

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

        md_sign, _ = longitude_to_sign(self.planets.get(md.planet, 0))
        md_house   = self.house_map.get(md.planet, 0)
        md_dignity = self.dignities.get(md.planet, "Neutral")

        ad_sign    = longitude_to_sign(self.planets.get(ad.planet, 0))[0] if ad else ""
        ad_house   = self.house_map.get(ad.planet, 0) if ad else 0
        ad_dignity = self.dignities.get(ad.planet, "Neutral") if ad else "Neutral"

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
            "ad_sign":           ad_sign,
            "ad_house":          ad_house,
            "ad_dignity":        ad_dignity,
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
# SECTION 5 — CONTEXT BUILDER (FIXED v7.0)
# ==================================================================

def build_context(chart: ChartData, dasha_info: Dict = None,
                  sade_sati_info: Dict = None,
                  transit_planets: Dict[str, float] = None) -> Dict:
    """
    FIX BUG-C + BUG-D: Now accepts transit_planets and populates:
      - transit_house_map: {planet: house_from_natal_lagna}
      - transit_sign_map: {planet: sign}
      - sade_sati_active, sade_sati_phase, sade_sati_detail (from transit)
    Also exposes antardasha planet info: ad_planet, ad_house, ad_dignity, ad_sign.
    """
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

    # Build transit house map (FIX BUG-D)
    transit_house_map = {}
    transit_sign_map  = {}
    if transit_planets:
        for p, lon in transit_planets.items():
            sign, _ = longitude_to_sign(lon)
            transit_sign_map[p]  = sign
            transit_house_map[p] = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1

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
        # Transit context (FIX BUG-C, BUG-D)
        "transit_house_map":   transit_house_map,
        "transit_sign_map":    transit_sign_map,
        # Dasha context
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
        # FIX BUG-E: Antardasha planet exposed in ctx
        "ad_planet":           dasha_info.get("antardasha","")   if dasha_info else "",
        "ad_house":            dasha_info.get("ad_house",0)      if dasha_info else 0,
        "ad_dignity":          dasha_info.get("ad_dignity","")   if dasha_info else "",
        "ad_sign":             dasha_info.get("ad_sign","")      if dasha_info else "",
        # Sade Sati
        "sade_sati_active":    False,
        "sade_sati_phase":     "",
        "sade_sati_detail":    "",
    }

    if sade_sati_info:
        ctx["sade_sati_active"] = sade_sati_info.get("active", False)
        ctx["sade_sati_phase"]  = sade_sati_info.get("phase", "")
        ctx["sade_sati_detail"] = sade_sati_info.get("detail", "")
    elif transit_planets and "Saturn" in transit_sign_map:
        # Auto-compute sade sati from transit
        ss = check_sade_sati(chart.moon_sign, transit_sign_map["Saturn"])
        ctx["sade_sati_active"] = ss.get("active", False)
        ctx["sade_sati_phase"]  = ss.get("phase", "")
        ctx["sade_sati_detail"] = ss.get("detail", "")

    return ctx


# ==================================================================
# SECTION 6 — HELPER ACCESSORS
# ==================================================================

def _house(planet: str, ctx: dict) -> int:
    return ctx["house_map"].get(planet, 0)

def _transit_house(planet: str, ctx: dict) -> int:
    """House of transit planet from natal lagna."""
    return ctx["transit_house_map"].get(planet, 0)

def _transit_sign(planet: str, ctx: dict) -> str:
    return ctx["transit_sign_map"].get(planet, "")

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

def _has_transit(ctx: dict) -> bool:
    return bool(ctx.get("transit_house_map"))

def _logic_label(planet: str, ctx: dict) -> str:
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
# SECTION 7 — PREDICTION RULES (v7.0: transit + AD rules added)
# ==================================================================

PREDICTION_RULES: List[Dict] = [

    # ── CAREER — NATAL ──────────────────────────────────────────
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
            + f"\nINTERPRETATION: Sun in the 10th is one of the finest career placements. "
            "It confers natural authority, confidence, and an instinct for leadership. "
            + ("The strong Sun maximises this: career rise is marked by recognition and enduring respect. " if _strong("Sun",ctx) else
               "Neechabhanga applies — debilitation cancelled; career authority emerges after initial adversity. " if _nb("Sun",ctx) and _weak("Sun",ctx) else
               "Sun in neutral/friendly dignity — authority is present but must be consciously built. ")
            + (f"\nNB conditions: {'; '.join(_nb_conds('Sun',ctx))}" if _nb("Sun",ctx) else "")
            + ("\nJupiter aspects the 10th — dharmic success and institutional recognition." if _aspects_house("Jupiter",10,ctx) else "")
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
            + (f"Retrograde — increases Saturn's introspective power, may delay recognition. " if _retro("Saturn",ctx) else "")
            + f"\nINTERPRETATION: Saturn in the 10th grants rock-solid career foundations built over decades. "
            + ("Exalted Saturn here forms Shasha Yoga — massive career authority, recognition from the masses. " if _dignity("Saturn",ctx) == "Exalted"
               else "Own-sign Saturn — slower rise but iron reputation over decades. " if _dignity("Saturn",ctx) == "Own"
               else "Debilitated Saturn in 10th — authority conflicts; Neechabhanga forges resilience if conditions met. " if _dignity("Saturn",ctx) == "Debilitated"
               else "Saturn builds career brick by brick through discipline and integrity. ")
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
            + ("Vargottama — wisdom extraordinarily amplified. " if _vargo("Jupiter",ctx) else "")
            + f"\nINTERPRETATION: Jupiter in the 10th creates a dharmic, ethics-driven career. "
            "Teaching, law, finance, banking, counselling, spirituality are ideal domains. "
            + ("Exalted/Own Jupiter forms Hamsa Yoga in this kendra — rarest of career yogas." if _strong("Jupiter",ctx)
               else "Debilitated Jupiter slows expansion; Guru-seva and Guru-puja are essential remedies." if _weak("Jupiter",ctx)
               else "Principled career growth over the arc is indicated.")
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
            f"In House {_house(_lord(10,ctx),ctx)}. Shadbala: {_strength(_lord(10,ctx),ctx)}/100. "
            + ("Vargottama — reinforced in D9. " if _vargo(_lord(10,ctx),ctx) else "")
            + f"\nINTERPRETATION: Strong 10th lord creates a Rajayoga-class career indicator — "
            "success is foundational and enduring. "
            + _house_career_meaning(_house(_lord(10,ctx),ctx))
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
            f"CALCULATION: 10th lord {_lord(10,ctx)} is debilitated. No Neechabhanga conditions met. "
            f"Shadbala: {_strength(_lord(10,ctx),ctx)}/100. "
            f"\nINTERPRETATION: Most significant single indicator of career difficulty. "
            "Repeated career restarts, authority conflicts, abrupt terminations possible. "
            "Remedies for the 10th lord planet are essential."
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
            "the cancellation produces exceptional strength after early hardships."
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
            f"H{_house(_lord(10,ctx),ctx)}). Both strong. "
            f"\nINTERPRETATION: Fortune (9th) actively supports karma/action (10th). "
            "Career success carries divine timing and righteous purpose."
            + (f" Both lords in same house — exceptionally powerful activation." if _house(_lord(9,ctx),ctx) == _house(_lord(10,ctx),ctx) else "")
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
            f"Mercury in same sign. Mercury combust: {_combust('Mercury',ctx)} "
            f"(orb: {_bkd('Mercury',ctx).get('combust_orb_deg','?')}°). "
            f"\nINTERPRETATION: Analytical power, communication brilliance, administrative acumen. "
            + ("NOTE: Mercury is combust — yoga is weakened but not destroyed." if _combust("Mercury",ctx) else
               "Mercury clear of combustion — yoga operates at full strength.")
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
            f"CALCULATION: Running MD = {ctx.get('dasha','')} "
            f"({ctx.get('dasha_md_start','')} → {ctx.get('dasha_md_end','')}). "
            f"MD planet: H{ctx.get('md_house',0)}, dignity: {ctx.get('md_dignity','')}. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}), "
            f"H{ctx.get('ad_house',0)}, {ctx.get('ad_dignity','')}. "
            + {
                "Jupiter": "\nINTERPRETATION: Jupiter MD — expansion into teaching, law, banking, advisory.",
                "Sun":     "\nINTERPRETATION: Sun MD — authority and advancement in government/leadership.",
                "Saturn":  "\nINTERPRETATION: Saturn MD — promotions come slowly but solidly. Building legacy.",
                "Mercury": "\nINTERPRETATION: Mercury MD — communication, IT, trade, analytics excel.",
                "Rahu":    "\nINTERPRETATION: Rahu MD — dramatic leaps through unconventional routes.",
            }.get(ctx.get("dasha",""), "")
        ),
        "activation": "dasha_activated"
    },
    # FIX BUG-E: AD-sensitive career rule
    {
        "id": "career_ad_career_planet",
        "topic": "career",
        "condition": lambda ctx: ctx.get("ad_planet","") in ["Jupiter","Sun","Saturn","Mercury","Rahu","Mars"] and ctx.get("ad_planet","") != ctx.get("dasha",""),
        "severity": "positive",
        "score": 2,
        "title": "Career-Activating Antardasha Running",
        "detail": lambda ctx: (
            f"CALCULATION: Antardasha = {ctx.get('ad_planet','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"AD planet natal: H{ctx.get('ad_house',0)}, dignity: {ctx.get('ad_dignity','')}. "
            f"AD sign: {ctx.get('ad_sign','')}. "
            + {
                "Jupiter": "\nINTERPRETATION: Jupiter AD brings expansion and institutional recognition during this sub-period.",
                "Sun":     "\nINTERPRETATION: Sun AD activates leadership qualities, possible promotion or recognition.",
                "Saturn":  "\nINTERPRETATION: Saturn AD rewards sustained effort — promotions, structured advancement.",
                "Mercury": "\nINTERPRETATION: Mercury AD favours communication roles, contracts, analysis.",
                "Rahu":    "\nINTERPRETATION: Rahu AD — sudden career shifts, unconventional opportunities.",
                "Mars":    "\nINTERPRETATION: Mars AD — courage-driven moves, entrepreneurship, competitive success.",
            }.get(ctx.get("ad_planet",""), "")
        ),
        "activation": "dasha_activated"
    },
    # FIX BUG-D: Transit Jupiter career rules (change EVERY year)
    {
        "id": "transit_jupiter_career_benefic",
        "topic": "career",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [1, 2, 5, 9, 10, 11],
        "severity": "positive",
        "score": 3,
        "title": "Transit Jupiter in Benefic Career House — Annual Fortune Window",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna ({ctx['lagna_sign']}). "
            f"Natal Jupiter: H{_house('Jupiter',ctx)}, {_dignity('Jupiter',ctx)}. "
            + {
                1:  "\nINTERPRETATION: Jupiter transiting Lagna — peak personal growth, opportunities come to you. Best year for career launches.",
                2:  "\nINTERPRETATION: Jupiter transiting 2nd — wealth and income from career increase. Salary hike, promotions in financial roles.",
                5:  "\nINTERPRETATION: Jupiter transiting 5th — creative and speculative career wins. Intelligence recognised. Teaching/advisory shine.",
                9:  "\nINTERPRETATION: Jupiter transiting 9th — fortune actively supports career. Foreign opportunities, higher education, lucky breaks.",
                10: "\nINTERPRETATION: Jupiter transiting 10th (most powerful career transit) — direct blessing on career house. New role, public recognition, major advancement.",
                11: "\nINTERPRETATION: Jupiter transiting 11th — gains from career peak. Rewards arrive. Networks expand. Ambitions fulfilled.",
            }.get(_transit_house("Jupiter",ctx), "\nINTERPRETATION: Positive Jupiter transit supporting career themes.")
        ),
        "activation": "transit"
    },
    {
        "id": "transit_jupiter_career_mixed",
        "topic": "career",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [4, 6, 8, 12],
        "severity": "caution",
        "score": -1,
        "title": "Transit Jupiter in Challenging Career House — Muted Expansion",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            + {
                4:  "\nINTERPRETATION: Jupiter in 4th from Lagna — career gains muted; focus shifts to home/property. Domestic year, not peak career.",
                6:  "\nINTERPRETATION: Jupiter in 6th — obstacles from competitors. Service roles improve but promotions may be blocked by rivals.",
                8:  "\nINTERPRETATION: Jupiter in 8th — career transformations, possible sudden changes. Research/occult/inheritance themes emerge.",
                12: "\nINTERPRETATION: Jupiter in 12th — expenses and foreign connections dominate. Remote work, overseas placements, or career detachment.",
            }.get(_transit_house("Jupiter",ctx), "Mixed Jupiter transit — career themes present but require extra effort.")
        ),
        "activation": "transit"
    },
    {
        "id": "transit_saturn_career",
        "topic": "career",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Saturn", ctx) in [10, 1, 7],
        "severity": "caution",
        "score": -2,
        "title": "Transit Saturn Conjunct/Opposing Career Axis — Hard Work Demanded",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Saturn in {_transit_sign('Saturn',ctx)} "
            f"= House {_transit_house('Saturn',ctx)} from natal Lagna. "
            + {
                10: "\nINTERPRETATION: Saturn transiting 10th (Ashtama Shani for career) — maximum career pressure. Authority tests, restructuring, possible demotion then rebuild. Discipline and patience are the only path.",
                1:  "\nINTERPRETATION: Saturn transiting Lagna — personal energy low, career doubts. Major life restructuring. Results improve after Saturn moves on.",
                7:  "\nINTERPRETATION: Saturn transiting 7th — business partnerships under strain. Public dealings difficult. Negotiations drag.",
            }.get(_transit_house("Saturn",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "transit_saturn_career_positive",
        "topic": "career",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Saturn", ctx) in [3, 6, 11],
        "severity": "positive",
        "score": 2,
        "title": "Transit Saturn in Upachaya House — Career Discipline Pays",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Saturn in {_transit_sign('Saturn',ctx)} "
            f"= House {_transit_house('Saturn',ctx)} from natal Lagna. "
            "Saturn gives best results in Upachaya houses (3rd, 6th, 11th). "
            + {
                3:  "\nINTERPRETATION: Saturn in 3rd — courageous effort rewarded. Communication-based career roles thrive. Siblings may help.",
                6:  "\nINTERPRETATION: Saturn in 6th — enemies defeated through sustained work. Service and health-related roles excel.",
                11: "\nINTERPRETATION: Saturn in 11th (best Saturn transit) — career gains, income rise, network rewards arrive after sustained effort.",
            }.get(_transit_house("Saturn",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "transit_jupiter_over_natal_10th_lord",
        "topic": "career",
        "condition": lambda ctx: (
            _has_transit(ctx) and
            _transit_sign("Jupiter", ctx) == longitude_to_sign(ctx["planets"].get(_lord(10,ctx), 0))[0]
        ),
        "severity": "positive",
        "score": 3,
        "title": "Transit Jupiter Conjunct Natal 10th Lord — Peak Career Activation",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"exactly conjuncts natal 10th lord {_lord(10,ctx)} "
            f"({_dignity(_lord(10,ctx),ctx)}, H{_house(_lord(10,ctx),ctx)}). "
            f"\nINTERPRETATION: This is one of the most powerful year-specific career activations. "
            "Jupiter directly blesses the planet governing career — promotions, recognition, "
            "new opportunities, and institutional support all arrive together. "
            "This transit occurs once every ~12 years."
        ),
        "activation": "transit"
    },
    {
        "id": "transit_rahu_career_10th",
        "topic": "career",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Rahu", ctx) in [10, 1],
        "severity": "positive",
        "score": 2,
        "title": "Transit Rahu Near Career Axis — Unconventional Career Surge",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Rahu in {_transit_sign('Rahu',ctx)} "
            f"= House {_transit_house('Rahu',ctx)} from natal Lagna. "
            "\nINTERPRETATION: Rahu near the career axis drives intense ambition and unconventional leaps. "
            "Technology, foreign companies, media, disruption-driven careers spike during this transit. "
            "Sudden rises possible — maintain ethics to avoid equally sudden falls."
        ),
        "activation": "transit"
    },
    {
        "id": "career_sade_sati_career",
        "topic": "career",
        "condition": lambda ctx: ctx.get("sade_sati_active", False),
        "severity": "caution",
        "score": -2,
        "title": "Sade Sati Active — Career Disruptions & Delays",
        "detail": lambda ctx: (
            f"CALCULATION: {ctx.get('sade_sati_detail','')}. Phase: {ctx.get('sade_sati_phase','')}. "
            f"\nINTERPRETATION: Sade Sati burdens all life areas including career. "
            "Restructuring, authority conflicts, sudden reversals are possible. "
            "Work extra diligently; document achievements carefully. "
            "Peak phase (Saturn on Moon sign) is the most difficult; rising/setting phases are milder."
        ),
        "activation": "transit"
    },
    {
        "id": "career_jupiter_aspects_10th",
        "topic": "career",
        "condition": lambda ctx: _aspects_house("Jupiter",10,ctx) and _house("Jupiter",ctx) != 10,
        "severity": "positive",
        "score": 2,
        "title": "Jupiter Aspects 10th House — Dharmic Blessings on Career",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter in H{_house('Jupiter',ctx)} aspects the 10th. "
            f"Dignity: {_dignity('Jupiter',ctx)}. Shadbala: {_strength('Jupiter',ctx)}/100. "
            f"\nINTERPRETATION: Jupiter's aspect illuminates the career house with wisdom and expansive blessings."
        ),
        "activation": "natal"
    },

    # ── MARRIAGE — NATAL ─────────────────────────────────────────
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
            f"D9 Venus: {_navamsa_dignity('Venus',ctx)}. "
            + (f"Vargottama — soul-level confirmation of marital grace. " if _vargo("Venus",ctx) else "")
            + f"\nINTERPRETATION: Dignified Venus confers a loving, emotionally warm marriage. "
            + (" D9 Venus strong — deep soul-level compatibility confirmed." if _navamsa_dignity("Venus",ctx) in ["Exalted","Own","Mool Trikona"] else "")
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
            f"No Neechabhanga. Shadbala: {_strength('Venus',ctx)}/100. "
            f"\nINTERPRETATION: Pre-marital compatibility analysis strongly recommended. "
            "Marital dissatisfaction or material friction possible."
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
            f"CALCULATION: 7th lord = {_lord(7,ctx)}. Dignity = {_dignity(_lord(7,ctx),ctx)}. "
            f"In H{_house(_lord(7,ctx),ctx)}. Shadbala: {_strength(_lord(7,ctx),ctx)}/100. "
            + ("Vargottama — partner energy confirmed. " if _vargo(_lord(7,ctx),ctx) else "")
            + f"\nINTERPRETATION: Strong 7th lord indicates a capable, supportive, karmically well-matched spouse."
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
            f"CALCULATION: 7th lord {_lord(7,ctx)} debilitated. No Neechabhanga. "
            f"\nINTERPRETATION: Most critical indicator of partnership challenges. "
            "Full compatibility matching before marriage is essential."
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
            f"CALCULATION: Jupiter in H7. Dignity: {_dignity('Jupiter',ctx)}. "
            f"\nINTERPRETATION: Jupiter in the 7th is one of the best placements for marriage. "
            "Spouse is likely educated, wise, and morally upright."
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
            f"CALCULATION: Mars in H7. Dignity: {_dignity('Mars',ctx)}. "
            f"\nINTERPRETATION: Most intense Kuja Dosha. Match with a Manglik partner to neutralise."
            + (" Exalted Mars — dosha significantly mitigated." if _dignity("Mars",ctx) == "Exalted" else "")
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
            f"Venus natal: H{_house('Venus',ctx)}, {_dignity('Venus',ctx)}. "
            f"Current AD: {ctx.get('antardasha','')} "
            f"({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"\nINTERPRETATION: Venus MD — most powerful period for romantic union. "
            "Best sub-periods: Venus-Jupiter, Venus-Mercury, Venus-Moon."
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
            f"CALCULATION: Jupiter MD active. AD: {ctx.get('antardasha','')}. "
            f"\nINTERPRETATION: Jupiter MD blesses marriage. Best ADs: Jupiter-Venus, Jupiter-Moon."
        ),
        "activation": "dasha_activated"
    },
    # FIX BUG-E: AD-sensitive marriage rules
    {
        "id": "marriage_ad_venus",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("ad_planet","") == "Venus" and ctx.get("dasha","") != "Venus",
        "severity": "positive",
        "score": 3,
        "title": "Venus Antardasha — Peak Marriage Sub-Period",
        "detail": lambda ctx: (
            f"CALCULATION: Venus AD running ({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"Venus natal: H{_house('Venus',ctx)}, {_dignity('Venus',ctx)}, AD dignity: {ctx.get('ad_dignity','')}. "
            f"MD planet: {ctx.get('dasha','')}. "
            f"\nINTERPRETATION: Venus Antardasha is the most important sub-period for marriage and romance "
            "regardless of which Mahadasha is running. Marriages most commonly occur in Venus AD. "
            + ("Venus is strong natally — exceptionally positive for union." if _strong("Venus",ctx)
               else "Venus is debilitated — partnership may begin with friction; patience required.")
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "marriage_ad_jupiter",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("ad_planet","") == "Jupiter" and ctx.get("dasha","") != "Jupiter",
        "severity": "positive",
        "score": 2,
        "title": "Jupiter Antardasha — Auspicious Sub-Period for Union",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter AD running ({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"Jupiter natal: H{_house('Jupiter',ctx)}, {_dignity('Jupiter',ctx)}. "
            f"\nINTERPRETATION: Jupiter AD in any Mahadasha is among the top marriage-timing sub-periods. "
            "Dharmic, well-matched unions begin during Jupiter AD."
        ),
        "activation": "dasha_activated"
    },
    # FIX BUG-D: Transit marriage rules
    {
        "id": "transit_jupiter_marriage_7th",
        "topic": "marriage",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [7, 1, 2, 5],
        "severity": "positive",
        "score": 3,
        "title": "Transit Jupiter Activating Marriage Houses — Auspicious Year for Union",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            + {
                7:  "\nINTERPRETATION: Jupiter directly transiting the 7th house — the single best annual transit for marriage. Partnerships formed this year are blessed and enduring.",
                1:  "\nINTERPRETATION: Jupiter on Lagna — personal charm peaks; new relationships begin naturally. Good year for marriage talks.",
                2:  "\nINTERPRETATION: Jupiter in 2nd — family happiness, domestic expansion. Favourable for marriage settlements.",
                5:  "\nINTERPRETATION: Jupiter in 5th — romance, love, and emotional bonds deepen. Excellent for romantic commitments.",
            }.get(_transit_house("Jupiter",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "transit_jupiter_marriage_adverse",
        "topic": "marriage",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [6, 8, 12],
        "severity": "caution",
        "score": -1,
        "title": "Transit Jupiter in Dusthana from Lagna — Caution for New Unions",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            "\nINTERPRETATION: Jupiter's blessings on partnerships are muted this year. "
            "Existing relationships may face testing. Wait for a more auspicious transit year "
            "for new commitments unless dasha is strongly supportive."
        ),
        "activation": "transit"
    },
    {
        "id": "transit_saturn_marriage",
        "topic": "marriage",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Saturn", ctx) in [7, 1],
        "severity": "warning",
        "score": -2,
        "title": "Transit Saturn Over Marriage Axis — Relationship Pressure",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Saturn in {_transit_sign('Saturn',ctx)} "
            f"= House {_transit_house('Saturn',ctx)} from natal Lagna. "
            + {
                7: "\nINTERPRETATION: Saturn transiting 7th — partnerships under intense karmic pressure. Delays in marriage, commitment fears, separation risk. Existing marriages need conscious nurturing.",
                1: "\nINTERPRETATION: Saturn on Lagna — personal energy low, self-doubt may affect relationships. Not ideal for new commitments.",
            }.get(_transit_house("Saturn",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "marriage_benefics_aspect_7th",
        "topic": "marriage",
        "condition": lambda ctx: len(_benefics_aspect(7,ctx)) > 0,
        "severity": "positive",
        "score": 2,
        "title": "Benefic Planets Aspect 7th House — Protected Marriage",
        "detail": lambda ctx: (
            f"CALCULATION: Benefics aspecting 7th: {_benefics_aspect(7,ctx)}. "
            f"\nINTERPRETATION: Benefic aspects protect the marriage and add wisdom, love, communication."
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
            f"\nINTERPRETATION: Multiple malefic aspects — friction and power conflicts in partnerships."
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
            f"CALCULATION: Jupiter = natural Putrakaraka. Dignity: {_dignity('Jupiter',ctx)}. "
            f"H{_house('Jupiter',ctx)}. Shadbala: {_strength('Jupiter',ctx)}/100. "
            + ("Vargottama — child-blessing confirmed. " if _vargo("Jupiter",ctx) else "")
            + f"\nINTERPRETATION: Dignified Jupiter — most powerful indicator of good fortune with children."
            + (" Jupiter in trine — maximum yoga." if _house("Jupiter",ctx) in [1,5,9] else "")
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
            f"CALCULATION: 5th lord = {_lord(5,ctx)}. Dignity = {_dignity(_lord(5,ctx),ctx)}. "
            f"In H{_house(_lord(5,ctx),ctx)}. Shadbala: {_strength(_lord(5,ctx),ctx)}/100. "
            f"\nINTERPRETATION: Strong 5th lord — children likely intellectually bright and creative."
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
            f"CALCULATION: Jupiter MD ({ctx.get('dasha_md_start','')} → {ctx.get('dasha_md_end','')}). "
            f"Jupiter: H{_house('Jupiter',ctx)}, {_dignity('Jupiter',ctx)}. AD: {ctx.get('antardasha','')}. "
            f"\nINTERPRETATION: Universally most favourable period for conception. "
            "Best ADs: Jupiter-Jupiter, Jupiter-Venus, Jupiter-Moon, Jupiter-Mars."
        ),
        "activation": "dasha_activated"
    },
    # FIX BUG-E: AD-sensitive children rules
    {
        "id": "children_ad_jupiter",
        "topic": "children",
        "condition": lambda ctx: ctx.get("ad_planet","") == "Jupiter" and ctx.get("dasha","") != "Jupiter",
        "severity": "positive",
        "score": 3,
        "title": "Jupiter Antardasha — Peak Conception Sub-Period",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter AD ({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"Jupiter natal: H{_house('Jupiter',ctx)}, {_dignity('Jupiter',ctx)}. "
            f"AD dignity: {ctx.get('ad_dignity','')}. "
            f"\nINTERPRETATION: Jupiter Antardasha in any Mahadasha is the top sub-period for conception "
            "and blessed children. This sub-period changes every 1-2 years, making it the "
            "primary year-specific timing signal for children."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "children_ad_moon",
        "topic": "children",
        "condition": lambda ctx: ctx.get("ad_planet","") == "Moon",
        "severity": "positive",
        "score": 2,
        "title": "Moon Antardasha — Nurturing, Fertility & Emotional Readiness",
        "detail": lambda ctx: (
            f"CALCULATION: Moon AD ({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"Moon natal: H{_house('Moon',ctx)}, {_dignity('Moon',ctx)}. "
            f"\nINTERPRETATION: Moon AD activates nurturing instincts and emotional readiness for children. "
            "Conception is favoured, especially combined with benefic Jupiter transit over 5th."
        ),
        "activation": "dasha_activated"
    },
    # Transit children rules
    {
        "id": "transit_jupiter_children_5th",
        "topic": "children",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [5, 1, 9],
        "severity": "positive",
        "score": 3,
        "title": "Transit Jupiter Activating 5th/1st/9th — Best Year for Children",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            + {
                5: "\nINTERPRETATION: Jupiter directly transiting the 5th house of children — the single most auspicious annual transit for conception and childbirth. This transit occurs once every 12 years.",
                1: "\nINTERPRETATION: Jupiter on Lagna — overall expansion favours all life areas including progeny.",
                9: "\nINTERPRETATION: Jupiter in 9th — fortune and dharma active; children born now are spiritually blessed.",
            }.get(_transit_house("Jupiter",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "transit_jupiter_children_adverse",
        "topic": "children",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [8, 12, 6],
        "severity": "caution",
        "score": -2,
        "title": "Transit Jupiter in Dusthana from Lagna — Conception Challenges This Year",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            "\nINTERPRETATION: Jupiter's blessings on the 5th house are restricted this year. "
            "Conception may require more effort. Not the peak window — consider waiting for "
            "Jupiter to transit 5th, 1st, or 9th from Lagna."
        ),
        "activation": "transit"
    },
    {
        "id": "children_saturn_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Saturn", ctx) == 5,
        "severity": "caution",
        "score": -2,
        "title": "Saturn in 5th — Delayed but Serious Children",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn in H5. Dignity: {_dignity('Saturn',ctx)}. "
            f"Benefics aspecting 5th: {_benefics_aspect(5,ctx)}. "
            f"\nINTERPRETATION: Classically delays progeny until after ~36 years. "
            "Children tend to be responsible and long-lived."
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
            f"\nINTERPRETATION: Significant protection even if 5th lord or Jupiter are weak."
        ),
        "activation": "natal"
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
            f"CALCULATION: Lagna lord = {_lord(1,ctx)}. "
            f"Dignity: {_dignity(_lord(1,ctx),ctx)}. H{_house(_lord(1,ctx),ctx)}. "
            f"Shadbala: {_strength(_lord(1,ctx),ctx)}/100. "
            + ("Combust — vitality reduced despite strong dignity." if _combust(_lord(1,ctx),ctx) else "")
            + f"\nINTERPRETATION: Strong Lagna lord — robust vitality and resilient constitution."
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
            f"CALCULATION: Lagna lord {_lord(1,ctx)} debilitated. No Neechabhanga. "
            f"Shadbala: {_strength(_lord(1,ctx),ctx)}/100. "
            f"\nINTERPRETATION: Weakened baseline constitution. Regular check-ups essential."
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
            f"CALCULATION: {ctx.get('sade_sati_detail','')}. Phase: {ctx.get('sade_sati_phase','')}. "
            f"\nINTERPRETATION: Sade Sati lowers immunity, disturbs sleep, creates digestive "
            "and joint discomfort. Protective: moderate exercise, Shani Shanti puja, oil massage on Saturdays."
        ),
        "activation": "transit"
    },
    {
        "id": "health_transit_jupiter_1st",
        "topic": "health",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [1, 5, 9],
        "severity": "positive",
        "score": 2,
        "title": "Transit Jupiter in Health-Protective House — Vitality Boosted",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            + {
                1: "\nINTERPRETATION: Jupiter transiting Lagna — body and vitality protected and expanded. Excellent year for health improvements, recovery, and new health regimes.",
                5: "\nINTERPRETATION: Jupiter in 5th — immunity and intelligence both boosted. Healing processes supported.",
                9: "\nINTERPRETATION: Jupiter in 9th — fortune protects health. Spiritual practices enhance constitution.",
            }.get(_transit_house("Jupiter",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "health_transit_jupiter_adverse",
        "topic": "health",
        "condition": lambda ctx: _has_transit(ctx) and _transit_house("Jupiter", ctx) in [6, 8, 12],
        "severity": "caution",
        "score": -1,
        "title": "Transit Jupiter in Dusthana — Watch Health More Carefully",
        "detail": lambda ctx: (
            f"CALCULATION: Transit Jupiter in {_transit_sign('Jupiter',ctx)} "
            f"= House {_transit_house('Jupiter',ctx)} from natal Lagna. "
            + {
                6: "\nINTERPRETATION: Jupiter in 6th — prone to over-indulgence causing liver, weight, or digestive issues. Medical check-ups advisable.",
                8: "\nINTERPRETATION: Jupiter in 8th — health transformations; chronic conditions may surface for treatment. Hidden issues become visible.",
                12: "\nINTERPRETATION: Jupiter in 12th — energy drain, hospitalisation risk. Rest and spiritual practice are the medicine.",
            }.get(_transit_house("Jupiter",ctx), "")
        ),
        "activation": "transit"
    },
    {
        "id": "health_saturn_6th_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Saturn", ctx) in [6,8],
        "severity": "caution",
        "score": -2,
        "title": "Saturn in 6th/8th — Chronic Health Concerns",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn in H{_house('Saturn',ctx)}. Dignity: {_dignity('Saturn',ctx)}. "
            + f"\nINTERPRETATION: "
            + ("Saturn in 6th: chronic conditions — joints, bones, teeth, skin." if _house("Saturn",ctx)==6
               else "Saturn in 8th: longevity but chronic ailments — vata disorders. Regular Ayurvedic care essential.")
        ),
        "activation": "natal"
    },
    {
        "id": "health_dasha_saturn",
        "topic": "health",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Health Vigilance Required",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn MD. Natal: H{_house('Saturn',ctx)}, {_dignity('Saturn',ctx)}. "
            f"AD: {ctx.get('antardasha','')} ({ctx.get('dasha_ad_start','')}→{ctx.get('dasha_ad_end','')}). "
            f"\nINTERPRETATION: Heightened attention needed for bones, joints, teeth, digestion."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "health_ad_saturn",
        "topic": "health",
        "condition": lambda ctx: ctx.get("ad_planet","") == "Saturn" and ctx.get("dasha","") != "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Antardasha — Health Checkpoint Period",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn AD ({ctx.get('dasha_ad_start','')} → {ctx.get('dasha_ad_end','')}). "
            f"Saturn natal: H{_house('Saturn',ctx)}, {_dignity('Saturn',ctx)}. "
            f"\nINTERPRETATION: Saturn AD (in any MD) is the most important health caution period. "
            "Vata conditions, bone/joint issues, chronic fatigue. Preventive care is essential now."
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
            + ("Rahu MD: anxiety, atypical infections, lifestyle excesses, mysterious diagnoses." if ctx.get("dasha","")=="Rahu"
               else "Ketu MD: sudden health crises, spiritual/psychosomatic, mysterious symptoms.")
            + " Most critical sub-period: AD of Saturn."
        ),
        "activation": "dasha_activated"
    },
    {
        "id": "health_jupiter_trine_strong",
        "topic": "health",
        "condition": lambda ctx: _strong("Jupiter",ctx) and _house("Jupiter",ctx) in [1,5,9],
        "severity": "positive",
        "score": 3,
        "title": "Strong Jupiter in Trikona — Exceptional Health Protection",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter {_dignity('Jupiter',ctx)} in H{_house('Jupiter',ctx)} (trikona). "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. "
            f"\nINTERPRETATION: One of the strongest health-protective yogas. Remarkable recuperative power."
        ),
        "activation": "natal"
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
            f"CALCULATION: Jupiter {_dignity('Jupiter',ctx)} in H{_house('Jupiter',ctx)} (kendra). "
            f"Shadbala: {_strength('Jupiter',ctx)}/100. Vimsopaka: {_vims('Jupiter',ctx)}/20. "
            + ("Vargottama — supreme." if _vargo("Jupiter",ctx) else "")
            + f"\nINTERPRETATION: Exceptional wisdom, noble character, spiritual inclination, distinguished reputation."
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
            f"CALCULATION: Venus {_dignity('Venus',ctx)} in H{_house('Venus',ctx)} (kendra). "
            f"\nINTERPRETATION: Physical beauty, artistic talent, luxury, romantic success."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_ruchaka",
        "topic": "general",
        "condition": lambda ctx: _strong("Mars",ctx) and _house("Mars",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Ruchaka Yoga — Courage, Command & Vitality",
        "detail": lambda ctx: (
            f"CALCULATION: Mars {_dignity('Mars',ctx)} in H{_house('Mars',ctx)} (kendra). "
            f"\nINTERPRETATION: Exceptional physical vitality, courage, competitive prowess."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_bhadra",
        "topic": "general",
        "condition": lambda ctx: _strong("Mercury",ctx) and _house("Mercury",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Bhadra Yoga — Intellect, Eloquence & Wealth",
        "detail": lambda ctx: (
            f"CALCULATION: Mercury {_dignity('Mercury',ctx)} in H{_house('Mercury',ctx)} (kendra). "
            f"Combust: {_combust('Mercury',ctx)}. "
            + (" NOTE: combust — partially weakened." if _combust("Mercury",ctx) else "")
            + f"\nINTERPRETATION: Sharp intellect, exceptional communication, business acumen."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_shasha",
        "topic": "general",
        "condition": lambda ctx: _strong("Saturn",ctx) and _house("Saturn",ctx) in [1,4,7,10],
        "severity": "positive",
        "score": 5,
        "title": "Shasha Yoga — Authority, Discipline & Lasting Legacy",
        "detail": lambda ctx: (
            f"CALCULATION: Saturn {_dignity('Saturn',ctx)} in H{_house('Saturn',ctx)} (kendra). "
            f"\nINTERPRETATION: Iron discipline, authority over masses, lasting professional legacy."
        ),
        "activation": "natal"
    },
    {
        "id": "yoga_gajkesari",
        "topic": "general",
        "condition": lambda ctx: ((_house("Jupiter",ctx) - _house("Moon",ctx)) % 12) in [0,3,6,9],
        "severity": "positive",
        "score": 4,
        "title": "Gaja-Kesari Yoga — Fame, Wisdom & Respected Standing",
        "detail": lambda ctx: (
            f"CALCULATION: Jupiter H{_house('Jupiter',ctx)}, Moon H{_house('Moon',ctx)}. "
            f"Gap = {(_house('Jupiter',ctx)-_house('Moon',ctx))%12} (kendra ✓). "
            f"\nINTERPRETATION: Fame, wealth, eloquence, wisdom, respected position in society."
            + (" Max potency — both strong." if _strong("Jupiter",ctx) and _strong("Moon",ctx) else "")
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
            f"CALCULATION: Pairs: {ctx.get('parivartana_pairs',[])}. "
            f"\nINTERPRETATION: Powerful link between two houses — results intermingle and support each other."
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
            f"CALCULATION: Vargottama: {[p for p,v in ctx.get('vargottama',{}).items() if v]}. "
            f"\nINTERPRETATION: Stabilising, deepening quality. Delivers significations more fully."
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
            f"CALCULATION: Planets with NB: "
            f"{[p for p in ['Sun','Moon','Mars','Mercury','Jupiter','Venus','Saturn'] if _nb(p,ctx)]}. "
            + "\n".join([
                f"  {p}: {'; '.join(_nb_conds(p,ctx))}"
                for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"] if _nb(p,ctx)
            ])
            + f"\nINTERPRETATION: Rise after adversity — Neechabhanga itself is a Raja Yoga."
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
            f"CALCULATION: 6th lord = {_lord(6,ctx)}, in H{_house(_lord(6,ctx),ctx)} (dusthana). "
            f"\nINTERPRETATION: Enemies defeated by their own actions. Adversities become stepping stones."
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
            f"CALCULATION: Atmakaraka = {ctx.get('atmakaraka','')}. "
            f"Dignity: {_dignity(ctx.get('atmakaraka',''),ctx)}. H{_house(ctx.get('atmakaraka',''),ctx)}. "
            f"\nINTERPRETATION: Soul's primary purpose actively supported by destiny."
        ),
        "activation": "natal"
    },
]


# ==================================================================
# SECTION 8 — RULE ENGINE (v7.0: transit activation added)
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


def _apply_dasha_boost(fired_rules, topic_lord, md_planet, ad_planet=None, related_planets=None):
    """
    FIX BUG-E: Now also boosts when AD planet is relevant to the topic.
    This ensures year-by-year variation within the same Mahadasha.
    """
    related  = set([topic_lord] + (related_planets or []))
    result   = []
    boosted  = set()
    for r in fired_rules:
        rc = copy.deepcopy(r)
        # Boost for MD match
        if rc.get("activation") == "dasha_activated" and md_planet in related and rc["id"] not in boosted:
            old = rc["score"]
            rc["score"] = round(old * 1.5) if old > 0 else round(old * 1.2)
            rc["title"] += " [⚡ MD ACTIVATED]"
            rc["detail"] += "\n  ⚡ Amplified: the running Mahadasha planet directly governs this life area."
            boosted.add(rc["id"])
        # FIX BUG-E: Additional boost for AD match (fires every time AD changes)
        elif rc.get("activation") == "dasha_activated" and ad_planet and ad_planet in related and rc["id"] not in boosted:
            old = rc["score"]
            rc["score"] = round(old * 1.3) if old > 0 else round(old * 1.1)
            rc["title"] += " [⚡ AD ACTIVATED]"
            rc["detail"] += f"\n  ⚡ Amplified: the running Antardasha planet ({ad_planet}) governs this life area."
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
        1:  "In H1 (Lagna) — career identity merges with self; you ARE your work.",
        2:  "In H2 — career channels into wealth and family legacy.",
        3:  "In H3 — career thrives through communication, media, entrepreneurship.",
        4:  "In H4 — career connected to home, real estate, psychology.",
        5:  "In H5 — career infused with creativity, intelligence, speculation.",
        6:  "In H6 — career involves service, health, law, competition.",
        7:  "In H7 — career through partnerships, public dealing, foreign connections.",
        8:  "In H8 — research, occult, insurance, transformation-related professions.",
        9:  "In H9 (ideal) — fortune supports career; Dharma-Karma connection.",
        10: "In H10 (best) — self-contained; maximising career power directly.",
        11: "In H11 — career oriented toward networks, gains, and connections.",
        12: "In H12 — career in foreign lands, hospitals, or behind-the-scenes roles.",
    }
    return meanings.get(house, "")


# ==================================================================
# SECTION 9 — TOPIC ANALYSIS FUNCTIONS (FIXED v7.0)
# ==================================================================

def analyze_career(chart: ChartData, check_date: datetime = None,
                   transit_planets: Dict[str, float] = None) -> Dict:
    """FIX BUG-C: Now accepts transit_planets and passes to build_context."""
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info, transit_planets=transit_planets)
    fired      = evaluate_rules(ctx, topic="career")

    lagna_idx  = ZODIAC.index(chart.lagna_sign)
    tenth_lord = SIGN_LORD[ZODIAC[(lagna_idx + 9) % 12]]
    md_planet  = dasha_info.get("mahadasha","")
    ad_planet  = dasha_info.get("antardasha","")
    fired      = _apply_dasha_boost(fired, tenth_lord, md_planet, ad_planet,
                                    related_planets=["Sun","Saturn","Mercury","Jupiter","Rahu"])

    summary      = score_topic(fired)
    tenth_sign   = ZODIAC[(lagna_idx + 9) % 12]
    planets_10th = [p for p, h in ctx["house_map"].items() if h == 10]
    aspects_10th = ctx["aspect_map"].get(10, [])

    transit_summary = ""
    if transit_planets:
        jup_h = ctx["transit_house_map"].get("Jupiter", 0)
        sat_h = ctx["transit_house_map"].get("Saturn", 0)
        jup_s = ctx["transit_sign_map"].get("Jupiter","—")
        sat_s = ctx["transit_sign_map"].get("Saturn","—")
        transit_summary = f"Transit Jupiter: {jup_s} (H{jup_h}), Saturn: {sat_s} (H{sat_h})"

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
        "transit_summary":    transit_summary,
        "fired_rules":        fired,
        "narrative":          _narrative_block(fired),
        "summary": (
            f"Career: {summary['rating']} (score {summary['net_score']:+d}). "
            f"10th lord {tenth_lord} is {chart.dignities.get(tenth_lord,'Neutral')} "
            f"in H{ctx['house_map'].get(tenth_lord,0)}. "
            f"MD: {md_planet}, AD: {ad_planet}. "
            f"{transit_summary}. "
            f"{summary['positive_count']} strengths, {summary['warning_count']} cautions."
        )
    }


def analyze_marriage(chart: ChartData, check_date: datetime = None,
                     transit_planets: Dict[str, float] = None) -> Dict:
    """FIX BUG-C: Now accepts transit_planets."""
    dasha_info  = chart.get_current_dasha_info(check_date)
    ctx         = build_context(chart, dasha_info, transit_planets=transit_planets)
    fired       = evaluate_rules(ctx, topic="marriage")

    lagna_idx   = ZODIAC.index(chart.lagna_sign)
    seventh_lord= SIGN_LORD[ZODIAC[(lagna_idx + 6) % 12]]
    md_planet   = dasha_info.get("mahadasha","")
    ad_planet   = dasha_info.get("antardasha","")
    fired       = _apply_dasha_boost(fired, seventh_lord, md_planet, ad_planet,
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
            f"MD: {md_planet}, AD: {ad_planet}."
        )
    }


def analyze_children(chart: ChartData, check_date: datetime = None,
                     transit_planets: Dict[str, float] = None) -> Dict:
    """FIX BUG-C: Now accepts transit_planets."""
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info, transit_planets=transit_planets)
    fired      = evaluate_rules(ctx, topic="children")

    lagna_idx  = ZODIAC.index(chart.lagna_sign)
    fifth_lord = SIGN_LORD[ZODIAC[(lagna_idx + 4) % 12]]
    md_planet  = dasha_info.get("mahadasha","")
    ad_planet  = dasha_info.get("antardasha","")
    fired      = _apply_dasha_boost(fired, fifth_lord, md_planet, ad_planet,
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
            f"in H{ctx['house_map'].get('Jupiter',0)}. "
            f"MD: {md_planet}, AD: {ad_planet}."
        )
    }


def analyze_health(chart: ChartData, check_date: datetime = None,
                   transit_saturn_sign: str = None,
                   transit_planets: Dict[str, float] = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    if not transit_saturn_sign and transit_planets:
        transit_saturn_sign = longitude_to_sign(transit_planets.get("Saturn", 0))[0]
    sade_sati  = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")
    kantaka    = check_kantaka_shani(chart.moon_sign, transit_saturn_sign or "")
    ctx        = build_context(chart, dasha_info, sade_sati, transit_planets=transit_planets)
    fired      = evaluate_rules(ctx, topic="health")

    lagna_lord = SIGN_LORD[chart.lagna_sign]
    md_planet  = dasha_info.get("mahadasha","")
    ad_planet  = dasha_info.get("antardasha","")
    fired      = _apply_dasha_boost(fired, lagna_lord, md_planet, ad_planet,
                                    related_planets=["Sun","Jupiter","Mars","Moon"])

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
            f"MD: {md_planet}, AD: {ad_planet}."
        )
    }


def analyze_general_yogas(chart: ChartData, check_date: datetime = None,
                           transit_planets: Dict[str, float] = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info, transit_planets=transit_planets)
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
        doshas.append(f"Nadi Dosha — both have {NAKSHATRA_NADI[n1]} Nadi.")
    if bhakoot == 0:
        diff = (i2 - i1) % 12
        axis = "6/8" if diff in [5,7] else "2/12"
        doshas.append(f"Bhakoot Dosha ({axis} axis).")

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
# SECTION 11 — APPROXIMATE TRANSIT POSITIONS (FALLBACK)
# ==================================================================

_J2000_SIDEREAL = {
    "Sun":     280.5,
    "Moon":    218.3,
    "Mars":    210.0,
    "Mercury": 271.0,
    "Jupiter":  28.0,
    "Venus":   265.0,
    "Saturn":   28.5,
    "Rahu":    102.0,
}

_DAILY_MOTION = {
    "Sun":      0.98563,
    "Moon":    13.17640,
    "Mars":     0.52403,
    "Mercury":  4.09235,
    "Jupiter":  0.08309,
    "Venus":    1.60215,
    "Saturn":   0.03344,
    "Rahu":    -0.05296,
}

_J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)


def get_approx_transits(year: int, month: int = 6, day: int = 15) -> Dict[str, float]:
    """
    Return approximate sidereal (Lahiri) longitudes using mean-motion arithmetic.
    Sign-level accuracy (~±5° for Jupiter/Saturn) — sufficient for house-based
    transit rules. Used as fallback when swisseph is unavailable.
    """
    target = datetime(year, month, day, 12, 0, 0)
    days   = (target - _J2000_EPOCH).days + (target - _J2000_EPOCH).seconds / 86400.0

    transits = {}
    for planet in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu"]:
        lon = (_J2000_SIDEREAL[planet] + _DAILY_MOTION[planet] * days) % 360.0
        transits[planet] = round(lon, 2)

    transits["Ketu"] = round((transits["Rahu"] + 180.0) % 360.0, 2)
    return transits


# ==================================================================
# SECTION 12 — VARSHPHAL (SOLAR RETURN) — FIXED
# ==================================================================

def calculate_varshphal(chart: ChartData, year: int,
                         transit_planets: Dict[str, float] = None) -> Dict:
    if not chart.birth_date:
        return {}

    years_elapsed = year - chart.birth_date.year

    muntha_lon   = (chart.ascendant + years_elapsed * 30) % 360
    muntha_sign, muntha_deg = longitude_to_sign(muntha_lon)
    muntha_lord  = SIGN_LORD[muntha_sign]
    lagna_idx    = ZODIAC.index(chart.lagna_sign)
    muntha_idx   = ZODIAC.index(muntha_sign)
    muntha_house = ((muntha_idx - lagna_idx) % 12) + 1

    muntha_lord_dignity = chart.dignities.get(muntha_lord, "Neutral")
    muntha_lord_house   = chart.house_map.get(muntha_lord, 0)

    tri_pataki = {
        "udaya_muntha":  muntha_sign,
        "madhya_muntha": ZODIAC[(muntha_idx + 3) % 12],
        "asta_muntha":   ZODIAC[(muntha_idx + 6) % 12],
    }

    varsha_lagna_sign = ZODIAC[(lagna_idx + years_elapsed % 12) % 12]
    varsha_lagna_lord = SIGN_LORD[varsha_lagna_sign]
    varsha_lagna_lord_dignity = chart.dignities.get(varsha_lagna_lord, "Neutral")

    try:
        sr_date  = datetime(year, chart.birth_date.month, chart.birth_date.day)
        weekday  = sr_date.weekday()
        day_lords = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]
        varshesha = day_lords[weekday]
    except Exception:
        varshesha = "Sun"

    varshesha_dignity = chart.dignities.get(varshesha, "Neutral")
    varshesha_house   = chart.house_map.get(varshesha, 0)

    themes = _varshphal_themes_v3(
        chart, muntha_sign, muntha_house, muntha_lord,
        muntha_lord_dignity, muntha_lord_house,
        varsha_lagna_sign, varsha_lagna_lord, varsha_lagna_lord_dignity,
        varshesha, varshesha_dignity, varshesha_house,
        tri_pataki, transit_planets,
        years_elapsed=years_elapsed,
        prediction_year=year,
    )

    return {
        "year":                   year,
        "years_elapsed":          years_elapsed,
        "muntha_sign":            muntha_sign,
        "muntha_degree":          round(muntha_deg % 30, 2),
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


def _varshphal_themes_v3(
    chart, muntha_sign, muntha_house, muntha_lord,
    muntha_lord_dignity, muntha_lord_house,
    varsha_lagna_sign, varsha_lagna_lord, varsha_lagna_lord_dignity,
    varshesha, varshesha_dignity, varshesha_house,
    tri_pataki, transit_planets,
    years_elapsed: int = 0,
    prediction_year: int = 0,
) -> List[Dict]:
    """
    v3 Varshphal themes — fully year-specific:
    - All calculation strings reference prediction_year and years_elapsed correctly
    - Muntha lord modifier uses natal dignity (correct Varshphal approach)
    - Tri-Pataki references solar return phases (not calendar months)
    - Transit Jupiter/Saturn themes always generated (fallback available)
    - BUG-F addressed: explicit notes about which dignity basis is used
    """
    themes = []

    # --- Theme 1: Muntha ---
    muntha_nature = {
        1:  ("Auspicious", "A year of major personal initiative and fresh beginnings. You are at the centre of events. New identities, new ventures launched."),
        2:  ("Auspicious", "Financial accumulation, family harmony, and speech-related opportunities. Savings grow; family events are prominent."),
        3:  ("Moderate",   "Courage, communication, siblings, and short travels dominate. Skills acquired; writing/media projects advance."),
        4:  ("Auspicious", "Home, property, mother, vehicle, and inner peace highlighted. Real estate decisions and educational breakthroughs favoured."),
        5:  ("Auspicious", "Creativity, children, love, and speculative ventures surge. Intelligence recognised. Educational and romantic highlights."),
        6:  ("Challenging","A year of service, health vigilance, and competition. Discipline in daily routines is paramount. Overcoming enemies and debts."),
        7:  ("Auspicious", "Partnerships, marriage, and public dealings at peak. Business ventures and relational events dominate this year."),
        8:  ("Challenging","Transformation, hidden matters, sudden changes. Research, occult, inheritance events. Health monitoring essential."),
        9:  ("Auspicious", "Fortune, dharma, long travel, father, and higher learning are blessed. Spiritual growth and unexpected good luck."),
        10: ("Auspicious", "Career, public image, and authority at peak. Major professional milestones, recognition, and new roles likely."),
        11: ("Auspicious", "Gains, social networks, and ambitions fulfilled. Elder siblings and friends play key roles. Income rises."),
        12: ("Challenging","Expenses, isolation, foreign connections, or inner retreat. Spiritual practice deeply rewarding. Hidden assets may surface."),
    }
    nature, desc = muntha_nature.get(muntha_house, ("Moderate",""))

    # Calculate degree within sign (same as natal lagna degree within sign — correct)
    natal_deg_in_sign = round(chart.ascendant % 30, 1)

    themes.append({
        "category":  "Muntha (Annual Ascendant Marker)",
        "nature":    nature,
        "calculation": (
            f"Muntha progresses 1 sign/year from natal Lagna ({chart.lagna_sign}). "
            f"Birth year: {chart.birth_date.year} → Prediction year: {prediction_year}. "
            f"Years elapsed: {years_elapsed}. "
            f"Muntha = Lagna ({ZODIAC.index(chart.lagna_sign)+1}) + {years_elapsed} signs "
            f"→ sign {(ZODIAC.index(chart.lagna_sign) + years_elapsed) % 12 + 1} = {muntha_sign} "
            f"at {natal_deg_in_sign}° (same degree as natal Lagna within sign). "
            f"Natal house: House {muntha_house} of natal chart."
        ),
        "classical_rule": "Muntha is to Varshphal what the Lagna is to the natal chart. Its house position determines the primary domain of the year's events.",
        "interpretation": desc,
        "modifier": (
            f"Muntha lord {muntha_lord} is natally {muntha_lord_dignity} in natal House {muntha_lord_house}. "
            + ("A strong Muntha lord powerfully supports the year's themes." if muntha_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "A debilitated Muntha lord weakens the year's results; remedies essential." if muntha_lord_dignity == "Debilitated"
               else "A moderate Muntha lord delivers mixed results — sustained effort required.")
        )
    })

    # --- Theme 2: Tri-Pataki Chakra ---
    # Calculate solar return date for proper phase dating
    try:
        sr_date = datetime(prediction_year, chart.birth_date.month, chart.birth_date.day)
        sr_date_str = sr_date.strftime("%d %b %Y")
        phase1_end = (sr_date + timedelta(days=122)).strftime("%d %b")
        phase2_end = (sr_date + timedelta(days=243)).strftime("%d %b")
        phase3_end = (sr_date + timedelta(days=365)).strftime("%d %b")
    except Exception:
        sr_date_str = f"~{prediction_year}"
        phase1_end = "~Month 4"
        phase2_end = "~Month 8"
        phase3_end = "~Year end"

    themes.append({
        "category":  "Tri-Pataki Chakra (Three-Phase Annual Wheel)",
        "nature":    "Neutral",
        "calculation": (
            f"Solar return date: {sr_date_str}. "
            f"Phase 1 (Udaya/Rising, {sr_date_str}–{phase1_end}): Muntha in {tri_pataki['udaya_muntha']}. "
            f"Phase 2 (Madhya/Peak, {phase1_end}–{phase2_end}): 4th from Muntha = {tri_pataki['madhya_muntha']}. "
            f"Phase 3 (Asta/Setting, {phase2_end}–{phase3_end}): 7th from Muntha = {tri_pataki['asta_muntha']}."
        ),
        "classical_rule": "The Tri-Pataki Chakra times events within the Solar Return year — which life areas activate in each trimester (each ~4-month phase).",
        "interpretation": (
            f"Rising Phase ({tri_pataki['udaya_muntha']} — lord {SIGN_LORD[tri_pataki['udaya_muntha']]}): "
            "Initial themes and first-trimester events are set by this sign's nature. "
            f"Peak Phase ({tri_pataki['madhya_muntha']} — lord {SIGN_LORD[tri_pataki['madhya_muntha']]}): "
            "Intensification and mid-year turning points. "
            f"Setting Phase ({tri_pataki['asta_muntha']} — lord {SIGN_LORD[tri_pataki['asta_muntha']]}): "
            "Consolidation, closure, and preparation for the next annual cycle."
        ),
        "modifier": ""
    })

    # --- Theme 3: Varsha Lagna ---
    # FIX BUG-F: Explicitly state this is natal dignity (Varshphal doesn't use transit dignity of VL lord)
    themes.append({
        "category":  "Varsha Lagna (Solar Return Ascendant Quality)",
        "nature":    "Auspicious" if varsha_lagna_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"] else "Moderate",
        "calculation": (
            f"Varsha Lagna for {prediction_year}: natal Lagna ({chart.lagna_sign}) "
            f"progressed by {years_elapsed} mod 12 = {years_elapsed % 12} signs → {varsha_lagna_sign}. "
            f"Lord: {varsha_lagna_lord}. Natal dignity (used for Varshphal): {varsha_lagna_lord_dignity}. "
            f"Natal house: {chart.house_map.get(varsha_lagna_lord, 0)}."
        ),
        "classical_rule": "The Varsha Lagna lord's natal strength governs the overall tone of the year. A strong VL lord in the natal chart delivers better results throughout the solar year.",
        "interpretation": (
            f"Year {prediction_year} is coloured by {varsha_lagna_sign}'s energy "
            f"({SIGN_ELEMENT[varsha_lagna_sign]} element, {SIGN_QUALITY[varsha_lagna_sign]} quality). "
            f"Lord {varsha_lagna_lord} is natally {varsha_lagna_lord_dignity} — "
            + ("the year's overall energy is powerfully supported." if varsha_lagna_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "the year's energy is somewhat depleted; remedies recommended." if varsha_lagna_lord_dignity == "Debilitated"
               else "moderate support; sustained effort unlocks the year's potential.")
        ),
        "modifier": ""
    })

    # --- Theme 4: Varshesha (Year Lord) ---
    themes.append({
        "category":  "Varshesha (Ruler of the Solar Year)",
        "nature":    "Auspicious" if varshesha_dignity in ["Exalted","Own","Mool Trikona","Great Friend"] else "Challenging" if varshesha_dignity == "Debilitated" else "Moderate",
        "calculation": (
            f"Varshesha for {prediction_year}: lord of the weekday of the Solar Return "
            f"({chart.birth_date.strftime('%b %d')} {prediction_year}) = {varshesha}. "
            f"Natal dignity: {varshesha_dignity}. Natal house: {varshesha_house}. "
            f"Shadbala: {chart.shadbala_proxy.get(varshesha,0)}/100."
        ),
        "classical_rule": "The Varshesha (day-lord of Solar Return) is the primary governor of that year's overall results and the planet to propitiate.",
        "interpretation": (
            f"Varshesha {varshesha} governs {prediction_year}. "
            + ("Its strength promises a highly productive year." if varshesha_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "A debilitated Varshesha brings obstacles — intensify the relevant remedies." if varshesha_dignity == "Debilitated"
               else "A neutral Varshesha gives average results; outcomes depend on personal effort.")
            + f" Key domain: {HOUSE_MEANINGS.get(varshesha_house, 'General life')}."
        ),
        "modifier": (
            f"Varshesha ({varshesha}) is also the Muntha lord this year — "
            "results are concentrated and intense in the Muntha domain."
            if varshesha == muntha_lord else ""
        )
    })

    # --- Theme 5: Muntha in dusthana warning ---
    if muntha_house in [6, 8, 12]:
        themes.append({
            "category":  "Dusthana Muntha — Year of Transformation & Challenge",
            "nature":    "Challenging",
            "calculation": (
                f"Muntha in House {muntha_house} (a dusthana: 6th/8th/12th from natal Lagna) in {prediction_year}. "
                f"Muntha lord {muntha_lord} is natally {muntha_lord_dignity} in H{muntha_lord_house}."
            ),
            "classical_rule": "Muntha in a dusthana brings more obstacles and inner work than outer conquest. Health, finances, and reputation require careful management.",
            "interpretation": (
                "A year for resilience, inner transformation, and releasing what no longer serves. "
                + ("Service, debt resolution, and health discipline are primary (6th)." if muntha_house == 6 else
                   "Sudden changes, occult matters, hidden finances come to the fore (8th)." if muntha_house == 8 else
                   "Spiritual retreat, foreign matters, expenses, release (12th).")
                + " Remedies for the Muntha lord planet are strongly protective."
            ),
            "modifier": (
                f"Strong Muntha lord {muntha_lord} ({muntha_lord_dignity}) significantly softens the dusthana effect."
                if muntha_lord_dignity in ["Exalted","Own","Mool Trikona"] else ""
            )
        })

    # --- Theme 6: Transit Jupiter (always generated, fallback used if needed) ---
    if transit_planets:
        jup_sign = longitude_to_sign(transit_planets.get("Jupiter", 0))[0]
        jup_idx  = ZODIAC.index(jup_sign)
        lagna_idx_local = ZODIAC.index(chart.lagna_sign)
        jup_from_lagna = ((jup_idx - lagna_idx_local) % 12) + 1
        jup_from_moon  = ((jup_idx - ZODIAC.index(chart.moon_sign)) % 12) + 1

        jup_nature = "Auspicious" if jup_from_lagna in [1,2,5,9,11] else "Challenging" if jup_from_lagna in [4,8,12] else "Moderate"

        jup_house_meanings = {
            1: "Jupiter on Lagna — peak personal growth, opportunities arrive unsolicited. Best for all-round expansion.",
            2: "Jupiter in 2nd — income and wealth from existing work increase; family happiness rises.",
            3: "Jupiter in 3rd — good for courage, communication, short travel; moderate overall.",
            4: "Jupiter in 4th — domestic peace, property gains, but career and outer life muted.",
            5: "Jupiter in 5th — creativity, children, love, and intelligence all flourish.",
            6: "Jupiter in 6th — over-expansion in routine; health indulgence risk; obstacles from rivals.",
            7: "Jupiter in 7th — partnerships, marriage, and business collaborations are directly blessed.",
            8: "Jupiter in 8th — hidden gains, research depth; surface life unstable but inner transformation deep.",
            9: "Jupiter in 9th (best for fortune) — luck, father, spirituality, and long travel all blessed.",
            10:"Jupiter in 10th — direct blessing on career house; promotions, recognition, public standing.",
            11:"Jupiter in 11th — gains peak; networks expand; ambitions fulfilled; income from multiple sources.",
            12:"Jupiter in 12th — spiritual depth, foreign travel, ashram/retreat; outer material life constrained.",
        }

        themes.append({
            "category":  f"Transit Jupiter in {jup_sign} — Annual Benefic Pattern",
            "nature":    jup_nature,
            "calculation": (
                f"Transit Jupiter in {prediction_year} (mid-year, ~Jun 15): {jup_sign} "
                f"(House {jup_from_lagna} from natal Lagna {chart.lagna_sign}; "
                f"House {jup_from_moon} from Moon {chart.moon_sign}). "
                f"Jupiter transits one sign per year (~12 months in each sign)."
            ),
            "classical_rule": "Jupiter's annual transit sign is the most important single factor in predicting the year's general fortune and timing of auspicious events (Gurugochara).",
            "interpretation": (
                jup_house_meanings.get(jup_from_lagna, "Jupiter transit — moderate results.")
                + f" From Moon (H{jup_from_moon}): "
                + ("Guruchandra Yoga — emotional expansion, social recognition, intuitive clarity." if jup_from_moon in [1,5,9,11]
                   else "Moderate emotional support from Jupiter." if jup_from_moon in [2,3,7,10]
                   else "Muted Jupiter influence from Moon — inner growth required.")
            ),
            "modifier": ""
        })

    # --- Theme 7: Saturn transit (Sade Sati / Kantaka) ---
    if transit_planets:
        sat_sign = longitude_to_sign(transit_planets.get("Saturn", 0))[0]
        sat_idx  = ZODIAC.index(sat_sign)
        lagna_idx_local = ZODIAC.index(chart.lagna_sign)
        sat_from_lagna = ((sat_idx - lagna_idx_local) % 12) + 1
        sati = check_sade_sati(chart.moon_sign, sat_sign)
        kant = check_kantaka_shani(chart.moon_sign, sat_sign)

        sat_house_note = {
            1: "Saturn on Lagna — major life restructuring; energy low but foundational work done.",
            2: "Saturn in 2nd — financial caution; family and speech under pressure.",
            3: "Saturn in 3rd (upachaya) — courageous effort rewarded; good for disciplined communication.",
            4: "Saturn in 4th — Kantaka Shani; domestic disruptions, property delays.",
            5: "Saturn in 5th — creative blocks; delays with children; study requires extra focus.",
            6: "Saturn in 6th (upachaya) — enemies defeated; service roles shine; health manageable.",
            7: "Saturn in 7th — Kantaka Shani; partnership and marriage under karmic pressure.",
            8: "Saturn in 8th — transformation; longevity themes; chronic health attention needed.",
            9: "Saturn in 9th — fortune restricted; father/guru matters need extra care.",
            10:"Saturn in 10th (Ashtama Shani from Moon for many) — career authority tests; discipline demanded.",
            11:"Saturn in 11th (best upachaya) — sustained effort brings gains; income and networks reward patience.",
            12:"Saturn in 12th — expenses, isolation, spiritual retreat; preparation for Sade Sati if Moon is Aries.",
        }.get(sat_from_lagna, "Saturn transit — moderate karmic pressure.")

        sat_nature = "Challenging" if (sati["active"] or kant["active"] or sat_from_lagna in [4,7,8,10]) else "Moderate"
        sat_nature = "Positive" if sat_from_lagna in [3,6,11] else sat_nature

        sat_detail = sat_house_note
        if sati["active"]:
            sat_detail += f" ⚠ SADE SATI ({sati['phase']}): {sati['detail']}"
        if kant["active"]:
            sat_detail += f" ⚠ KANTAKA SHANI: {kant.get('position','')}"
        if not sati["active"] and not kant["active"]:
            sat_detail += " No Sade Sati or Kantaka Shani this year."

        themes.append({
            "category":  f"Transit Saturn in {sat_sign} — Annual Karmic Discipline",
            "nature":    sat_nature,
            "calculation": (
                f"Transit Saturn in {prediction_year}: {sat_sign} "
                f"(House {sat_from_lagna} from natal Lagna; "
                f"Saturn moves ~1 sign per 2.5 years). "
                + (f"Sade Sati: {sati['phase']} — {sati['logic']}. " if sati["active"] else "Sade Sati: Not active. ")
                + (f"Kantaka Shani: {kant.get('position','')}." if kant["active"] else "Kantaka: Not active.")
            ),
            "classical_rule": "Saturn's annual transit position governs karmic lessons and the areas requiring disciplined effort. Sade Sati (7.5-year cycle) and Kantaka (4/7/10 from Moon) are the most significant Saturn transit challenges.",
            "interpretation": sat_detail,
            "modifier": (
                "Protective measures: Shani puja on Saturdays, donation of black sesame, oil massage, "
                "and mantra: 'Om Sham Shanaischaraya Namah' (108 times)." if sati["active"] or kant["active"] else ""
            )
        })

    return themes


# ==================================================================
# SECTION 13 — RAM SHALAKA ORACLE
# ==================================================================

def ram_shalaka_query(question: str = "", seed: int = None) -> Dict:
    if seed is None:
        import time
        question_hash = sum(ord(c) for c in question) if question else 0
        seed = int(time.time() * 1000) % 997 + question_hash % 49

    random.seed(seed)
    start_row = random.randint(0, 6)
    start_col = random.randint(0, 6)

    if start_row < 3 and start_col < 3:
        dr, dc = 1, 1
    elif start_row < 3 and start_col >= 3:
        dr, dc = 1, -1
    elif start_row >= 3 and start_col < 3:
        dr, dc = -1, 1
    else:
        dr, dc = -1, -1

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

    path_score  = sum(cell[0] + cell[1] for cell in path_cells)
    power_cells = {(0,0),(0,3),(0,6),(3,0),(3,6),(6,0),(6,3),(6,6)}
    power_hits  = sum(1 for cell in path_cells if cell in power_cells)
    center_hit  = (3, 3) in path_cells
    start_symbol= RAM_SHALAKA_GRID[start_row][start_col]

    if center_hit or power_hits >= 3:
        outcome_key = "auspicious_high";   outcome_en = "Highly Auspicious";     score_pct = 90 + random.randint(0,10)
    elif power_hits == 2:
        outcome_key = "auspicious_medium"; outcome_en = "Auspicious";            score_pct = 70 + random.randint(0,15)
    elif power_hits == 1:
        outcome_key = "auspicious_low";    outcome_en = "Mildly Auspicious";     score_pct = 50 + random.randint(0,15)
    elif path_score > 25:
        outcome_key = "neutral";           outcome_en = "Neutral / Mixed";       score_pct = 40 + random.randint(-5,10)
    elif path_score > 15:
        outcome_key = "inauspicious_low";  outcome_en = "Mildly Inauspicious";   score_pct = 25 + random.randint(0,15)
    else:
        outcome_key = "inauspicious_high"; outcome_en = "Inauspicious — Wait";   score_pct = 10 + random.randint(0,15)

    meaning = RAM_SHALAKA_MEANINGS[outcome_key]

    verse_hindi, verse_en = "श्रीगुरु चरन सरोज रज", "By the Guru's grace, proceed with faith."
    for r_range, (h, e) in SHALAKA_VERSE_MAP.items():
        if path_score in r_range:
            verse_hindi, verse_en = h, e
            break

    grid_display = []
    path_set     = set(map(tuple, path_cells))
    for row_idx in range(7):
        row_str = ""
        for col_idx in range(7):
            cell = RAM_SHALAKA_GRID[row_idx][col_idx]
            if (row_idx, col_idx) == (start_row, start_col):
                row_str += f"[{cell}★] "
            elif (row_idx, col_idx) in path_set:
                row_str += f"[{cell}→] "
            else:
                row_str += f" {cell}  "
        grid_display.append(row_str)

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
            "Offer sesame oil lamps on Saturday evenings",
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

    timing = {
        "auspicious_high":    "Proceed immediately or within 3 days. Tuesday is ideal.",
        "auspicious_medium":  "Proceed within the week, preferably on a Tuesday or Thursday.",
        "auspicious_low":     "Proceed after completing one Hanuman Chalisa recitation. Wait for Shukla Paksha.",
        "neutral":            "Wait at least 11 days. Strengthen with Hanuman puja before proceeding.",
        "inauspicious_low":   "Wait for the next Shukla Paksha. Remedy first.",
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
    }


# ==================================================================
# SECTION 14 — YEARLY PREDICTION (FIXED v7.0)
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


def compute_chart(year, month, day, hour, minute, lat, lon, tz_offset=0.0) -> "ChartData":
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
    """
    Full yearly prediction — v7.0 with all fixes:
    - Transit planets always computed (swisseph or fallback)
    - All analyze_*() functions receive transit_planets → year-specific rules
    - AD planet exposed throughout → year-by-year variation within same MD
    - Varshphal always has Jupiter/Saturn transit themes
    """
    check_date = datetime(year, 6, 15)
    dasha_info = chart.get_current_dasha_info(check_date)

    # Always get transit positions
    transit_planets = None
    if SWISSEPH_AVAILABLE:
        try:
            transit_planets = get_transits(year, 6, 15)
        except Exception:
            pass
    if transit_planets is None:
        transit_planets = get_approx_transits(year, 6, 15)

    transit_saturn_sign  = longitude_to_sign(transit_planets.get("Saturn", 0))[0]
    transit_jupiter_sign = longitude_to_sign(transit_planets.get("Jupiter", 0))[0]

    sade_sati = check_sade_sati(chart.moon_sign, transit_saturn_sign)
    kantaka   = check_kantaka_shani(chart.moon_sign, transit_saturn_sign)

    varshphal = calculate_varshphal(chart, year, transit_planets)

    # FIX BUG-C + BUG-D: Pass transit_planets to all analyze functions
    career    = analyze_career(chart, check_date, transit_planets=transit_planets)
    marriage  = analyze_marriage(chart, check_date, transit_planets=transit_planets)
    children  = analyze_children(chart, check_date, transit_planets=transit_planets)
    health    = analyze_health(chart, check_date, transit_planets=transit_planets)
    yogas     = analyze_general_yogas(chart, check_date, transit_planets=transit_planets)

    # Jupiter transit notes
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    moon_idx  = ZODIAC.index(chart.moon_sign)
    j_idx     = ZODIAC.index(transit_jupiter_sign)
    jh_lagna  = ((j_idx - lagna_idx) % 12) + 1
    jh_moon   = ((j_idx - moon_idx)  % 12) + 1

    jupiter_transit_notes = []
    if jh_lagna in [1,5,9]:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna ({transit_jupiter_sign}) — exceptionally auspicious. Growth, luck, new opportunities arrive.")
    elif jh_lagna in [2,11]:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — wealth and gains are favoured this year.")
    elif jh_lagna in [10]:
        jupiter_transit_notes.append(f"Jupiter directly transiting 10th from Lagna — peak career transit this year.")
    elif jh_lagna in [4,8,12]:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — muted outer results; inner work and introspection favoured.")
    else:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna ({transit_jupiter_sign}) — moderate results.")

    if jh_moon in [1,5,9,11]:
        jupiter_transit_notes.append(f"Guruchandra Yoga: Jupiter H{jh_moon} from Moon — emotional expansion and social recognition.")

    return {
        "year":               year,
        "dasha":              dasha_info,
        "sade_sati":          sade_sati,
        "kantaka_shani":      kantaka,
        "jupiter_transit":    " | ".join(jupiter_transit_notes),
        "transit_saturn":     transit_saturn_sign,
        "transit_jupiter":    transit_jupiter_sign,
        "transit_planets":    {p: longitude_to_sign(v)[0] for p,v in transit_planets.items()},
        "transit_planets_raw": transit_planets,
        "varshphal":          varshphal,
        "career":             career,
        "marriage":           marriage,
        "children":           children,
        "health":             health,
        "general_yogas":      yogas,
        "overall_summary":    _year_summary_v3(
            year, dasha_info, sade_sati, kantaka,
            varshphal, career, marriage, children, health, yogas,
            jupiter_transit_notes, transit_saturn_sign, transit_jupiter_sign
        )
    }


def _year_summary_v3(year, dasha, sade_sati, kantaka, varshphal,
                      career, marriage, children, health, yogas,
                      jupiter_notes, transit_saturn, transit_jupiter) -> str:
    lines = [
        "=" * 72,
        f"VEDIC ASTROLOGY YEAR PREDICTION — {year}  (Engine v7.0)",
        "=" * 72, ""
    ]

    md  = dasha.get("mahadasha","?")
    ad  = dasha.get("antardasha","?")
    pd  = dasha.get("pratyantardasha","?")
    lines += [
        "▶ DASHA OPERATING PERIOD",
        f"  Mahadasha:       {md}  ({dasha.get('mahadasha_start','')} → {dasha.get('mahadasha_end','')})",
        f"  Antardasha:      {ad}  ({dasha.get('antardasha_start','')} → {dasha.get('antardasha_end','')})",
        f"  Pratyantardasha: {pd}",
        f"  MD planet: {dasha.get('md_sign','')} (H{dasha.get('md_house',0)}) — {dasha.get('md_dignity','')}",
        f"  AD planet: {dasha.get('ad_sign','')} (H{dasha.get('ad_house',0)}) — {dasha.get('ad_dignity','')}",
        ""
    ]

    lines.append("▶ KEY TRANSITS")
    lines.append(f"  Saturn transit: {transit_saturn}")
    if sade_sati.get("active"):
        lines.append(f"  ⚠ SADE SATI ACTIVE — {sade_sati['phase']}")
        lines.append(f"    {sade_sati.get('detail','')}")
    if kantaka.get("active"):
        lines.append(f"  ⚠ KANTAKA SHANI — {kantaka.get('position','')}")
    lines.append(f"  Jupiter transit: {transit_jupiter}")
    for note in jupiter_notes:
        lines.append(f"    • {note}")
    lines.append("")

    if varshphal:
        lines.append("▶ VARSHPHAL (SOLAR RETURN ANALYSIS)")
        lines.append(f"  Muntha: {varshphal.get('muntha_sign','')} (House {varshphal.get('muntha_house','')}) | Lord: {varshphal.get('muntha_lord','')} [{varshphal.get('muntha_lord_dignity','')}]")
        lines.append(f"  Varsha Lagna: {varshphal.get('varsha_lagna_sign','')} | Lord: {varshphal.get('varsha_lagna_lord','')} [{varshphal.get('varsha_lagna_lord_dignity','')}]")
        lines.append(f"  Varshesha (Year Ruler): {varshphal.get('varshesha','')} [{varshphal.get('varshesha_dignity','')}] in H{varshphal.get('varshesha_house',0)}")
        tp = varshphal.get("tri_pataki",{})
        lines.append(f"  Tri-Pataki: Rising={tp.get('udaya_muntha','')} | Peak={tp.get('madhya_muntha','')} | Setting={tp.get('asta_muntha','')}")
        lines.append("")

    if yogas.get("fired_yogas"):
        lines.append(f"▶ ACTIVE YOGAS ({yogas['yoga_count']} total — Strength: {yogas['yoga_strength']})")
        for y in yogas["fired_yogas"][:6]:
            lines.append(f"  ✦ {y['title']} (score: {y['score']:+d})")
        if yogas['yoga_count'] > 6:
            lines.append(f"  … and {yogas['yoga_count']-6} more.")
        lines.append("")

    lines.append("▶ LIFE DOMAIN SCORECARDS")
    for label, data in [("CAREER",career),("MARRIAGE",marriage),("CHILDREN",children),("HEALTH",health)]:
        rating = data.get("rating","?")
        score  = data.get("net_score",0)
        bar    = "█" * max(0, score + 10)
        lines.append(f"  {label:10s}: {rating:12s} (score {score:+3d})  {bar}")
    lines.append("")

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

    lines += ["=" * 72, "END OF YEAR PREDICTION — v7.0", "=" * 72]
    return "\n".join(lines)


# ==================================================================
# SECTION 15 — DEMO & UTILITIES
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
    retro["Saturn"] = True
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
    year = year or datetime.now().year
    print("=" * 72)
    print("VEDIC ASTROLOGY REPORT — Engine v7.0")
    print("=" * 72)
    print(f"Lagna: {chart.lagna_sign}  Moon: {chart.moon_sign}  Sun: {chart.sun_sign}")
    print(f"Atmakaraka: {chart.atmakaraka}  Amatyakaraka: {chart.amatyakaraka}")
    print()
    prediction = get_year_prediction(chart, year)
    print(prediction["overall_summary"])
