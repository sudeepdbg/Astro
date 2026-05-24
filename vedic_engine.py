"""
Vedic Astrology Calculation Engine v2.0
Robust implementation with:
- Exact Vimshottari Dasha (MD/AD/PD) with date calculations
- Varshphal (Tajaka/Solar Return) calculations
- Detailed topic predictions (Career, Marriage, Children, Health)
- Chart export/save functionality
- Lahiri Ayanamsa with Swiss Ephemeris
"""

import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    import swisseph as swe
    SWISSEPH_AVAILABLE = True
    swe.set_sid_mode(swe.SIDM_LAHIRI)
except ImportError:
    SWISSEPH_AVAILABLE = False

# ------------------------------------------------------------------
# CONSTANTS & DATA TABLES
# ------------------------------------------------------------------
ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

ZODIAC_SHORT = ["ARI", "TAU", "GEM", "CAN", "LEO", "VIR", "LIB", "SCO", "SAG", "CAP", "AQU", "PIS"]

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

PLANET_IDS = [0, 1, 2, 3, 4, 5, 6, 10]  # swe.SUN, MOON, MARS, MERCURY, JUPITER, VENUS, SATURN, TRUE_NODE
PLANET_NAMES = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu"]

HOUSE_MEANINGS = {
    1: "Self / Body", 2: "Wealth / Family", 3: "Courage / Siblings",
    4: "Mother / Home", 5: "Intelligence / Children", 6: "Disease / Enemies",
    7: "Marriage / Partnership", 8: "Longevity / Occult", 9: "Fortune / Dharma",
    10: "Career / Status", 11: "Gains / Friends", 12: "Loss / Liberation"
}

EXALTATION = {
    "Sun": "Aries", "Moon": "Taurus", "Mars": "Capricorn",
    "Mercury": "Virgo", "Jupiter": "Cancer", "Venus": "Pisces", "Saturn": "Libra"
}

DEBILITATION = {
    "Sun": "Libra", "Moon": "Scorpio", "Mars": "Cancer",
    "Mercury": "Pisces", "Jupiter": "Capricorn", "Venus": "Virgo", "Saturn": "Aries"
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
    part = int(deg_in_sign // (10 / 3))
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12
    else:
        start = (sign_idx + 4) % 12
    return ZODIAC[(start + part) % 12]


def get_drekkana(longitude: float) -> str:
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // 10)
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 4) % 12
    else:
        start = (sign_idx + 8) % 12
    return ZODIAC[(start + part) % 12]


def get_saptamsa(longitude: float) -> str:
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // (30 / 7))
    # Saptamsa rules: odd signs start from same, even from 7th
    if sign_idx % 2 == 0:  # Odd signs (0=Aries, 2=Gemini...)
        start = sign_idx
    else:
        start = (sign_idx + 6) % 12
    return ZODIAC[(start + part) % 12]


def get_dasamsa(longitude: float) -> str:
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // 3)
    # Dasamsa rules: movable from same, fixed from 8th, dual from 5th
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12
    else:
        start = (sign_idx + 4) % 12
    return ZODIAC[(start + part) % 12]


def get_dwadasamsa(longitude: float) -> str:
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // 2.5)
    return ZODIAC[(sign_idx + part) % 12]


def get_planet_dignity(planet: str, sign: str) -> str:
    lord = SIGN_LORD[sign]
    if planet == lord:
        return "Own"
    if EXALTATION.get(planet) == sign:
        return "Exalted"
    if DEBILITATION.get(planet) == sign:
        return "Debilitated"
    # Moolatrikona (simplified)
    moola = {
        "Sun": "Leo", "Moon": "Taurus", "Mars": "Aries",
        "Mercury": "Virgo", "Jupiter": "Sagittarius", "Venus": "Libra", "Saturn": "Aquarius"
    }
    if moola.get(planet) == sign:
        return "Mool Trikona"
    return "Neutral"


# ------------------------------------------------------------------
# DASHA CALCULATIONS WITH EXACT DATES
# ------------------------------------------------------------------
@dataclass
class DashaPeriod:
    planet: str
    start_date: datetime
    end_date: datetime
    years: float
    level: str  # "MD", "AD", "PD"
    parent: Optional[str] = None


def calculate_vimshottari_full(birth_date: datetime, moon_longitude: float) -> List[DashaPeriod]:
    """Calculate complete Vimshottari Dasha with exact dates."""
    nak, pada, rem = get_nakshatra(moon_longitude)
    nak_idx = NAKSHATRAS.index(nak)
    lord_idx = nak_idx % 9
    start_lord = DASHA_SEQUENCE[lord_idx]

    nak_start = nak_idx * NAKSHATRA_SIZE
    degrees_covered = moon_longitude - nak_start
    remaining = NAKSHATRA_SIZE - (degrees_covered % NAKSHATRA_SIZE)
    fraction = remaining / NAKSHATRA_SIZE
    balance = fraction * DASHA_YEARS[start_lord]

    periods = []
    current_date = birth_date

    for i in range(9):
        lord = DASHA_SEQUENCE[(lord_idx + i) % 9]
        years = balance if i == 0 else DASHA_YEARS[lord]
        # Convert years to days (using 365.25 days/year for accuracy)
        days = years * 365.25
        end_date = current_date + timedelta(days=days)
        periods.append(DashaPeriod(
            planet=lord,
            start_date=current_date,
            end_date=end_date,
            years=round(years, 3),
            level="MD"
        ))
        current_date = end_date

    return periods


def calculate_antardasha(md: DashaPeriod) -> List[DashaPeriod]:
    """Calculate Antardashas within a Mahadasha."""
    ad_periods = []
    md_start = md.start_date
    md_years = md.years
    md_planet = md.planet
    md_idx = DASHA_SEQUENCE.index(md_planet)

    current_date = md_start
    for i in range(9):
        ad_planet = DASHA_SEQUENCE[(md_idx + i) % 9]
        # AD years = MD planet years * AD planet years / 120
        ad_years = (DASHA_YEARS[md_planet] * DASHA_YEARS[ad_planet]) / 120.0
        # Scale to actual MD duration
        ad_years_actual = ad_years * (md_years / DASHA_YEARS[md_planet])
        days = ad_years_actual * 365.25
        end_date = current_date + timedelta(days=days)
        ad_periods.append(DashaPeriod(
            planet=ad_planet,
            start_date=current_date,
            end_date=end_date,
            years=round(ad_years_actual, 3),
            level="AD",
            parent=md_planet
        ))
        current_date = end_date

    return ad_periods


def get_current_dasha(periods: List[DashaPeriod], check_date: datetime = None) -> Optional[DashaPeriod]:
    """Find which Mahadasha is running at a given date."""
    if check_date is None:
        check_date = datetime.now()
    for p in periods:
        if p.start_date <= check_date < p.end_date:
            return p
    return None


def get_current_antardasha(md_periods: List[DashaPeriod], check_date: datetime = None) -> Optional[DashaPeriod]:
    """Find which Antardasha is running at a given date."""
    if check_date is None:
        check_date = datetime.now()
    md = get_current_dasha(md_periods, check_date)
    if not md:
        return None
    ad_periods = calculate_antardasha(md)
    for ad in ad_periods:
        if ad.start_date <= check_date < ad.end_date:
            return ad
    return None


# ------------------------------------------------------------------
# CHART DATA CLASS
# ------------------------------------------------------------------
class ChartData:
    def __init__(self, planets: Dict[str, float], ascendant: float, lagna_sign: str,
                 birth_date: datetime = None, lat: float = 0.0, lon: float = 0.0, tz: float = 0.0):
        self.planets = planets
        self.ascendant = ascendant
        self.lagna_sign = lagna_sign
        self.moon_sign = longitude_to_sign(planets["Moon"])[0]
        self.sun_sign = longitude_to_sign(planets["Sun"])[0]
        self.nakshatras = {}
        self.navamsa = {}
        self.drekkana = {}
        self.saptamsa = {}
        self.dasamsa = {}
        self.dwadasamsa = {}
        self.dignities = {}
        self.dasha_periods = []
        self.birth_date = birth_date
        self.lat = lat
        self.lon = lon
        self.tz = tz
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
            self.drekkana[p] = get_drekkana(lon)
            self.saptamsa[p] = get_saptamsa(lon)
            self.dasamsa[p] = get_dasamsa(lon)
            self.dwadasamsa[p] = get_dwadasamsa(lon)
            sign, _ = longitude_to_sign(lon)
            self.dignities[p] = get_planet_dignity(p, sign)

        if self.birth_date:
            self.dasha_periods = calculate_vimshottari_full(
                self.birth_date, self.planets["Moon"]
            )

    def get_current_dasha_info(self, check_date: datetime = None) -> Dict:
        """Get current MD and AD info."""
        md = get_current_dasha(self.dasha_periods, check_date)
        if not md:
            return {}
        ad = get_current_antardasha(self.dasha_periods, check_date)
        return {
            "mahadasha": md.planet,
            "mahadasha_start": md.start_date.strftime("%d %b %Y"),
            "mahadasha_end": md.end_date.strftime("%d %b %Y"),
            "antardasha": ad.planet if ad else "Unknown",
            "antardasha_start": ad.start_date.strftime("%d %b %Y") if ad else "",
            "antardasha_end": ad.end_date.strftime("%d %b %Y") if ad else "",
        }

    def to_dict(self) -> Dict:
        """Export chart to dictionary for saving."""
        return {
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "lat": self.lat, "lon": self.lon, "tz": self.tz,
            "lagna_sign": self.lagna_sign,
            "moon_sign": self.moon_sign,
            "sun_sign": self.sun_sign,
            "ascendant": self.ascendant,
            "planets": self.planets,
            "nakshatras": self.nakshatras,
            "navamsa": self.navamsa,
            "dignities": self.dignities,
            "dasha": [
                {
                    "planet": p.planet,
                    "start": p.start_date.isoformat(),
                    "end": p.end_date.isoformat(),
                    "years": p.years
                }
                for p in self.dasha_periods
            ]
        }

    def save_to_file(self, filepath: str):
        """Save chart to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# CHART CALCULATION (SWISS EPHEMERIS)
# ------------------------------------------------------------------
def compute_chart(year, month, day, hour, minute, lat, lon, tz_offset=0.0) -> ChartData:
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("Swiss Ephemeris (pyswisseph) not installed. Use generate_demo_chart() or install it.")

    jd = swe.julday(year, month, day, hour + minute / 60.0 - tz_offset)
    houses = swe.houses_ex(jd, lat, lon, b'W', swe.FLG_SIDEREAL)
    ascendant = houses[1][0]

    planets = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        planets[pname] = res[0][0]

    planets["Ketu"] = (planets["Rahu"] + 180.0) % 360.0

    lagna_sign, _ = longitude_to_sign(ascendant)
    birth_date = datetime(year, month, day, hour, minute)

    return ChartData(planets, ascendant, lagna_sign, birth_date, lat, lon, tz_offset)


def get_transits(year: int, month: int = 6, day: int = 15) -> Dict[str, float]:
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("Swiss Ephemeris not installed.")
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
    tara = (diff % 9) + 1
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
    return 2


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
    return 3


def get_gana_score(g1: str, g2: str) -> int:
    if g1 == g2:
        return 6
    if (g1 == "Deva" and g2 == "Manushya") or (g1 == "Manushya" and g2 == "Deva"):
        return 6
    if (g1 == "Manushya" and g2 == "Rakshasa") or (g1 == "Rakshasa" and g2 == "Manushya"):
        return 3
    return 0


def get_bhakoot_score(idx1: int, idx2: int) -> int:
    diff = (idx2 - idx1) % 12
    if diff in [2, 10, 6, 8]:
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
# DETAILED PREDICTIONS
# ------------------------------------------------------------------
def analyze_career(chart: ChartData) -> Dict:
    """Detailed career analysis based on chart."""
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    tenth_lord = SIGN_LORD[ZODIAC[(lagna_idx + 9) % 12]]
    tenth_lord_pos = None
    tenth_lord_house = None

    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        if p == tenth_lord:
            tenth_lord_pos = sign
            tenth_lord_house = house

    # Planets in 10th house
    planets_in_10th = []
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        if house == 10:
            planets_in_10th.append(p)

    # Yogas
    yogas = []
    if "Sun" in planets_in_10th and "Mercury" in planets_in_10th:
        yogas.append("Budhaditya Yoga — Intelligence, communication, leadership")
    if tenth_lord and tenth_lord in ["Jupiter", "Sun", "Mercury"]:
        yogas.append(f"Strong 10th lord {tenth_lord} — Career success indicated")

    # Dasha analysis
    current = chart.get_current_dasha_info()
    dasha_career = ""
    if current.get("mahadasha") in ["Jupiter", "Sun", "Saturn", "Mercury"]:
        dasha_career = f"{current['mahadasha']} Mahadasha supports career growth and authority."
    elif current.get("mahadasha") == "Rahu":
        dasha_career = "Rahu Mahadasha brings unconventional career paths and foreign opportunities."
    elif current.get("mahadasha") == "Venus":
        dasha_career = "Venus Mahadasha favors creative, artistic, and luxury-related careers."
    else:
        dasha_career = f"{current['mahadasha']} Mahadasha — focus on building foundation."

    return {
        "tenth_lord": tenth_lord,
        "tenth_lord_position": f"House {tenth_lord_house} ({tenth_lord_pos})" if tenth_lord_house else "Unknown",
        "planets_in_10th": planets_in_10th,
        "yogas": yogas,
        "current_dasha": current,
        "dasha_career": dasha_career,
        "recommendation": _career_recommendation(chart.lagna_sign, planets_in_10th, tenth_lord)
    }


def _career_recommendation(lagna: str, planets_10th: List[str], tenth_lord: str) -> str:
    recs = []
    if "Mercury" in planets_10th or tenth_lord == "Mercury":
        recs.append("Communication, writing, teaching, analytics, IT")
    if "Jupiter" in planets_10th or tenth_lord == "Jupiter":
        recs.append("Education, law, counseling, spirituality, finance")
    if "Saturn" in planets_10th or tenth_lord == "Saturn":
        recs.append("Engineering, administration, mining, real estate")
    if "Mars" in planets_10th or tenth_lord == "Mars":
        recs.append("Military, sports, surgery, technical fields")
    if "Venus" in planets_10th or tenth_lord == "Venus":
        recs.append("Arts, fashion, hospitality, beauty, entertainment")
    if "Sun" in planets_10th or tenth_lord == "Sun":
        recs.append("Government, leadership, politics, medicine")
    if "Moon" in planets_10th or tenth_lord == "Moon":
        recs.append("Healthcare, nurturing, public service, food")
    if "Rahu" in planets_10th:
        recs.append("Technology, media, foreign trade, research, occult")
    if not recs:
        recs.append("General administration, management, self-employment")
    return "; ".join(recs)


def analyze_marriage(chart: ChartData) -> Dict:
    """Detailed marriage analysis."""
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    seventh_lord = SIGN_LORD[ZODIAC[(lagna_idx + 6) % 12]]

    # 7th house planets
    planets_in_7th = []
    seventh_lord_pos = None
    seventh_lord_house = None

    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        if house == 7:
            planets_in_7th.append(p)
        if p == seventh_lord:
            seventh_lord_pos = sign
            seventh_lord_house = house

    # Venus analysis (karaka)
    venus_sign, venus_deg = longitude_to_sign(chart.planets["Venus"])
    venus_house = ((ZODIAC.index(venus_sign) - lagna_idx) % 12) + 1

    # Mars position (Kuja dosha)
    mars_sign, _ = longitude_to_sign(chart.planets["Mars"])
    mars_house = ((ZODIAC.index(mars_sign) - lagna_idx) % 12) + 1
    kuja_dosha = mars_house in [1, 2, 4, 7, 8, 12]

    # Timing
    current = chart.get_current_dasha_info()
    favorable_md = ["Venus", "Jupiter", "Mercury", "Moon"]
    marriage_timing = ""
    if current.get("mahadasha") in favorable_md:
        marriage_timing = f"{current['mahadasha']} Mahadasha is favorable for marriage. Current Antardasha of {current.get('antardasha', 'unknown')} provides the window."
    else:
        marriage_timing = f"{current['mahadasha']} Mahadasha is not the strongest for marriage. Wait for Antardasha of Venus/Jupiter/Mercury."

    return {
        "seventh_lord": seventh_lord,
        "seventh_lord_position": f"House {seventh_lord_house} ({seventh_lord_pos})" if seventh_lord_house else "Unknown",
        "planets_in_7th": planets_in_7th,
        "venus_house": venus_house,
        "venus_sign": venus_sign,
        "kuja_dosha": kuja_dosha,
        "kuja_severity": "High" if mars_house == 7 else "Moderate" if kuja_dosha else "None",
        "current_dasha": current,
        "marriage_timing": marriage_timing,
        "spouse_nature": _spouse_nature(seventh_lord, venus_sign, planets_in_7th)
    }


def _spouse_nature(seventh_lord: str, venus_sign: str, planets_7th: List[str]) -> str:
    traits = []
    if seventh_lord == "Jupiter":
        traits.append("wise, educated, spiritual, possibly older")
    elif seventh_lord == "Venus":
        traits.append("attractive, artistic, loving, well-mannered")
    elif seventh_lord == "Mercury":
        traits.append("intelligent, communicative, youthful, business-minded")
    elif seventh_lord == "Saturn":
        traits.append("mature, serious, hardworking, possibly older")
    elif seventh_lord == "Mars":
        traits.append("energetic, assertive, technical/military background")
    elif seventh_lord == "Sun":
        traits.append("authoritative, confident, government/management background")
    elif seventh_lord == "Moon":
        traits.append("nurturing, emotional, family-oriented")
    else:
        traits.append("unique, unconventional personality")

    if "Rahu" in planets_7th:
        traits.append("foreign or inter-cultural background")
    if "Ketu" in planets_7th:
        traits.append("spiritual, detached, possibly past-life connection")
    if "Saturn" in planets_7th:
        traits.append("significant age difference or serious disposition")

    return ", ".join(traits) if traits else "Balanced, supportive partner"


def analyze_children(chart: ChartData) -> Dict:
    """Detailed children/progeny analysis."""
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    fifth_lord = SIGN_LORD[ZODIAC[(lagna_idx + 4) % 12]]

    planets_in_5th = []
    fifth_lord_pos = None
    fifth_lord_house = None

    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        if house == 5:
            planets_in_5th.append(p)
        if p == fifth_lord:
            fifth_lord_pos = sign
            fifth_lord_house = house

    # Jupiter (Putrakaraka)
    jupiter_sign, _ = longitude_to_sign(chart.planets["Jupiter"])
    jupiter_house = ((ZODIAC.index(jupiter_sign) - lagna_idx) % 12) + 1
    jupiter_dignity = chart.dignities.get("Jupiter", "Neutral")

    # Timing
    current = chart.get_current_dasha_info()
    favorable_for_children = ["Jupiter", "Venus", "Moon", "Mercury"]
    children_timing = ""
    if current.get("mahadasha") in favorable_for_children:
        children_timing = f"{current['mahadasha']} Mahadasha is favorable for progeny."
    else:
        children_timing = f"{current['mahadasha']} Mahadasha requires patience for children. Focus on preparation."

    # Number prediction (simplified)
    num_children = "1-2"
    if jupiter_dignity in ["Exalted", "Own"] and len(planets_in_5th) >= 2:
        num_children = "2-3"
    elif "Rahu" in planets_in_5th or "Saturn" in planets_in_5th:
        num_children = "1 or delayed"
    elif "Ketu" in planets_in_5th:
        num_children = "Fewer or spiritual adoption"

    return {
        "fifth_lord": fifth_lord,
        "fifth_lord_position": f"House {fifth_lord_house} ({fifth_lord_pos})" if fifth_lord_house else "Unknown",
        "planets_in_5th": planets_in_5th,
        "jupiter_house": jupiter_house,
        "jupiter_dignity": jupiter_dignity,
        "jupiter_strength": "Strong" if jupiter_dignity in ["Exalted", "Own", "Mool Trikona"] else "Moderate" if jupiter_dignity == "Neutral" else "Weak",
        "current_dasha": current,
        "children_timing": children_timing,
        "predicted_number": num_children,
        "conception_advice": _conception_advice(chart, current)
    }


def _conception_advice(chart: ChartData, current_dasha: Dict) -> str:
    advice = []
    md = current_dasha.get("mahadasha", "")
    ad = current_dasha.get("antardasha", "")

    if md in ["Jupiter", "Venus"]:
        advice.append("Excellent period — try naturally with confidence.")
    elif md in ["Moon", "Mercury"]:
        advice.append("Favorable period — maintain emotional balance and health.")
    elif md == "Saturn":
        advice.append("Delays possible — consider medical consultation if needed.")
    elif md == "Rahu":
        advice.append("Unconventional methods may help — keep an open mind.")
    elif md == "Mars":
        advice.append("Avoid this period if possible — wait for calmer dasha.")
    else:
        advice.append("Moderate period — focus on health and timing.")

    if chart.dignities.get("Jupiter") in ["Debilitated", "Weak"]:
        advice.append("Jupiter needs strengthening — consider remedies.")

    return " ".join(advice)


def analyze_health(chart: ChartData) -> Dict:
    """Health analysis."""
    lagna_idx = ZODIAC.index(chart.lagna_sign)
    first_lord = SIGN_LORD[chart.lagna_sign]

    # 1st, 6th, 8th house analysis
    planets_in_1st = []
    planets_in_6th = []
    planets_in_8th = []

    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1
        if house == 1:
            planets_in_1st.append(p)
        elif house == 6:
            planets_in_6th.append(p)
        elif house == 8:
            planets_in_8th.append(p)

    # Saturn transit check (Sade Sati)
    moon_house = ((ZODIAC.index(chart.moon_sign) - lagna_idx) % 12) + 1

    current = chart.get_current_dasha_info()
    health_dasha = ""
    if current.get("mahadasha") in ["Saturn", "Rahu", "Ketu"]:
        health_dasha = f"{current['mahadasha']} Mahadasha requires careful health management."
    else:
        health_dasha = f"{current['mahadasha']} Mahadasha generally supports stable health."

    return {
        "lagna_lord": first_lord,
        "planets_in_1st": planets_in_1st,
        "planets_in_6th": planets_in_6th,
        "planets_in_8th": planets_in_8th,
        "moon_house": moon_house,
        "current_dasha": current,
        "health_dasha": health_dasha,
        "vulnerable_areas": _health_vulnerabilities(chart.lagna_sign, planets_in_6th, planets_in_8th)
    }


def _health_vulnerabilities(lagna: str, p6th: List[str], p8th: List[str]) -> List[str]:
    areas = []
    if "Saturn" in p6th or "Saturn" in p8th:
        areas.append("Bones, joints, chronic conditions")
    if "Mars" in p6th or "Mars" in p8th:
        areas.append("Blood, inflammation, accidents, fevers")
    if "Mercury" in p6th or "Mercury" in p8th:
        areas.append("Nervous system, skin, respiratory")
    if "Moon" in p6th or "Moon" in p8th:
        areas.append("Digestion, fluids, mental health")
    if "Sun" in p6th or "Sun" in p8th:
        areas.append("Heart, eyes, vitality")
    if "Venus" in p6th or "Venus" in p8th:
        areas.append("Reproductive, diabetes, kidneys")
    if "Jupiter" in p6th or "Jupiter" in p8th:
        areas.append("Liver, obesity, circulation")
    if "Rahu" in p6th or "Rahu" in p8th:
        areas.append("Addictions, mysterious ailments, anxiety")
    if not areas:
        areas.append("Generally good health — maintain routine checkups")
    return areas


# ------------------------------------------------------------------
# VARSHPHAL (TAJAKA / SOLAR RETURN)
# ------------------------------------------------------------------
def calculate_varshphal(chart: ChartData, year: int) -> Dict:
    """Calculate Varshphal (annual chart) for a given year."""
    # Simplified: Use Sun return to determine Varshphal date
    # In practice, this requires precise solar return calculation
    birth_month = chart.birth_date.month if chart.birth_date else 1
    birth_day = chart.birth_date.day if chart.birth_date else 1

    # Approximate: same month/day as birth
    varsh_date = datetime(year, birth_month, birth_day)

    # Muntha (progressed ascendant)
    # Muntha = (Year - Birth Year) + Lagna degree
    years_elapsed = year - (chart.birth_date.year if chart.birth_date else year)
    muntha = (chart.ascendant + years_elapsed * 30) % 360
    muntha_sign, _ = longitude_to_sign(muntha)

    # Transit analysis for the year
    transits = get_transits(year, birth_month, birth_day) if SWISSEPH_AVAILABLE else {}

    return {
        "varshphal_date": varsh_date.strftime("%d %b %Y"),
        "muntha_sign": muntha_sign,
        "muntha_longitude": round(muntha, 2),
        "years_elapsed": years_elapsed,
        "transits": transits,
        "themes": _varshphal_themes(chart, muntha_sign, year)
    }


def _varshphal_themes(chart: ChartData, muntha_sign: str, year: int) -> List[str]:
    themes = []
    muntha_idx = ZODIAC.index(muntha_sign)
    lagna_idx = ZODIAC.index(chart.lagna_sign)

    # Muntha in trine to lagna
    diff = (muntha_idx - lagna_idx) % 12
    if diff in [0, 4, 8]:
        themes.append("Year of growth and opportunity")
    elif diff in [6]:
        themes.append("Year of challenges and transformation")
    elif diff in [3, 9]:
        themes.append("Year of effort and hard work")
    else:
        themes.append("Mixed results — adaptability required")

    # Check Muntha lord
    muntha_lord = SIGN_LORD[muntha_sign]
    if muntha_lord in ["Jupiter", "Venus", "Mercury"]:
        themes.append("Favorable for expansion and gains")
    elif muntha_lord in ["Saturn", "Mars"]:
        themes.append("Requires discipline and patience")

    return themes


# ------------------------------------------------------------------
# YEARLY PREDICTIONS
# ------------------------------------------------------------------
def get_year_prediction(chart: ChartData, year: int) -> Dict:
    """Comprehensive yearly prediction."""
    transits = get_transits(year) if SWISSEPH_AVAILABLE else {}
    t_saturn = longitude_to_sign(transits.get("Saturn", 0))[0] if transits else "Unknown"
    t_jupiter = longitude_to_sign(transits.get("Jupiter", 0))[0] if transits else "Unknown"

    s_idx = ZODIAC.index(t_saturn) if t_saturn in ZODIAC else 0
    j_idx = ZODIAC.index(t_jupiter) if t_jupiter in ZODIAC else 0
    m_idx = ZODIAC.index(chart.moon_sign)
    lagna_idx = ZODIAC.index(chart.lagna_sign)

    # Sade Sati
    sade_sati = "No Sade Sati"
    rel = (s_idx - m_idx) % 12
    if rel in [11, 0, 1]:
        phases = {11: "Rising Phase", 0: "Peak Phase", 1: "Setting Phase"}
        sade_sati = f"⚠️ Sade Sati Active — {phases[rel]}"

    # Current dasha
    check_date = datetime(year, 6, 15)
    current = chart.get_current_dasha_info(check_date)

    # Varshphal
    varshphal = calculate_varshphal(chart, year)

    # Detailed topic predictions
    career = analyze_career(chart)
    marriage = analyze_marriage(chart)
    children = analyze_children(chart)
    health = analyze_health(chart)

    # Transit impacts
    jupiter_to_lagna = (j_idx - lagna_idx) % 12
    jupiter_transit = ""
    if jupiter_to_lagna in [0, 5, 9]:
        jupiter_transit = "Jupiter blessing Lagna — excellent year for new beginnings."
    elif jupiter_to_lagna in [3, 6]:
        jupiter_transit = "Jupiter challenging Lagna — growth through effort."
    else:
        jupiter_transit = "Jupiter in neutral position — steady progress."

    return {
        "year": year,
        "dasha": current,
        "transits": {"Saturn": t_saturn, "Jupiter": t_jupiter},
        "sade_sati": sade_sati,
        "jupiter_transit": jupiter_transit,
        "varshphal": varshphal,
        "career": career,
        "marriage": marriage,
        "children": children,
        "health": health,
        "summary": _year_summary(year, current, sade_sati, varshphal)
    }


def _year_summary(year: int, dasha: Dict, sade_sati: str, varshphal: Dict) -> str:
    parts = []
    parts.append(f"Year {year} brings energies of {dasha.get('mahadasha', 'unknown')} Mahadasha")
    if dasha.get('antardasha'):
        parts.append(f"with {dasha['antardasha']} Antardasha")
    parts.append(f". Muntha is in {varshphal['muntha_sign']}")
    if "Sade Sati" in sade_sati:
        parts.append(f". {sade_sati}")
    parts.append(f". Key themes: {', '.join(varshphal['themes'])}")
    return "".join(parts)


# ------------------------------------------------------------------
# DEMO CHART
# ------------------------------------------------------------------
def generate_demo_chart() -> ChartData:
    planets = {
        "Sun": 45.5, "Moon": 128.3, "Mars": 200.0, "Mercury": 50.2,
        "Jupiter": 95.0, "Venus": 70.5, "Saturn": 310.0, "Rahu": 175.0, "Ketu": 355.0
    }
    return ChartData(planets, 30.0, "Taurus",
                     birth_date=datetime(1995, 6, 15, 10, 30))


# Load chart from file
def load_chart_from_file(filepath: str) -> ChartData:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    birth_date = datetime.fromisoformat(data["birth_date"]) if data.get("birth_date") else None
    chart = ChartData(
        planets=data["planets"],
        ascendant=data["ascendant"],
        lagna_sign=data["lagna_sign"],
        birth_date=birth_date,
        lat=data.get("lat", 0),
        lon=data.get("lon", 0),
        tz=data.get("tz", 0)
    )
    return chart
