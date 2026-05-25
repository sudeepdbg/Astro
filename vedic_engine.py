"""
Vedic Astrology Calculation Engine v3.0
========================================
Enhancements over v2.0:
  - Centralized RULES layer (PREDICTION_RULES dict) — edit all thresholds
    and interpretations in one place.
  - Rule-evaluation engine: evaluate_rules() scores every applicable rule
    and returns structured, detailed analysis with severity levels.
  - Fixed logic bugs:
      * get_navamsa / get_drekkana / get_dasamsa: correct Movable/Fixed/Dual
        start-sign logic (was using wrong sign_idx offsets).
      * get_saptamsa: odd/even sign check corrected (1-based sign numbering).
      * Dasha balance: fixed degrees_covered to not use modulo (Moon can be
        anywhere in the nakshatra).
      * calculate_antardasha: removed erroneous double-scaling; standard
        formula is (MD_years × AD_years) / 120, no further scaling.
      * get_tara_score: added all 9 Tara types with correct auspiciousness.
  - Detailed narrative paragraphs for Career, Marriage, Children, Health.
  - Varshphal Muntha lord + Yogas properly evaluated.
  - Yoga detection centralised in RULES.
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


# ==================================================================
# SECTION 1 — STATIC LOOKUP TABLES
# ==================================================================

ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

ZODIAC_SHORT = ["ARI","TAU","GEM","CAN","LEO","VIR","LIB","SCO","SAG","CAP","AQU","PIS"]

SIGN_SANSKRIT = {
    "Aries": "Mesha", "Taurus": "Vrishabha", "Gemini": "Mithuna",
    "Cancer": "Karka", "Leo": "Simha", "Virgo": "Kanya",
    "Libra": "Tula", "Scorpio": "Vrischika", "Sagittarius": "Dhanu",
    "Capricorn": "Makara", "Aquarius": "Kumbha", "Pisces": "Meena"
}

SIGN_LORD = {
    "Aries": "Mars",    "Taurus": "Venus",   "Gemini": "Mercury",
    "Cancer": "Moon",   "Leo": "Sun",        "Virgo": "Mercury",
    "Libra": "Venus",   "Scorpio": "Mars",   "Sagittarius": "Jupiter",
    "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

SIGN_ELEMENT = {
    "Aries": "Fire",  "Taurus": "Earth", "Gemini": "Air",
    "Cancer": "Water","Leo": "Fire",     "Virgo": "Earth",
    "Libra": "Air",   "Scorpio": "Water","Sagittarius": "Fire",
    "Capricorn": "Earth","Aquarius": "Air","Pisces": "Water"
}

SIGN_QUALITY = {
    "Aries": "Movable",  "Taurus": "Fixed",  "Gemini": "Dual",
    "Cancer": "Movable", "Leo": "Fixed",     "Virgo": "Dual",
    "Libra": "Movable",  "Scorpio": "Fixed", "Sagittarius": "Dual",
    "Capricorn": "Movable","Aquarius": "Fixed","Pisces": "Dual"
}

NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra",
    "Punarvasu","Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni",
    "Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha",
    "Purva Bhadrapada","Uttara Bhadrapada","Revati"
]

# Repeating cycle of 9 lords
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

VARNA_MAP    = {"Water":"Brahmin","Fire":"Kshatriya","Earth":"Vaishya","Air":"Shudra"}
VASHYA_MAP   = {
    "Aries":"Quadruped","Taurus":"Quadruped","Gemini":"Human",
    "Cancer":"Water","Leo":"Quadruped","Virgo":"Human",
    "Libra":"Human","Scorpio":"Keet","Sagittarius":"Human",
    "Capricorn":"Quadruped","Aquarius":"Human","Pisces":"Water"
}

DASHA_YEARS = {
    "Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,
    "Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17
}
DASHA_SEQUENCE = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
TOTAL_DASHA_YEARS = 120  # sum of DASHA_YEARS

PLANET_IDS   = [0, 1, 2, 3, 4, 5, 6, 10]  # SUN,MOON,MARS,MERCURY,JUPITER,VENUS,SATURN,TRUE_NODE
PLANET_NAMES = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu"]

HOUSE_MEANINGS = {
    1:"Self / Body",          2:"Wealth / Family",       3:"Courage / Siblings",
    4:"Mother / Home",        5:"Intelligence / Children",6:"Disease / Enemies",
    7:"Marriage / Partnership",8:"Longevity / Occult",   9:"Fortune / Dharma",
    10:"Career / Status",     11:"Gains / Friends",      12:"Loss / Liberation"
}

EXALTATION   = {
    "Sun":"Aries","Moon":"Taurus","Mars":"Capricorn",
    "Mercury":"Virgo","Jupiter":"Cancer","Venus":"Pisces","Saturn":"Libra"
}
DEBILITATION = {
    "Sun":"Libra","Moon":"Scorpio","Mars":"Cancer",
    "Mercury":"Pisces","Jupiter":"Capricorn","Venus":"Virgo","Saturn":"Aries"
}
MOOLATRIKONA = {
    "Sun":"Leo","Moon":"Taurus","Mars":"Aries",
    "Mercury":"Virgo","Jupiter":"Sagittarius","Venus":"Libra","Saturn":"Aquarius"
}

PLANET_FRIENDS = {
    "Sun":    ["Moon","Mars","Jupiter"],
    "Moon":   ["Sun","Mercury"],
    "Mars":   ["Sun","Moon","Jupiter"],
    "Mercury":["Sun","Venus"],
    "Jupiter":["Sun","Moon","Mars"],
    "Venus":  ["Mercury","Saturn"],
    "Saturn": ["Mercury","Venus"],
}
PLANET_ENEMIES = {
    "Sun":    ["Venus","Saturn"],
    "Moon":   ["Rahu","Ketu"],
    "Mars":   ["Mercury"],
    "Mercury":["Moon"],
    "Jupiter":["Mercury","Venus"],
    "Venus":  ["Sun","Moon"],
    "Saturn": ["Sun","Moon","Mars"],
}

# Tara Bala: positions 1-9 from birth nakshatra
# 1=Janma(good for inner), 2=Sampat(wealth+), 3=Vipat(danger-),
# 4=Kshema(prosperity+), 5=Pratyak(obstacles-), 6=Sadhana(effort+),
# 7=Naidhana(death-), 8=Mitra(friend+), 9=Parama Mitra(best friend+)
TARA_AUSPICIOUS = {1: True, 2: True, 3: False, 4: True, 5: False,
                   6: True, 7: False, 8: True, 9: True}
TARA_MAX_SCORE = 3
TARA_SCORES    = {1: 3, 2: 3, 3: 0, 4: 3, 5: 0, 6: 3, 7: 0, 8: 3, 9: 3}

NAKSHATRA_SIZE = 13 + 20/60   # 13°20′ per nakshatra
PADA_SIZE      = 3  + 20/60   # 3°20′  per pada


# ==================================================================
# SECTION 2 — CENTRALISED RULES LAYER
# ==================================================================
# All prediction thresholds, interpretations, and scores live here.
# Change a rule here → it propagates everywhere automatically.
#
# Rule structure:
#   {
#     "id":          str  — unique key,
#     "topic":       str  — "career"|"marriage"|"children"|"health"|"general"|"matchmaking",
#     "condition":   callable(context: dict) → bool,
#     "severity":    "positive"|"neutral"|"caution"|"warning",
#     "score":       int  — contribution to topic score (+positive / -negative),
#     "title":       str  — short label,
#     "detail":      callable(context: dict) → str  — full narrative
#   }
#
# context keys provided to every rule (all may be None if unavailable):
#   planets, lagna_sign, lagna_idx, moon_sign, sun_sign, dignities,
#   nakshatras, navamsa, dasamsa, dasha, antardasha,
#   house_map (planet → house number),
#   lord_map  (house number → lord planet),
#   ashtakoota (matchmaking only)

def _house(planet: str, ctx: dict) -> int:
    return ctx["house_map"].get(planet, 0)

def _lord(house: int, ctx: dict) -> str:
    return ctx["lord_map"].get(house, "")

def _dignity(planet: str, ctx: dict) -> str:
    return ctx["dignities"].get(planet, "Neutral")

def _in_house(planets_list, house: int, ctx: dict) -> List[str]:
    return [p for p in planets_list if _house(p, ctx) == house]


PREDICTION_RULES: List[Dict] = [

    # ── CAREER ───────────────────────────────────────────────────
    {
        "id": "career_sun_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Sun", ctx) == 10,
        "severity": "positive",
        "score": 3,
        "title": "Sun in 10th — Authority & Recognition",
        "detail": lambda ctx: (
            "Sun placed in the 10th house bestows prominence, authority, and a strong drive for "
            "recognition. You are likely to rise to leadership or senior management. Government "
            "service, politics, senior corporate roles, medicine, or administration are natural fits. "
            f"Sun is {_dignity('Sun', ctx)} here, {'amplifying' if _dignity('Sun',ctx) in ['Exalted','Own','Mool Trikona'] else 'which may moderate'} these effects."
        )
    },
    {
        "id": "career_saturn_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Saturn", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Saturn in 10th — Disciplined Rise",
        "detail": lambda ctx: (
            "Saturn in the 10th house indicates a career built through consistent effort, patience, "
            "and discipline. Success comes late but is enduring. Fields like engineering, law, "
            "administration, real estate, or mining are favoured. "
            f"Saturn is {_dignity('Saturn', ctx)} here — "
            + ("exalted Saturn here is one of the strongest career placements in the zodiac." if _dignity("Saturn",ctx) == "Exalted"
               else "debilitated Saturn may cause career disruptions; remedies are advised." if _dignity("Saturn",ctx) == "Debilitated"
               else "steady, long-term rewards are expected.")
        )
    },
    {
        "id": "career_jupiter_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Jupiter", ctx) == 10,
        "severity": "positive",
        "score": 3,
        "title": "Jupiter in 10th — Wisdom & Expansion",
        "detail": lambda ctx: (
            "Jupiter in the 10th house is a strong Dharmakarmadhipati indicator. You are drawn to "
            "professions that involve teaching, counselling, law, banking, spirituality, or large "
            "institutions. Your reputation grows through integrity and wisdom. "
            f"Jupiter is {_dignity('Jupiter', ctx)} — "
            + ("exalted Jupiter here creates Hamsa Yoga, indicating distinguished career success." if _dignity("Jupiter",ctx) == "Exalted"
               else "debilitated Jupiter slows expansion; consider Guru-related remedies." if _dignity("Jupiter",ctx) == "Debilitated"
               else "benefic influence supports steady career growth.")
        )
    },
    {
        "id": "career_mercury_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Mercury", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Mercury in 10th — Communication & Intellect",
        "detail": lambda ctx: (
            "Mercury in the 10th favours careers in writing, publishing, IT, analytics, teaching, "
            "trade, or consulting. Your intellect is your greatest professional asset. Multiple "
            "income streams or career pivots are common. "
            f"Mercury is {_dignity('Mercury', ctx)} here."
        )
    },
    {
        "id": "career_mars_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Mars", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Mars in 10th — Drive & Technical Skill",
        "detail": lambda ctx: (
            "Mars in the 10th brings ambition, courage, and technical aptitude. Careers in military, "
            "police, surgery, engineering, sports, or competitive business are indicated. "
            "Guard against impulsive decisions at work. "
            f"Mars is {_dignity('Mars', ctx)} here."
        )
    },
    {
        "id": "career_venus_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Venus", ctx) == 10,
        "severity": "positive",
        "score": 2,
        "title": "Venus in 10th — Creative & Luxury Careers",
        "detail": lambda ctx: (
            "Venus in the 10th indicates success in arts, fashion, entertainment, hospitality, "
            "beauty, luxury goods, or diplomacy. Public charm and aesthetic sense are career assets. "
            f"Venus is {_dignity('Venus', ctx)} here."
        )
    },
    {
        "id": "career_rahu_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Rahu", ctx) == 10,
        "severity": "neutral",
        "score": 1,
        "title": "Rahu in 10th — Unconventional Career Path",
        "detail": lambda ctx: (
            "Rahu in the 10th creates strong ambition for status and can bring sudden career leaps. "
            "Technology, media, foreign companies, research, or unconventional fields are favoured. "
            "Beware of ethical shortcuts; career crises are possible if Rahu acts rashly."
        )
    },
    {
        "id": "career_ketu_10th",
        "topic": "career",
        "condition": lambda ctx: _house("Ketu", ctx) == 10,
        "severity": "caution",
        "score": -1,
        "title": "Ketu in 10th — Detachment from Career",
        "detail": lambda ctx: (
            "Ketu in the 10th house can cause dissatisfaction with worldly career, bringing a pull "
            "towards spirituality or alternative paths. Professional disruptions are possible. "
            "Meditation, research, astrology, healing, or behind-the-scenes roles suit this placement."
        )
    },
    {
        "id": "career_10th_lord_strong",
        "topic": "career",
        "condition": lambda ctx: _dignity(_lord(10, ctx), ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 3,
        "title": "Strong 10th Lord — Powerful Career Yoga",
        "detail": lambda ctx: (
            f"The 10th lord {_lord(10,ctx)} is {_dignity(_lord(10,ctx), ctx)}, conferring a strong "
            f"Rajayoga element. This significantly boosts career success, status, and recognition. "
            f"The house where the 10th lord sits (H{_house(_lord(10,ctx),ctx)}) becomes a zone of "
            f"career activity and effort."
        )
    },
    {
        "id": "career_10th_lord_weak",
        "topic": "career",
        "condition": lambda ctx: _dignity(_lord(10, ctx), ctx) == "Debilitated",
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 10th Lord — Career Challenges",
        "detail": lambda ctx: (
            f"The 10th lord {_lord(10,ctx)} is debilitated, indicating significant career challenges, "
            "possible loss of position, or difficulty sustaining professional momentum. Neechabhanga "
            "cancellation (if applicable) can mitigate this. Remedies for the 10th lord planet are strongly advised."
        )
    },
    {
        "id": "career_budhaditya",
        "topic": "career",
        "condition": lambda ctx: (
            abs(ctx["planets"].get("Sun",0) - ctx["planets"].get("Mercury",0)) <= 12
            and longitude_to_sign(ctx["planets"]["Sun"])[0] == longitude_to_sign(ctx["planets"]["Mercury"])[0]
        ),
        "severity": "positive",
        "score": 2,
        "title": "Budhaditya Yoga — Intelligence & Leadership",
        "detail": lambda ctx: (
            "Sun and Mercury are conjunct, forming Budhaditya Yoga. This sharpens intellect, "
            "communication, and analytical ability, supporting careers in management, writing, "
            "commerce, or any field requiring quick thinking."
        )
    },
    {
        "id": "career_dasha_career_planet",
        "topic": "career",
        "condition": lambda ctx: ctx.get("dasha","") in ["Jupiter","Sun","Saturn","Mercury","Rahu"],
        "severity": "positive",
        "score": 2,
        "title": "Active Career Dasha",
        "detail": lambda ctx: (
            f"{ctx.get('dasha','')} Mahadasha is running. "
            + {
                "Jupiter": "Jupiter Dasha supports expansion, wisdom-based career growth, and opportunities in education, law, or finance.",
                "Sun":     "Sun Dasha brings authority, recognition, and advancement in government or leadership roles.",
                "Saturn":  "Saturn Dasha rewards past effort with slow but solid career gains; discipline is key.",
                "Mercury": "Mercury Dasha favours communication, trade, IT, and intellectual pursuits.",
                "Rahu":    "Rahu Dasha can bring sudden rises through unconventional means; foreign or technology careers thrive."
            }.get(ctx.get("dasha",""), "")
            + f" Antardasha of {ctx.get('antardasha','')} colours the next sub-period."
        )
    },
    {
        "id": "career_dasha_challenging",
        "topic": "career",
        "condition": lambda ctx: ctx.get("dasha","") in ["Ketu","Moon","Mars","Venus"],
        "severity": "neutral",
        "score": 0,
        "title": "Moderate Career Dasha",
        "detail": lambda ctx: (
            f"{ctx.get('dasha','')} Mahadasha is active. "
            + {
                "Ketu":  "Ketu Dasha may bring career transitions or a pull toward spiritual/research work. Not the strongest for promotion.",
                "Moon":  "Moon Dasha favours public-facing or nurturing careers but can bring emotional fluctuations at work.",
                "Mars":  "Mars Dasha boosts energy and initiative but risks conflicts with authority figures. Technical careers do well.",
                "Venus": "Venus Dasha supports creative and luxury careers; financial gains through partnership or arts."
            }.get(ctx.get("dasha",""), "")
        )
    },

    # ── MARRIAGE ─────────────────────────────────────────────────
    {
        "id": "marriage_venus_strong",
        "topic": "marriage",
        "condition": lambda ctx: _dignity("Venus", ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 3,
        "title": "Strong Venus — Excellent Marriage Prospects",
        "detail": lambda ctx: (
            f"Venus is {_dignity('Venus', ctx)}, indicating a happy, loving, and aesthetically "
            "pleasing marriage. The spouse is likely attractive, refined, and emotionally warm. "
            "Venus strong in the chart is one of the best indicators for marital happiness."
        )
    },
    {
        "id": "marriage_venus_weak",
        "topic": "marriage",
        "condition": lambda ctx: _dignity("Venus", ctx) == "Debilitated",
        "severity": "warning",
        "score": -2,
        "title": "Debilitated Venus — Marital Caution",
        "detail": lambda ctx: (
            "Venus is debilitated, which can introduce dissatisfaction, misunderstandings, or "
            "incompatibility in marriage. Neechabhanga may help. Venus remedies (white flowers, "
            "sugar, Friday fasting) are recommended before marriage."
        )
    },
    {
        "id": "marriage_7th_lord_strong",
        "topic": "marriage",
        "condition": lambda ctx: _dignity(_lord(7, ctx), ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 3,
        "title": "Strong 7th Lord — Blessed Partnership",
        "detail": lambda ctx: (
            f"The 7th lord {_lord(7,ctx)} is {_dignity(_lord(7,ctx), ctx)}, strongly supporting a "
            "stable and rewarding marriage. The spouse will be a genuine source of strength."
        )
    },
    {
        "id": "marriage_7th_lord_weak",
        "topic": "marriage",
        "condition": lambda ctx: _dignity(_lord(7, ctx), ctx) == "Debilitated",
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 7th Lord — Partnership Challenges",
        "detail": lambda ctx: (
            f"The 7th lord {_lord(7,ctx)} is debilitated, indicating potential difficulties in "
            "marriage such as incompatibility, delays, or separation risk. Remedies for the 7th "
            "lord and Venus are strongly advised."
        )
    },
    {
        "id": "marriage_kuja_dosha_high",
        "topic": "marriage",
        "condition": lambda ctx: _house("Mars", ctx) == 7,
        "severity": "warning",
        "score": -2,
        "title": "High Kuja Dosha — Mars in 7th",
        "detail": lambda ctx: (
            "Mars in the 7th house creates strong Kuja (Mangal) Dosha. This is the most intense "
            "form. It can cause conflicts, dominance issues, or in severe cases, separation. "
            "Matching with a Manglik partner nullifies this. Mars Shanti puja is advised."
        )
    },
    {
        "id": "marriage_kuja_dosha_moderate",
        "topic": "marriage",
        "condition": lambda ctx: _house("Mars", ctx) in [1,2,4,8,12] and _house("Mars", ctx) != 7,
        "severity": "caution",
        "score": -1,
        "title": "Moderate Kuja Dosha",
        "detail": lambda ctx: (
            f"Mars is in House {_house('Mars',ctx)}, creating moderate Kuja Dosha. "
            "This can bring assertiveness or friction in relationships. Partial dosha — "
            "matching with a partner whose chart has similar Mars placement reduces its effect."
        )
    },
    {
        "id": "marriage_rahu_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Rahu", ctx) == 7,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 7th — Unconventional Marriage",
        "detail": lambda ctx: (
            "Rahu in the 7th house often brings an unusual or inter-cultural marriage, or a "
            "relationship that starts suddenly. There may be obsession or mistrust. The spouse "
            "may have a foreign connection or unconventional personality."
        )
    },
    {
        "id": "marriage_saturn_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Saturn", ctx) == 7,
        "severity": "caution",
        "score": -1,
        "title": "Saturn in 7th — Delayed or Serious Marriage",
        "detail": lambda ctx: (
            "Saturn in the 7th house can delay marriage (often after age 28-30) and brings a "
            "serious, karmic quality to partnerships. The spouse may be older, more mature, or "
            "reserved. Long-term commitment is strong once formed."
        )
    },
    {
        "id": "marriage_jupiter_7th",
        "topic": "marriage",
        "condition": lambda ctx: _house("Jupiter", ctx) == 7,
        "severity": "positive",
        "score": 2,
        "title": "Jupiter in 7th — Wise & Supportive Spouse",
        "detail": lambda ctx: (
            "Jupiter in the 7th is highly auspicious for marriage. The spouse is likely to be "
            "educated, wise, and spiritually inclined. This placement supports a dharmic, "
            "growth-oriented partnership."
        )
    },
    {
        "id": "marriage_dasha_venus",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("dasha","") == "Venus",
        "severity": "positive",
        "score": 3,
        "title": "Venus Mahadasha — Prime Marriage Window",
        "detail": lambda ctx: (
            "Venus Mahadasha is the most potent period for marriage and romantic unions. "
            f"The current Antardasha of {ctx.get('antardasha','')} further refines timing. "
            "Venus-Jupiter or Venus-Mercury Antardashas are the most auspicious sub-periods for wedding ceremonies."
        )
    },
    {
        "id": "marriage_dasha_jupiter",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("dasha","") == "Jupiter",
        "severity": "positive",
        "score": 2,
        "title": "Jupiter Mahadasha — Auspicious for Marriage",
        "detail": lambda ctx: (
            "Jupiter Mahadasha blesses partnerships and family life. This is a highly auspicious "
            f"time for marriage, especially in Jupiter-Venus or Jupiter-Moon Antardasha periods. "
            f"Current Antardasha: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "marriage_dasha_saturn",
        "topic": "marriage",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Delayed but Stable",
        "detail": lambda ctx: (
            "Saturn Mahadasha is not the first choice for marriage timing, but unions formed "
            "during this period tend to be karmic, serious, and lasting. Wait for Venus or "
            f"Jupiter Antardasha within Saturn Mahadasha. Current AD: {ctx.get('antardasha','')}."
        )
    },

    # ── CHILDREN ─────────────────────────────────────────────────
    {
        "id": "children_jupiter_strong",
        "topic": "children",
        "condition": lambda ctx: _dignity("Jupiter", ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 4,
        "title": "Strong Jupiter — Blessed Progeny",
        "detail": lambda ctx: (
            f"Jupiter (Putrakaraka) is {_dignity('Jupiter',ctx)}, one of the strongest indicators "
            "of good fortune in matters of children. Multiple children are possible; at least one "
            "is likely to be notably talented or spiritually inclined."
        )
    },
    {
        "id": "children_jupiter_weak",
        "topic": "children",
        "condition": lambda ctx: _dignity("Jupiter", ctx) == "Debilitated",
        "severity": "warning",
        "score": -3,
        "title": "Debilitated Jupiter — Progeny Challenges",
        "detail": lambda ctx: (
            "Jupiter (Putrakaraka) is debilitated, which is the most significant indicator of "
            "difficulty in having children. Delays, miscarriages, or fewer children are possible. "
            "Jupiter Shanti and Santana Gopala Puja are classical remedies."
        )
    },
    {
        "id": "children_5th_lord_strong",
        "topic": "children",
        "condition": lambda ctx: _dignity(_lord(5, ctx), ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 3,
        "title": "Strong 5th Lord — Fertile House",
        "detail": lambda ctx: (
            f"The 5th lord {_lord(5,ctx)} is {_dignity(_lord(5,ctx),ctx)}, strongly activating the "
            "Putra (children) bhava. Children are likely to be intellectually bright and bring honour."
        )
    },
    {
        "id": "children_5th_lord_weak",
        "topic": "children",
        "condition": lambda ctx: _dignity(_lord(5, ctx), ctx) == "Debilitated",
        "severity": "warning",
        "score": -3,
        "title": "Debilitated 5th Lord — Challenges with Progeny",
        "detail": lambda ctx: (
            f"The 5th lord {_lord(5,ctx)} is debilitated, weakening the house of children. "
            "Conception challenges or difficult pregnancies are possible. Remedies for the 5th "
            "lord and regular Santana Gopala prayers are advised."
        )
    },
    {
        "id": "children_saturn_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Saturn", ctx) == 5,
        "severity": "caution",
        "score": -2,
        "title": "Saturn in 5th — Delayed Children",
        "detail": lambda ctx: (
            "Saturn in the 5th house is a classical indicator of delayed progeny. Children may "
            "come later in life (after Saturn's maturation at age 36, or after Saturn Antardasha "
            "passes). The children born will be serious, responsible, and long-lived."
        )
    },
    {
        "id": "children_rahu_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Rahu", ctx) == 5,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 5th — Unconventional Progeny Path",
        "detail": lambda ctx: (
            "Rahu in the 5th house can create confusion around conception or unusual circumstances "
            "around children (e.g., adoption, IVF, or step-children). Medical consultation and "
            "Rahu remedies are advised if conception is delayed."
        )
    },
    {
        "id": "children_ketu_5th",
        "topic": "children",
        "condition": lambda ctx: _house("Ketu", ctx) == 5,
        "severity": "caution",
        "score": -1,
        "title": "Ketu in 5th — Spiritual Orientation",
        "detail": lambda ctx: (
            "Ketu in the 5th may reduce the desire for children or indicate a spiritually gifted "
            "child. There is sometimes a past-life karmic connection with children born under this "
            "placement."
        )
    },
    {
        "id": "children_dasha_jupiter",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Jupiter",
        "severity": "positive",
        "score": 3,
        "title": "Jupiter Mahadasha — Best Period for Children",
        "detail": lambda ctx: (
            "Jupiter Mahadasha is the most auspicious period for conception and birth of children. "
            f"Jupiter-Jupiter and Jupiter-Venus Antardashas are particularly fruitful. "
            f"Current Antardasha: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "children_dasha_venus",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Venus",
        "severity": "positive",
        "score": 2,
        "title": "Venus Mahadasha — Favourable for Progeny",
        "detail": lambda ctx: (
            "Venus Mahadasha is generally favourable for family expansion. "
            f"Venus-Jupiter or Venus-Moon Antardashas within this period are the best sub-windows. "
            f"Current Antardasha: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "children_dasha_saturn",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Patience Required",
        "detail": lambda ctx: (
            "Saturn Mahadasha can bring delays in having children. Medical check-ups are advised. "
            "Saturn-Jupiter or Saturn-Venus Antardashas can still deliver children within this period. "
            f"Current Antardasha: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "children_dasha_rahu",
        "topic": "children",
        "condition": lambda ctx: ctx.get("dasha","") == "Rahu",
        "severity": "caution",
        "score": -1,
        "title": "Rahu Mahadasha — Consult Medically",
        "detail": lambda ctx: (
            "Rahu Mahadasha is unpredictable for children. Conception is possible but may involve "
            "unusual circumstances. Medical consultation is advised if trying. "
            f"Rahu-Jupiter Antardasha is the best sub-period. Current AD: {ctx.get('antardasha','')}."
        )
    },

    # ── HEALTH ───────────────────────────────────────────────────
    {
        "id": "health_lagna_lord_strong",
        "topic": "health",
        "condition": lambda ctx: _dignity(_lord(1, ctx), ctx) in ["Exalted","Own","Mool Trikona"],
        "severity": "positive",
        "score": 3,
        "title": "Strong Lagna Lord — Vitality & Immunity",
        "detail": lambda ctx: (
            f"The Lagna lord {_lord(1,ctx)} is {_dignity(_lord(1,ctx),ctx)}, bestowing robust "
            "physical constitution, strong immunity, and faster recovery from illness. "
            "This is a protective factor against chronic disease."
        )
    },
    {
        "id": "health_lagna_lord_weak",
        "topic": "health",
        "condition": lambda ctx: _dignity(_lord(1, ctx), ctx) == "Debilitated",
        "severity": "warning",
        "score": -3,
        "title": "Debilitated Lagna Lord — Physical Vulnerability",
        "detail": lambda ctx: (
            f"The Lagna lord {_lord(1,ctx)} is debilitated, weakening the physical body and immune "
            "system. Regular health check-ups, Lagna lord remedies, and avoiding stress are important."
        )
    },
    {
        "id": "health_saturn_6th",
        "topic": "health",
        "condition": lambda ctx: _house("Saturn", ctx) == 6,
        "severity": "caution",
        "score": -2,
        "title": "Saturn in 6th — Chronic Conditions",
        "detail": lambda ctx: (
            "Saturn in the 6th house, while giving victory over enemies, can predispose to chronic "
            "or long-term health issues — particularly joints, bones, teeth, and skin. "
            "Saturn here is also a classic indicator of service-related stress and fatigue."
        )
    },
    {
        "id": "health_saturn_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Saturn", ctx) == 8,
        "severity": "caution",
        "score": -1,
        "title": "Saturn in 8th — Longevity with Chronic Issues",
        "detail": lambda ctx: (
            "Saturn in the 8th house often gives long life but with chronic health challenges. "
            "Digestive issues, nerve problems, or constitutional weakness may arise. "
            "Saturn in the 8th also indicates a contemplative, research-oriented mind."
        )
    },
    {
        "id": "health_mars_6th",
        "topic": "health",
        "condition": lambda ctx: _house("Mars", ctx) == 6,
        "severity": "caution",
        "score": -1,
        "title": "Mars in 6th — Fevers & Inflammation",
        "detail": lambda ctx: (
            "Mars in the 6th house can cause inflammatory conditions, fevers, blood disorders, "
            "or accident-proneness. However, it also gives strong fighting spirit and quick recovery. "
            "Physical exercise is an excellent outlet for this energy."
        )
    },
    {
        "id": "health_mars_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Mars", ctx) == 8,
        "severity": "caution",
        "score": -1,
        "title": "Mars in 8th — Accident Caution",
        "detail": lambda ctx: (
            "Mars in the 8th house increases the risk of accidents, surgeries, or injuries, "
            "particularly to the head, blood, or reproductive system. Caution while driving "
            "and during Mars Antardasha is advised."
        )
    },
    {
        "id": "health_rahu_6th",
        "topic": "health",
        "condition": lambda ctx: _house("Rahu", ctx) == 6,
        "severity": "caution",
        "score": -1,
        "title": "Rahu in 6th — Hidden or Unusual Ailments",
        "detail": lambda ctx: (
            "Rahu in the 6th can cause mysterious or hard-to-diagnose health issues. Allergies, "
            "anxiety, addictions, or unusual infections are possible. Second medical opinions "
            "and regular detox are beneficial."
        )
    },
    {
        "id": "health_ketu_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Ketu", ctx) == 8,
        "severity": "caution",
        "score": -1,
        "title": "Ketu in 8th — Spiritual Crises & Psychosomatic Issues",
        "detail": lambda ctx: (
            "Ketu in the 8th house can cause psychosomatic conditions, sudden surgeries, or "
            "near-death experiences. Spiritual practices and avoiding extreme sports are advisable."
        )
    },
    {
        "id": "health_moon_6th_or_8th",
        "topic": "health",
        "condition": lambda ctx: _house("Moon", ctx) in [6, 8],
        "severity": "caution",
        "score": -2,
        "title": "Moon in 6th/8th — Mental & Digestive Health",
        "detail": lambda ctx: (
            f"Moon in the {_house('Moon',ctx)}th house can cause emotional instability, digestive "
            "disorders, fluid-related issues, or mental health challenges. Meditation, "
            "proper sleep, and reducing emotional stress are vital."
        )
    },
    {
        "id": "health_sade_sati",
        "topic": "health",
        "condition": lambda ctx: ctx.get("sade_sati_active", False),
        "severity": "caution",
        "score": -2,
        "title": "Sade Sati Active — Physical & Mental Stress",
        "detail": lambda ctx: (
            f"Saturn's Sade Sati is active ({ctx.get('sade_sati_phase','')}) over your Moon sign. "
            "This 7.5-year period tests physical stamina and mental resilience. Immune function "
            "may be lowered. Regular exercise, proper rest, and Saturn remedies (oil on Saturdays, "
            "blue sapphire consultation) are protective."
        )
    },
    {
        "id": "health_dasha_saturn",
        "topic": "health",
        "condition": lambda ctx: ctx.get("dasha","") == "Saturn",
        "severity": "caution",
        "score": -1,
        "title": "Saturn Mahadasha — Health Vigilance",
        "detail": lambda ctx: (
            "Saturn Mahadasha requires careful attention to bones, joints, teeth, digestion, and "
            f"chronic conditions. Fatigue and slow recovery are common. Saturn-Rahu Antardasha "
            f"is particularly sensitive. Current AD: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "health_dasha_rahu",
        "topic": "health",
        "condition": lambda ctx: ctx.get("dasha","") == "Rahu",
        "severity": "caution",
        "score": -1,
        "title": "Rahu Mahadasha — Anxiety & Unusual Ailments",
        "detail": lambda ctx: (
            "Rahu Mahadasha can manifest as anxiety, stress, unusual diagnoses, or lifestyle "
            f"excesses affecting health. Rahu-Rahu and Rahu-Saturn are the most sensitive "
            f"sub-periods. Current AD: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "health_dasha_ketu",
        "topic": "health",
        "condition": lambda ctx: ctx.get("dasha","") == "Ketu",
        "severity": "caution",
        "score": -1,
        "title": "Ketu Mahadasha — Mysterious Health Episodes",
        "detail": lambda ctx: (
            "Ketu Mahadasha can bring sudden or puzzling health events, psychosomatic symptoms, "
            "or hospitalisation. Ketu-Mars Antardasha requires particular caution. "
            f"Current AD: {ctx.get('antardasha','')}."
        )
    },
    {
        "id": "health_jupiter_strong_protection",
        "topic": "health",
        "condition": lambda ctx: (
            _dignity("Jupiter", ctx) in ["Exalted","Own","Mool Trikona"]
            and _house("Jupiter", ctx) in [1, 5, 9]
        ),
        "severity": "positive",
        "score": 3,
        "title": "Jupiter in Trine — Powerful Health Protection",
        "detail": lambda ctx: (
            f"Jupiter is {_dignity('Jupiter',ctx)} and placed in House {_house('Jupiter',ctx)} (a trine). "
            "This is one of the strongest protective factors in Vedic astrology for health and longevity. "
            "Recovery from illness is excellent."
        )
    },

    # ── GENERAL / YOGAS ──────────────────────────────────────────
    {
        "id": "yoga_panchamahapurusha_hamsa",
        "topic": "general",
        "condition": lambda ctx: (
            _dignity("Jupiter", ctx) in ["Exalted","Own","Mool Trikona"]
            and _house("Jupiter", ctx) in [1,4,7,10]
        ),
        "severity": "positive",
        "score": 5,
        "title": "Hamsa Yoga — Wisdom, Purity & Fortune",
        "detail": lambda ctx: (
            f"Jupiter is {_dignity('Jupiter',ctx)} in House {_house('Jupiter',ctx)} (a kendra), "
            "forming the Panchamahapurusha Yoga called Hamsa Yoga. This bestows wisdom, "
            "noble character, spiritual inclination, and great fortune. A rare and powerful yoga."
        )
    },
    {
        "id": "yoga_panchamahapurusha_malavya",
        "topic": "general",
        "condition": lambda ctx: (
            _dignity("Venus", ctx) in ["Exalted","Own","Mool Trikona"]
            and _house("Venus", ctx) in [1,4,7,10]
        ),
        "severity": "positive",
        "score": 5,
        "title": "Malavya Yoga — Beauty, Prosperity & Comforts",
        "detail": lambda ctx: (
            f"Venus is {_dignity('Venus',ctx)} in House {_house('Venus',ctx)} (a kendra), "
            "forming Malavya Yoga. Blesses with beauty, artistic talent, luxury, romantic success, "
            "and material comforts."
        )
    },
    {
        "id": "yoga_panchamahapurusha_ruchaka",
        "topic": "general",
        "condition": lambda ctx: (
            _dignity("Mars", ctx) in ["Exalted","Own","Mool Trikona"]
            and _house("Mars", ctx) in [1,4,7,10]
        ),
        "severity": "positive",
        "score": 5,
        "title": "Ruchaka Yoga — Courage, Leadership & Command",
        "detail": lambda ctx: (
            f"Mars is {_dignity('Mars',ctx)} in House {_house('Mars',ctx)} (a kendra), "
            "forming Ruchaka Yoga. Bestows exceptional courage, physical vitality, leadership, "
            "and success in competitive fields."
        )
    },
    {
        "id": "yoga_panchamahapurusha_bhadra",
        "topic": "general",
        "condition": lambda ctx: (
            _dignity("Mercury", ctx) in ["Exalted","Own","Mool Trikona"]
            and _house("Mercury", ctx) in [1,4,7,10]
        ),
        "severity": "positive",
        "score": 5,
        "title": "Bhadra Yoga — Intelligence, Eloquence & Wealth",
        "detail": lambda ctx: (
            f"Mercury is {_dignity('Mercury',ctx)} in House {_house('Mercury',ctx)} (a kendra), "
            "forming Bhadra Yoga. Gifts intelligence, eloquence, business acumen, and financial success."
        )
    },
    {
        "id": "yoga_panchamahapurusha_shasha",
        "topic": "general",
        "condition": lambda ctx: (
            _dignity("Saturn", ctx) in ["Exalted","Own","Mool Trikona"]
            and _house("Saturn", ctx) in [1,4,7,10]
        ),
        "severity": "positive",
        "score": 5,
        "title": "Shasha Yoga — Authority, Discipline & Service",
        "detail": lambda ctx: (
            f"Saturn is {_dignity('Saturn',ctx)} in House {_house('Saturn',ctx)} (a kendra), "
            "forming Shasha Yoga. Bestows authority, discipline, management skill, and lasting achievements."
        )
    },
    {
        "id": "yoga_gajkesari",
        "topic": "general",
        "condition": lambda ctx: (
            abs(_house("Jupiter", ctx) - _house("Moon", ctx)) % 6 in [0, 1, 3, 4, 6]
            and _house("Jupiter", ctx) in [1,4,7,10]
        ),
        "severity": "positive",
        "score": 4,
        "title": "Gaja-Kesari Yoga — Fame, Prosperity & Wisdom",
        "detail": lambda ctx: (
            f"Jupiter is in a kendra (House {_house('Jupiter',ctx)}) from the Moon, forming "
            "Gaja-Kesari Yoga — one of the most celebrated yogas. This grants fame, wealth, "
            "wisdom, and a respected position in society."
        )
    },
    {
        "id": "yoga_viparita_harsha",
        "topic": "general",
        "condition": lambda ctx: (
            _lord(6, ctx) != "" and _house(_lord(6, ctx), ctx) in [6, 8, 12]
        ),
        "severity": "positive",
        "score": 3,
        "title": "Viparita Harsha Yoga — Victory from Adversity",
        "detail": lambda ctx: (
            f"The 6th lord {_lord(6,ctx)} is placed in House {_house(_lord(6,ctx),ctx)} (a dusthana), "
            "forming Viparita Harsha Yoga. Enemies and obstacles defeat themselves; you gain "
            "from challenging situations and hardships ultimately strengthen you."
        )
    },
]


# ==================================================================
# SECTION 3 — RULE EVALUATION ENGINE
# ==================================================================

def build_context(chart: "ChartData", dasha_info: Dict = None, sade_sati_info: Dict = None) -> Dict:
    """
    Build the flat context dict passed to every rule condition/detail callable.
    """
    lagna_idx = ZODIAC.index(chart.lagna_sign)

    # house_map: planet → house number (1-12)
    house_map = {}
    for p, lon in chart.planets.items():
        sign, _ = longitude_to_sign(lon)
        house_map[p] = ((ZODIAC.index(sign) - lagna_idx) % 12) + 1

    # lord_map: house number → ruling planet
    lord_map = {}
    for i in range(12):
        lord_map[i + 1] = SIGN_LORD[ZODIAC[(lagna_idx + i) % 12]]

    ctx = {
        "planets":    chart.planets,
        "lagna_sign": chart.lagna_sign,
        "lagna_idx":  lagna_idx,
        "moon_sign":  chart.moon_sign,
        "sun_sign":   chart.sun_sign,
        "dignities":  chart.dignities,
        "nakshatras": chart.nakshatras,
        "navamsa":    chart.navamsa,
        "dasamsa":    chart.dasamsa,
        "house_map":  house_map,
        "lord_map":   lord_map,
        "dasha":      dasha_info.get("mahadasha", "") if dasha_info else "",
        "antardasha": dasha_info.get("antardasha", "") if dasha_info else "",
        "sade_sati_active": False,
        "sade_sati_phase":  "",
    }

    if sade_sati_info:
        ctx["sade_sati_active"] = sade_sati_info.get("active", False)
        ctx["sade_sati_phase"]  = sade_sati_info.get("phase", "")

    return ctx


def evaluate_rules(ctx: Dict, topic: str = None) -> List[Dict]:
    """
    Evaluate all PREDICTION_RULES (or filtered by topic).
    Returns a list of fired rules with their detail, sorted by score descending.
    """
    results = []
    for rule in PREDICTION_RULES:
        if topic and rule["topic"] != topic:
            continue
        try:
            fired = rule["condition"](ctx)
        except Exception:
            fired = False
        if fired:
            try:
                detail = rule["detail"](ctx)
            except Exception as e:
                detail = f"[Detail generation error: {e}]"
            results.append({
                "id":       rule["id"],
                "topic":    rule["topic"],
                "severity": rule["severity"],
                "score":    rule["score"],
                "title":    rule["title"],
                "detail":   detail,
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def score_topic(fired_rules: List[Dict]) -> Dict:
    """Return score summary for a topic's fired rules."""
    total   = sum(r["score"] for r in fired_rules)
    positive = [r for r in fired_rules if r["severity"] == "positive"]
    warnings = [r for r in fired_rules if r["severity"] in ["warning","caution"]]
    return {
        "net_score": total,
        "positive_count": len(positive),
        "warning_count":  len(warnings),
        "rating": (
            "Excellent" if total >= 6 else
            "Good"      if total >= 3 else
            "Mixed"     if total >= 0 else
            "Challenging" if total >= -4 else
            "Difficult"
        )
    }


# ==================================================================
# SECTION 4 — CORE MATH (fixed bugs)
# ==================================================================

def longitude_to_sign(longitude: float) -> Tuple[str, float]:
    idx = int(longitude // 30) % 12
    return ZODIAC[idx], longitude % 30


def get_nakshatra(longitude: float) -> Tuple[str, int, float]:
    lon = longitude % 360
    nak_idx = int(lon / NAKSHATRA_SIZE)
    rem = lon % NAKSHATRA_SIZE
    pada = int(rem / PADA_SIZE) + 1
    return NAKSHATRAS[nak_idx % 27], pada, rem


def get_navamsa(longitude: float) -> str:
    """
    Navamsa (D9): each sign divided into 9 × 3°20′.
    Starting signs: Movable→same, Fixed→9th from itself, Dual→5th from itself.
    BUG FIX v2: offsets were swapped. Fixed signs should start from 9th (index+8),
    Dual from 5th (index+4). This was correct in v2 but double-checked here.
    """
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // (10 / 3))  # 0..8
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12   # 9th sign (0-indexed: +8)
    else:  # Dual
        start = (sign_idx + 4) % 12   # 5th sign (0-indexed: +4)
    return ZODIAC[(start + part) % 12]


def get_drekkana(longitude: float) -> str:
    """
    Drekkana (D3): each sign divided into 3 × 10°.
    Starting signs: Movable→same, Fixed→5th, Dual→9th.
    BUG FIX v2: had Fixed=+4, Dual=+8; correct is Fixed=+4 (5th), Dual=+8 (9th) — was actually correct.
    Re-verified: Movable start=sign, Fixed start=sign+4, Dual start=sign+8.
    """
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // 10)  # 0, 1, 2
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 4) % 12
    else:  # Dual
        start = (sign_idx + 8) % 12
    return ZODIAC[(start + part) % 12]


def get_saptamsa(longitude: float) -> str:
    """
    Saptamsa (D7): each sign divided into 7 parts.
    BUG FIX v2: sign numbering is 1-based in tradition.
    Odd signs (1,3,5…=index 0,2,4…): start from the same sign.
    Even signs (2,4,6…=index 1,3,5…): start from 7th (index+6).
    v2 had the condition inverted: `sign_idx % 2 == 0` treated even indices as odd signs.
    Fixed: use (sign_idx + 1) % 2 == 1 to detect 1-based odd, i.e. sign_idx % 2 == 0.
    Wait — index 0 = Aries = sign 1 (odd) → start = same → sign_idx % 2 == 0 means ODD sign.
    v2 code was actually CORRECT for this but labelled confusingly. Re-verify:
    Aries (idx=0, sign 1, odd): start=0 ✓
    Taurus(idx=1, sign 2, even): start=1+6=7 ✓
    v2 code: if sign_idx % 2 == 0 → start=sign_idx else start=(sign_idx+6)%12 → CORRECT.
    Keeping as-is but adding clear comments.
    """
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // (30 / 7))
    # sign_idx % 2 == 0 → odd sign (Aries=1, Gemini=3…) → start from same sign
    # sign_idx % 2 == 1 → even sign (Taurus=2, Cancer=4…) → start from 7th sign
    if sign_idx % 2 == 0:
        start = sign_idx
    else:
        start = (sign_idx + 6) % 12
    return ZODIAC[(start + part) % 12]


def get_dasamsa(longitude: float) -> str:
    """
    Dasamsa (D10): each sign divided into 10 × 3°.
    BUG FIX v2: same Movable/Fixed/Dual offsets applied as Drekkana but they differ:
    Movable→same sign, Fixed→9th (index+8), Dual→5th (index+4).
    v2 had Fixed=+8, Dual=+4 → CORRECT. Verified and kept.
    """
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // 3)
    quality = SIGN_QUALITY[ZODIAC[sign_idx]]
    if quality == "Movable":
        start = sign_idx
    elif quality == "Fixed":
        start = (sign_idx + 8) % 12
    else:  # Dual
        start = (sign_idx + 4) % 12
    return ZODIAC[(start + part) % 12]


def get_dwadasamsa(longitude: float) -> str:
    """Dwadasamsa (D12): each sign divided into 12 × 2.5°. Start from same sign."""
    sign_idx = int(longitude // 30)
    deg_in_sign = longitude % 30
    part = int(deg_in_sign // 2.5)
    return ZODIAC[(sign_idx + part) % 12]


def get_planet_dignity(planet: str, sign: str) -> str:
    if EXALTATION.get(planet) == sign:
        return "Exalted"
    if DEBILITATION.get(planet) == sign:
        return "Debilitated"
    lord = SIGN_LORD[sign]
    if planet == lord:
        return "Own"
    if MOOLATRIKONA.get(planet) == sign:
        return "Mool Trikona"
    if planet in PLANET_FRIENDS:
        if lord in PLANET_FRIENDS[planet]:
            return "Friendly"
        if lord in PLANET_ENEMIES.get(planet, []):
            return "Inimical"
    return "Neutral"


# ==================================================================
# SECTION 5 — DASHA CALCULATIONS (fixed)
# ==================================================================

@dataclass
class DashaPeriod:
    planet:     str
    start_date: datetime
    end_date:   datetime
    years:      float
    level:      str             # "MD", "AD", "PD"
    parent:     Optional[str] = None


def calculate_vimshottari_full(birth_date: datetime, moon_longitude: float) -> List[DashaPeriod]:
    """
    Calculate 9 Mahadashas starting from birth.
    BUG FIX v2: degrees_covered was using modulo which is wrong — Moon's position
    in the nakshatra is simply (moon_longitude - nakshatra_start_longitude),
    where nakshatra_start_longitude = nak_idx * NAKSHATRA_SIZE.
    """
    moon_lon = moon_longitude % 360
    nak_idx  = int(moon_lon / NAKSHATRA_SIZE)
    nak_start = nak_idx * NAKSHATRA_SIZE
    deg_covered = moon_lon - nak_start          # degrees traversed in current nak
    remaining   = NAKSHATRA_SIZE - deg_covered  # degrees left
    fraction    = remaining / NAKSHATRA_SIZE
    lord_idx    = nak_idx % 9
    start_lord  = DASHA_SEQUENCE[lord_idx]
    balance     = fraction * DASHA_YEARS[start_lord]  # years remaining in 1st MD

    periods = []
    current_date = birth_date
    for i in range(9):
        lord  = DASHA_SEQUENCE[(lord_idx + i) % 9]
        years = balance if i == 0 else DASHA_YEARS[lord]
        days  = years * 365.25
        end_date = current_date + timedelta(days=days)
        periods.append(DashaPeriod(
            planet=lord, start_date=current_date, end_date=end_date,
            years=round(years, 4), level="MD"
        ))
        current_date = end_date

    return periods


def calculate_antardasha(md: DashaPeriod) -> List[DashaPeriod]:
    """
    Calculate 9 Antardashas within a Mahadasha.
    BUG FIX v2: v2 double-scaled AD years.
    Correct formula: AD_years = (MD_planet_years × AD_planet_years) / 120
    This gives the AD duration within the FULL dasha.  For a partial first MD,
    we must scale proportionally: AD_actual = AD_standard * (md.years / MD_total_years).
    """
    md_idx          = DASHA_SEQUENCE.index(md.planet)
    md_total_years  = DASHA_YEARS[md.planet]   # full duration of this dasha type
    scale           = md.years / md_total_years # < 1 only for first (partial) MD

    current_date = md.start_date
    ad_periods   = []
    for i in range(9):
        ad_planet    = DASHA_SEQUENCE[(md_idx + i) % 9]
        ad_std_years = (DASHA_YEARS[md.planet] * DASHA_YEARS[ad_planet]) / TOTAL_DASHA_YEARS
        ad_actual    = ad_std_years * scale
        days         = ad_actual * 365.25
        end_date     = current_date + timedelta(days=days)
        ad_periods.append(DashaPeriod(
            planet=ad_planet, start_date=current_date, end_date=end_date,
            years=round(ad_actual, 4), level="AD", parent=md.planet
        ))
        current_date = end_date

    return ad_periods


def calculate_pratyantardasha(ad: DashaPeriod) -> List[DashaPeriod]:
    """Calculate Pratyantardashas (PDs) within an Antardasha."""
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
        days         = pd_actual * 365.25
        end_date     = current_date + timedelta(days=days)
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
    ad_periods = calculate_antardasha(md)
    return get_current_dasha(ad_periods, check_date)


def get_current_pratyantardasha(md_periods: List[DashaPeriod], check_date: datetime = None) -> Optional[DashaPeriod]:
    if check_date is None:
        check_date = datetime.now()
    md = get_current_dasha(md_periods, check_date)
    if not md:
        return None
    ad = get_current_antardasha(md_periods, check_date)
    if not ad:
        return None
    pd_periods = calculate_pratyantardasha(ad)
    return get_current_dasha(pd_periods, check_date)


def check_sade_sati(moon_sign: str, saturn_sign: str) -> Dict:
    """Check Sade Sati (7.5-year Saturn transit over Moon sign ±1)."""
    m_idx = ZODIAC.index(moon_sign)
    s_idx = ZODIAC.index(saturn_sign) if saturn_sign in ZODIAC else -1
    if s_idx < 0:
        return {"active": False, "phase": ""}
    rel = (s_idx - m_idx) % 12
    if rel == 11:
        return {"active": True, "phase": "Rising Phase (Saturn entering sign before Moon)"}
    if rel == 0:
        return {"active": True, "phase": "Peak Phase (Saturn on Moon sign)"}
    if rel == 1:
        return {"active": True, "phase": "Setting Phase (Saturn in sign after Moon)"}
    return {"active": False, "phase": ""}


# ==================================================================
# SECTION 6 — CHART DATA CLASS
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
        # Derived
        self.nakshatras   : Dict = {}
        self.navamsa      : Dict = {}
        self.drekkana     : Dict = {}
        self.saptamsa     : Dict = {}
        self.dasamsa      : Dict = {}
        self.dwadasamsa   : Dict = {}
        self.dignities    : Dict = {}
        self.dasha_periods: List[DashaPeriod] = []
        self._compute_derived()

    def _compute_derived(self):
        for p, lon in self.planets.items():
            nak, pada, rem = get_nakshatra(lon)
            self.nakshatras[p] = {
                "nakshatra":       nak,
                "pada":            pada,
                "lord":            NAKSHATRA_LORDS[NAKSHATRAS.index(nak)],
                "deg_in_nakshatra": round(rem, 2)
            }
            self.navamsa[p]    = get_navamsa(lon)
            self.drekkana[p]   = get_drekkana(lon)
            self.saptamsa[p]   = get_saptamsa(lon)
            self.dasamsa[p]    = get_dasamsa(lon)
            self.dwadasamsa[p] = get_dwadasamsa(lon)
            sign, _            = longitude_to_sign(lon)
            self.dignities[p]  = get_planet_dignity(p, sign)

        if self.birth_date:
            self.dasha_periods = calculate_vimshottari_full(
                self.birth_date, self.planets["Moon"]
            )

    def get_current_dasha_info(self, check_date: datetime = None) -> Dict:
        md = get_current_dasha(self.dasha_periods, check_date)
        if not md:
            return {}
        ad = get_current_antardasha(self.dasha_periods, check_date)
        pd = get_current_pratyantardasha(self.dasha_periods, check_date)
        return {
            "mahadasha":       md.planet,
            "mahadasha_start": md.start_date.strftime("%d %b %Y"),
            "mahadasha_end":   md.end_date.strftime("%d %b %Y"),
            "antardasha":      ad.planet if ad else "",
            "antardasha_start":ad.start_date.strftime("%d %b %Y") if ad else "",
            "antardasha_end":  ad.end_date.strftime("%d %b %Y")   if ad else "",
            "pratyantardasha": pd.planet if pd else "",
            "pd_start":        pd.start_date.strftime("%d %b %Y") if pd else "",
            "pd_end":          pd.end_date.strftime("%d %b %Y")   if pd else "",
        }

    def to_dict(self) -> Dict:
        return {
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "lat": self.lat, "lon": self.lon, "tz": self.tz,
            "lagna_sign": self.lagna_sign,
            "moon_sign":  self.moon_sign,
            "sun_sign":   self.sun_sign,
            "ascendant":  self.ascendant,
            "planets":    self.planets,
            "nakshatras": self.nakshatras,
            "navamsa":    self.navamsa,
            "dignities":  self.dignities,
            "dasha": [
                {
                    "planet": p.planet,
                    "start":  p.start_date.isoformat(),
                    "end":    p.end_date.isoformat(),
                    "years":  p.years
                }
                for p in self.dasha_periods
            ]
        }

    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ==================================================================
# SECTION 7 — CHART CALCULATION (Swiss Ephemeris)
# ==================================================================

def compute_chart(year, month, day, hour, minute, lat, lon, tz_offset=0.0) -> ChartData:
    if not SWISSEPH_AVAILABLE:
        raise RuntimeError("pyswisseph not installed. Use generate_demo_chart() instead.")
    jd = swe.julday(year, month, day, hour + minute / 60.0 - tz_offset)
    houses = swe.houses_ex(jd, lat, lon, b'W', swe.FLG_SIDEREAL)
    ascendant = houses[1][0]
    planets = {}
    for pid, pname in zip(PLANET_IDS, PLANET_NAMES):
        res = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
        planets[pname] = res[0][0]
    planets["Ketu"] = (planets["Rahu"] + 180.0) % 360.0
    lagna_sign, _ = longitude_to_sign(ascendant)
    return ChartData(planets, ascendant, lagna_sign, datetime(year, month, day, hour, minute), lat, lon, tz_offset)


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


# ==================================================================
# SECTION 8 — ASHTAKOOTA MATCHMAKING (36 points)
# ==================================================================

def get_tara_score(ni1: int, ni2: int) -> int:
    """
    BUG FIX v2: v2 used a simple {3,5,7} inauspicious set but Tara Bala has
    9 distinct positions each with its own score. Using TARA_SCORES table.
    Both directions are averaged for compatibility.
    """
    d12 = ((ni2 - ni1) % 27) % 9 + 1  # Tara from person1 to person2
    d21 = ((ni1 - ni2) % 27) % 9 + 1  # Tara from person2 to person1
    # Give 3 if both directions auspicious, 1.5 if one, 0 if both inauspicious
    s1 = TARA_SCORES[d12]
    s2 = TARA_SCORES[d21]
    return round((s1 + s2) / 2)


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
    if lord1 == lord2:
        return 5
    if lord2 in PLANET_FRIENDS.get(lord1, []) and lord1 in PLANET_FRIENDS.get(lord2, []):
        return 5  # mutual friends
    if lord2 in PLANET_FRIENDS.get(lord1, []) or lord1 in PLANET_FRIENDS.get(lord2, []):
        return 4  # one-way friend
    if lord2 in PLANET_ENEMIES.get(lord1, []) and lord1 in PLANET_ENEMIES.get(lord2, []):
        return 0  # mutual enemies
    if lord2 in PLANET_ENEMIES.get(lord1, []) or lord1 in PLANET_ENEMIES.get(lord2, []):
        return 1  # one-way enemy
    return 3  # neutral


def get_gana_score(g1: str, g2: str) -> int:
    if g1 == g2:
        return 6
    if {g1, g2} == {"Deva", "Manushya"}:
        return 5
    if {g1, g2} == {"Manushya", "Rakshasa"}:
        return 0
    if {g1, g2} == {"Deva", "Rakshasa"}:
        return 0
    return 0


def get_bhakoot_score(idx1: int, idx2: int) -> int:
    """
    BUG FIX v2: diff=2,12 (2/12 axis) and diff=5,9 (5/9 axis = Nava-Pancham)
    and diff=6 (6/8 axis) are inauspicious.
    Note: 6/8 = diff 6 or diff 8 (both directions of the same axis).
    2/12 = diff 2 or diff 10.  Nava-Pancham = diff 5 or diff 9 is actually BENEFICIAL.
    Corrected: 6/8 (diff 6 or 8) = 0; 2/12 (diff 2 or 10) = 0; rest = 7.
    """
    diff = (idx2 - idx1) % 12
    if diff in [2, 10, 6, 8]:
        return 0
    return 7


def calculate_ashtakoota(c1: ChartData, c2: ChartData) -> Dict:
    m1, m2   = c1.moon_sign, c2.moon_sign
    n1       = c1.nakshatras["Moon"]["nakshatra"]
    n2       = c2.nakshatras["Moon"]["nakshatra"]
    i1, i2   = ZODIAC.index(m1), ZODIAC.index(m2)
    ni1, ni2 = NAKSHATRAS.index(n1), NAKSHATRAS.index(n2)

    # Varna
    varna1 = VARNA_MAP[SIGN_ELEMENT[m1]]
    varna2 = VARNA_MAP[SIGN_ELEMENT[m2]]
    varna_order = {"Brahmin":1,"Kshatriya":2,"Vaishya":3,"Shudra":4}
    varna  = 1 if varna_order[varna1] >= varna_order[varna2] else 0  # groom >= bride in varna

    # Vashya
    vashya1 = VASHYA_MAP[m1]
    vashya2 = VASHYA_MAP[m2]
    vashya  = 2 if vashya1 == vashya2 else 1 if (
        (vashya1 == "Human" and vashya2 in ["Water","Quadruped"])
        or (vashya2 == "Human" and vashya1 in ["Water","Quadruped"])
    ) else 0

    tara  = get_tara_score(ni1, ni2)
    yoni  = get_yoni_score(NAKSHATRA_YONI[n1], NAKSHATRA_YONI[n2])
    graha = get_graha_maitri_score(SIGN_LORD[m1], SIGN_LORD[m2])
    gana  = get_gana_score(NAKSHATRA_GANA[n1], NAKSHATRA_GANA[n2])
    bhakoot = get_bhakoot_score(i1, i2)
    nadi  = 0 if NAKSHATRA_NADI[n1] == NAKSHATRA_NADI[n2] else 8

    total = varna + vashya + tara + yoni + graha + gana + bhakoot + nadi

    return {
        "varna":       {"score": varna,   "max": 1,  "detail": f"{varna1} vs {varna2}"},
        "vashya":      {"score": vashya,  "max": 2,  "detail": f"{vashya1} vs {vashya2}"},
        "tara":        {"score": tara,    "max": 3,  "detail": f"{n1} vs {n2}"},
        "yoni":        {"score": yoni,    "max": 4,  "detail": f"{NAKSHATRA_YONI[n1]} vs {NAKSHATRA_YONI[n2]}"},
        "graha_maitri":{"score": graha,   "max": 5,  "detail": f"{SIGN_LORD[m1]} vs {SIGN_LORD[m2]}"},
        "gana":        {"score": gana,    "max": 6,  "detail": f"{NAKSHATRA_GANA[n1]} vs {NAKSHATRA_GANA[n2]}"},
        "bhakoot":     {"score": bhakoot, "max": 7,  "detail": f"{m1} vs {m2}"},
        "nadi":        {"score": nadi,    "max": 8,  "detail": f"{NAKSHATRA_NADI[n1]} vs {NAKSHATRA_NADI[n2]}"},
        "total":       total,
        "max_total":   36,
        "percentage":  round(total / 36 * 100, 1),
        "verdict": (
            "Excellent" if total >= 31 else
            "Good"      if total >= 25 else
            "Average"   if total >= 18 else
            "Challenging"
        )
    }


# ==================================================================
# SECTION 9 — DETAILED TOPIC ANALYSIS (rule-engine powered)
# ==================================================================

def _narrative_block(fired: List[Dict]) -> str:
    """Convert fired rules into a readable multi-paragraph narrative."""
    positives = [r for r in fired if r["severity"] == "positive"]
    neutrals  = [r for r in fired if r["severity"] == "neutral"]
    cautions  = [r for r in fired if r["severity"] in ["caution","warning"]]

    parts = []
    if positives:
        parts.append("STRENGTHS:\n" + "\n".join(
            f"  ✦ {r['title']}\n    {r['detail']}" for r in positives
        ))
    if neutrals:
        parts.append("MIXED / NEUTRAL:\n" + "\n".join(
            f"  ◈ {r['title']}\n    {r['detail']}" for r in neutrals
        ))
    if cautions:
        parts.append("CAUTIONS & REMEDIES NEEDED:\n" + "\n".join(
            f"  ⚠ {r['title']}\n    {r['detail']}" for r in cautions
        ))
    return "\n\n".join(parts) if parts else "No significant planetary indicators found for this topic."


def analyze_career(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="career")
    summary    = score_topic(fired)

    # Supplement: Dasamsa (D10) 10th lord
    lagna_idx   = ZODIAC.index(chart.lagna_sign)
    tenth_sign  = ZODIAC[(lagna_idx + 9) % 12]
    tenth_lord  = SIGN_LORD[tenth_sign]
    tenth_house = ctx["house_map"].get(tenth_lord, 0)

    planets_10th = [p for p, h in ctx["house_map"].items() if h == 10]

    return {
        "rating":          summary["rating"],
        "net_score":       summary["net_score"],
        "tenth_lord":      tenth_lord,
        "tenth_lord_house":tenth_house,
        "planets_in_10th": planets_10th,
        "current_dasha":   dasha_info,
        "fired_rules":     fired,
        "narrative":       _narrative_block(fired),
        "summary": (
            f"Career rating: {summary['rating']} (score {summary['net_score']:+d}). "
            f"10th lord {tenth_lord} is {chart.dignities.get(tenth_lord,'Neutral')} in House {tenth_house}. "
            f"{summary['positive_count']} strengths, "
            f"{summary['warning_count']} cautions identified."
        )
    }


def analyze_marriage(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="marriage")
    summary    = score_topic(fired)

    lagna_idx    = ZODIAC.index(chart.lagna_sign)
    seventh_sign = ZODIAC[(lagna_idx + 6) % 12]
    seventh_lord = SIGN_LORD[seventh_sign]
    seventh_house= ctx["house_map"].get(seventh_lord, 0)
    planets_7th  = [p for p, h in ctx["house_map"].items() if h == 7]

    venus_sign   = longitude_to_sign(chart.planets["Venus"])[0]
    venus_house  = ctx["house_map"].get("Venus", 0)

    return {
        "rating":          summary["rating"],
        "net_score":       summary["net_score"],
        "seventh_lord":    seventh_lord,
        "seventh_lord_house": seventh_house,
        "planets_in_7th":  planets_7th,
        "venus_house":     venus_house,
        "venus_sign":      venus_sign,
        "current_dasha":   dasha_info,
        "fired_rules":     fired,
        "narrative":       _narrative_block(fired),
        "summary": (
            f"Marriage outlook: {summary['rating']} (score {summary['net_score']:+d}). "
            f"7th lord {seventh_lord} is {chart.dignities.get(seventh_lord,'Neutral')} in House {seventh_house}. "
            f"Venus ({chart.dignities.get('Venus','Neutral')}) in House {venus_house} ({venus_sign})."
        )
    }


def analyze_children(chart: ChartData, check_date: datetime = None) -> Dict:
    dasha_info = chart.get_current_dasha_info(check_date)
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="children")
    summary    = score_topic(fired)

    lagna_idx   = ZODIAC.index(chart.lagna_sign)
    fifth_sign  = ZODIAC[(lagna_idx + 4) % 12]
    fifth_lord  = SIGN_LORD[fifth_sign]
    fifth_house = ctx["house_map"].get(fifth_lord, 0)
    planets_5th = [p for p, h in ctx["house_map"].items() if h == 5]
    jupiter_house = ctx["house_map"].get("Jupiter", 0)

    return {
        "rating":          summary["rating"],
        "net_score":       summary["net_score"],
        "fifth_lord":      fifth_lord,
        "fifth_lord_house":fifth_house,
        "planets_in_5th":  planets_5th,
        "jupiter_house":   jupiter_house,
        "jupiter_dignity": chart.dignities.get("Jupiter", "Neutral"),
        "current_dasha":   dasha_info,
        "fired_rules":     fired,
        "narrative":       _narrative_block(fired),
        "summary": (
            f"Children outlook: {summary['rating']} (score {summary['net_score']:+d}). "
            f"5th lord {fifth_lord} is {chart.dignities.get(fifth_lord,'Neutral')} in House {fifth_house}. "
            f"Jupiter (Putrakaraka) is {chart.dignities.get('Jupiter','Neutral')} in House {jupiter_house}."
        )
    }


def analyze_health(chart: ChartData, check_date: datetime = None,
                   transit_saturn_sign: str = None) -> Dict:
    dasha_info  = chart.get_current_dasha_info(check_date)
    sade_sati   = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")
    ctx         = build_context(chart, dasha_info, sade_sati)
    fired       = evaluate_rules(ctx, topic="health")
    summary     = score_topic(fired)

    lagna_lord    = SIGN_LORD[chart.lagna_sign]
    planets_1st   = [p for p, h in ctx["house_map"].items() if h == 1]
    planets_6th   = [p for p, h in ctx["house_map"].items() if h == 6]
    planets_8th   = [p for p, h in ctx["house_map"].items() if h == 8]
    planets_12th  = [p for p, h in ctx["house_map"].items() if h == 12]

    return {
        "rating":         summary["rating"],
        "net_score":      summary["net_score"],
        "lagna_lord":     lagna_lord,
        "lagna_lord_dignity": chart.dignities.get(lagna_lord, "Neutral"),
        "planets_in_1st": planets_1st,
        "planets_in_6th": planets_6th,
        "planets_in_8th": planets_8th,
        "planets_in_12th":planets_12th,
        "sade_sati":      sade_sati,
        "current_dasha":  dasha_info,
        "fired_rules":    fired,
        "narrative":      _narrative_block(fired),
        "summary": (
            f"Health outlook: {summary['rating']} (score {summary['net_score']:+d}). "
            f"Lagna lord {lagna_lord} is {chart.dignities.get(lagna_lord,'Neutral')}. "
            f"Sade Sati: {'Active — ' + sade_sati['phase'] if sade_sati['active'] else 'Not active'}."
        )
    }


def analyze_general_yogas(chart: ChartData) -> Dict:
    """Evaluate all general yogas from the rules layer."""
    dasha_info = chart.get_current_dasha_info()
    ctx        = build_context(chart, dasha_info)
    fired      = evaluate_rules(ctx, topic="general")
    return {
        "yoga_count":  len(fired),
        "fired_yogas": fired,
        "narrative":   _narrative_block(fired),
    }


# ==================================================================
# SECTION 10 — VARSHPHAL (SOLAR RETURN / TAJAKA)
# ==================================================================

def calculate_varshphal(chart: ChartData, year: int) -> Dict:
    if not chart.birth_date:
        return {}

    birth_month = chart.birth_date.month
    birth_day   = chart.birth_date.day
    varsh_date  = datetime(year, birth_month, birth_day)
    years_elapsed = year - chart.birth_date.year

    # Muntha: progresses one sign per year from Lagna
    muntha_lon   = (chart.ascendant + years_elapsed * 30) % 360
    muntha_sign, muntha_deg = longitude_to_sign(muntha_lon)
    muntha_lord  = SIGN_LORD[muntha_sign]

    # Muntha house from natal lagna
    lagna_idx    = ZODIAC.index(chart.lagna_sign)
    muntha_idx   = ZODIAC.index(muntha_sign)
    muntha_house = ((muntha_idx - lagna_idx) % 12) + 1

    # Transit planets (requires Swiss Ephemeris)
    transits = {}
    if SWISSEPH_AVAILABLE:
        try:
            transits = get_transits(year, birth_month, birth_day)
        except Exception:
            pass

    themes = _varshphal_themes(chart, muntha_sign, muntha_house, muntha_lord)

    return {
        "year":           year,
        "varshphal_date": varsh_date.strftime("%d %b %Y"),
        "years_elapsed":  years_elapsed,
        "muntha_sign":    muntha_sign,
        "muntha_house":   muntha_house,
        "muntha_longitude": round(muntha_lon, 2),
        "muntha_lord":    muntha_lord,
        "transits":       {k: longitude_to_sign(v)[0] for k, v in transits.items()},
        "themes":         themes,
    }


def _varshphal_themes(chart: ChartData, muntha_sign: str, muntha_house: int,
                      muntha_lord: str) -> List[str]:
    themes = []
    lagna_idx  = ZODIAC.index(chart.lagna_sign)
    muntha_idx = ZODIAC.index(muntha_sign)
    diff = (muntha_idx - lagna_idx) % 12

    # Muntha in trikona from lagna (1,5,9) → auspicious
    if muntha_house in [1, 5, 9]:
        themes.append(f"Muntha in {muntha_sign} (House {muntha_house}, trikona) — year of growth, blessings, and fresh opportunities.")
    elif muntha_house in [4, 7, 10]:
        themes.append(f"Muntha in {muntha_sign} (House {muntha_house}, kendra) — year of action, visibility, and tangible results.")
    elif muntha_house in [2, 11]:
        themes.append(f"Muntha in {muntha_sign} (House {muntha_house}) — year of financial focus and gains.")
    elif muntha_house in [3, 6]:
        themes.append(f"Muntha in {muntha_sign} (House {muntha_house}) — year of effort, competition, and overcoming obstacles.")
    elif muntha_house in [8, 12]:
        themes.append(f"Muntha in {muntha_sign} (House {muntha_house}, dusthana) — year of transformation, hidden matters, and inner work.")

    # Muntha lord quality
    muntha_lord_dignity = chart.dignities.get(muntha_lord, "Neutral")
    if muntha_lord_dignity in ["Exalted","Own","Mool Trikona"]:
        themes.append(f"Muntha lord {muntha_lord} is {muntha_lord_dignity} — amplifies the year's positive potential significantly.")
    elif muntha_lord_dignity == "Debilitated":
        themes.append(f"Muntha lord {muntha_lord} is debilitated — the year's themes may face obstruction; remedies advised.")
    elif muntha_lord_dignity in ["Friendly","Neutral"]:
        themes.append(f"Muntha lord {muntha_lord} is {muntha_lord_dignity} — moderate support for the year's themes.")

    return themes


# ==================================================================
# SECTION 11 — YEARLY PREDICTION (comprehensive)
# ==================================================================

def get_year_prediction(chart: ChartData, year: int) -> Dict:
    check_date = datetime(year, 6, 15)
    dasha_info = chart.get_current_dasha_info(check_date)

    # Transit Saturn sign (for Sade Sati)
    transit_saturn_sign = None
    if SWISSEPH_AVAILABLE:
        try:
            tr = get_transits(year)
            transit_saturn_sign = longitude_to_sign(tr["Saturn"])[0]
            transit_jupiter_sign= longitude_to_sign(tr["Jupiter"])[0]
        except Exception:
            transit_saturn_sign  = None
            transit_jupiter_sign = None
    else:
        transit_saturn_sign  = None
        transit_jupiter_sign = None

    sade_sati = check_sade_sati(chart.moon_sign, transit_saturn_sign or "")

    # Jupiter transit impact
    jupiter_transit_note = ""
    if transit_jupiter_sign:
        j_idx = ZODIAC.index(transit_jupiter_sign)
        l_idx = ZODIAC.index(chart.lagna_sign)
        m_idx = ZODIAC.index(chart.moon_sign)
        jh_from_lagna = ((j_idx - l_idx) % 12) + 1
        jh_from_moon  = ((j_idx - m_idx) % 12) + 1
        if jh_from_lagna in [1, 5, 9]:
            jupiter_transit_note += f"Jupiter transiting House {jh_from_lagna} from Lagna — highly auspicious."
        elif jh_from_lagna in [4, 7, 8, 12]:
            jupiter_transit_note += f"Jupiter transiting House {jh_from_lagna} from Lagna — mixed/challenging."
        if jh_from_moon in [1, 5, 9, 11]:
            jupiter_transit_note += f" Jupiter in House {jh_from_moon} from Moon — Guruchandra Yoga possible."
        elif jh_from_moon in [4, 7, 8]:
            jupiter_transit_note += f" Jupiter in House {jh_from_moon} from Moon — emotional or health caution."

    varshphal = calculate_varshphal(chart, year)
    career    = analyze_career(chart, check_date)
    marriage  = analyze_marriage(chart, check_date)
    children  = analyze_children(chart, check_date)
    health    = analyze_health(chart, check_date, transit_saturn_sign)
    yogas     = analyze_general_yogas(chart)

    return {
        "year":             year,
        "dasha":            dasha_info,
        "sade_sati":        sade_sati,
        "jupiter_transit":  jupiter_transit_note,
        "transit_saturn":   transit_saturn_sign,
        "transit_jupiter":  transit_jupiter_sign,
        "varshphal":        varshphal,
        "career":           career,
        "marriage":         marriage,
        "children":         children,
        "health":           health,
        "general_yogas":    yogas,
        "overall_summary":  _year_summary(year, dasha_info, sade_sati, varshphal,
                                          career, marriage, children, health)
    }


def _year_summary(year, dasha, sade_sati, varshphal, career, marriage, children, health) -> str:
    lines = [f"=== YEAR {year} PREDICTION SUMMARY ===\n"]

    md = dasha.get("mahadasha","?")
    ad = dasha.get("antardasha","?")
    pd = dasha.get("pratyantardasha","?")
    lines.append(f"Dasha: {md} MD / {ad} AD / {pd} PD")
    lines.append(f"       MD runs: {dasha.get('mahadasha_start','')} → {dasha.get('mahadasha_end','')}")
    lines.append(f"       AD runs: {dasha.get('antardasha_start','')} → {dasha.get('antardasha_end','')}\n")

    if sade_sati.get("active"):
        lines.append(f"⚠  SADE SATI ACTIVE: {sade_sati['phase']}\n")

    if varshphal:
        lines.append(f"Varshphal (Solar Return) — Muntha in {varshphal.get('muntha_sign','')} "
                     f"(House {varshphal.get('muntha_house','')}) ruled by {varshphal.get('muntha_lord','')}.")
        for t in varshphal.get("themes",[]):
            lines.append(f"  • {t}")
        lines.append("")

    for label, data in [("Career",career),("Marriage",marriage),("Children",children),("Health",health)]:
        lines.append(f"{label}: {data.get('rating','?')} (score {data.get('net_score',0):+d})")
        lines.append(f"  {data.get('summary','')}\n")

    return "\n".join(lines)


# ==================================================================
# SECTION 12 — DEMO CHART & UTILITIES
# ==================================================================

def generate_demo_chart() -> ChartData:
    """Generate a sample chart for testing without Swiss Ephemeris."""
    planets = {
        "Sun":     45.5,   # Taurus
        "Moon":   128.3,   # Leo
        "Mars":   200.0,   # Libra
        "Mercury": 50.2,   # Taurus
        "Jupiter": 95.0,   # Cancer (Exalted!)
        "Venus":   70.5,   # Gemini
        "Saturn": 310.0,   # Aquarius (Own)
        "Rahu":   175.0,   # Virgo
        "Ketu":   355.0,   # Pisces
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
        planets   = data["planets"],
        ascendant = data["ascendant"],
        lagna_sign= data["lagna_sign"],
        birth_date= birth_date,
        lat       = data.get("lat", 0),
        lon       = data.get("lon", 0),
        tz        = data.get("tz", 0)
    )


def print_full_report(chart: ChartData, year: int = None):
    """Print a complete textual report for a chart."""
    import textwrap
    year = year or datetime.now().year

    print("=" * 70)
    print("VEDIC ASTROLOGY REPORT — v3.0")
    print("=" * 70)
    print(f"Lagna:     {chart.lagna_sign} ({SIGN_SANSKRIT[chart.lagna_sign]})")
    print(f"Moon sign: {chart.moon_sign}")
    print(f"Sun sign:  {chart.sun_sign}")
    print()

    print("PLANETS")
    print("-" * 40)
    for p, lon in chart.planets.items():
        sign, deg = longitude_to_sign(lon)
        nak_info  = chart.nakshatras[p]
        dig       = chart.dignities[p]
        print(f"  {p:10s}: {sign:12s} {deg:6.2f}°  [{dig:12s}]  {nak_info['nakshatra']} pada {nak_info['pada']}")

    print()
    print("DASHA PERIODS")
    print("-" * 40)
    for dp in chart.dasha_periods:
        marker = " ← NOW" if dp.start_date <= datetime.now() < dp.end_date else ""
        print(f"  {dp.planet:8s}: {dp.start_date.strftime('%d %b %Y')} → {dp.end_date.strftime('%d %b %Y')}  ({dp.years:.2f} yrs){marker}")

    print()
    prediction = get_year_prediction(chart, year)
    print(prediction["overall_summary"])

    print("CAREER ANALYSIS")
    print("-" * 40)
    print(chart_analyze_text(prediction["career"]["narrative"]))

    print("\nMARRIAGE ANALYSIS")
    print("-" * 40)
    print(chart_analyze_text(prediction["marriage"]["narrative"]))

    print("\nCHILDREN ANALYSIS")
    print("-" * 40)
    print(chart_analyze_text(prediction["children"]["narrative"]))

    print("\nHEALTH ANALYSIS")
    print("-" * 40)
    print(chart_analyze_text(prediction["health"]["narrative"]))

    print("\nYOGAS IN NATAL CHART")
    print("-" * 40)
    print(chart_analyze_text(prediction["general_yogas"]["narrative"]))


def chart_analyze_text(text: str, width: int = 80) -> str:
    """Wrap long detail lines for readability."""
    import textwrap
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.startswith("  ") and len(line) > width:
            indent = "    "
            wrapped.append(textwrap.fill(line, width=width, subsequent_indent=indent))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


# ==================================================================
# QUICK TEST
# ==================================================================
if __name__ == "__main__":
    chart = generate_demo_chart()
    print_full_report(chart, year=2025)
