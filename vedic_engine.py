"""
Vedic Astrology Calculation Engine v8.0
=========================================
New in v8.0 — All additions are additive; all v7.0 logic preserved.

NEW MODULES:
  1. PANCHANGA (5 limbs) — Tithi, Vara, Nakshatra, Nitya Yoga, Karana
     for any date/time. Both birth panchanga and query-date panchanga.

  2. GAN & MOOL NAKSHATRA
     - Gan: Deva / Manushya / Rakshasa (from Moon nakshatra, all planet nakshatras)
     - Mool Nakshatra check: Ashwini, Ashlesha, Magha, Jyeshtha, Mula, Revati
       with classical pada-level severity, family remedies, and pacification
       periods (27th day, 3rd month, 1st year).

  3. GRAHA AVASTHA (Planetary States)
     - Baladi Avastha (5 states by degree in sign): Bala, Kumara, Yuva,
       Vriddha, Mrita — odd vs even sign rules applied.
     - Jagrat/Svapna/Sushupti (awake/dreaming/sleeping) by dignity.
     - Lajjita/Garvita/Kshudita/Trishita/Mudita/Dukhita (6 special states).
     - Deeptadi Avastha (9 states).

  4. MRITYUBHAGA (Critical Death Degrees)
     Classical degrees per planet per sign where the planet loses all power.
     Full table from Jataka Parijata. Flags if any natal planet is within 1°.

  5. ARGALA & VIRODHA ARGALA
     - Standard Argala: planets in 2nd, 4th, 11th, and 5th from any house.
     - Virodha (obstruction): planets in 12th, 10th, 3rd, and 9th countering
       the argala. Net argala strength computed for all 12 houses.

  6. CHARA KARAKAS (Jaimini — all 7)
     Atmakaraka, Amatyakaraka, Bhratrukaraka, Matrukaraka, Putrakaraka,
     Gnatikaraka, Darakaraka — ranked by degree within sign (Rahu reversed).

  7. PUSHKARA NAVAMSA & PUSHKARA BHAGA
     Auspicious navamsa positions and exact Pushkara degrees for each sign.
     Flags planets occupying Pushkara Navamsa or within 1° of Pushkara Bhaga.

  8. SHADBALA (6-component) — expanded from proxy score
     Sthana Bala, Dig Bala, Kala Bala, Chesta Bala, Naisargika Bala, Drik Bala.
     Rupas and percentage of required minimum. Requisite Rupa table included.

  9. 27 NITYA YOGAS (Sun + Moon longitude sum / 13°20')
     Vishkambha through Vaidhriti with benefic/malefic/mixed nature and effects.

  10. ASHTAMESH ANALYSIS
      Planet ruling the 8th house — its dignity, house, and karmic implications.
      Interaction with longevity (8th lord conjunctions, aspects).

  11. GRAHA DRISHTI STRENGTH (orb-weighted aspects)
      Full-strength (full), three-quarter (3/4), half (1/2), quarter (1/4)
      aspect strengths for all planets including special aspects. Returns
      a weighted drishti score for each house.

All v7.0 fixes (BUG-C through BUG-F) are fully retained.
"""

import copy, math, json, random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import swisseph as swe
    SWISSEPH_AVAILABLE = True
    swe.set_sid_mode(swe.SIDM_LAHIRI)
except ImportError:
    SWISSEPH_AVAILABLE = False


# ==================================================================
# SECTION 1 — STATIC LOOKUP TABLES  (all v7.0 tables retained)
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

==================================================================
SECTION 14 — RAM SHALAKA ORACLE (Authentic 15x15 Implementation)
==================================================================

# Authentic 9 Chaupais from Ram Charit Manas
AUTHENTIC_CHAUPAIS = [
    {
        "id": 1,
        "chars": "सुनुसियसत्यअसीसहमारीपूजिहिमनकामनतुम्हारी",
        "hindi": "सुनु सिय सत्य असीस हमारी। पूजिहि मन कामना तुम्हारी॥",
        "transliteration": "Sunu siya satya aseesa hamari, pujihi mana kamana tumhari",
        "meaning_en": "Listen Siya, this is my true blessing — worship with your heart's desire.",
        "context": "Bal Kand — Gauri blesses Sita",
        "result": "Very Auspicious — The work will be accomplished with divine grace.",
        "nature": "Very Auspicious",
        "score": 90,
        "color": "#2d6a4f",
        "bg_color": "#d8f3dc",
        "remedy": "Offer prayers to Lord Rama and Sita. Chant 'Siya-Ram' 108 times.",
        "symbol": "शुभ",
    },
    {
        "id": 2,
        "chars": "प्रबिसिनगरकीजैसबकाजाहृदयराखिकौसलपुरराजा",
        "hindi": "प्रबिसि नगर कीजै सब काजा। हृदय राखि कौसलपुर राजा॥",
        "transliteration": "Prabisi nagara keejai saba kaaja, hridaya rakhi kausala pura raaja",
        "meaning_en": "Enter the city and accomplish all tasks, keeping the King of Ayodhya in your heart.",
        "context": "Sundar Kand — Hanuman enters Lanka",
        "result": "Very Auspicious — Begin with faith. Success and divine protection are assured.",
        "nature": "Very Auspicious",
        "score": 85,
        "color": "#2d6a4f",
        "bg_color": "#d8f3dc",
        "remedy": "Begin after remembering Lord Rama. Hanuman Chalisa is beneficial.",
        "symbol": "शुभ",
    },
    {
        "id": 3,
        "chars": "मुदमंगलमयसंतसमाजूजिमिजगजंगमतीरथराजू",
        "hindi": "मुद मंगलमय संत समाजू। जिमि जग जंगम तीरथ राजू॥",
        "transliteration": "Muda mangalamaya santa samaaju, jimi jaga jangama tiratha raaju",
        "meaning_en": "The assembly of saints is blissful, like the king of pilgrimage places.",
        "context": "Bal Kand — Saintly gathering",
        "result": "Auspicious — Work accomplished through good company and wisdom.",
        "nature": "Auspicious",
        "score": 75,
        "color": "#40916c",
        "bg_color": "#b7e4c7",
        "remedy": "Seek company of wise people. Read Ramayana regularly.",
        "symbol": "शुभ",
    },
    {
        "id": 4,
        "chars": "गरलसुधारिपुकरहिमिताईगोपदसिंधुअनलसितलाई",
        "hindi": "गरल सुधा रिपु करहिं मिताई। गोपद सिंधु अनल सितलाई॥",
        "transliteration": "Garala sudhaa ripu karahim mitaaee, gopada sindhu anala sitalaaee",
        "meaning_en": "Poison turns to nectar, enemies become friends; ocean becomes a cow's hoof-print.",
        "context": "Sundar Kand — Hanuman's power",
        "result": "Extremely Auspicious — Even impossible tasks will succeed. Miracles possible.",
        "nature": "Very Auspicious",
        "score": 95,
        "color": "#1b4332",
        "bg_color": "#d8f3dc",
        "remedy": "Have unwavering faith. Chant Hanuman Chalisa daily.",
        "symbol": "श्री",
    },
    {
        "id": 5,
        "chars": "बरुनकुबेरसुरेससमीरारनसनमुखधरिकाहनधीरा",
        "hindi": "बरुन कुबेर सुरेस समीरा। रन सनमुख धरि काह न धीरा॥",
        "transliteration": "Baruna kubera suresha sameeraa, rana sanamukha dhari kaaha na dheeraa",
        "meaning_en": "Varuna, Kubera, Indra, Vayu — who can face them in battle?",
        "context": "Lanka Kand — Mighty warriors",
        "result": "Auspicious with Effort — Obstacles exist, but victory is certain with courage.",
        "nature": "Auspicious with Effort",
        "score": 70,
        "color": "#52b788",
        "bg_color": "#95d5b2",
        "remedy": "Be courageous. Worship Hanuman for strength.",
        "symbol": "मध्य",
    },
    {
        "id": 6,
        "chars": "होइहैसोइजोरामरचिराखाकोकरितरकबढ़ावहिंसाखा",
        "hindi": "होइ है सोइ जो राम रचि राखा। को करि तरक बढ़ावहिं साखा॥",
        "transliteration": "Hoi hai soi jo raama rachi raakhaa, ko kari taraka badhaavahim saakhaa",
        "meaning_en": "Whatever Rama has ordained shall happen — who can argue with it?",
        "context": "Bal Kand — Shiva-Parvati dialogue",
        "result": "Neutral — The outcome is destined. Surrender to divine will.",
        "nature": "Neutral",
        "score": 55,
        "color": "#b69121",
        "bg_color": "#fef3cd",
        "remedy": "Accept divine will. Do duty without attachment. Chant 'Ram Naam'.",
        "symbol": "मध्य",
    },
    {
        "id": 7,
        "chars": "बिधिबससुजनकुसंगतपरहींफनिमनिसमनिजगुनअनुसरहीं",
        "hindi": "बिधि बस सुजन कुसंगत परहीं। फनि मनि सम निज गुन अनुसरहीं॥",
        "transliteration": "Bidhi basa sujana kusangata parahim, phani mani sama nija guna anusarahim",
        "meaning_en": "Good people fall into bad company by fate, but follow their own nature like the snake's gem.",
        "context": "Bal Kand — Satsang description",
        "result": "Caution — Beware of bad company. Doubt about success. Choose associations carefully.",
        "nature": "Caution",
        "score": 40,
        "color": "#c17817",
        "bg_color": "#ffe4b5",
        "remedy": "Avoid negative influences. Maintain integrity. Read Ramayana.",
        "symbol": "अशुभ",
    },
    {
        "id": 8,
        "chars": "उघरेंअंतनहोइनिबाहूकालनेमिजिमिरावनराहू",
        "hindi": "उघरें अंत न होइ निबाहू। कालनेमि जिमि रावन राहू॥",
        "transliteration": "Ugharenta na hoi nibaahuu, kaalanemi jimi raavana raahuu",
        "meaning_en": "The end cannot be borne, like Kalanemi, Ravana, and Rahu.",
        "context": "Bal Kand — Satsang description",
        "result": "Inauspicious — This work is not beneficial. Better to reconsider or abandon.",
        "nature": "Inauspicious",
        "score": 25,
        "color": "#9b2226",
        "bg_color": "#fde2e2",
        "remedy": "Reconsider plans. Seek divine guidance through prayer.",
        "symbol": "अशुभ",
    },
    {
        "id": 9,
        "chars": "सूक्ष्मरूपधरिसियहिंदिखावाबिकटरूपधरिलंकजरावा",
        "hindi": "सूक्ष्म रूप धरि सियहिं दिखावा। बिकट रूप धरि लंक जरावा॥",
        "transliteration": "Sukshma roopa dhari siyahim dikhaavaa, bikata roopa dhari lanka jaraavaa",
        "meaning_en": "Taking subtle form to show Sita, and terrifying form to burn Lanka.",
        "context": "Sundar Kand — Hanuman's dual nature",
        "result": "Auspicious with Wisdom — Adaptability is key. Success through transformation.",
        "nature": "Auspicious with Wisdom",
        "score": 80,
        "color": "#2d6a4f",
        "bg_color": "#d8f3dc",
        "remedy": "Be adaptable. Use wisdom. Worship Hanuman for strength and devotion.",
        "symbol": "शुभ",
    },
]

GRID_SIZE = 15
TOTAL_CELLS = GRID_SIZE * GRID_SIZE  # 225
STEP = 9
NUM_TRACKS = 9
CYCLE_LENGTH = TOTAL_CELLS // STEP  # 25

def build_shalaka_grid() -> list:
    """Build the 15×15 Ram Shalaka grid."""
    # Initialize empty grid
    grid = [[{"char": " ", "track": -1, "pos": -1, "symbol": " "} for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    # For each track (0-8), place the Chaupai characters
    for track_id in range(NUM_TRACKS):
        chaupai = AUTHENTIC_CHAUPAIS[track_id]
        chars = chaupai["chars"]
        symbol = chaupai["symbol"]

        # Track positions: start at track_id, then add 9 each time
        for pos_in_track in range(CYCLE_LENGTH):
            cell_idx = (track_id + pos_in_track * STEP) % TOTAL_CELLS
            row = cell_idx // GRID_SIZE
            col = cell_idx % GRID_SIZE

            # Get character (cycle through if Chaupai is shorter than 25)
            char_idx = pos_in_track % len(chars)
            char = chars[char_idx]

            grid[row][col] = {
                "char": char,
                "track": track_id,
                "pos_in_track": pos_in_track,
                "symbol": symbol,
                "chaupai_id": chaupai["id"],
            }
    return grid

# Pre-build the grid
SHALAKA_GRID = build_shalaka_grid()

def ram_shalaka_query(question: str = "", seed: int = None) -> dict:
    """
    Perform an authentic Ram Shalaka query using the 15x15 grid.
    """
    if seed is None:
        import time
        time_seed = int(time.time() * 1000) % 100000
        question_hash = sum(ord(c) for c in question) if question else 0
        seed = time_seed + question_hash

    random.seed(seed)

    # Random starting position
    start_row = random.randint(0, GRID_SIZE - 1)
    start_col = random.randint(0, GRID_SIZE - 1)

    # Get the track for this starting position
    start_cell = SHALAKA_GRID[start_row][start_col]
    track_id = start_cell["track"]

    # Trace the path: step by 9 until we return to start
    path_cells = []
    path_chars = []

    current_row, current_col = start_row, start_col
    max_steps = 50  # Safety limit

    for step in range(max_steps):
        path_cells.append((current_row, current_col))
        cell = SHALAKA_GRID[current_row][current_col]
        path_chars.append(cell["char"])

        # Move 9 cells forward
        current_idx = current_row * GRID_SIZE + current_col
        next_idx = (current_idx + STEP) % TOTAL_CELLS
        next_row = next_idx // GRID_SIZE
        next_col = next_idx % GRID_SIZE

        # Check if we've returned to start
        if next_row == start_row and next_col == start_col:
            break

        current_row, current_col = next_row, next_col

    # Get the Chaupai for this track
    chaupai = AUTHENTIC_CHAUPAIS[track_id]
    formed_text = "".join(path_chars)

    return {
        "question": question,
        "start_row": start_row,
        "start_col": start_col,
        "track_id": track_id,
        "chaupai": chaupai,
        "path_cells": path_cells,
        "path_chars": path_chars,
        "formed_text": formed_text,
        "score": chaupai["score"],
        "nature": chaupai["nature"],
        "result": chaupai["result"],
        "remedy": chaupai["remedy"],
        "meaning_en": chaupai["meaning_en"],
        "context": chaupai["context"],
        "transliteration": chaupai["transliteration"],
        "symbol": chaupai["symbol"],
        "grid": SHALAKA_GRID  # Include grid for UI rendering
    }


# ==================================================================
# SECTION 1B — NEW v8.0 LOOKUP TABLES
# ==================================================================

# ── 27 Nitya Yogas (Sun longitude + Moon longitude, divide by 13°20') ─
NITYA_YOGAS = [
    ("Vishkambha",  "Inauspicious",  "Obstacles and delays. Avoid important beginnings."),
    ("Preeti",      "Auspicious",    "Love, friendship, and positive relationships. Good for affection."),
    ("Ayushman",    "Auspicious",    "Longevity and vitality. Good for health matters."),
    ("Saubhagya",   "Auspicious",    "Fortune and prosperity. Excellent for new ventures."),
    ("Shobhana",    "Auspicious",    "Beauty and splendour. Favourable for creative work."),
    ("Atiganda",    "Inauspicious",  "Accidents and obstructions. Caution with travel."),
    ("Sukarma",     "Auspicious",    "Noble deeds. Good for charitable acts and righteous work."),
    ("Dhriti",      "Auspicious",    "Stability and determination. Good for sustained effort."),
    ("Shula",       "Inauspicious",  "Pain and sorrow. Avoid confrontations."),
    ("Ganda",       "Inauspicious",  "Danger and disruptions. Exercise great caution."),
    ("Vriddhi",     "Auspicious",    "Growth and increase. Excellent for investments."),
    ("Dhruva",      "Auspicious",    "Fixed and stable. Good for permanent agreements."),
    ("Vyaghata",    "Inauspicious",  "Destruction and harm. Avoid risky activities."),
    ("Harshana",    "Auspicious",    "Joy and delight. Good for celebrations."),
    ("Vajra",       "Mixed",         "Hard like a diamond — obstacles cut through by courage."),
    ("Siddhi",      "Auspicious",    "Success and accomplishment. Highly auspicious."),
    ("Vyatipata",   "Inauspicious",  "Calamity. Very inauspicious — delay important matters."),
    ("Variyan",     "Auspicious",    "Comfort and luxury. Good for pleasurable activities."),
    ("Parigha",     "Inauspicious",  "Barrier and obstruction. Things feel blocked."),
    ("Shiva",       "Auspicious",    "Auspicious. Blessed by Shiva. Good for all auspicious deeds."),
    ("Siddha",      "Auspicious",    "Perfection and mastery. Excellent for learning and skill."),
    ("Sadhya",      "Auspicious",    "Achievable goals. Moderate auspiciousness."),
    ("Shubha",      "Auspicious",    "Auspicious and good. Favourable for most activities."),
    ("Shukla",      "Auspicious",    "Purity and brightness. Good for religious acts."),
    ("Brahma",      "Auspicious",    "Divine support. Excellent for all important beginnings."),
    ("Indra",       "Auspicious",    "Power and leadership. Good for authority-related matters."),
    ("Vaidhriti",   "Inauspicious",  "Separation and loss. Avoid major decisions."),
]

# ── Tithis (30 lunar days) ─────────────────────────────────────────
TITHIS = [
    ("Pratipada",    1,  "Shukla",  "Auspicious",    "New beginnings; Lord Agni. Good for starting ventures."),
    ("Dwitiya",      2,  "Shukla",  "Auspicious",    "Construction, agriculture; Lord Brahma."),
    ("Tritiya",      3,  "Shukla",  "Auspicious",    "Cutting hair, clothes; Lord Gauri."),
    ("Chaturthi",    4,  "Shukla",  "Mixed",         "Lord Ganesha. Avoid auspicious events; good for obstacles-removal work."),
    ("Panchami",     5,  "Shukla",  "Auspicious",    "Medicine, learning; Lord Naga."),
    ("Shashthi",     6,  "Shukla",  "Auspicious",    "Journey, war; Lord Kartik."),
    ("Saptami",      7,  "Shukla",  "Auspicious",    "Travel, vehicles; Lord Surya."),
    ("Ashtami",      8,  "Shukla",  "Mixed",         "Mixed results; Lord Shiva / Rudra."),
    ("Navami",       9,  "Shukla",  "Mixed",         "Mixed; Lord Durga. Avoid marriage."),
    ("Dashami",      10, "Shukla",  "Auspicious",    "Dharmic work; Lord Yama."),
    ("Ekadashi",     11, "Shukla",  "Auspicious",    "Fasting; Lord Vishnu. Highly auspicious."),
    ("Dwadashi",     12, "Shukla",  "Auspicious",    "Donation; Lord Vishnu."),
    ("Trayodashi",   13, "Shukla",  "Auspicious",    "Pleasure, weapons; Lord Kama / Shiva."),
    ("Chaturdashi",  14, "Shukla",  "Mixed",         "Destructive; Lord Shiva / Kali. Avoid auspicious."),
    ("Purnima",      15, "Shukla",  "Auspicious",    "Full Moon; Lord Chandra. Highly auspicious."),
    ("Pratipada",    1,  "Krishna", "Mixed",         "Krishna Paksha beginning; descent of blessings."),
    ("Dwitiya",      2,  "Krishna", "Auspicious",    "Moderate auspiciousness."),
    ("Tritiya",      3,  "Krishna", "Auspicious",    "Good for most activities."),
    ("Chaturthi",    4,  "Krishna", "Inauspicious",  "Avoid auspicious events."),
    ("Panchami",     5,  "Krishna", "Auspicious",    "Moderate."),
    ("Shashthi",     6,  "Krishna", "Mixed",         "Mixed results."),
    ("Saptami",      7,  "Krishna", "Auspicious",    "Moderate."),
    ("Ashtami",      8,  "Krishna", "Inauspicious",  "Avoid major events."),
    ("Navami",       9,  "Krishna", "Inauspicious",  "Avoid."),
    ("Dashami",      10, "Krishna", "Auspicious",    "Good for dharmic work."),
    ("Ekadashi",     11, "Krishna", "Auspicious",    "Fasting; Lord Vishnu."),
    ("Dwadashi",     12, "Krishna", "Auspicious",    "Donation."),
    ("Trayodashi",   13, "Krishna", "Mixed",         "Mixed."),
    ("Chaturdashi",  14, "Krishna", "Inauspicious",  "Most inauspicious Krishna tithi."),
    ("Amavasya",     15, "Krishna", "Mixed",         "New Moon; ancestor propitiation. Avoid new ventures."),
]

# ── Karana (half-tithi) — 11 types ────────────────────────────────
KARANAS = [
    ("Bava",      "Movable", "Auspicious",   "Good for all works."),
    ("Balava",    "Movable", "Auspicious",   "Pleasures, leisure."),
    ("Kaulava",   "Movable", "Auspicious",   "Family, friends."),
    ("Taitila",   "Movable", "Auspicious",   "Agriculture, trade."),
    ("Gara",      "Movable", "Auspicious",   "Domestic work."),
    ("Vanija",    "Movable", "Auspicious",   "Commerce, trade — best for business."),
    ("Vishti",    "Movable", "Inauspicious", "Bhadra — avoid all auspicious work."),
    ("Shakuni",   "Fixed",   "Mixed",        "Occult and healing."),
    ("Chatushpada","Fixed",  "Mixed",        "Four-footed beings, animals."),
    ("Naga",      "Fixed",   "Inauspicious", "Avoid; serpent energy."),
    ("Kimstughna","Fixed",   "Auspicious",   "First karana of Shukla Pratipada — auspicious."),
]

# ── Vara (weekday lords) ───────────────────────────────────────────
VARA_DATA = [
    ("Monday",    "Moon",    "Auspicious",   "Travel, trade, white flowers, silver."),
    ("Tuesday",   "Mars",    "Mixed",        "Courage, war, red items; avoid auspicious."),
    ("Wednesday", "Mercury", "Auspicious",   "Learning, commerce, writing, green."),
    ("Thursday",  "Jupiter", "Auspicious",   "Most auspicious day. Guru puja, yellow, gold."),
    ("Friday",    "Venus",   "Auspicious",   "Love, art, beauty, white; marriage."),
    ("Saturday",  "Saturn",  "Inauspicious", "Discipline, karma; avoid new starts. Shani puja."),
    ("Sunday",    "Sun",     "Mixed",        "Authority, government; moderate for most."),
]

# ── Mool Nakshatra table with pada-level detail ────────────────────
MOOL_NAKSHATRAS = {
    "Ashwini": {
        "pada_effects": {
            1: ("Severe", "Child's life threatened in first month. Immediate Ashwini Kumaras puja required."),
            2: ("Moderate","Father's wellbeing affected. Paternal remedies recommended."),
            3: ("Mild",   "Minor disturbances. Routine Ashwini puja sufficient."),
            4: ("Benign", "Auspicious Mool pada. No serious effects."),
        },
        "deity": "Ashwini Kumaras",
        "remedy": "Puja on 27th day after birth; gold horse donated; Ashwini Sukta chanted.",
        "pacification_days": [27, 90, 365],
    },
    "Ashlesha": {
        "pada_effects": {
            1: ("Severe", "Threat to maternal family. Sarpa (serpent) puja mandatory."),
            2: ("Moderate","Maternal uncle affected. Naga dosha remedies."),
            3: ("Mild",   "Minor maternal side disturbances."),
            4: ("Benign", "Lucky pada for Ashlesha. Wealth indicated."),
        },
        "deity": "Naga (Sarpa Devata)",
        "remedy": "Sarpa puja; milk offered to serpent images; Naga Panchami observance.",
        "pacification_days": [27, 90, 365],
    },
    "Magha": {
        "pada_effects": {
            1: ("Severe", "Paternal lineage affected; grandfather's health. Pitru puja essential."),
            2: ("Moderate","Father affected. Pitru tarpan and Gaya shraddha."),
            3: ("Mild",   "Minor ancestral disturbances."),
            4: ("Benign", "Auspicious pada — prosperity for family."),
        },
        "deity": "Pitru (Ancestors)",
        "remedy": "Pitru puja, shraddha on 27th day; black sesame and water offered to ancestors.",
        "pacification_days": [27, 90, 365],
    },
    "Jyeshtha": {
        "pada_effects": {
            1: ("Severe", "Elder sibling affected. Indra puja and elder sibling remedies."),
            2: ("Moderate","Head of family affected."),
            3: ("Mild",   "Minor disturbances in family."),
            4: ("Benign", "Prosperity for elder siblings."),
        },
        "deity": "Indra",
        "remedy": "Indra puja; rice and milk offering; elder sibling performs Abhisheka.",
        "pacification_days": [27, 90, 365],
    },
    "Mula": {
        "pada_effects": {
            1: ("Severe", "Father-in-law severely affected. Nairriti puja and Ketu remedies mandatory."),
            2: ("Moderate","Mother-in-law affected."),
            3: ("Mild",   "Minor in-law disturbances."),
            4: ("Benign", "Auspicious — prosperity for the native."),
        },
        "deity": "Nairriti (Nirriti)",
        "remedy": "Nairriti puja on 27th day; Mula Shanti homa; 108 Ketu mantras for 40 days.",
        "pacification_days": [27, 90, 365],
    },
    "Revati": {
        "pada_effects": {
            1: ("Severe", "Native's own life at risk in first month. Pushan puja and Mercury remedies."),
            2: ("Moderate","Maternal side affected."),
            3: ("Mild",   "Minor disturbances."),
            4: ("Benign", "Most auspicious Revati pada — great fortune."),
        },
        "deity": "Pushan",
        "remedy": "Pushan puja; green lentils donated; Mercury yantra installed.",
        "pacification_days": [27, 90, 365],
    },
}

# ── Mrityubhaga (critical/death degrees) — from Jataka Parijata ───
# {planet: {sign: critical_degree_within_sign}}
MRITYUBHAGA = {
    "Sun": {
        "Aries":20,"Taurus":9,"Gemini":12,"Cancer":6,"Leo":8,"Virgo":24,
        "Libra":16,"Scorpio":17,"Sagittarius":22,"Capricorn":2,"Aquarius":3,"Pisces":23
    },
    "Moon": {
        "Aries":26,"Taurus":12,"Gemini":13,"Cancer":25,"Leo":24,"Virgo":11,
        "Libra":26,"Scorpio":14,"Sagittarius":13,"Capricorn":25,"Aquarius":5,"Pisces":12
    },
    "Mars": {
        "Aries":27,"Taurus":22,"Gemini":18,"Cancer":25,"Leo":19,"Virgo":28,
        "Libra":8,"Scorpio":18,"Sagittarius":29,"Capricorn":28,"Aquarius":20,"Pisces":24
    },
    "Mercury": {
        "Aries":5,"Taurus":6,"Gemini":14,"Cancer":13,"Leo":9,"Virgo":17,
        "Libra":8,"Scorpio":27,"Sagittarius":4,"Capricorn":12,"Aquarius":17,"Pisces":19
    },
    "Jupiter": {
        "Aries":11,"Taurus":7,"Gemini":12,"Cancer":4,"Leo":8,"Virgo":15,
        "Libra":14,"Scorpio":18,"Sagittarius":9,"Capricorn":25,"Aquarius":21,"Pisces":16
    },
    "Venus": {
        "Aries":24,"Taurus":11,"Gemini":11,"Cancer":16,"Leo":16,"Virgo":21,
        "Libra":5,"Scorpio":15,"Sagittarius":14,"Capricorn":21,"Aquarius":17,"Pisces":16
    },
    "Saturn": {
        "Aries":14,"Taurus":27,"Gemini":20,"Cancer":22,"Leo":11,"Virgo":25,
        "Libra":20,"Scorpio":25,"Sagittarius":22,"Capricorn":24,"Aquarius":14,"Pisces":17
    },
}

# ── Pushkara Navamsa — auspicious navamsa divisions ───────────────
# For each sign, the navamsa positions that are Pushkara
PUSHKARA_NAVAMSA = {
    "Aries":        [3, 8],   # 3rd and 8th navamsa (Leo and Sagittarius)
    "Taurus":       [4, 8],
    "Gemini":       [1, 6],
    "Cancer":       [2, 6],
    "Leo":          [1, 7],
    "Virgo":        [3, 8],
    "Libra":        [4, 8],
    "Scorpio":      [2, 6],
    "Sagittarius":  [1, 8],
    "Capricorn":    [5, 8],
    "Aquarius":     [1, 8],
    "Pisces":       [4, 7],
}

# Pushkara Bhaga — specific auspicious degrees within each sign
PUSHKARA_BHAGA = {
    "Aries":21,"Taurus":14,"Gemini":18,"Cancer":8,"Leo":15,"Virgo":22,
    "Libra":7,"Scorpio":11,"Sagittarius":17,"Capricorn":28,"Aquarius":20,"Pisces":24
}

# ── Naisargika (natural) Bala — permanent strength in Rupas ───────
NAISARGIKA_BALA = {
    "Sun":60,"Moon":51.43,"Venus":45,"Jupiter":34.29,
    "Mercury":25.71,"Mars":17.14,"Saturn":8.57
}

# ── Requisite Rupas (minimum Shadbala needed) ─────────────────────
REQUISITE_RUPAS = {
    "Sun":390,"Moon":360,"Mars":300,"Mercury":420,
    "Jupiter":390,"Venus":330,"Saturn":300
}

# ── Graha Avastha — Baladi (5 states by degree in sign) ───────────
# Odd signs: Bala 0-6°, Kumara 6-12°, Yuva 12-18°, Vriddha 18-24°, Mrita 24-30°
# Even signs: reversed
BALADI_STATES = ["Bala","Kumara","Yuva","Vriddha","Mrita"]
BALADI_EFFECTS = {
    "Bala":    ("Infant — learning, potential", 40),
    "Kumara":  ("Youth — active, developing",   60),
    "Yuva":    ("Adult — full power, prime",    100),
    "Vriddha": ("Elder — wisdom, declining",    60),
    "Mrita":   ("Dead — very weak, inert",       10),
}

# ── Deeptadi Avastha (9 states by dignity + situation) ────────────
DEEPTADI_STATES = {
    "Deepta":   ("Radiant — exalted, full power",             100),
    "Swastha":  ("Comfortable — own sign",                     75),
    "Pramudita":("Joyful — great friend's sign",               50),
    "Shanta":   ("Peaceful — friendly sign",                   30),
    "Dina":     ("Distressed — neutral sign",                  20),
    "Dukhita":  ("Sad — inimical sign",                        10),
    "Vikala":   ("Disabled — combust or defeated",             5),
    "Khala":    ("Wicked — debilitated",                       3),
    "Kopa":     ("Angry — debilitated and afflicted",          1),
}


# ==================================================================
# SECTION 2 — CORE MATH (all v7.0 functions retained)
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
# SECTION 2B — NEW v8.0 CALCULATION FUNCTIONS
# ==================================================================

def get_panchanga(sun_lon: float, moon_lon: float,
                  weekday: int = None,
                  birth_date: datetime = None) -> Dict:
    """
    Compute all 5 Panchanga limbs from Sun and Moon sidereal longitudes.

    Args:
        sun_lon:    Sidereal Sun longitude (0-360)
        moon_lon:   Sidereal Moon longitude (0-360)
        weekday:    Python weekday() — 0=Monday; if None, derived from birth_date
        birth_date: datetime object (used for weekday if weekday is None)

    Returns dict with: vara, tithi, nakshatra, yoga, karana
    """
    # 1. VARA (weekday)
    if weekday is None and birth_date is not None:
        weekday = birth_date.weekday()  # 0=Monday
    vara_index = weekday if weekday is not None else 0
    vara_name, vara_lord, vara_nature, vara_desc = VARA_DATA[vara_index]

    # 2. TITHI — Moon - Sun difference, each tithi = 12°
    moon_sun_diff = (moon_lon - sun_lon) % 360
    tithi_index   = int(moon_sun_diff / 12)  # 0-29
    tithi_index   = min(tithi_index, 29)
    tithi_row     = TITHIS[tithi_index]
    tithi_name, tithi_num, paksha, tithi_nature, tithi_desc = tithi_row
    tithi_degree_elapsed = (moon_sun_diff % 12)
    tithi_pct = round(tithi_degree_elapsed / 12 * 100, 1)

    # 3. NAKSHATRA — Moon's nakshatra (primary limb)
    nak_name, pada, deg_in_nak = get_nakshatra(moon_lon)
    nak_lord = NAKSHATRA_LORDS[NAKSHATRAS.index(nak_name)]
    nak_gana = NAKSHATRA_GANA[nak_name]
    nak_nadi = NAKSHATRA_NADI[nak_name]

    # 4. NITYA YOGA — (Sun lon + Moon lon) / (13°20')
    yoga_lon    = (sun_lon + moon_lon) % 360
    yoga_index  = int(yoga_lon / NAKSHATRA_SIZE) % 27
    yoga_name, yoga_nature, yoga_desc = NITYA_YOGAS[yoga_index]

    # 5. KARANA — half tithi; 11 types cycling through 60 half-tithis
    # First karana of month = Kimstughna; karanas 1-7 repeat; 8-10 are fixed at end
    half_tithi = int(moon_sun_diff / 6)  # 0-59
    if half_tithi == 0:
        karana_idx = 10  # Kimstughna
    elif half_tithi >= 57:
        # Last 3 half-tithis: Shakuni, Chatushpada, Naga
        karana_idx = 7 + (half_tithi - 57)
    else:
        karana_idx = ((half_tithi - 1) % 7)
    karana_idx = min(karana_idx, 10)
    karana_name, karana_type, karana_nature, karana_desc = KARANAS[karana_idx]

    return {
        "vara": {
            "name":   vara_name,
            "lord":   vara_lord,
            "nature": vara_nature,
            "detail": vara_desc,
        },
        "tithi": {
            "name":    tithi_name,
            "number":  tithi_num,
            "paksha":  paksha,
            "nature":  tithi_nature,
            "detail":  tithi_desc,
            "elapsed_pct": tithi_pct,
            "full_name": f"{paksha} {tithi_name} ({tithi_num})",
        },
        "nakshatra": {
            "name":   nak_name,
            "pada":   pada,
            "lord":   nak_lord,
            "gana":   nak_gana,
            "nadi":   nak_nadi,
            "yoni":   NAKSHATRA_YONI.get(nak_name,""),
            "deg_in_nakshatra": round(deg_in_nak, 2),
        },
        "yoga": {
            "name":   yoga_name,
            "nature": yoga_nature,
            "detail": yoga_desc,
            "index":  yoga_index + 1,
        },
        "karana": {
            "name":   karana_name,
            "type":   karana_type,
            "nature": karana_nature,
            "detail": karana_desc,
        },
        "panchanga_quality": _panchanga_quality(vara_nature, tithi_nature, yoga_nature, karana_nature),
    }


def _panchanga_quality(vara_n, tithi_n, yoga_n, karana_n) -> str:
    """Overall panchanga quality for the day."""
    score = 0
    for n in [vara_n, tithi_n, yoga_n, karana_n]:
        if n == "Auspicious":
            score += 1
        elif n == "Inauspicious":
            score -= 1
    if score >= 3:
        return "Highly Auspicious"
    elif score >= 1:
        return "Auspicious"
    elif score == 0:
        return "Mixed"
    else:
        return "Inauspicious"


def check_mool_nakshatra(moon_longitude: float) -> Dict:
    """
    Check if Moon is in a Mool Nakshatra (Ganda Mool).
    Returns detailed analysis including pada-level severity and remedies.
    """
    nak_name, pada, deg_in_nak = get_nakshatra(moon_longitude)

    if nak_name not in MOOL_NAKSHATRAS:
        return {
            "is_mool": False,
            "nakshatra": nak_name,
            "pada": pada,
            "message": f"Moon in {nak_name} (Pada {pada}) — not a Mool Nakshatra."
        }

    mool_data = MOOL_NAKSHATRAS[nak_name]
    severity_label, severity_effect = mool_data["pada_effects"][pada]
    pacification_days = mool_data["pacification_days"]

    return {
        "is_mool": True,
        "nakshatra": nak_name,
        "pada": pada,
        "severity": severity_label,
        "effect": severity_effect,
        "deity": mool_data["deity"],
        "remedy": mool_data["remedy"],
        "pacification_milestones": {
            f"Day {d}": f"Important pacification puja on day {d} after birth"
            for d in pacification_days
        },
        "classical_rule": (
            f"{nak_name} is a Ganda Mool Nakshatra (junction point of signs/dashas). "
            f"Pada {pada} severity: {severity_label}. "
            "Classical texts prescribe Mool Shanti within 27 days of birth."
        ),
        "message": (
            f"Moon in {nak_name} Pada {pada} — Ganda Mool Nakshatra detected. "
            f"Severity: {severity_label}. {severity_effect}"
        )
    }


def get_all_planet_ganas(chart_nakshatras: Dict) -> Dict:
    """Return Gan (Deva/Manushya/Rakshasa) for every planet in the chart."""
    result = {}
    for planet, nak_data in chart_nakshatras.items():
        nak = nak_data.get("nakshatra","")
        result[planet] = NAKSHATRA_GANA.get(nak, "Unknown")
    return result


def get_chara_karakas(planets: Dict[str, float]) -> Dict:
    """
    Compute all 7 Jaimini Chara Karakas (excluding Ketu).
    Ranked by degree within sign (Rahu counted backwards: 30 - deg).
    Returns: AK, AmK, BK, MK, PK, GK, DK
    """
    KARAKA_NAMES = [
        "Atmakaraka (AK)",
        "Amatyakaraka (AmK)",
        "Bhratrukaraka (BK)",
        "Matrukaraka (MK)",
        "Putrakaraka (PK)",
        "Gnatikaraka (GK)",
        "Darakaraka (DK)",
    ]
    KARAKA_MEANINGS = [
        "Soul's purpose; the self; primary life driver",
        "Career, livelihood, minister; means of sustenance",
        "Siblings, courage, communication",
        "Mother, home, education, emotional roots",
        "Children, creativity, intelligence, past merit",
        "Relatives, disease, competition, transformation",
        "Spouse, partnerships, desires, fulfillment",
    ]

    relevant = {p: v for p, v in planets.items() if p not in ("Ketu",)}
    deg_map = {}
    for p, lon in relevant.items():
        deg = lon % 30
        if p == "Rahu":
            deg = 30 - deg  # Rahu counted in reverse
        deg_map[p] = round(deg, 4)

    # Sort descending by degree
    sorted_planets = sorted(deg_map.keys(), key=lambda p: deg_map[p], reverse=True)

    karakas = {}
    for i, karaka_name in enumerate(KARAKA_NAMES):
        if i < len(sorted_planets):
            planet = sorted_planets[i]
            sign, _ = longitude_to_sign(planets[planet])
            karakas[karaka_name] = {
                "planet":  planet,
                "degree":  deg_map[planet],
                "sign":    sign,
                "meaning": KARAKA_MEANINGS[i],
            }

    return karakas


def get_baladi_avastha(planet: str, longitude: float) -> Dict:
    """
    Baladi Avastha — 5-state system by degree position within sign.
    Odd signs (Aries, Gemini...): Bala→Kumara→Yuva→Vriddha→Mrita (0→6→12→18→24→30)
    Even signs (Taurus, Cancer...): reversed
    """
    sign, deg_in_sign = longitude_to_sign(longitude)
    sign_idx  = ZODIAC.index(sign)
    is_odd    = (sign_idx % 2 == 0)   # Aries=0 is odd sign (1st)
    segment   = int(deg_in_sign / 6)
    segment   = min(segment, 4)

    if is_odd:
        state = BALADI_STATES[segment]
    else:
        state = BALADI_STATES[4 - segment]

    effect, strength_pct = BALADI_EFFECTS[state]
    return {
        "state":        state,
        "effect":       effect,
        "strength_pct": strength_pct,
        "degree_in_sign": round(deg_in_sign, 2),
        "sign_parity": "Odd (forward)" if is_odd else "Even (reversed)",
    }


def get_deeptadi_avastha(planet: str, dignity: str,
                          combust: bool = False, defeated: bool = False) -> Dict:
    """Map dignity to Deeptadi Avastha (9-state system)."""
    if combust or defeated:
        state = "Vikala"
    elif dignity == "Exalted":
        state = "Deepta"
    elif dignity in ("Own", "Mool Trikona"):
        state = "Swastha"
    elif dignity == "Great Friend":
        state = "Pramudita"
    elif dignity == "Friendly":
        state = "Shanta"
    elif dignity == "Neutral":
        state = "Dina"
    elif dignity == "Inimical":
        state = "Dukhita"
    elif dignity == "Debilitated":
        state = "Khala"
    else:
        state = "Dina"

    effect, strength = DEEPTADI_STATES.get(state, ("Unknown", 20))
    return {
        "state":    state,
        "effect":   effect,
        "strength": strength,
    }


def get_jagrat_svapna_sushupti(planet: str, dignity: str) -> str:
    """
    Jagrat (awake/strong), Svapna (dreaming/moderate), Sushupti (sleeping/weak).
    Based on dignity classification.
    """
    if dignity in ("Exalted", "Own", "Mool Trikona"):
        return "Jagrat (Awake) — planet fully conscious and operative"
    elif dignity in ("Great Friend", "Friendly"):
        return "Svapna (Dreaming) — planet moderately active"
    else:
        return "Sushupti (Sleeping) — planet weakened, results delayed"


def check_mrityubhaga(planet: str, longitude: float) -> Dict:
    """
    Check if planet is at or near its Mrityubhaga (death degree).
    Returns within_orb=True if within 1° of critical degree.
    """
    if planet not in MRITYUBHAGA:
        return {"has_mrityubhaga": False, "planet": planet}

    sign, deg_in_sign = longitude_to_sign(longitude)
    critical_deg = MRITYUBHAGA[planet].get(sign, None)

    if critical_deg is None:
        return {"has_mrityubhaga": False, "planet": planet, "sign": sign}

    orb = abs(deg_in_sign - critical_deg)
    within_orb = orb <= 1.0
    exact = orb < 0.1

    return {
        "has_mrityubhaga":  within_orb,
        "planet":           planet,
        "sign":             sign,
        "planet_degree":    round(deg_in_sign, 2),
        "critical_degree":  critical_deg,
        "orb":              round(orb, 2),
        "exact":            exact,
        "severity": "Critical (exact)" if exact else "Severe (within 1°)" if within_orb else "Clear",
        "interpretation": (
            f"{planet} at {round(deg_in_sign,2)}° {sign} — "
            + (f"AT MRITYUBHAGA ({critical_deg}° {sign}). Planet severely weakened. "
               "Results of this planet largely fail to materialise. Strong remedies essential."
               if within_orb else
               f"Clear of Mrityubhaga ({critical_deg}° {sign}).")
        )
    }


def get_pushkara_analysis(planet: str, longitude: float) -> Dict:
    """
    Check Pushkara Navamsa and Pushkara Bhaga for a planet.
    Pushkara Navamsa: auspicious navamsa positions for each sign.
    Pushkara Bhaga: specific degree within sign that is highly auspicious.
    """
    sign, deg_in_sign = longitude_to_sign(longitude)
    sign_idx  = ZODIAC.index(sign)
    deg_in_sign_precise = longitude % 30

    # Navamsa number (1-9)
    navamsa_num = int(deg_in_sign_precise / (30/9)) + 1
    navamsa_num = min(navamsa_num, 9)

    pushkara_navamsas = PUSHKARA_NAVAMSA.get(sign, [])
    is_pushkara_navamsa = navamsa_num in pushkara_navamsas

    # Pushkara Bhaga
    pb_degree = PUSHKARA_BHAGA.get(sign, -1)
    pb_orb    = abs(deg_in_sign_precise - pb_degree) if pb_degree >= 0 else 99
    is_pushkara_bhaga = pb_orb <= 1.0

    return {
        "planet":               planet,
        "sign":                 sign,
        "degree_in_sign":       round(deg_in_sign_precise, 2),
        "navamsa_number":       navamsa_num,
        "is_pushkara_navamsa":  is_pushkara_navamsa,
        "pushkara_navamsas_for_sign": pushkara_navamsas,
        "pushkara_bhaga_degree": pb_degree,
        "pushkara_bhaga_orb":   round(pb_orb, 2),
        "is_pushkara_bhaga":    is_pushkara_bhaga,
        "interpretation": (
            f"{planet} in {sign} Navamsa {navamsa_num}: "
            + ("PUSHKARA NAVAMSA — highly auspicious position, planet gives excellent results. " if is_pushkara_navamsa else "")
            + ("PUSHKARA BHAGA — at the most auspicious degree of the sign (within 1°). " if is_pushkara_bhaga else "")
            + ("No Pushkara position." if not is_pushkara_navamsa and not is_pushkara_bhaga else "")
        )
    }


def calculate_argala(house_map: Dict[str, int]) -> Dict:
    """
    Calculate Argala (intervention/obstruction) for each of the 12 houses.

    Argala houses (intervention positions):
      2nd, 4th, 11th from any house = positive argala
      5th from any house = secondary argala

    Virodha Argala (obstruction, cancels argala):
      12th, 10th, 3rd, 9th from any house (counter to 2nd, 4th, 11th, 5th)

    Returns net argala score and details for each house.
    """
    # Build sign-level planet positions
    sign_planets: Dict[int, List[str]] = {i: [] for i in range(1, 13)}
    for planet, house in house_map.items():
        sign_planets[house].append(planet)

    BENEFICS  = {"Jupiter", "Venus", "Mercury", "Moon"}
    MALEFICS  = {"Sun", "Saturn", "Mars", "Rahu", "Ketu"}

    def planet_score(planets_in_house: List[str]) -> float:
        score = 0.0
        for p in planets_in_house:
            score += 1.5 if p in BENEFICS else -0.5 if p in MALEFICS else 1.0
        return score

    results = {}
    for h in range(1, 13):
        argala_pos  = [(h + 1) % 12 or 12,   # 2nd
                       (h + 3) % 12 or 12,   # 4th
                       (h + 10) % 12 or 12,  # 11th
                       (h + 4) % 12 or 12]   # 5th (secondary)
        virodha_pos = [(h + 11) % 12 or 12,  # 12th (counters 2nd)
                       (h + 9)  % 12 or 12,  # 10th (counters 4th)
                       (h + 2)  % 12 or 12,  # 3rd  (counters 11th)
                       (h + 8)  % 12 or 12]  # 9th  (counters 5th)

        argala_details  = []
        virodha_details = []
        net_score = 0.0

        for i, (ap, vp) in enumerate(zip(argala_pos, virodha_pos)):
            arg_planets = sign_planets[ap]
            vir_planets = sign_planets[vp]
            arg_score   = planet_score(arg_planets)
            vir_score   = planet_score(vir_planets)

            kind = ["2nd", "4th", "11th", "5th"][i]
            if arg_planets:
                argala_details.append(
                    f"H{ap} ({kind} from H{h}): {', '.join(arg_planets)} "
                    f"[raw score {arg_score:+.1f}]"
                )
            if vir_planets:
                virodha_details.append(
                    f"H{vp} (Virodha {kind}): {', '.join(vir_planets)} "
                    f"[obstructs {arg_score:+.1f}]"
                )

            # Net: argala reduced by virodha (virodha must exceed argala to cancel)
            if vir_score >= arg_score and arg_score > 0:
                net_score += 0  # argala cancelled
            else:
                net_score += (arg_score - max(vir_score, 0))

        results[h] = {
            "argala_sources":  argala_details,
            "virodha_sources": virodha_details,
            "net_score":       round(net_score, 2),
            "status": (
                "Strong Argala" if net_score >= 3 else
                "Moderate Argala" if net_score >= 1 else
                "Weak/Neutral" if net_score >= 0 else
                "Virodha Dominant (blocked)"
            )
        }
    return results


def calculate_graha_drishti_strength(house_map: Dict[str, int]) -> Dict[int, float]:
    """
    Compute weighted aspect (Drishti) strength arriving at each house.
    Weights: 7th aspect = 1.0, special aspects (Mars 4/8, Jupiter 5/9, Saturn 3/10) = 0.75
    Returns {house: total_drishti_strength}
    """
    ASPECT_WEIGHT = {
        7: 1.0,
        4: 0.75, 8: 0.75,   # Mars
        5: 0.75, 9: 0.75,   # Jupiter, Rahu, Ketu
        3: 0.75, 10: 0.75,  # Saturn
    }
    house_drishti: Dict[int, float] = {h: 0.0 for h in range(1, 13)}

    for planet, p_house in house_map.items():
        # Universal 7th aspect
        target_7 = ((p_house - 1 + 6) % 12) + 1
        house_drishti[target_7] += ASPECT_WEIGHT[7]

        # Special aspects
        if planet in SPECIAL_ASPECTS:
            for offset in SPECIAL_ASPECTS[planet]:
                target = ((p_house - 1 + offset - 1) % 12) + 1
                house_drishti[target] += ASPECT_WEIGHT.get(offset, 0.5)

    return {h: round(v, 2) for h, v in house_drishti.items()}


def get_ashtamesh_analysis(chart_data: "ChartData") -> Dict:
    """
    Analyse the 8th house lord (Ashtamesh) — longevity, transformation,
    hidden matters, sudden events.
    """
    lagna_idx   = ZODIAC.index(chart_data.lagna_sign)
    eighth_sign = ZODIAC[(lagna_idx + 7) % 12]
    ashtamesh   = SIGN_LORD[eighth_sign]
    am_house    = chart_data.house_map.get(ashtamesh, 0)
    am_dignity  = chart_data.dignities.get(ashtamesh, "Neutral")
    am_strength = chart_data.shadbala_proxy.get(ashtamesh, 45)

    placement_interp = {
        1:  "8th lord in Lagna — strong longevity yoga; native connected to research/occult. Rajayoga possible.",
        2:  "8th lord in 2nd — wealth through inheritance; family affected by sudden events; speech issues.",
        3:  "8th lord in 3rd — sudden changes in communication/siblings; journalistic or investigative career.",
        4:  "8th lord in 4th — mother's health; property disputes; domestic secrets.",
        5:  "8th lord in 5th — children's health watch; speculation losses; past-life karma active.",
        6:  "8th lord in 6th — Viparita Raja Yoga (Harsha); enemies defeated; medical/legal field favoured.",
        7:  "8th lord in 7th — spouse health watch; business partnerships face hidden complications.",
        8:  "8th lord in 8th — strong 8th house; longevity very strong; occult mastery; inheritance.",
        9:  "8th lord in 9th — father's health watch; fortune comes through transformation; spiritual depth.",
        10: "8th lord in 10th — career in research, medicine, law; public life involves hidden/confidential work.",
        11: "8th lord in 11th — gains through inheritance/insurance; elder sibling health watch; secret networks.",
        12: "8th lord in 12th — expenses from hidden sources; spiritual liberation; foreign settlement possible.",
    }

    return {
        "eighth_sign":     eighth_sign,
        "ashtamesh":       ashtamesh,
        "ashtamesh_house": am_house,
        "ashtamesh_dignity": am_dignity,
        "ashtamesh_strength": am_strength,
        "is_viparita_raja_yoga": am_house in [6, 8, 12],
        "longevity_quality": (
            "Strong" if am_dignity in ["Exalted","Own","Mool Trikona"] else
            "Moderate" if am_dignity in ["Great Friend","Friendly","Neutral"] else
            "Challenged"
        ),
        "interpretation": placement_interp.get(am_house, ""),
        "note": (
            "Viparita Raja Yoga (Sarala) — 8th lord in 8th: exceptional hidden strength and longevity."
            if am_house == 8 else
            "Viparita Raja Yoga (Harsha) — 8th lord in 6th: enemies defeated; rise through service."
            if am_house == 6 else
            "Viparita Raja Yoga (Vimala) — 8th lord in 12th: liberation-oriented; frugal but spiritually rich."
            if am_house == 12 else ""
        )
    }


def calculate_shadbala_extended(chart_data: "ChartData") -> Dict:
    """
    Extended Shadbala proxy incorporating all 6 components.
    Full mathematical Shadbala requires precise birth time; this gives
    a well-structured multi-component estimate.

    Components:
      1. Sthana Bala (positional strength) — from dignity
      2. Dig Bala (directional strength) — from house position
      3. Kala Bala (temporal strength) — benefic/malefic day-night
      4. Chesta Bala (motional strength) — retrograde/direct
      5. Naisargika Bala (natural strength) — fixed by planet
      6. Drik Bala (aspectual strength) — benefic vs malefic aspects
    """
    results = {}
    classical_planets = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]

    for planet in classical_planets:
        if planet not in chart_data.planets:
            continue

        lon     = chart_data.planets[planet]
        sign, _ = longitude_to_sign(lon)
        dignity = chart_data.dignities.get(planet, "Neutral")
        house   = chart_data.house_map.get(planet, 6)
        retro   = chart_data.retrograde.get(planet, False)

        # 1. Sthana Bala (max 60 Shashtiamsas → scaled to 100)
        sthana = DIGNITY_STRENGTH.get(dignity, 45) * 0.6

        # 2. Dig Bala
        dig_factor = get_directional_strength(planet, house)
        dig_bala   = 60 * dig_factor

        # 3. Kala Bala — simplified: benefics strong by day, malefics by night
        BENEFICS_K = {"Jupiter","Venus","Mercury","Moon"}
        # Day birth approximated by Sun in houses 7-12 (above horizon)
        sun_house = chart_data.house_map.get("Sun", 1)
        is_day = sun_house in range(7, 13)
        if (planet in BENEFICS_K and is_day) or (planet not in BENEFICS_K and not is_day):
            kala_bala = 50
        else:
            kala_bala = 30

        # 4. Chesta Bala — retrograde planets get max chesta bala
        chesta_bala = 60 if retro else 30

        # 5. Naisargika Bala (fixed)
        naisargika = NAISARGIKA_BALA.get(planet, 25)

        # 6. Drik Bala — count benefic vs malefic aspects
        BENEFIC_PLANETS = {"Jupiter","Venus","Mercury","Moon"}
        MALEFIC_PLANETS = {"Saturn","Mars","Rahu","Ketu","Sun"}
        aspect_list = chart_data.house_map  # reuse via aspect_map
        drik = 0
        for asp_planet, asp_house in chart_data.house_map.items():
            if asp_planet == planet:
                continue
            # Check if asp_planet aspects planet's house
            p_house = house
            # 7th aspect
            if ((asp_house - 1 + 6) % 12) + 1 == p_house:
                drik += 15 if asp_planet in BENEFIC_PLANETS else -15
            # Special aspects
            if asp_planet in SPECIAL_ASPECTS:
                for offset in SPECIAL_ASPECTS[asp_planet]:
                    if ((asp_house - 1 + offset - 1) % 12) + 1 == p_house:
                        drik += 10 if asp_planet in BENEFIC_PLANETS else -10
        drik_bala = max(0, min(60, 30 + drik))

        # Total (sum of components)
        total_rupas = sthana + dig_bala + kala_bala + chesta_bala + naisargika + drik_bala
        requisite   = REQUISITE_RUPAS.get(planet, 360)
        adequacy_pct = round(total_rupas / requisite * 100, 1)

        results[planet] = {
            "sthana_bala":    round(sthana, 1),
            "dig_bala":       round(dig_bala, 1),
            "kala_bala":      round(kala_bala, 1),
            "chesta_bala":    round(chesta_bala, 1),
            "naisargika_bala":round(naisargika, 1),
            "drik_bala":      round(drik_bala, 1),
            "total_rupas":    round(total_rupas, 1),
            "requisite_rupas":requisite,
            "adequacy_pct":   adequacy_pct,
            "is_shadbala_adequate": adequacy_pct >= 100,
            "status": (
                "Excellent" if adequacy_pct >= 130 else
                "Adequate"  if adequacy_pct >= 100 else
                "Weak"      if adequacy_pct >= 70  else
                "Very Weak"
            )
        }

    return results


def get_nitya_yoga(sun_lon: float, moon_lon: float) -> Dict:
    """Compute Nitya Yoga from Sun + Moon longitude."""
    yoga_lon   = (sun_lon + moon_lon) % 360
    yoga_index = int(yoga_lon / NAKSHATRA_SIZE) % 27
    name, nature, desc = NITYA_YOGAS[yoga_index]
    return {
        "index":  yoga_index + 1,
        "name":   name,
        "nature": nature,
        "detail": desc,
        "calculation": (
            f"(Sun {round(sun_lon,2)}° + Moon {round(moon_lon,2)}°) mod 360 = "
            f"{round(yoga_lon,2)}° ÷ 13°20' = Yoga {yoga_index+1}: {name}"
        )
    }


# ==================================================================
# SECTION 3 — DASHA CALCULATIONS (unchanged from v7.0)
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
# SECTION 4 — CHART DATA CLASS (extended for v8.0)
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

        # v8.0 additions
        self.panchanga:            Dict = {}
        self.mool_nakshatra:       Dict = {}
        self.planet_ganas:         Dict = {}
        self.baladi_avasthas:      Dict = {}
        self.deeptadi_avasthas:    Dict = {}
        self.mrityubhaga_flags:    Dict = {}
        self.pushkara_analysis:    Dict = {}
        self.argala:               Dict = {}
        self.chara_karakas:        Dict = {}
        self.shadbala_extended:    Dict = {}
        self.graha_drishti_strength: Dict = {}
        self.ashtamesh:            Dict = {}
        self.nitya_yoga:           Dict = {}

        self._house_map = None
        self._lord_map  = None

        self._compute_derived()
        self._compute_v8_extended()

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

    def _compute_v8_extended(self):
        """Compute all v8.0 additional data."""
        # Panchanga
        if self.birth_date:
            self.panchanga = get_panchanga(
                self.planets["Sun"],
                self.planets["Moon"],
                birth_date=self.birth_date
            )

        # Mool Nakshatra
        self.mool_nakshatra = check_mool_nakshatra(self.planets["Moon"])

        # Planet Ganas
        self.planet_ganas = get_all_planet_ganas(self.nakshatras)

        # Baladi and Deeptadi Avasthas
        for p, lon in self.planets.items():
            self.baladi_avasthas[p] = get_baladi_avastha(p, lon)
            combust_flag = self.shadbala_breakdown.get(p, {}).get("combust", False)
            defeated = any(g["loser"] == p for g in self.graha_yuddha)
            self.deeptadi_avasthas[p] = get_deeptadi_avastha(
                p, self.dignities.get(p, "Neutral"), combust_flag, defeated
            )

        # Mrityubhaga
        for p, lon in self.planets.items():
            self.mrityubhaga_flags[p] = check_mrityubhaga(p, lon)

        # Pushkara
        for p, lon in self.planets.items():
            self.pushkara_analysis[p] = get_pushkara_analysis(p, lon)

        # Argala
        self.argala = calculate_argala(self.house_map)

        # Chara Karakas (Jaimini)
        self.chara_karakas = get_chara_karakas(self.planets)

        # Shadbala Extended
        self.shadbala_extended = calculate_shadbala_extended(self)

        # Graha Drishti Strength
        self.graha_drishti_strength = calculate_graha_drishti_strength(self.house_map)

        # Ashtamesh
        self.ashtamesh = get_ashtamesh_analysis(self)

        # Nitya Yoga
        self.nitya_yoga = get_nitya_yoga(self.planets["Sun"], self.planets["Moon"])

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

    def get_extended_report(self) -> str:
        """Generate comprehensive v8.0 extended report string."""
        lines = [
            "=" * 72,
            "EXTENDED VEDIC CHART ANALYSIS — v8.0",
            "=" * 72, ""
        ]

        # Panchanga
        p = self.panchanga
        if p:
            lines += [
                "▶ PANCHANGA (FIVE LIMBS OF THE DAY)",
                f"  Vara (Weekday):   {p['vara']['name']} — Lord: {p['vara']['lord']} | {p['vara']['nature']}",
                f"                    {p['vara']['detail']}",
                f"  Tithi (Lunar day):{p['tithi']['full_name']} | {p['tithi']['nature']} ({p['tithi']['elapsed_pct']}% elapsed)",
                f"                    {p['tithi']['detail']}",
                f"  Nakshatra:        {p['nakshatra']['name']} Pada {p['nakshatra']['pada']} | Lord: {p['nakshatra']['lord']}",
                f"                    Gana: {p['nakshatra']['gana']} | Nadi: {p['nakshatra']['nadi']} | Yoni: {p['nakshatra']['yoni']}",
                f"  Yoga (Nitya):     {p['yoga']['name']} ({p['yoga']['index']}/27) | {p['yoga']['nature']}",
                f"                    {p['yoga']['detail']}",
                f"  Karana:           {p['karana']['name']} ({p['karana']['type']}) | {p['karana']['nature']}",
                f"                    {p['karana']['detail']}",
                f"  Overall Quality:  {p['panchanga_quality']}",
                ""
            ]

        # Mool Nakshatra
        m = self.mool_nakshatra
        if m["is_mool"]:
            lines += [
                "▶ GANDA MOOL NAKSHATRA ⚠",
                f"  Moon in: {m['nakshatra']} Pada {m['pada']}",
                f"  Severity: {m['severity']}",
                f"  Effect:   {m['effect']}",
                f"  Deity:    {m['deity']}",
                f"  Remedy:   {m['remedy']}",
                f"  Pacification milestones: Days {', '.join(str(d) for d in MOOL_NAKSHATRAS.get(m['nakshatra'],{}).get('pacification_days',[]))}",
                ""
            ]
        else:
            lines += ["▶ GANDA MOOL: Moon not in Mool Nakshatra — no special concern.", ""]

        # Planet Ganas
        lines.append("▶ PLANET GANAS (Nakshatra-based temperament)")
        for planet, gana in self.planet_ganas.items():
            nak = self.nakshatras.get(planet, {}).get("nakshatra","")
            lines.append(f"  {planet:10s}: {gana:10s} (nakshatra: {nak})")
        moon_gana = self.planet_ganas.get("Moon","")
        lines.append(f"  → Moon Gana ({moon_gana}): defines primary temperament for compatibility matching.")
        lines.append("")

        # Chara Karakas
        lines.append("▶ JAIMINI CHARA KARAKAS (7 Soul-Purpose Planets)")
        for karaka, data in self.chara_karakas.items():
            lines.append(
                f"  {karaka:35s}: {data['planet']:8s} ({data['degree']:.2f}° in {data['sign']})"
            )
            lines.append(f"    → {data['meaning']}")
        lines.append("")

        # Nitya Yoga
        ny = self.nitya_yoga
        lines += [
            "▶ NITYA YOGA (Birth Day Yoga)",
            f"  Yoga: {ny['name']} ({ny['index']}/27) | Nature: {ny['nature']}",
            f"  {ny['detail']}",
            f"  Calculation: {ny['calculation']}",
            ""
        ]

        # Graha Avastha
        lines.append("▶ GRAHA AVASTHA (Planetary States)")
        lines.append("  {'Planet':<10} {'Baladi':^12} {'Strength':^8} {'Deeptadi':^14} {'Str':^5} {'Jagrat State'}")
        lines.append("  " + "-" * 72)
        for p_name in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"]:
            if p_name not in self.planets:
                continue
            bal  = self.baladi_avasthas.get(p_name, {})
            dee  = self.deeptadi_avasthas.get(p_name, {})
            jss  = get_jagrat_svapna_sushupti(p_name, self.dignities.get(p_name,"Neutral"))
            jss_short = jss.split("—")[0].strip()
            lines.append(
                f"  {p_name:<10} {bal.get('state','?'):^12} {bal.get('strength_pct',0):^8}%"
                f" {dee.get('state','?'):^14} {dee.get('strength',0):^5} {jss_short}"
            )
        lines.append("")

        # Mrityubhaga flags
        mb_flagged = [p for p, d in self.mrityubhaga_flags.items() if d.get("has_mrityubhaga")]
        lines.append("▶ MRITYUBHAGA (Critical Death Degrees)")
        if mb_flagged:
            for p in mb_flagged:
                d = self.mrityubhaga_flags[p]
                lines.append(f"  ⚠ {d['interpretation']}")
        else:
            lines.append("  All planets clear of Mrityubhaga degrees.")
        lines.append("")

        # Pushkara
        pk_found = [p for p, d in self.pushkara_analysis.items()
                    if d.get("is_pushkara_navamsa") or d.get("is_pushkara_bhaga")]
        lines.append("▶ PUSHKARA NAVAMSA / PUSHKARA BHAGA")
        if pk_found:
            for p in pk_found:
                d = self.pushkara_analysis[p]
                lines.append(f"  ✦ {d['interpretation']}")
        else:
            lines.append("  No planets in Pushkara Navamsa or Pushkara Bhaga.")
        lines.append("")

        # Shadbala Extended
        lines.append("▶ SHADBALA EXTENDED (6-Component Strength)")
        lines.append(f"  {'Planet':<10} {'Sthana':>7} {'Dig':>6} {'Kala':>6} {'Chesta':>7} {'Naisa':>7} {'Drik':>6} {'Total':>7} {'Req':>6} {'%':>6} {'Status'}")
        lines.append("  " + "-" * 80)
        for p_name in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]:
            d = self.shadbala_extended.get(p_name, {})
            if d:
                lines.append(
                    f"  {p_name:<10} {d['sthana_bala']:>7.1f} {d['dig_bala']:>6.1f}"
                    f" {d['kala_bala']:>6.1f} {d['chesta_bala']:>7.1f}"
                    f" {d['naisargika_bala']:>7.1f} {d['drik_bala']:>6.1f}"
                    f" {d['total_rupas']:>7.1f} {d['requisite_rupas']:>6} {d['adequacy_pct']:>6.1f}%"
                    f" {d['status']}"
                )
        lines.append("")

        # Argala (top 3 houses by score)
        lines.append("▶ ARGALA ANALYSIS (Top Houses by Intervention Strength)")
        sorted_argala = sorted(self.argala.items(), key=lambda x: x[1]["net_score"], reverse=True)
        for h, data in sorted_argala[:6]:
            if data["net_score"] != 0:
                lines.append(f"  House {h:2d}: {data['status']:25s} (net: {data['net_score']:+.1f})")
                for src in data["argala_sources"][:2]:
                    lines.append(f"          + {src}")
                for vir in data["virodha_sources"][:1]:
                    lines.append(f"          - {vir}")
        lines.append("")

        # Graha Drishti Strength
        lines.append("▶ GRAHA DRISHTI STRENGTH (Aspect Weight per House)")
        for h in range(1, 13):
            strength = self.graha_drishti_strength.get(h, 0)
            bar = "█" * int(strength * 2)
            lines.append(f"  H{h:2d}: {strength:4.1f}  {bar}")
        lines.append("")

        # Ashtamesh
        am = self.ashtamesh
        lines += [
            "▶ ASHTAMESH (8th Lord — Longevity & Transformation)",
            f"  8th Sign:   {am['eighth_sign']}",
            f"  Ashtamesh:  {am['ashtamesh']} | Dignity: {am['ashtamesh_dignity']} | House: {am['ashtamesh_house']}",
            f"  Longevity:  {am['longevity_quality']}",
            f"  Placement:  {am['interpretation']}",
        ]
        if am.get("note"):
            lines.append(f"  ✦ {am['note']}")
        lines.append("")

        lines += ["=" * 72, "END OF EXTENDED REPORT v8.0", "=" * 72]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        base = {
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
            ],
            # v8.0
            "panchanga":           self.panchanga,
            "mool_nakshatra":      self.mool_nakshatra,
            "planet_ganas":        self.planet_ganas,
            "chara_karakas":       self.chara_karakas,
            "nitya_yoga":          self.nitya_yoga,
            "baladi_avasthas":     self.baladi_avasthas,
            "deeptadi_avasthas":   self.deeptadi_avasthas,
            "mrityubhaga_flags":   self.mrityubhaga_flags,
            "pushkara_analysis":   self.pushkara_analysis,
            "argala":              self.argala,
            "shadbala_extended":   self.shadbala_extended,
            "graha_drishti_strength": self.graha_drishti_strength,
            "ashtamesh":           self.ashtamesh,
        }
        return base

    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ==================================================================
# SECTIONS 5-14: All v7.0 code retained identically below
# (build_context, helper accessors, prediction rules, rule engine,
#  topic analysis functions, Ashtakoota, transit, Varshphal,
#  Ram Shalaka, yearly prediction, demo utilities)
# ==================================================================

def build_context(chart: "ChartData", dasha_info: Dict = None,
                  sade_sati_info: Dict = None,
                  transit_planets: Dict[str, float] = None) -> Dict:
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
        "transit_house_map":   transit_house_map,
        "transit_sign_map":    transit_sign_map,
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
        "ad_planet":           dasha_info.get("antardasha","")   if dasha_info else "",
        "ad_house":            dasha_info.get("ad_house",0)      if dasha_info else 0,
        "ad_dignity":          dasha_info.get("ad_dignity","")   if dasha_info else "",
        "ad_sign":             dasha_info.get("ad_sign","")      if dasha_info else "",
        "sade_sati_active":    False,
        "sade_sati_phase":     "",
        "sade_sati_detail":    "",
    }

    if sade_sati_info:
        ctx["sade_sati_active"] = sade_sati_info.get("active", False)
        ctx["sade_sati_phase"]  = sade_sati_info.get("phase", "")
        ctx["sade_sati_detail"] = sade_sati_info.get("detail", "")
    elif transit_planets and "Saturn" in transit_sign_map:
        ss = check_sade_sati(chart.moon_sign, transit_sign_map["Saturn"])
        ctx["sade_sati_active"] = ss.get("active", False)
        ctx["sade_sati_phase"]  = ss.get("phase", "")
        ctx["sade_sati_detail"] = ss.get("detail", "")

    return ctx


# ── Helper accessors (all from v7.0) ──────────────────────────────

def _house(planet: str, ctx: dict) -> int:
    return ctx["house_map"].get(planet, 0)

def _transit_house(planet: str, ctx: dict) -> int:
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


# ── Prediction Rules (all v7.0 rules retained, unchanged) ─────────
# [Full PREDICTION_RULES list from v7.0 is preserved here in production use.
#  For brevity in this listing, the rules are referenced by inclusion — the
#  complete ~600-line PREDICTION_RULES list from v7.0 Section 7 is unchanged
#  and must be inserted here when deploying. The rule engine (evaluate_rules,
#  score_topic, _apply_dasha_boost, _narrative_block) also unchanged.]

# To keep this file self-contained for review, we include a condensed reference:
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
]


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
    related  = set([topic_lord] + (related_planets or []))
    result   = []
    boosted  = set()
    for r in fired_rules:
        rc = copy.deepcopy(r)
        if rc.get("activation") == "dasha_activated" and md_planet in related and rc["id"] not in boosted:
            old = rc["score"]
            rc["score"] = round(old * 1.5) if old > 0 else round(old * 1.2)
            rc["title"] += " [⚡ MD ACTIVATED]"
            rc["detail"] += "\n  ⚡ Amplified: the running Mahadasha planet directly governs this life area."
            boosted.add(rc["id"])
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


# ── Analysis functions (v7.0 — unchanged) ─────────────────────────

def analyze_career(chart: "ChartData", check_date: datetime = None,
                   transit_planets: Dict[str, float] = None) -> Dict:
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


def analyze_marriage(chart: "ChartData", check_date: datetime = None,
                     transit_planets: Dict[str, float] = None) -> Dict:
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


def analyze_children(chart: "ChartData", check_date: datetime = None,
                     transit_planets: Dict[str, float] = None) -> Dict:
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


def analyze_health(chart: "ChartData", check_date: datetime = None,
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


def analyze_general_yogas(chart: "ChartData", check_date: datetime = None,
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


# ── Ashtakoota Matchmaking (v7.0 — unchanged) ─────────────────────

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


def calculate_ashtakoota(c1: "ChartData", c2: "ChartData",
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


# ── Approximate transit positions (v7.0 fallback) ─────────────────

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
    target = datetime(year, month, day, 12, 0, 0)
    days   = (target - _J2000_EPOCH).days + (target - _J2000_EPOCH).seconds / 86400.0
    transits = {}
    for planet in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu"]:
        lon = (_J2000_SIDEREAL[planet] + _DAILY_MOTION[planet] * days) % 360.0
        transits[planet] = round(lon, 2)
    transits["Ketu"] = round((transits["Rahu"] + 180.0) % 360.0, 2)
    return transits


# ── Varshphal (v7.0 — retained) ───────────────────────────────────

def calculate_varshphal(chart: "ChartData", year: int,
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
    themes = []
    muntha_nature = {
        1:("Auspicious","Major personal initiative and fresh beginnings."),
        2:("Auspicious","Financial accumulation and family harmony."),
        3:("Moderate","Courage, communication, siblings dominate."),
        4:("Auspicious","Home, property, mother, inner peace highlighted."),
        5:("Auspicious","Creativity, children, love, speculation surge."),
        6:("Challenging","Service, health vigilance, competition."),
        7:("Auspicious","Partnerships and public dealings at peak."),
        8:("Challenging","Transformation, sudden changes, occult."),
        9:("Auspicious","Fortune, dharma, long travel, father blessed."),
        10:("Auspicious","Career and authority at peak."),
        11:("Auspicious","Gains, social networks, ambitions fulfilled."),
        12:("Challenging","Expenses, isolation, foreign connections."),
    }
    nature, desc = muntha_nature.get(muntha_house, ("Moderate",""))
    natal_deg_in_sign = round(chart.ascendant % 30, 1)

    themes.append({
        "category": "Muntha (Annual Ascendant Marker)",
        "nature": nature,
        "calculation": (
            f"Muntha progresses 1 sign/year from natal Lagna ({chart.lagna_sign}). "
            f"Years elapsed: {years_elapsed}. Muntha = {muntha_sign} at {natal_deg_in_sign}°. "
            f"Natal house: {muntha_house}."
        ),
        "classical_rule": "Muntha is to Varshphal what Lagna is to the natal chart.",
        "interpretation": desc,
        "modifier": (
            f"Muntha lord {muntha_lord} is natally {muntha_lord_dignity} in natal H{muntha_lord_house}. "
            + ("Strong Muntha lord powerfully supports the year's themes." if muntha_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "Debilitated Muntha lord weakens results; remedies essential." if muntha_lord_dignity == "Debilitated"
               else "Moderate Muntha lord delivers mixed results.")
        )
    })

    try:
        sr_date = datetime(prediction_year, chart.birth_date.month, chart.birth_date.day)
        sr_date_str = sr_date.strftime("%d %b %Y")
        phase1_end = (sr_date + timedelta(days=122)).strftime("%d %b")
        phase2_end = (sr_date + timedelta(days=243)).strftime("%d %b")
        phase3_end = (sr_date + timedelta(days=365)).strftime("%d %b")
    except Exception:
        sr_date_str, phase1_end, phase2_end, phase3_end = f"~{prediction_year}", "~M4", "~M8", "~M12"

    themes.append({
        "category": "Tri-Pataki Chakra",
        "nature": "Neutral",
        "calculation": (
            f"Solar return: {sr_date_str}. "
            f"Phase 1 ({sr_date_str}–{phase1_end}): {tri_pataki['udaya_muntha']}. "
            f"Phase 2 ({phase1_end}–{phase2_end}): {tri_pataki['madhya_muntha']}. "
            f"Phase 3 ({phase2_end}–{phase3_end}): {tri_pataki['asta_muntha']}."
        ),
        "classical_rule": "Times events within the solar year by trimester.",
        "interpretation": (
            f"Rising Phase: {tri_pataki['udaya_muntha']} — initial themes. "
            f"Peak Phase: {tri_pataki['madhya_muntha']} — mid-year. "
            f"Setting Phase: {tri_pataki['asta_muntha']} — consolidation."
        ),
        "modifier": ""
    })

    themes.append({
        "category": "Varsha Lagna",
        "nature": "Auspicious" if varsha_lagna_lord_dignity in ["Exalted","Own","Mool Trikona","Great Friend"] else "Moderate",
        "calculation": (
            f"Varsha Lagna: {varsha_lagna_sign}. Lord: {varsha_lagna_lord}. "
            f"Natal dignity: {varsha_lagna_lord_dignity}."
        ),
        "classical_rule": "Varsha Lagna lord's natal strength governs the year's overall tone.",
        "interpretation": (
            f"Year {prediction_year} coloured by {varsha_lagna_sign} energy. "
            f"Lord {varsha_lagna_lord} natally {varsha_lagna_lord_dignity}."
        ),
        "modifier": ""
    })

    themes.append({
        "category": "Varshesha (Year Lord)",
        "nature": "Auspicious" if varshesha_dignity in ["Exalted","Own","Mool Trikona","Great Friend"] else "Challenging" if varshesha_dignity == "Debilitated" else "Moderate",
        "calculation": (
            f"Varshesha: {varshesha}. Natal dignity: {varshesha_dignity}. H{varshesha_house}."
        ),
        "classical_rule": "Day-lord of Solar Return governs the year's overall results.",
        "interpretation": (
            f"Varshesha {varshesha} governs {prediction_year}. "
            + ("Productive year." if varshesha_dignity in ["Exalted","Own","Mool Trikona","Great Friend"]
               else "Obstacles — intensify remedies." if varshesha_dignity == "Debilitated"
               else "Average results; effort required.")
        ),
        "modifier": (
            f"Varshesha ({varshesha}) is also Muntha lord — results concentrated."
            if varshesha == muntha_lord else ""
        )
    })

    if muntha_house in [6, 8, 12]:
        themes.append({
            "category": "Dusthana Muntha — Year of Transformation",
            "nature": "Challenging",
            "calculation": f"Muntha in H{muntha_house} (dusthana) in {prediction_year}.",
            "classical_rule": "Muntha in dusthana brings obstacles and inner work.",
            "interpretation": (
                "A year for resilience and inner transformation. "
                + ("Service, health, competition (6th)." if muntha_house==6 else
                   "Sudden changes, occult, hidden finances (8th)." if muntha_house==8 else
                   "Expenses, spiritual retreat, foreign matters (12th).")
            ),
            "modifier": (
                f"Strong Muntha lord {muntha_lord} softens the dusthana effect."
                if muntha_lord_dignity in ["Exalted","Own","Mool Trikona"] else ""
            )
        })

    if transit_planets:
        jup_sign = longitude_to_sign(transit_planets.get("Jupiter", 0))[0]
        jup_idx  = ZODIAC.index(jup_sign)
        lagna_idx_local = ZODIAC.index(chart.lagna_sign)
        jup_from_lagna = ((jup_idx - lagna_idx_local) % 12) + 1
        jup_from_moon  = ((jup_idx - ZODIAC.index(chart.moon_sign)) % 12) + 1
        jup_nature = "Auspicious" if jup_from_lagna in [1,2,5,9,11] else "Challenging" if jup_from_lagna in [4,8,12] else "Moderate"

        themes.append({
            "category": f"Transit Jupiter in {jup_sign}",
            "nature": jup_nature,
            "calculation": (
                f"Transit Jupiter: {jup_sign} (H{jup_from_lagna} from Lagna; H{jup_from_moon} from Moon)."
            ),
            "classical_rule": "Jupiter's annual transit is the primary fortune indicator.",
            "interpretation": (
                f"From Lagna (H{jup_from_lagna}): "
                + {1:"Peak personal growth.",2:"Wealth from career.",5:"Creativity/children.",
                   7:"Partnership blessed.",9:"Fortune active.",10:"Career peak.",11:"Gains peak.",
                   4:"Domestic focus.",6:"Rival obstacles.",8:"Transformation.",12:"Foreign/spiritual."
                   }.get(jup_from_lagna, "Moderate transit.")
                + f" From Moon (H{jup_from_moon}): "
                + ("Guruchandra Yoga — emotional expansion." if jup_from_moon in [1,5,9,11] else "Moderate Moon influence.")
            ),
            "modifier": ""
        })

        sat_sign = longitude_to_sign(transit_planets.get("Saturn", 0))[0]
        sat_idx  = ZODIAC.index(sat_sign)
        sat_from_lagna = ((sat_idx - lagna_idx_local) % 12) + 1
        sati = check_sade_sati(chart.moon_sign, sat_sign)
        kant = check_kantaka_shani(chart.moon_sign, sat_sign)
        sat_nature = "Challenging" if (sati["active"] or kant["active"] or sat_from_lagna in [4,7,8,10]) else "Positive" if sat_from_lagna in [3,6,11] else "Moderate"

        themes.append({
            "category": f"Transit Saturn in {sat_sign}",
            "nature": sat_nature,
            "calculation": (
                f"Transit Saturn: {sat_sign} (H{sat_from_lagna} from Lagna). "
                + (f"Sade Sati: {sati['phase']}. " if sati["active"] else "Sade Sati: Not active. ")
                + (f"Kantaka: {kant.get('position','')}." if kant["active"] else "Kantaka: Not active.")
            ),
            "classical_rule": "Saturn transit governs karmic lessons and discipline areas.",
            "interpretation": (
                {3:"Upachaya — courageous effort rewarded.",6:"Upachaya — enemies defeated.",
                 11:"Best Saturn transit — gains arrive.",1:"Lagna — restructuring.",
                 4:"Kantaka/domestic disruptions.",7:"Kantaka/partnerships.",
                 8:"Transformation.",10:"Authority tests.",12:"Retreat/expenses."
                 }.get(sat_from_lagna, "Karmic pressure.")
                + (" ⚠ SADE SATI active." if sati["active"] else "")
                + (" ⚠ KANTAKA SHANI active." if kant["active"] else "")
            ),
            "modifier": (
                "Protective: Shani puja, black sesame donation, oil massage Saturdays."
                if sati["active"] or kant["active"] else ""
            )
        })

    return themes


# ── Ram Shalaka Oracle (v7.0 — unchanged) ─────────────────────────

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
        outcome_key = "auspicious_high";   outcome_en = "Highly Auspicious";   score_pct = 90 + random.randint(0,10)
    elif power_hits == 2:
        outcome_key = "auspicious_medium"; outcome_en = "Auspicious";          score_pct = 70 + random.randint(0,15)
    elif power_hits == 1:
        outcome_key = "auspicious_low";    outcome_en = "Mildly Auspicious";   score_pct = 50 + random.randint(0,15)
    elif path_score > 25:
        outcome_key = "neutral";           outcome_en = "Neutral / Mixed";     score_pct = 40 + random.randint(-5,10)
    elif path_score > 15:
        outcome_key = "inauspicious_low";  outcome_en = "Mildly Inauspicious"; score_pct = 25 + random.randint(0,15)
    else:
        outcome_key = "inauspicious_high"; outcome_en = "Inauspicious — Wait"; score_pct = 10 + random.randint(0,15)

    meaning = RAM_SHALAKA_MEANINGS[outcome_key]
    verse_hindi, verse_en = "श्रीगुरु चरन सरोज रज", "By the Guru's grace, proceed with faith."
    for r_range, (h, e) in SHALAKA_VERSE_MAP.items():
        if path_score in r_range:
            verse_hindi, verse_en = h, e
            break

    grid_display = []
    path_set = set(map(tuple, path_cells))
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
    }


# ── Yearly Prediction (v7.0 — unchanged) ──────────────────────────

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


def get_year_prediction(chart: "ChartData", year: int) -> Dict:
    check_date = datetime(year, 6, 15)
    dasha_info = chart.get_current_dasha_info(check_date)

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
    career    = analyze_career(chart, check_date, transit_planets=transit_planets)
    marriage  = analyze_marriage(chart, check_date, transit_planets=transit_planets)
    children  = analyze_children(chart, check_date, transit_planets=transit_planets)
    health    = analyze_health(chart, check_date, transit_planets=transit_planets)
    yogas     = analyze_general_yogas(chart, check_date, transit_planets=transit_planets)

    lagna_idx = ZODIAC.index(chart.lagna_sign)
    moon_idx  = ZODIAC.index(chart.moon_sign)
    j_idx     = ZODIAC.index(transit_jupiter_sign)
    jh_lagna  = ((j_idx - lagna_idx) % 12) + 1
    jh_moon   = ((j_idx - moon_idx)  % 12) + 1

    jupiter_transit_notes = []
    if jh_lagna in [1,5,9]:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — exceptionally auspicious.")
    elif jh_lagna in [2,11]:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} — wealth and gains favoured.")
    elif jh_lagna == 10:
        jupiter_transit_notes.append(f"Jupiter transiting 10th — peak career transit.")
    elif jh_lagna in [4,8,12]:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} — muted outer results; inner focus.")
    else:
        jupiter_transit_notes.append(f"Jupiter in H{jh_lagna} from Lagna — moderate results.")

    if jh_moon in [1,5,9,11]:
        jupiter_transit_notes.append(f"Guruchandra Yoga: Jupiter H{jh_moon} from Moon.")

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
    }


# ==================================================================
# SECTION 15 — DEMO & UTILITIES
# ==================================================================

def generate_demo_chart() -> "ChartData":
    """Generate a demo chart for testing all v8.0 features."""
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


def load_chart_from_file(filepath: str) -> "ChartData":
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


def print_full_report(chart: "ChartData", year: int = None):
    """Print complete report — natal + extended (v8.0) + yearly prediction."""
    year = year or datetime.now().year
    print("=" * 72)
    print("VEDIC ASTROLOGY COMPLETE REPORT — Engine v8.0")
    print("=" * 72)
    print(f"Lagna: {chart.lagna_sign}  Moon: {chart.moon_sign}  Sun: {chart.sun_sign}")
    print(f"Atmakaraka: {chart.atmakaraka}  Amatyakaraka: {chart.amatyakaraka}")

    # Birth Panchanga
    if chart.panchanga:
        p = chart.panchanga
        print(f"\nBirth Panchanga: {p['vara']['name']} | {p['tithi']['full_name']} | "
              f"{p['nakshatra']['name']} P{p['nakshatra']['pada']} | "
              f"{p['yoga']['name']} Yoga | {p['karana']['name']} Karana")
        print(f"Overall Panchanga Quality: {p['panchanga_quality']}")

    # Mool Nakshatra
    m = chart.mool_nakshatra
    if m["is_mool"]:
        print(f"\n⚠ GANDA MOOL: {m['nakshatra']} Pada {m['pada']} — {m['severity']}")
        print(f"  {m['effect']}")

    # Nitya Yoga
    ny = chart.nitya_yoga
    print(f"\nNitya Yoga: {ny['name']} — {ny['nature']} ({ny['detail']})")

    # Chara Karakas
    print("\nChara Karakas:")
    for karaka, data in chart.chara_karakas.items():
        print(f"  {karaka:35s}: {data['planet']} ({data['degree']:.2f}° {data['sign']})")

    # Extended report
    print()
    print(chart.get_extended_report())

    # Yearly prediction
    prediction = get_year_prediction(chart, year)
    print(f"\n{'='*72}")
    print(f"YEAR {year} PREDICTION (v8.0)")
    print(f"{'='*72}")
    dasha = prediction["dasha"]
    print(f"MD: {dasha.get('mahadasha','')} | AD: {dasha.get('antardasha','')} | PD: {dasha.get('pratyantardasha','')}")
    print(f"Career:   {prediction['career']['rating']} ({prediction['career']['net_score']:+d})")
    print(f"Marriage: {prediction['marriage']['rating']} ({prediction['marriage']['net_score']:+d})")
    print(f"Children: {prediction['children']['rating']} ({prediction['children']['net_score']:+d})")
    print(f"Health:   {prediction['health']['rating']} ({prediction['health']['net_score']:+d})")


# ==================================================================
# QUICK REFERENCE — v8.0 NEW FUNCTIONS SUMMARY
# ==================================================================
"""
NEW FUNCTIONS (all importable):

  get_panchanga(sun_lon, moon_lon, weekday=None, birth_date=None)
    → Full 5-limb Panchanga dict (Vara, Tithi, Nakshatra, Yoga, Karana)

  check_mool_nakshatra(moon_longitude)
    → Ganda Mool check with pada-level severity, deity, remedy, milestones

  get_all_planet_ganas(chart_nakshatras)
    → Deva/Manushya/Rakshasa for every planet

  get_chara_karakas(planets)
    → All 7 Jaimini Chara Karakas ranked by degree (Rahu reversed)

  get_baladi_avastha(planet, longitude)
    → Bala/Kumara/Yuva/Vriddha/Mrita state with strength percentage

  get_deeptadi_avastha(planet, dignity, combust, defeated)
    → Deepta/Swastha/Pramudita/.../Kopa (9-state system)

  get_jagrat_svapna_sushupti(planet, dignity)
    → Jagrat/Svapna/Sushupti state string

  check_mrityubhaga(planet, longitude)
    → Mrityubhaga check with orb, severity, interpretation

  get_pushkara_analysis(planet, longitude)
    → Pushkara Navamsa and Pushkara Bhaga check

  calculate_argala(house_map)
    → Net argala score for all 12 houses with virodha analysis

  calculate_graha_drishti_strength(house_map)
    → Weighted aspect strength arriving at each house

  get_ashtamesh_analysis(chart_data)
    → 8th lord analysis, longevity, Viparita Raja Yoga check

  calculate_shadbala_extended(chart_data)
    → 6-component Shadbala with adequacy percentage

  get_nitya_yoga(sun_lon, moon_lon)
    → 27 Nitya Yoga identification with nature and effects

NEW ChartData ATTRIBUTES:
  .panchanga, .mool_nakshatra, .planet_ganas, .chara_karakas
  .baladi_avasthas, .deeptadi_avasthas, .mrityubhaga_flags
  .pushkara_analysis, .argala, .shadbala_extended
  .graha_drishti_strength, .ashtamesh, .nitya_yoga

NEW ChartData METHODS:
  .get_extended_report() → Full v8.0 extended analysis string
"""

if __name__ == "__main__":
    chart = generate_demo_chart()
    print_full_report(chart, year=2025)
