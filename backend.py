from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Dict, List
import math
import logging
from enum import Enum
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nadi Astrology API",
    description="Free Nadi Astrology Prediction API with Hindi/English support",
    version="2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sudeepdbg.github.io", "http://localhost:3000", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ZODIAC_SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
ZODIAC_SIGNS_HINDI = ["मेष", "वृषभ", "मिथुन", "कर्क", "सिंह", "कन्या", "तुला", "वृश्चिक", "धनु", "मकर", "कुंभ", "मीन"]
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
NAKSHATRAS_HINDI = ["अश्विनी", "भरणी", "कृत्तिका", "रोहिणी", "मृगशिरा", "आर्द्रा", "पुनर्वसु", "पुष्य", "आश्लेषा", "मघा", "पूर्व फाल्गुनी", "उत्तर फाल्गुनी", "हस्त", "चित्रा", "स्वाति", "विशाखा", "अनुराधा", "ज्येष्ठा", "मूल", "पूर्वाषाढ़ा", "उत्तराषाढ़ा", "श्रवण", "धनिष्ठा", "शतभिषा", "पूर्व भाद्रपद", "उत्तर भाद्रपद", "रेवती"]

class Language(str, Enum):
    ENGLISH = "English"
    HINDI = "Hindi"

class BirthDetails(BaseModel):
    name: str = Field(..., min_length=1)
    date: str 
    time: str 
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    language: Language = Language.ENGLISH

    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Use YYYY-MM-DD")

class PlanetaryPosition(BaseModel):
    planet: str
    planet_hindi: str
    longitude: float
    sign: str
    sign_hindi: str
    house: int
    nakshatra: str
    nakshatra_hindi: str

class NadiPrediction(BaseModel):
    birth_details: BirthDetails
    planetary_positions: List[PlanetaryPosition]
    ascendant: str
    ascendant_hindi: str
    moon_sign: str
    moon_sign_hindi: str
    prediction: str
    timestamp: str

class AstrologyCalculator:
    @staticmethod
    def calculate_julian_day(year, month, day, hour, minute):
        if month <= 2:
            year -= 1
            month += 12
        a = year // 100
        b = 2 - a + (a // 4)
        jd = (365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += (hour + minute / 60.0) / 24.0
        return jd

    @staticmethod
    def get_planet_pos(jd, planet):
        # Simplified orbital logic for "Free Forever" mode
        offsets = {"Sun": 280, "Moon": 218, "Mars": 44, "Mercury": 77, "Jupiter": 34, "Venus": 131, "Saturn": 49}
        speeds = {"Sun": 0.98, "Moon": 13.17, "Mars": 0.52, "Mercury": 4.09, "Jupiter": 0.08, "Venus": 1.6, "Saturn": 0.03}
        n = jd - 2451545.0
        pos = (offsets.get(planet, 0) + speeds.get(planet, 0.1) * n) % 360
        return pos

@app.post("/predict", response_model=NadiPrediction)
async def generate_prediction(details: BirthDetails):
    try:
        dt = datetime.strptime(f"{details.date} {details.time}", "%Y-%m-%d %H:%M")
        jd = AstrologyCalculator.calculate_julian_day(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        
        positions = []
        planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
        planets_hi = ["सूर्य", "चंद्र", "मंगल", "बुध", "गुरु", "शुक्र", "शनि"]

        for en, hi in zip(planets, planets_hi):
            lon = AstrologyCalculator.get_planet_pos(jd, en)
            s_idx = int(lon / 30) % 12
            n_idx = int(lon / 13.33) % 27
            positions.append(PlanetaryPosition(
                planet=en, planet_hindi=hi, longitude=round(lon, 2),
                sign=ZODIAC_SIGNS[s_idx], sign_hindi=ZODIAC_SIGNS_HINDI[s_idx],
                house=(s_idx + 1), nakshatra=NAKSHATRAS[n_idx], nakshatra_hindi=NAKSHATRAS_HINDI[n_idx]
            ))

        moon = next(p for p in positions if p.planet == "Moon")
        sun = next(p for p in positions if p.planet == "Sun")

        # Language Logic
        if details.language == Language.HINDI:
            pred = f"नमस्ते {details.name}। आपके नक्षत्र {moon.nakshatra_hindi} और राशि {moon.sign_hindi} के अनुसार, आपका जीवन आध्यात्मिक शांति और करियर में सफलता की ओर अग्रसर है। सूर्य की {sun.sign_hindi} में स्थिति आपके व्यक्तित्व में तेज प्रदान करती है।"
        else:
            pred = f"Greetings {details.name}. Based on your {moon.nakshatra} Nakshatra and {moon.sign} Moon sign, your life is moving towards spiritual peace and career success. Sun in {sun.sign} grants strength to your personality."

        return NadiPrediction(
            birth_details=details,
            planetary_positions=positions,
            ascendant=ZODIAC_SIGNS[0], ascendant_hindi=ZODIAC_SIGNS_HINDI[0],
            moon_sign=moon.sign, moon_sign_hindi=moon.sign_hindi,
            prediction=pred,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "active", "version": "2.0"}
