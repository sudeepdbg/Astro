from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List
import math
import logging
from enum import Enum
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nadi Astrology API",
    description="Nadi Astrology Prediction API with Hindi/English support",
    version="2.0"
)

# CORS configuration - Essential for GitHub Pages to talk to Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

ZODIAC_SIGNS_HINDI = [
    "मेष", "वृषभ", "मिथुन", "कर्क", "सिंह", "कन्या",
    "तुला", "वृश्चिक", "धनु", "मकर", "कुंभ", "मीन"
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

NAKSHATRAS_HINDI = [
    "अश्विनी", "भरणी", "कृत्तिका", "रोहिणी", "मृगशिरा", "आर्द्रा",
    "पुनर्वसु", "पुष्य", "आश्लेषा", "मघा", "पूर्व फाल्गुनी", "उत्तर फाल्गुनी",
    "हस्त", "चित्रा", "स्वाति", "विशाखा", "अनुराधा", "ज्येष्ठा",
    "मूल", "पूर्वाषाढ़ा", "उत्तरषाढ़ा", "श्रवण", "धनिष्ठा", "शतभिषा",
    "पूर्व भाद्रपद", "उत्तर भाद्रपद", "रेवती"
]

PLANETS_HINDI = {
    "Sun": "सूर्य", "Moon": "चंद्र", "Mars": "मंगल", "Mercury": "बुध",
    "Jupiter": "गुरु", "Venus": "शुक्र", "Saturn": "शनि"
}

# --- Models ---
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
            # Ensures backend understands the date string from frontend
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

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

# --- Business Logic ---
class AstrologyCalculator:
    @staticmethod
    def calculate_julian_day(year: int, month: int, day: int, hour: int, minute: int) -> float:
        if month <= 2:
            year -= 1
            month += 12
        a = year // 100
        b = 2 - a + (a // 4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += (hour + minute / 60.0) / 24.0
        return jd

    @staticmethod
    def get_planet_position(jd: float, planet: str) -> float:
        orbital_data = {
            "Sun": {"offset": 280.460, "speed": 0.9856474},
            "Moon": {"offset": 218.316, "speed": 13.176396},
            "Mars": {"offset": 44.0, "speed": 0.5240},
            "Mercury": {"offset": 77.0, "speed": 4.0923},
            "Jupiter": {"offset": 34.0, "speed": 0.0831},
            "Venus": {"offset": 131.0, "speed": 1.6021},
            "Saturn": {"offset": 49.0, "speed": 0.0334}
        }
        data = orbital_data.get(planet, {"offset": 0, "speed": 0.1})
        n = jd - 2451545.0
        return (data["offset"] + data["speed"] * n) % 360

def generate_nadi_prediction_text(name, m_sign, m_sign_hi, m_nak, m_nak_hi, s_sign, s_sign_hi, j_sign, j_sign_hi, lang):
    if lang == Language.HINDI:
        return f"प्रिय {name},\n\n🌙 जीवन का उद्देश्य 🌙\nआपका चंद्रमा {m_sign_hi} राशि और {m_nak_hi} नक्षत्र में है। यह शांति और आध्यात्मिक ज्ञान की खोज दर्शाता है।\n\n💼 करियर 💼\nसूर्य की {s_sign_hi} में स्थिति नेतृत्व गुण प्रदान करती है। गुरु {j_sign_hi} में होने से शिक्षा या परामर्श में सफलता मिलेगी।\n\n🕉️ आध्यात्मिक मार्ग 🕉️\n{m_nak_hi} प्राचीन ज्ञान से गहरा संबंध प्रकट करता है। ब्रह्मांड की योजना पर विश्वास करें।\n\nशुभम अस्तु 🙏"
    else:
        return f"Dear {name},\n\n🌙 LIFE PURPOSE 🌙\nYour Moon in {m_sign} and {m_nak} Nakshatra reveals a soul seeking wisdom and peace.\n\n💼 CAREER 💼\nThe Sun in {s_sign} bestows leadership. With Jupiter in {j_sign}, you will find success in education or guidance fields.\n\n🕉️ SPIRITUAL PATH 🕉️\nYour {m_nak} Nakshatra shows a deep connection to ancient traditions. Trust the divine plan.\n\nBlessings 🙏"

# --- Endpoints ---
@app.post("/predict", response_model=NadiPrediction)
async def generate_prediction(details: BirthDetails):
    try:
        dt = datetime.strptime(f"{details.date} {details.time}", "%Y-%m-%d %H:%M")
        jd = AstrologyCalculator.calculate_julian_day(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        
        positions = []
        planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
        
        for p_name in planets:
            lon = AstrologyCalculator.get_planet_position(jd, p_name)
            s_idx = int(lon / 30) % 12
            n_idx = int(lon / 13.333333) % 27
            
            positions.append(PlanetaryPosition(
                planet=p_name, planet_hindi=PLANETS_HINDI[p_name], longitude=round(lon, 2),
                sign=ZODIAC_SIGNS[s_idx], sign_hindi=ZODIAC_SIGNS_HINDI[s_idx],
                house=(s_idx + 1), nakshatra=NAKSHATRAS[n_idx], nakshatra_hindi=NAKSHATRAS_HINDI[n_idx]
            ))
        
        sun = next(p for p in positions if p.planet == "Sun")
        moon = next(p for p in positions if p.planet == "Moon")
        jupiter = next(p for p in positions if p.planet == "Jupiter")
        
        text = generate_nadi_prediction_text(
            details.name, moon.sign, moon.sign_hindi, moon.nakshatra, moon.nakshatra_hindi,
            sun.sign, sun.sign_hindi, jupiter.sign, jupiter.sign_hindi, details.language
        )
        
        return NadiPrediction(
            birth_details=details, planetary_positions=positions,
            ascendant=sun.sign, ascendant_hindi=sun.sign_hindi,
            moon_sign=moon.sign, moon_sign_hindi=moon.sign_hindi,
            prediction=text, timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "active", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
