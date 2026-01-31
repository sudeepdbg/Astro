# -*- coding: utf-8 -*-
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

# CORS configuration - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
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
    "मूल", "पूर्वाषाढ़ा", "उत्तराषाढ़ा", "श्रवण", "धनिष्ठा", "शतभिषा",
    "पूर्व भाद्रपद", "उत्तर भाद्रपद", "रेवती"
]

PLANETS_HINDI = {
    "Sun": "सूर्य",
    "Moon": "चंद्र",
    "Mars": "मंगल",
    "Mercury": "बुध",
    "Jupiter": "गुरु",
    "Venus": "शुक्र",
    "Saturn": "शनि"
}

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

class AstrologyCalculator:
    @staticmethod
    def calculate_julian_day(year: int, month: int, day: int, hour: int, minute: int) -> float:
        """Calculate Julian Day Number"""
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
        """Calculate simplified planetary position"""
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
    """Generate detailed Nadi prediction"""
    if lang == Language.HINDI:
        return f"""प्रिय {name},

🌙 जीवन का उद्देश्य (धर्म) 🌙
आपका चंद्रमा {m_sign_hi} राशि में {m_nak_hi} नक्षत्र में स्थित है। यह दर्शाता है कि आपकी आत्मा शांति और आध्यात्मिक ज्ञान की खोज में है। आपका जीवन उद्देश्य दूसरों की सेवा करना और उन्हें अपने दयालु स्वभाव से प्रेरित करना है।

💼 करियर और समृद्धि 💼
सूर्य की {s_sign_hi} में स्थिति आपके व्यक्तित्व में नेतृत्व के गुण प्रदान करती है। गुरु {j_sign_hi} में होने से आपको शिक्षा, परामर्श या आध्यात्मिक मार्गदर्शन के क्षेत्र में सफलता मिलेगी। जब आप अपने सच्चे उद्देश्य के साथ जुड़ते हैं तो धन की प्राप्ति होती है।

❤️ संबंध और परिवार ❤️
चंद्रमा की स्थिति आपको भावनात्मक गहराई और देखभाल करने की क्षमता प्रदान करती है। आपके संबंध करुणा और समझ के माध्यम से फलते-फूलते हैं। एक महत्वपूर्ण साझेदारी उभरेगी जो आपके जीवन में गहरा आनंद लाएगी।

🏥 स्वास्थ्य और दीर्घायु 🏥
ग्रहों की स्थिति मजबूत जीवन शक्ति का संकेत देती है जब आप संतुलन बनाए रखते हैं। नियमित ध्यान, योग और प्रकृति से जुड़ाव आपकी भलाई को बढ़ाएगा।

🕉️ आध्यात्मिक मार्ग 🕉️
आपका {m_nak_hi} नक्षत्र प्राचीन ज्ञान और रहस्यमय परंपराओं से गहरा संबंध प्रकट करता है। भक्ति, सेवा और चिंतन के माध्यम से आपकी आध्यात्मिक जागृति तेज होती है।

तारों ने बोल दिया है। ब्रह्मांड की योजना पर विश्वास करें।

ॐ शांति शांति शांति 🙏"""
    else:
        return f"""Dear {name},

🌙 LIFE PURPOSE (DHARMA) 🌙
Your Moon resides in {m_sign} sign within the {m_nak} Nakshatra. This reveals that your soul seeks peace and spiritual wisdom. Your life purpose is to serve others and inspire them through your compassionate nature.

💼 CAREER & PROSPERITY 💼
The Sun in {s_sign} bestows leadership qualities upon your personality. With Jupiter positioned in {j_sign}, you will find success in fields related to education, counseling, or spiritual guidance. Financial abundance flows when you align with your authentic purpose.

❤️ RELATIONSHIPS & FAMILY ❤️
The Moon's placement grants you emotional depth and nurturing abilities. Your relationships thrive through compassion and understanding. A significant partnership will emerge that brings profound joy to your life.

🏥 HEALTH & LONGEVITY 🏥
The planetary configuration indicates robust vitality when you maintain balance. Regular meditation, yogic practices, and connection with nature will significantly enhance your well-being.

🕉️ SPIRITUAL PATH 🕉️
Your {m_nak} Nakshatra reveals a deep connection to ancient wisdom and mystical traditions. Your spiritual awakening accelerates through devotional practices and service to humanity.

The stars have spoken. Trust in the universe's plan for you.

Om Shanti Shanti Shanti 🙏"""

@app.get("/")
def read_root():
    return {
        "message": "Nadi Astrology API",
        "version": "2.0",
        "status": "operational",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "version": "2.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=NadiPrediction)
async def generate_prediction(details: BirthDetails):
    try:
        logger.info(f"Generating prediction for {details.name}")
        
        # Parse date and time
        dt = datetime.strptime(f"{details.date} {details.time}", "%Y-%m-%d %H:%M")
        
        # Calculate Julian Day
        jd = AstrologyCalculator.calculate_julian_day(
            dt.year, dt.month, dt.day, dt.hour, dt.minute
        )
        
        # Calculate planetary positions
        positions = []
        planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
        
        for planet in planets:
            longitude = AstrologyCalculator.get_planet_position(jd, planet)
            sign_idx = int(longitude / 30) % 12
            nakshatra_idx = int(longitude / 13.333333) % 27
            
            positions.append(PlanetaryPosition(
                planet=planet,
                planet_hindi=PLANETS_HINDI[planet],
                longitude=round(longitude, 2),
                sign=ZODIAC_SIGNS[sign_idx],
                sign_hindi=ZODIAC_SIGNS_HINDI[sign_idx],
                house=(sign_idx + 1),
                nakshatra=NAKSHATRAS[nakshatra_idx],
                nakshatra_hindi=NAKSHATRAS_HINDI[nakshatra_idx]
            ))
        
        # Get specific planetary data
        sun_data = next(p for p in positions if p.planet == "Sun")
        moon_data = next(p for p in positions if p.planet == "Moon")
        jupiter_data = next(p for p in positions if p.planet == "Jupiter")
        
        # Generate prediction
        prediction_text = generate_nadi_prediction_text(
            name=details.name,
            m_sign=moon_data.sign,
            m_sign_hi=moon_data.sign_hindi,
            m_nak=moon_data.nakshatra,
            m_nak_hi=moon_data.nakshatra_hindi,
            s_sign=sun_data.sign,
            s_sign_hi=sun_data.sign_hindi,
            j_sign=jupiter_data.sign,
            j_sign_hi=jupiter_data.sign_hindi,
            lang=details.language
        )
        
        result = NadiPrediction(
            birth_details=details,
            planetary_positions=positions,
            ascendant=sun_data.sign,
            ascendant_hindi=sun_data.sign_hindi,
            moon_sign=moon_data.sign,
            moon_sign_hindi=moon_data.sign_hindi,
            prediction=prediction_text,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction generated successfully for {details.name}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
