from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import math

app = FastAPI(title="Nadi Astrology API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BirthDetails(BaseModel):
    name: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PlanetaryPosition(BaseModel):
    planet: str
    longitude: float
    sign: str
    house: int
    nakshatra: str

class NadiPrediction(BaseModel):
    birth_details: BirthDetails
    planetary_positions: list[PlanetaryPosition]
    ascendant: str
    moon_sign: str
    prediction: str
    timestamp: str

# Zodiac signs
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# 27 Nakshatras
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

def calculate_julian_day(year: int, month: int, day: int, hour: int, minute: int) -> float:
    """Calculate Julian Day Number"""
    if month <= 2:
        year -= 1
        month += 12
    
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    jd += (hour + minute / 60.0) / 24.0
    
    return jd

def calculate_sun_position(jd: float) -> float:
    """Simplified sun position calculation"""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360
    g_rad = math.radians(g)
    
    lambda_sun = (L + 1.915 * math.sin(g_rad) + 0.020 * math.sin(2 * g_rad)) % 360
    return lambda_sun

def calculate_moon_position(jd: float) -> float:
    """Simplified moon position calculation"""
    n = jd - 2451545.0
    L = (218.316 + 13.176396 * n) % 360
    return L

def get_zodiac_sign(longitude: float) -> str:
    """Get zodiac sign from celestial longitude"""
    sign_index = int(longitude / 30)
    return ZODIAC_SIGNS[sign_index % 12]

def get_nakshatra(longitude: float) -> str:
    """Get nakshatra from celestial longitude"""
    nakshatra_index = int(longitude / 13.333333)
    return NAKSHATRAS[nakshatra_index % 27]

def calculate_planetary_positions(birth_dt: datetime) -> list[PlanetaryPosition]:
    """Calculate approximate planetary positions"""
    jd = calculate_julian_day(
        birth_dt.year, birth_dt.month, birth_dt.day,
        birth_dt.hour, birth_dt.minute
    )
    
    sun_long = calculate_sun_position(jd)
    moon_long = calculate_moon_position(jd)
    
    # Simplified planetary calculations (approximations)
    positions = [
        PlanetaryPosition(
            planet="Sun",
            longitude=sun_long,
            sign=get_zodiac_sign(sun_long),
            house=1,  # Simplified
            nakshatra=get_nakshatra(sun_long)
        ),
        PlanetaryPosition(
            planet="Moon",
            longitude=moon_long,
            sign=get_zodiac_sign(moon_long),
            house=4,  # Simplified
            nakshatra=get_nakshatra(moon_long)
        ),
        PlanetaryPosition(
            planet="Mars",
            longitude=(sun_long + 45) % 360,
            sign=get_zodiac_sign((sun_long + 45) % 360),
            house=3,
            nakshatra=get_nakshatra((sun_long + 45) % 360)
        ),
        PlanetaryPosition(
            planet="Mercury",
            longitude=(sun_long + 15) % 360,
            sign=get_zodiac_sign((sun_long + 15) % 360),
            house=2,
            nakshatra=get_nakshatra((sun_long + 15) % 360)
        ),
        PlanetaryPosition(
            planet="Jupiter",
            longitude=(sun_long + 120) % 360,
            sign=get_zodiac_sign((sun_long + 120) % 360),
            house=9,
            nakshatra=get_nakshatra((sun_long + 120) % 360)
        ),
        PlanetaryPosition(
            planet="Venus",
            longitude=(sun_long - 25) % 360,
            sign=get_zodiac_sign((sun_long - 25) % 360),
            house=7,
            nakshatra=get_nakshatra((sun_long - 25) % 360)
        ),
        PlanetaryPosition(
            planet="Saturn",
            longitude=(sun_long + 240) % 360,
            sign=get_zodiac_sign((sun_long + 240) % 360),
            house=10,
            nakshatra=get_nakshatra((sun_long + 240) % 360)
        )
    ]
    
    return positions

def generate_nadi_prediction(positions: list[PlanetaryPosition], name: str) -> str:
    """Generate Nadi-style prediction based on planetary positions"""
    sun_pos = next(p for p in positions if p.planet == "Sun")
    moon_pos = next(p for p in positions if p.planet == "Moon")
    
    prediction = f"""The Ancient Nadi Leaves Reveal for {name}:

🌟 LIFE PURPOSE (DHARMA) 🌟
The Sun in {sun_pos.sign} and {sun_pos.nakshatra} nakshatra indicates you are destined to illuminate the path of {sun_pos.sign.lower()} qualities. Your soul's journey is to master the art of leadership and self-expression through creative endeavors.

💼 CAREER & PROSPERITY 💼
With celestial alignments favoring growth, your professional path will see significant transformation. Jupiter's grace suggests opportunities in teaching, counseling, or spiritual guidance. Financial abundance flows when you align with your authentic purpose.

❤️ RELATIONSHIPS & FAMILY ❤️
The Moon in {moon_pos.sign} blesses you with emotional depth and nurturing abilities. Your relationships thrive through compassion and understanding. A significant partnership will emerge that supports your spiritual evolution.

🏥 HEALTH & LONGEVITY 🏥
The planetary configuration suggests robust vitality when you maintain balance. Pay attention to {moon_pos.sign.lower()}-related areas of the body. Meditation and yogic practices will enhance your well-being significantly.

🕉️ SPIRITUAL PATH 🕉️
Your {moon_pos.nakshatra} nakshatra reveals a deep connection to ancient wisdom. Your spiritual awakening accelerates through devotional practices and service to humanity. The divine guides you toward self-realization.

May the cosmic forces bless your journey with wisdom, prosperity, and divine grace.
"""
    
    return prediction

@app.get("/")
def read_root():
    return {
        "message": "Nadi Astrology API",
        "version": "1.0",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=NadiPrediction)
async def generate_prediction(details: BirthDetails):
    try:
        # Parse birth date and time
        birth_date = datetime.strptime(details.date, "%Y-%m-%d")
        time_parts = details.time.split(":")
        birth_dt = birth_date.replace(
            hour=int(time_parts[0]),
            minute=int(time_parts[1])
        )
        
        # Calculate planetary positions
        positions = calculate_planetary_positions(birth_dt)
        
        # Extract key positions
        sun_sign = next(p.sign for p in positions if p.planet == "Sun")
        moon_sign = next(p.sign for p in positions if p.planet == "Moon")
        
        # Generate prediction
        prediction_text = generate_nadi_prediction(positions, details.name)
        
        return NadiPrediction(
            birth_details=details,
            planetary_positions=positions,
            ascendant=sun_sign,  # Simplified - actual ascendant calculation is more complex
            moon_sign=moon_sign,
            prediction=prediction_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
