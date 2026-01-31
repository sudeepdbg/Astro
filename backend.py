# -*- coding: utf-8 -*-
"""
Nadi Astrology API with Claude AI Integration
Works on Railway without needing Ollama
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List, Dict
import math
import logging
import random
from enum import Enum
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nadi Astrology API with AI",
    description="Nadi Astrology with AI-Powered Predictions (Claude API)",
    version="5.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONSTANTS ====================
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
    "Sun": "सूर्य", "Moon": "चंद्र", "Mars": "मंगल", "Mercury": "बुध",
    "Jupiter": "गुरु", "Venus": "शुक्र", "Saturn": "शनि"
}

# ==================== MODELS ====================
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

class ChatMessage(BaseModel):
    message: str
    language: Language = Language.ENGLISH
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    timestamp: str

# ==================== ASTROLOGY CALCULATOR ====================
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

# ==================== PREDICTION GENERATOR ====================
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

# ==================== ENDPOINTS ====================
@app.get("/")
def read_root():
    return {
        "message": "Nadi Astrology API with AI",
        "version": "5.0",
        "status": "operational",
        "features": ["Predictions", "AI Chat (Fallback Mode)"],
        "endpoints": ["/predict", "/chat", "/health"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "version": "5.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=NadiPrediction)
async def generate_prediction(details: BirthDetails):
    """Generate Nadi prediction"""
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

# ==================== CHAT ENDPOINT ====================
# In-memory storage for chat sessions
chat_sessions = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_astrobot(message: ChatMessage):
    """Chat endpoint with intelligent fallback"""
    try:
        logger.info(f"Chat request: {message.message[:50]}...")
        
        # Generate session ID
        session_id = message.session_id or f"session_{datetime.now().timestamp()}"
        
        # Analyze question and provide helpful response
        question_lower = message.message.lower()
        response_text = ""
        
        # Career questions
        if any(word in question_lower for word in ['career', 'job', 'work', 'business', 'profession', 'करियर', 'नौकरी', 'व्यवसाय']):
            if message.language == Language.HINDI:
                response_text = "💼 करियर के बारे में विस्तृत जानकारी के लिए:\n\n1. ऊपर अपना पूर्ण जन्म विवरण दर्ज करें\n2. 'भविष्यवाणी प्राप्त करें' बटन क्लिक करें\n3. मैं आपके 10वें भाव, सूर्य और बुध की स्थिति के आधार पर विस्तृत करियर मार्गदर्शन प्रदान करूंगा"
            else:
                response_text = "💼 For detailed career insights:\n\n1. Enter your complete birth details above\n2. Click 'Get Prediction' button\n3. I'll analyze your 10th house, Sun, and Mercury positions to provide comprehensive career guidance"
        
        # Relationship questions
        elif any(word in question_lower for word in ['love', 'marriage', 'relationship', 'partner', 'spouse', 'शादी', 'प्रेम', 'विवाह']):
            if message.language == Language.HINDI:
                response_text = "❤️ संबंधों के बारे में जानने के लिए:\n\n1. अपना जन्म विवरण ऊपर दर्ज करें\n2. मैं आपके 7वें भाव, शुक्र और चंद्रमा की स्थिति का विश्लेषण करूंगा\n3. आपको संबंधों, विवाह और साझेदारी के बारे में विस्तृत जानकारी मिलेगी"
            else:
                response_text = "❤️ To understand your relationships:\n\n1. Enter your birth details above\n2. I'll analyze your 7th house, Venus, and Moon positions\n3. You'll get detailed insights about love, marriage, and partnerships"
        
        # Health questions
        elif any(word in question_lower for word in ['health', 'illness', 'disease', 'fitness', 'स्वास्थ्य', 'बीमारी']):
            if message.language == Language.HINDI:
                response_text = "🏥 स्वास्थ्य के बारे में:\n\nअपना जन्म विवरण दें और मैं आपके:\n• 6वें भाव (रोग)\n• चंद्रमा (मन और शरीर)\n• सूर्य (जीवन शक्ति)\n\nकी स्थिति के आधार पर स्वास्थ्य मार्गदर्शन दूंगा।"
            else:
                response_text = "🏥 For health insights:\n\nProvide your birth details and I'll analyze:\n• 6th house (diseases)\n• Moon (mind and body)\n• Sun (vitality)\n\nto give you health guidance."
        
        # Money/finance questions
        elif any(word in question_lower for word in ['money', 'wealth', 'finance', 'income', 'salary', 'धन', 'पैसा', 'आय']):
            if message.language == Language.HINDI:
                response_text = "💰 धन और वित्त के बारे में:\n\nमैं आपके 2रे भाव (धन), 11वें भाव (लाभ) और बृहस्पति (समृद्धि) की स्थिति देखकर वित्तीय मार्गदर्शन दूंगा। कृपया ऊपर जन्म विवरण दर्ज करें।"
            else:
                response_text = "💰 For financial guidance:\n\nI'll analyze your 2nd house (wealth), 11th house (gains), and Jupiter (prosperity) positions. Please enter your birth details above."
        
        # Education questions
        elif any(word in question_lower for word in ['study', 'education', 'exam', 'degree', 'college', 'शिक्षा', 'पढ़ाई']):
            if message.language == Language.HINDI:
                response_text = "📚 शिक्षा के बारे में:\n\nमैं आपके 4थे भाव (शिक्षा), 5वें भाव (बुद्धि) और बुध (ज्ञान) की स्थिति देखकर शैक्षिक मार्गदर्शन दूंगा।"
            else:
                response_text = "📚 For education insights:\n\nI'll analyze your 4th house (education), 5th house (intelligence), and Mercury (knowledge) to provide educational guidance."
        
        # General greeting
        elif any(word in question_lower for word in ['hi', 'hello', 'hey', 'namaste', 'नमस्ते', 'हेलो']):
            if message.language == Language.HINDI:
                response_text = "🙏 नमस्ते! मैं आपका नाड़ी ज्योतिष सहायक हूं।\n\nमैं आपकी मदद कर सकता हूं:\n• करियर मार्गदर्शन\n• संबंध और विवाह\n• स्वास्थ्य सलाह\n• वित्तीय मार्गदर्शन\n• शिक्षा मार्गदर्शन\n\nविस्तृत भविष्यवाणी के लिए कृपया ऊपर अपना जन्म विवरण दर्ज करें।"
            else:
                response_text = "🙏 Hello! I'm your Nadi Astrology assistant.\n\nI can help you with:\n• Career guidance\n• Relationships & marriage\n• Health advice\n• Financial guidance\n• Education guidance\n\nFor detailed predictions, please enter your birth details above."
        
        # Default response
        else:
            if message.language == Language.HINDI:
                response_text = "🔮 मैं नाड़ी ज्योतिष विशेषज्ञ हूं।\n\nविस्तृत और सटीक भविष्यवाणी के लिए:\n1. ऊपर अपना नाम, जन्म तिथि, समय और स्थान दर्ज करें\n2. अपनी पसंदीदा भाषा चुनें\n3. 'भविष्यवाणी प्राप्त करें' क्लिक करें\n\nमुझसे करियर, विवाह, स्वास्थ्य, धन या शिक्षा के बारे में भी पूछ सकते हैं।"
            else:
                response_text = "🔮 I'm a Nadi astrology expert.\n\nFor detailed and accurate predictions:\n1. Enter your name, birth date, time, and place above\n2. Select your preferred language\n3. Click 'Get Prediction'\n\nYou can also ask me about career, marriage, health, wealth, or education."
        
        # Store in session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].append({
            "user": message.message,
            "assistant": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 messages
        if len(chat_sessions[session_id]) > 10:
            chat_sessions[session_id] = chat_sessions[session_id][-10:]
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        error_msg = "I apologize, but I'm having trouble responding right now. Please try again."
        if message.language == Language.HINDI:
            error_msg = "माफ कीजिए, मुझे उत्तर देने में समस्या हो रही है। कृपया पुनः प्रयास करें।"
        
        return ChatResponse(
            response=error_msg,
            session_id=message.session_id,
            timestamp=datetime.now().isoformat()
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Nadi Astrology API v5.0")
    uvicorn.run(app, host="0.0.0.0", port=port)
