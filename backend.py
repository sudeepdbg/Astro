# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import math
import logging
import random
from enum import Enum
import os
import json
import httpx
import asyncio
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nadi Astrology API Pro + AI Chat",
    description="Enhanced Nadi Astrology with AI-Powered Predictions & Chat Assistant",
    version="4.0"
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
    "Jupiter": "गुरु", "Venus": "शुक्र", "Saturn": "शनि",
    "Rahu": "राहु", "Ketu": "केतु"
}

SIGN_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

NAKSHATRA_LORDS = [
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu",
    "Jupiter", "Saturn", "Mercury", "Ketu", "Venus", "Sun",
    "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury",
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu",
    "Jupiter", "Saturn", "Mercury"
]

# ==================== OLLAMA CONFIGURATION ====================
class OllamaConfig:
    """Configuration for Ollama LLM integration"""
    # Ollama server URL - defaults to localhost
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Model to use (small models for faster responses)
    # Available options: "mistral:7b", "llama3.2:3b", "gemma:7b", "phi:2.7b"
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral:7b")
    
    # Enable/disable LLM features
    ENABLE_LLM = os.getenv("ENABLE_LLM", "false").lower() == "true"
    
    # System prompt for astrology chatbot
    SYSTEM_PROMPT = """You are AstroBot, an expert Vedic astrology assistant with deep knowledge of Nadi Jyotish. 
    You analyze birth charts, provide personalized predictions, and answer astrology-related questions.
    
    Key knowledge areas:
    1. Planetary positions and their effects
    2. House significations (1st house: self, 10th: career, 5th: children, etc.)
    3. Nakshatras and their lords
    4. Yogas (planetary combinations)
    5. Dasha periods and transits
    
    Always respond in a compassionate, professional manner. If uncertain, say so rather than guessing.
    For predictions, focus on guidance and possibilities rather than absolute statements."""
    
    # Prompt templates for different tasks
    PROMPT_TEMPLATES = {
        "generic_prediction": """Based on this birth chart data, provide a personalized prediction:

Name: {name}
Birth Details: {date} at {time} in {location}
Ascendant (Lagna): {ascendant}
Moon Sign: {moon_sign} in {moon_nakshatra}
Sun Sign: {sun_sign}
Key Yogas: {yogas}
Current Dasha: {dasha}

Chart Highlights:
{chart_summary}

Please provide insights about:
1. Personality strengths and challenges
2. Career and financial prospects
3. Relationships and family life
4. Health considerations
5. Spiritual growth opportunities

Respond in {language} language.""",
        
        "chat_response": """Previous conversation context: {context}

Current user question: {question}

Astrological context for this user:
{astrology_context}

Provide a helpful, accurate response based on Vedic astrology principles."""
    }

# ==================== OLLAMA SERVICE ====================
class OllamaService:
    """Service for interacting with Ollama LLM"""
    
    @staticmethod
    async def generate_response(prompt: str, system_prompt: str = None, model: str = None) -> str:
        """Generate text using Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": model or OllamaConfig.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 800
                    }
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                
                response = await client.post(
                    f"{OllamaConfig.OLLAMA_BASE_URL}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "I apologize, but I couldn't generate a response at this time.")
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    @staticmethod
    async def check_ollama_available() -> bool:
        """Check if Ollama server is running"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OllamaConfig.OLLAMA_BASE_URL}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    @staticmethod
    async def get_available_models() -> List[str]:
        """Get list of available Ollama models"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{OllamaConfig.OLLAMA_BASE_URL}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [m.get("name") for m in models]
        except:
            return []
        return []
    
    @staticmethod
    async def generate_astrology_prediction(
        chart_data: Dict[str, Any], 
        language: str
    ) -> str:
        """Generate AI-powered astrology prediction"""
        
        # Prepare the prompt
        prompt_template = OllamaConfig.PROMPT_TEMPLATES["generic_prediction"]
        prompt = prompt_template.format(
            name=chart_data.get("name", "User"),
            date=chart_data.get("date", "Unknown"),
            time=chart_data.get("time", "Unknown"),
            location=chart_data.get("location", "Unknown"),
            ascendant=chart_data.get("ascendant", "Unknown"),
            moon_sign=chart_data.get("moon_sign", "Unknown"),
            moon_nakshatra=chart_data.get("moon_nakshatra", "Unknown"),
            sun_sign=chart_data.get("sun_sign", "Unknown"),
            yogas=", ".join(chart_data.get("yogas", [])),
            dasha=chart_data.get("dasha", "Unknown"),
            chart_summary=chart_data.get("chart_summary", "No chart summary available"),
            language=language
        )
        
        # Generate response
        response = await OllamaService.generate_response(
            prompt, 
            system_prompt=OllamaConfig.SYSTEM_PROMPT
        )
        
        if not response:
            # Fallback to template-based response
            if language == "Hindi":
                response = """मैं वर्तमान में AI सहायता प्रदान नहीं कर सकता। कृपया ऊपर दिए गए मानक भविष्यवाणी का उपयोग करें।"""
            else:
                response = """I'm unable to provide AI assistance at the moment. Please use the standard prediction above."""
        
        return response
    
    @staticmethod
    async def chat_response(
        question: str,
        context: Optional[str] = None,
        astrology_context: Optional[Dict[str, Any]] = None,
        language: str = "English"
    ) -> str:
        """Generate chat response for astrology questions"""
        
        # Prepare astrology context
        astro_context_str = "No astrological context available."
        if astrology_context:
            astro_context_str = json.dumps(astrology_context, indent=2)
        
        prompt_template = OllamaConfig.PROMPT_TEMPLATES["chat_response"]
        prompt = prompt_template.format(
            context=context or "No previous conversation.",
            question=question,
            astrology_context=astro_context_str,
            language=language
        )
        
        # Generate response
        response = await OllamaService.generate_response(
            prompt,
            system_prompt=OllamaConfig.SYSTEM_PROMPT
        )
        
        if not response:
            # Fallback responses
            if "career" in question.lower() or "job" in question.lower():
                if language == "Hindi":
                    response = "करियर के संबंध में, 10वें भाव और सूर्य की स्थिति महत्वपूर्ण है। विस्तृत विश्लेषण के लिए कृपया अपनी जन्म कुंडली साझा करें।"
                else:
                    response = "Regarding career, the 10th house and Sun's position are significant. For detailed analysis, please share your birth chart."
            elif "child" in question.lower() or "children" in question.lower():
                if language == "Hindi":
                    response = "संतान के विषय में पंचम भाव और चंद्रमा की स्थिति मुख्य हैं। अधिक जानकारी के लिए जन्म विवरण प्रदान करें।"
                else:
                    response = "For children matters, the 5th house and Moon's position are key. Provide birth details for more information."
            else:
                if language == "Hindi":
                    response = "मैं नाड़ी ज्योतिष विशेषज्ञ हूं। आपके प्रश्न का उत्तर देने के लिए मुझे अधिक संदर्भ की आवश्यकता है। कृपया अपना जन्म विवरण साझा करें।"
                else:
                    response = "I'm a Nadi astrology expert. I need more context to answer your question. Please share your birth details."
        
        return response

# ==================== DATA MODELS ====================
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
    prediction_type: str = Field(default="general", description="general, career, child")
    use_ai: bool = Field(default=False, description="Use AI for enhanced predictions")

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
    degree_in_sign: float
    nakshatra_pada: int
    is_retrograde: bool = False

class NadiPrediction(BaseModel):
    birth_details: BirthDetails
    planetary_positions: List[PlanetaryPosition]
    ascendant: str
    ascendant_hindi: str
    moon_sign: str
    moon_sign_hindi: str
    prediction: str
    ai_prediction: Optional[str] = None
    career_prediction: Optional[str] = None
    child_prediction: Optional[str] = None
    timestamp: str
    yogas: List[str] = []
    dasha_period: Optional[str] = None

class ChatMessage(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: str
    language: Language = Language.ENGLISH
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    timestamp: str
    is_astrology_related: bool = True

# ==================== ASTROLOGY CALCULATORS ====================
class EnhancedAstrologyCalculator:
    """More accurate astrology calculations with house system"""
    
    PLANETARY_DATA = {
        "Sun": {"mean_motion": 0.9856076686, "epoch_long": 280.46646},
        "Moon": {"mean_motion": 13.176396, "epoch_long": 218.31617},
        "Mars": {"mean_motion": 0.524032, "epoch_long": 355.433},
        "Mercury": {"mean_motion": 4.092334, "epoch_long": 234.96},
        "Jupiter": {"mean_motion": 0.083129, "epoch_long": 238.049},
        "Venus": {"mean_motion": 1.602136, "epoch_long": 342.768},
        "Saturn": {"mean_motion": 0.033496, "epoch_long": 345.324},
        "Rahu": {"mean_motion": -0.052953, "epoch_long": 95.9989},
        "Ketu": {"mean_motion": -0.052953, "epoch_long": 275.9989}
    }
    
    @staticmethod
    def calculate_julian_day(year: int, month: int, day: int, hour: int, minute: int) -> float:
        """More accurate Julian Day calculation"""
        if month <= 2:
            year -= 1
            month += 12
        
        a = year // 100
        b = 2 - a + (a // 4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += (hour + minute / 60.0) / 24.0
        return jd
    
    @staticmethod
    def calculate_planet_position(jd: float, planet: str) -> Tuple[float, bool]:
        """Calculate planetary position with retrograde simulation"""
        if planet not in EnhancedAstrologyCalculator.PLANETARY_DATA:
            return 0.0, False
        
        data = EnhancedAstrologyCalculator.PLANETARY_DATA[planet]
        n = jd - 2451545.0
        mean_long = (data["epoch_long"] + data["mean_motion"] * n) % 360
        
        perturbation = 0
        if planet == "Sun":
            g = math.radians((357.528 + 0.9856003 * n) % 360)
            perturbation = 1.915 * math.sin(g) + 0.020 * math.sin(2*g)
        elif planet == "Moon":
            D = math.radians((297.850 + 12.190749 * n) % 360)
            M = math.radians((357.528 + 0.9856003 * n) % 360)
            Mm = math.radians((134.963 + 13.064993 * n) % 360)
            perturbation = 6.289 * math.sin(Mm) + 1.274 * math.sin(2*D - Mm)
        
        true_long = (mean_long + perturbation) % 360
        
        is_retrograde = False
        if planet in ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]:
            retro_cycle = {
                "Mercury": 116, "Venus": 584, "Mars": 780, 
                "Jupiter": 399, "Saturn": 378
            }
            if planet in retro_cycle:
                cycle_day = n % retro_cycle[planet]
                is_retrograde = 20 < cycle_day < 50
        
        return true_long, is_retrograde
    
    @staticmethod
    def calculate_ascendant(jd: float, latitude: float, longitude: float) -> float:
        """Calculate ascendant (Lagna) more accurately"""
        t = (jd - 2451545.0) / 36525.0
        sidereal_time = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 
                        0.000387933 * t * t - t * t * t / 38710000.0) % 360
        
        lst = (sidereal_time + longitude) % 360
        
        epsilon = 23.4392911
        lst_rad = math.radians(lst)
        lat_rad = math.radians(latitude)
        epsilon_rad = math.radians(epsilon)
        
        asc_rad = math.atan2(
            math.sin(lst_rad),
            math.cos(lst_rad) * math.cos(epsilon_rad) + 
            math.tan(lat_rad) * math.sin(epsilon_rad)
        )
        
        ascendant = math.degrees(asc_rad) % 360
        return ascendant
    
    @staticmethod
    def calculate_houses(ascendant: float) -> List[float]:
        """Calculate house cusps (Equal House system)"""
        houses = []
        for i in range(12):
            house_cusp = (ascendant + i * 30) % 360
            houses.append(house_cusp)
        return houses
    
    @staticmethod
    def get_house_number(longitude: float, houses: List[float]) -> int:
        """Find which house a planet is in"""
        for i in range(12):
            start = houses[i]
            end = houses[(i + 1) % 12]
            if end < start:
                end += 360
            
            planet_long = longitude % 360
            if planet_long < start:
                planet_long += 360
            
            if start <= planet_long < end:
                return i + 1
        
        return 1

class PredictionGenerator:
    """Generate personalized predictions based on actual astrological factors"""
    
    SIGN_CHARACTERISTICS = {
        "Aries": {"element": "Fire", "quality": "Cardinal", "traits": ["courageous", "energetic", "impulsive"]},
        "Taurus": {"element": "Earth", "quality": "Fixed", "traits": ["reliable", "patient", "stubborn"]},
        "Gemini": {"element": "Air", "quality": "Mutable", "traits": ["communicative", "curious", "restless"]},
        "Cancer": {"element": "Water", "quality": "Cardinal", "traits": ["nurturing", "emotional", "protective"]},
        "Leo": {"element": "Fire", "quality": "Fixed", "traits": ["creative", "proud", "generous"]},
        "Virgo": {"element": "Earth", "quality": "Mutable", "traits": ["analytical", "practical", "critical"]},
        "Libra": {"element": "Air", "quality": "Cardinal", "traits": ["diplomatic", "harmonious", "indecisive"]},
        "Scorpio": {"element": "Water", "quality": "Fixed", "traits": ["intense", "passionate", "secretive"]},
        "Sagittarius": {"element": "Fire", "quality": "Mutable", "traits": ["optimistic", "adventurous", "blunt"]},
        "Capricorn": {"element": "Earth", "quality": "Cardinal", "traits": ["ambitious", "disciplined", "cautious"]},
        "Aquarius": {"element": "Air", "quality": "Fixed", "traits": ["innovative", "independent", "detached"]},
        "Pisces": {"element": "Water", "quality": "Mutable", "traits": ["compassionate", "intuitive", "dreamy"]}
    }
    
    CAREER_SUGGESTIONS = {
        "Sun": ["Leadership roles", "Government positions", "Management", "Entrepreneurship"],
        "Moon": ["Healthcare", "Psychology", "Hospitality", "Creative arts"],
        "Mars": ["Military", "Sports", "Engineering", "Police"],
        "Mercury": ["Writing", "Teaching", "IT", "Business"],
        "Jupiter": ["Education", "Law", "Finance", "Spiritual guidance"],
        "Venus": ["Arts", "Fashion", "Entertainment", "Diplomacy"],
        "Saturn": ["Research", "Science", "Construction", "Administration"]
    }
    
    @staticmethod
    def detect_yogas(positions: List[PlanetaryPosition]) -> List[str]:
        yogas = []
        planet_signs = {p.planet: p.sign for p in positions}
        
        if planet_signs.get("Sun") == "Leo" and planet_signs.get("Moon") == "Cancer":
            yogas.append("Raja Yoga")
        
        if planet_signs.get("Jupiter") in ["Cancer", "Sagittarius", "Pisces"]:
            yogas.append("Gaja Kesari Yoga")
        
        if planet_signs.get("Venus") == planet_signs.get("Jupiter"):
            yogas.append("Lakshmi Yoga")
        
        if planet_signs.get("Moon") in ["Cancer", "Taurus"] and planet_signs.get("Jupiter") in ["Cancer", "Taurus", "Sagittarius", "Pisces"]:
            yogas.append("Chandra-Mangala Yoga")
        
        return yogas
    
    @staticmethod
    def get_dasha_period(birth_dt: datetime, moon_nakshatra: str) -> str:
        nakshatra_index = NAKSHATRAS.index(moon_nakshatra) if moon_nakshatra in NAKSHATRAS else 0
        
        dasha_lords = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", 
                      "Jupiter", "Saturn", "Mercury"]
        
        starting_lord_index = nakshatra_index % 9
        years_since_birth = (datetime.now() - birth_dt).days / 365.25
        
        current_index = (starting_lord_index + int(years_since_birth / 6)) % 9
        return dasha_lords[current_index]
    
    @staticmethod
    def generate_general_prediction(name: str, positions: List[PlanetaryPosition], 
                                   language: Language) -> str:
        sun_pos = next(p for p in positions if p.planet == "Sun")
        moon_pos = next(p for p in positions if p.planet == "Moon")
        asc_pos = next(p for p in positions if p.house == 1)
        
        sun_char = PredictionGenerator.SIGN_CHARACTERISTICS.get(sun_pos.sign, {})
        moon_char = PredictionGenerator.SIGN_CHARACTERISTICS.get(moon_pos.sign, {})
        
        if language == Language.HINDI:
            return f"""प्रिय {name},

🌙 जीवन का उद्देश्य 🌙
आपका चंद्रमा {moon_pos.sign_hindi} राशि में {moon_pos.nakshatra_hindi} नक्षत्र के {moon_pos.nakshatra_pada} पाद में स्थित है। 
यह {', '.join(moon_char.get('traits', ['गहन']))} गुण प्रदर्शित करता है। 
आपकी आत्मा {moon_char.get('element', 'जल')} तत्व के माध्यम से {sun_char.get('quality', 'मौलिक')} ऊर्जा प्रकट करती है।

💫 व्यक्तित्व विश्लेषण 💫
लग्न {asc_pos.sign_hindi} और सूर्य {sun_pos.sign_hindi} के संयोग से आपमें प्राकृतिक नेतृत्व क्षमता है। 
चंद्रमा की स्थिति आपकी भावनात्मक बुद्धि को {moon_pos.degree_in_sign:.1f}° पर मजबूत करती है।

🌟 कुंडली की विशेषताएं 🌟
- चंद्र नक्षत्र: {moon_pos.nakshatra_hindi} (पाद {moon_pos.nakshatra_pada})
- सूर्य की डिग्री: {sun_pos.degree_in_sign:.1f}°
- ग्रहों की स्थिति: {len([p for p in positions if not p.is_retrograde])} सीधे, {len([p for p in positions if p.is_retrograde])} वक्री

🕉️ आध्यात्मिक मार्गदर्शन 🕉️
आपके {moon_pos.nakshatra_hindi} नक्षत्र का स्वामी {NAKSHATRA_LORDS[NAKSHATRAS.index(moon_pos.nakshatra)]} है, 
जो आपके आध्यात्मिक विकास को दर्शाता है।

धन्यवाद। ॐ शांति। 🙏"""
        else:
            return f"""Dear {name},

🌙 LIFE PURPOSE 🌙
Your Moon resides in {moon_pos.sign} sign within the {moon_pos.nakshatra} Nakshatra, pada {moon_pos.nakshatra_pada}. 
This reveals {', '.join(moon_char.get('traits', ['profound']))} qualities. 
Your soul expresses {sun_char.get('quality', 'cardinal')} energy through {moon_char.get('element', 'water')} element.

💫 PERSONALITY ANALYSIS 💫
With Ascendant {asc_pos.sign} and Sun in {sun_pos.sign}, you possess natural leadership qualities. 
Moon's position at {moon_pos.degree_in_sign:.1f}° strengthens your emotional intelligence.

🌟 CHART HIGHLIGHTS 🌟
- Moon Nakshatra: {moon_pos.nakshatra} (Pada {moon_pos.nakshatra_pada})
- Sun Degree: {sun_pos.degree_in_sign:.1f}°
- Planetary Status: {len([p for p in positions if not p.is_retrograde])} direct, {len([p for p in positions if p.is_retrograde])} retrograde

🕉️ SPIRITUAL GUIDANCE 🕉️
Your {moon_pos.nakshatra} is ruled by {NAKSHATRA_LORDS[NAKSHATRAS.index(moon_pos.nakshatra)]}, 
indicating your spiritual growth path.

Thank you. Om Shanti. 🙏"""
    
    @staticmethod
    def generate_career_prediction(positions: List[PlanetaryPosition], language: Language) -> str:
        tenth_house = [p for p in positions if p.house == 10]
        sun_pos = next(p for p in positions if p.planet == "Sun")
        jupiter_pos = next(p for p in positions if p.planet == "Jupiter")
        
        career_themes = []
        for planet in tenth_house:
            if planet.planet in PredictionGenerator.CAREER_SUGGESTIONS:
                career_themes.extend(PredictionGenerator.CAREER_SUGGESTIONS[planet.planet][:2])
        
        if not career_themes:
            career_themes = PredictionGenerator.CAREER_SUGGESTIONS.get(sun_pos.planet, ["Various professional fields"])
        
        if language == Language.HINDI:
            return f"""💼 कैरियर भविष्यवाणी 💼

प्रमुख क्षेत्र: {', '.join(career_themes[:3])}

सफलता के लिए सुझाव:
1. {sun_pos.sign_hindi} राशि में सूर्य: नेतृत्व भूमिकाएं अपनाएं
2. गुरु {jupiter_pos.sign_hindi} में: {jupiter_pos.house} भाव से संबंधित क्षेत्रों में विस्तार
3. 10वें भाव के ग्रह: {len(tenth_house)} ग्रह करियर में गतिशीलता दर्शाते हैं

शुभ समय: अगले 2-3 वर्षों में महत्वपूर्ण करियर बदलाव"""
        else:
            return f"""💼 CAREER PREDICTION 💼

Primary Fields: {', '.join(career_themes[:3])}

Success Suggestions:
1. Sun in {sun_pos.sign}: Embrace leadership roles
2. Jupiter in {jupiter_pos.sign}: Expand in areas related to {jupiter_pos.house}th house
3. 10th House Planets: {len(tenth_house)} planets indicate career dynamism

Auspicious Timing: Significant career shifts in next 2-3 years"""
    
    @staticmethod
    def generate_child_prediction(positions: List[PlanetaryPosition], language: Language) -> str:
        fifth_house = [p for p in positions if p.house == 5]
        moon_pos = next(p for p in positions if p.planet == "Moon")
        
        if language == Language.HINDI:
            return f"""👶 संतान भविष्यवाणी 👶

पंचम भाव विश्लेषण: {len(fifth_house)} ग्रह संतान क्षेत्र को प्रभावित कर रहे हैं

मुख्य संकेत:
1. चंद्रमा {moon_pos.sign_hindi} में: भावनात्मक संबंध मजबूत होंगे
2. {fifth_house[0].planet if fifth_house else 'चंद्रमा'} की स्थिति: संतान के स्वास्थ्य और विकास पर प्रभाव
3. शुभ समय: चंद्रमा की शुभ दशा में संतान सुख की प्राप्ति

ध्यान दें: यह सामान्य भविष्यवाणी है, व्यक्तिगत जन्म कुंडली परामर्श आवश्यक है"""
        else:
            return f"""👶 CHILD PREDICTION 👶

5th House Analysis: {len(fifth_house)} planets influencing children sector

Key Indicators:
1. Moon in {moon_pos.sign}: Strong emotional bonds with children
2. Position of {fifth_house[0].planet if fifth_house else 'Moon'}: Affects children's health and development
3. Auspicious Timing: Child blessings during favorable Moon periods

Note: This is general prediction, personal birth chart consultation recommended"""

# ==================== API ENDPOINTS ====================
@app.post("/predict", response_model=NadiPrediction)
async def generate_prediction(details: BirthDetails):
    try:
        logger.info(f"Generating {details.prediction_type} prediction for {details.name}")
        
        # Parse date and time
        dt = datetime.strptime(f"{details.date} {details.time}", "%Y-%m-%d %H:%M")
        
        # Calculate Julian Day
        jd = EnhancedAstrologyCalculator.calculate_julian_day(
            dt.year, dt.month, dt.day, dt.hour, dt.minute
        )
        
        # Use default coordinates if not provided
        latitude = details.latitude or 28.6139  # Default Delhi
        longitude = details.longitude or 77.2090
        
        # Calculate ascendant and houses
        ascendant = EnhancedAstrologyCalculator.calculate_ascendant(jd, latitude, longitude)
        houses = EnhancedAstrologyCalculator.calculate_houses(ascendant)
        
        # Calculate planetary positions
        positions = []
        planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
        
        for planet in planets:
            longitude, is_retrograde = EnhancedAstrologyCalculator.calculate_planet_position(jd, planet)
            
            sign_idx = int(longitude / 30) % 12
            degree_in_sign = longitude % 30
            nakshatra_idx = int(longitude / 13.333333) % 27
            nakshatra_pada = int((longitude % 13.333333) / 3.333333) + 1
            
            house_num = EnhancedAstrologyCalculator.get_house_number(longitude, houses)
            
            positions.append(PlanetaryPosition(
                planet=planet,
                planet_hindi=PLANETS_HINDI.get(planet, planet),
                longitude=round(longitude, 4),
                sign=ZODIAC_SIGNS[sign_idx],
                sign_hindi=ZODIAC_SIGNS_HINDI[sign_idx],
                house=house_num,
                nakshatra=NAKSHATRAS[nakshatra_idx],
                nakshatra_hindi=NAKSHATRAS_HINDI[nakshatra_idx],
                degree_in_sign=round(degree_in_sign, 2),
                nakshatra_pada=nakshatra_pada,
                is_retrograde=is_retrograde
            ))
        
        # Get ascendant sign
        asc_idx = int(ascendant / 30) % 12
        ascendant_sign = ZODIAC_SIGNS[asc_idx]
        ascendant_sign_hindi = ZODIAC_SIGNS_HINDI[asc_idx]
        
        # Get Moon data
        moon_data = next(p for p in positions if p.planet == "Moon")
        
        # Detect yogas
        yogas = PredictionGenerator.detect_yogas(positions)
        
        # Calculate dasha period
        dasha_period = PredictionGenerator.get_dasha_period(dt, moon_data.nakshatra)
        
        # Generate predictions based on type
        general_pred = PredictionGenerator.generate_general_prediction(
            details.name, positions, details.language
        )
        
        career_pred = None
        child_pred = None
        
        if details.prediction_type == "career":
            career_pred = PredictionGenerator.generate_career_prediction(positions, details.language)
        elif details.prediction_type == "child":
            child_pred = PredictionGenerator.generate_child_prediction(positions, details.language)
        else:
            career_pred = PredictionGenerator.generate_career_prediction(positions, details.language)
            child_pred = PredictionGenerator.generate_child_prediction(positions, details.language)
        
        # Combine predictions
        if details.language == Language.HINDI:
            full_prediction = f"""{general_pred}

{career_pred if career_pred else ''}

{child_pred if child_pred else ''}

🪐 विशेष योग: {', '.join(yogas) if yogas else 'कोई विशेष योग नहीं'}
📅 वर्तमान दशा: {dasha_period} दशा चल रही है"""
        else:
            full_prediction = f"""{general_pred}

{career_pred if career_pred else ''}

{child_pred if child_pred else ''}

🪐 SPECIAL YOGAS: {', '.join(yogas) if yogas else 'No special yogas'}
📅 CURRENT DASHA: Running {dasha_period} dasha period"""
        
        # Generate AI prediction if requested
        ai_prediction = None
        if details.use_ai and OllamaConfig.ENABLE_LLM:
            try:
                # Check if Ollama is available
                ollama_available = await OllamaService.check_ollama_available()
                
                if ollama_available:
                    # Prepare chart data for AI
                    chart_summary = []
                    for pos in positions[:7]:  # Limit to main planets
                        chart_summary.append(f"{pos.planet}: {pos.sign} in House {pos.house}")
                    
                    chart_data = {
                        "name": details.name,
                        "date": details.date,
                        "time": details.time,
                        "location": details.location,
                        "ascendant": ascendant_sign,
                        "moon_sign": moon_data.sign,
                        "moon_nakshatra": moon_data.nakshatra,
                        "sun_sign": next(p for p in positions if p.planet == "Sun").sign,
                        "yogas": yogas,
                        "dasha": dasha_period,
                        "chart_summary": "\n".join(chart_summary)
                    }
                    
                    ai_prediction = await OllamaService.generate_astrology_prediction(
                        chart_data, details.language.value
                    )
                else:
                    ai_prediction = "AI features are currently unavailable. Ollama server is not running."
            except Exception as e:
                logger.error(f"Error generating AI prediction: {e}")
                ai_prediction = "AI prediction temporarily unavailable due to technical issues."
        
        result = NadiPrediction(
            birth_details=details,
            planetary_positions=positions,
            ascendant=ascendant_sign,
            ascendant_hindi=ascendant_sign_hindi,
            moon_sign=moon_data.sign,
            moon_sign_hindi=moon_data.sign_hindi,
            prediction=full_prediction,
            ai_prediction=ai_prediction,
            career_prediction=career_pred,
            child_prediction=child_pred,
            timestamp=datetime.now().isoformat(),
            yogas=yogas,
            dasha_period=dasha_period
        )
        
        logger.info(f"Prediction generated successfully for {details.name}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# ==================== CHAT BOT ENDPOINTS ====================
# In-memory storage for chat sessions
chat_sessions = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_astrobot(message: ChatMessage):
    """Chat endpoint for astrology questions"""
    try:
        logger.info(f"Chat request from {message.user_id or 'anonymous'}")
        
        # Generate or retrieve session ID
        session_id = message.session_id or f"session_{datetime.now().timestamp()}"
        
        # Get session context if exists
        session_context = chat_sessions.get(session_id, {})
        
        # Prepare astrology context if provided
        astrology_context = message.context or session_context.get("astrology_data", {})
        
        # Check if Ollama is available
        ollama_available = await OllamaService.check_ollama_available() if OllamaConfig.ENABLE_LLM else False
        
        response_text = ""
        if ollama_available:
            # Generate response using Ollama
            response_text = await OllamaService.chat_response(
                question=message.message,
                context=session_context.get("history", ""),
                astrology_context=astrology_context,
                language=message.language.value
            )
        else:
            # Fallback response
            if message.language == Language.HINDI:
                response_text = "मैं वर्तमान में चैट सहायता प्रदान नहीं कर सकता। कृपया अपना जन्म विवरण दर्ज करें और विस्तृत भविष्यवाणी प्राप्त करें।"
            else:
                response_text = "I'm currently unable to provide chat assistance. Please enter your birth details to get detailed predictions."
        
        # Update session history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"history": "", "astrology_data": astrology_context}
        
        # Limit history length
        history = chat_sessions[session_id]["history"]
        new_history = f"{history}\nUser: {message.message}\nAssistant: {response_text}"
        if len(new_history) > 2000:
            new_history = new_history[-2000:]
        chat_sessions[session_id]["history"] = new_history
        
        # Check if astrology related
        astrology_keywords = ["birth", "chart", "horoscope", "planet", "sign", "rasi", 
                            "nakshatra", "house", "dasha", "yoga", "astrology", "jyotish"]
        is_astrology_related = any(keyword in message.message.lower() for keyword in astrology_keywords)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            is_astrology_related=is_astrology_related
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        error_msg = "I apologize, but I'm having trouble responding right now. Please try again later."
        if message.language == Language.HINDI:
            error_msg = "माफ कीजिए, मैं इस समय जवाब देने में असमर्थ हूं। कृपया कुछ देर बाद पुनः प्रयास करें।"
        
        return ChatResponse(
            response=error_msg,
            session_id=message.session_id,
            timestamp=datetime.now().isoformat(),
            is_astrology_related=False
        )

# ==================== OLLAMA STATUS ENDPOINTS ====================
@app.get("/ollama/status")
async def check_ollama_status():
    """Check if Ollama server is running and get available models"""
    try:
        ollama_available = await OllamaService.check_ollama_available()
        models = await OllamaService.get_available_models() if ollama_available else []
        
        return {
            "ollama_available": ollama_available,
            "models_available": models,
            "current_model": OllamaConfig.MODEL_NAME,
            "llm_enabled": OllamaConfig.ENABLE_LLM,
            "ollama_url": OllamaConfig.OLLAMA_BASE_URL
        }
    except Exception as e:
        return {
            "ollama_available": False,
            "error": str(e),
            "llm_enabled": OllamaConfig.ENABLE_LLM
        }

# ==================== ADDITIONAL ENDPOINTS ====================
@app.post("/predict/career", response_model=NadiPrediction)
async def generate_career_prediction(details: BirthDetails):
    details.prediction_type = "career"
    return await generate_prediction(details)

@app.post("/predict/child", response_model=NadiPrediction)
async def generate_child_prediction(details: BirthDetails):
    details.prediction_type = "child"
    return await generate_prediction(details)

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "version": "4.0",
        "features": ["General", "Career", "Child Predictions", "AI Chat", "Ollama Integration"],
        "llm_enabled": OllamaConfig.ENABLE_LLM,
        "timestamp": datetime.now().isoformat()
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Nadi Astrology API v4.0 with Ollama integration")
    logger.info(f"Ollama Enabled: {OllamaConfig.ENABLE_LLM}")
    if OllamaConfig.ENABLE_LLM:
        logger.info(f"Ollama URL: {OllamaConfig.OLLAMA_BASE_URL}")
        logger.info(f"Model: {OllamaConfig.MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=port)
