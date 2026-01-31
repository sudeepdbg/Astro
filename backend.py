from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import math
import os
from openai import OpenAI

app = FastAPI(title="Nadi Astrology AI API")

# 1. FIXED CORS - Handshake with your GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sudeepdbg.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Data Models
class BirthDetails(BaseModel):
    name: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    location: str
    language: str = "English"
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PlanetaryPosition(BaseModel):
    planet: str
    longitude: float
    sign: str

class NadiPrediction(BaseModel):
    birth_details: BirthDetails
    ascendant: str
    moon_sign: str
    prediction: str
    timestamp: str

# 3. AI Prediction Logic
# Ensure you have set OPENAI_API_KEY in Railway Variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_ai_nadi_prediction(positions, name, language):
    pos_summary = ", ".join([f"{p['planet']} in {p['sign']}" for p in positions])
    
    # Determine the prompt language based on user selection
    if language.lower() == "hindi":
        instruction = "Provide the detailed Nadi prediction in Hindi using Devnagari script."
    else:
        instruction = "Provide the detailed Nadi prediction in English."

    system_msg = "You are an expert Nadi Astrologer trained in Bhrigu Nandi Nadi (BNN) principles."
    user_msg = f"""
    Analyze these planetary positions for {name}: {pos_summary}.
    
    {instruction}
    
    Please include these specific sections:
    1. व्यक्तित्व और स्वभाव (Personality & Nature)
    2. करियर, शिक्षा और वित्त (Career, Education & Finance)
    3. स्वास्थ्य और महत्वपूर्ण उपाय (Health & Important Remedies/Upay)
    
    Use a professional and guiding tone.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"The cosmic signals are weak. (AI Error: {str(e)})"

# 4. Calculation Logic (Simplified for Prototype)
def calculate_positions(dt):
    # This generates a mock set of positions. 
    # In a full-scale app, replace with swiss-ephemeris library calls.
    return [
        {"planet": "Sun", "longitude": 120.5, "sign": "Leo"},
        {"planet": "Moon", "longitude": 45.2, "sign": "Taurus"},
        {"planet": "Jupiter", "longitude": 10.0, "sign": "Aries"},
        {"planet": "Saturn", "longitude": 315.0, "sign": "Aquarius"},
        {"planet": "Venus", "longitude": 60.0, "sign": "Gemini"}
    ]

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=NadiPrediction)
async def predict(details: BirthDetails):
    try:
        # Convert incoming data to datetime object
        birth_dt = datetime.strptime(f"{details.date} {details.time}", "%Y-%m-%d %H:%M")
        
        # 1. Calculate positions
        raw_positions = calculate_positions(birth_dt)
        
        # 2. Convert to PlanetaryPosition objects for the response
        planet_objs = [PlanetaryPosition(**p) for p in raw_positions]
        
        # 3. Get AI Prediction
        prediction_text = generate_ai_nadi_prediction(raw_positions, details.name, details.language)
        
        return NadiPrediction(
            birth_details=details,
            ascendant="Leo", # Mock data
            moon_sign="Taurus", # Mock data
            prediction=prediction_text,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        # This is where your previous error happened. 
        # Ensure there is NO text after the last parenthesis.
        raise HTTPException(status_code=400, detail=str(e))
