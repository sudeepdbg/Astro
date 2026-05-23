import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAM SHALAKA DATA ---
SHALAKA_CHOUPAIS = [
    {"text": "Sunu siya satya aseesa hamari, pujahi mana kamana tumhari", "meaning": "Success is certain, your wish will be fulfilled.", "type": "Positive"},
    {"text": "Prabishi nagara keeje saba kaaja, hridaya rakhi koushalapur raaja", "meaning": "Start your work, success will follow.", "type": "Positive"},
    {"text": "Hoeehai soee jo rama rachi raakhaa, ko kari taraka badhaavai saakhaa", "meaning": "Whatever God has planned will happen, do not worry.", "type": "Neutral"},
    {"text": "Garala sudha ripu karahi mitaee, gopada sindhu anala sitalaee", "meaning": "Even enemies will become friends; difficult tasks will become easy.", "type": "Very Positive"}
]

class BirthDetails(BaseModel):
    name: str
    date: str
    time: str
    location: str
    language: str = "English"

# --- AI CHAT FLOW (OpenRouter) ---
@app.post("/chat")
async def chat_with_ai(prompt: str, context: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"response": "AI is currently in meditation (API Key missing)."}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
                "messages": [
                    {"role": "system", "content": f"You are a Vedic Astrologer. Use this birth data: {context}"},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        return response.json()['choices'][0]['message']['content']

# --- CHILD & CAREER PREDICTION LOGIC ---
def get_advanced_predictions(moon_sign, sun_sign):
    # Simplified Astrological probability logic
    career = "Dominant success in leadership/tech" if sun_sign in ["Leo", "Aries"] else "Success in creative/service fields"
    # Child prediction based on odd/even sign logic (simplified)
    child_prob = "High probability of a Boy" if moon_sign in ["Aries", "Gemini", "Leo"] else "High probability of a Girl"
    return career, child_prob

@app.post("/predict")
async def predict(details: BirthDetails):
    # ... (Keep your existing planetary calculation logic here) ...
    # Mock data for demonstration of the flow
    m_sign = "Leo" 
    s_sign = "Aries"
    career, child = get_advanced_predictions(m_sign, s_sign)
    
    return {
        "prediction": f"General: Success ahead. \nCareer: {career}. \nChild: {child}.",
        "moon_sign": m_sign,
        "career": career,
        "child_prediction": child
    }

@app.get("/shalaka")
async def get_shalaka_answer():
    import random
    return random.choice(SHALAKA_CHOUPAIS)
