import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch




load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file")


genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


model_name = "tiiuae/falcon-rw-1b"
falcon_tokenizer = AutoTokenizer.from_pretrained(model_name)
falcon_model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
falcon_model.to(device)


# FastAPI app

app = FastAPI(title="GlobeTrotter API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
    ],
    allow_credentials=True,
    allow_methods=["*"],       
    allow_headers=["*"],      
)


# Request Schema

class TravelRequest(BaseModel):
    destination: str
    duration: int
    budget: str
    travel_style: str
    interests: list[str]
    accommodation: str
    transportation: str
    special_requests: str = "None"



# Prompt Template

TRAVEL_PROMPT_TEMPLATE = """
You are an expert travel planner. Create a comprehensive travel plan based on the following details:
DESTINATION: {destination}
DURATION: {duration} days
BUDGET: {budget}
TRAVEL_STYLE: {travel_style}
INTERESTS: {interests}
ACCOMMODATION_PREFERENCE: {accommodation}
TRANSPORTATION_PREFERENCE: {transportation}
SPECIAL_REQUESTS: {special_requests}

IMPORTANT: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON.
Do not use markdown formatting. The response must be parseable JSON.

Create a detailed travel plan in this exact JSON format:
{{
    "itinerary": [
        {{
            "day": "Day 1",
            "morning": "Activity description",
            "afternoon": "Activity description",
            "evening": "Activity description",
            "accommodation": "Hotel/Accommodation details",
            "meals": "Meal suggestions",
            "estimated_cost": "Cost estimate"
        }}
    ],
    "total_estimated_cost": "Total cost estimate",
    "travel_tips": ["Tip 1", "Tip 2", "Tip 3"],
    "packing_list": ["Item 1", "Item 2", "Item 3"],
    "emergency_contacts": {{
        "local_emergency": "Emergency number",
        "embassy": "Embassy contact if applicable",
        "hotel": "Hotel contact"
    }}
}}
"""



# Falcon Chat Function

def falcon_chat(prompt: str):
    inputs = falcon_tokenizer(prompt, return_tensors="pt").to(device)
    output = falcon_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=falcon_tokenizer.eos_token_id
    )
    response = falcon_tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()



# Summarize Travel Plan

def summarize_plan(plan):
    if isinstance(plan, dict):
        itinerary = plan.get("itinerary", [])
        main_places = set()
        for day in itinerary:
            for part in ["morning", "afternoon", "evening"]:
                val = day.get(part)
                if val:
                    place = val.split('.')[0].split(',')[0].strip()
                    if place:
                        main_places.add(place)
                if len(main_places) >= 5:
                    break
            if len(main_places) >= 5:
                break
        main_places_str = ', '.join(list(main_places))
        budget = plan.get("total_estimated_cost", "Not specified")
        tips = plan.get("travel_tips", [])
        highlight = tips[0] if tips else "Enjoy your trip!"
        summary_input = (
            f"Estimated budget: {budget}\n"
            f"Main places to visit: {main_places_str}\n"
            f"Highlight: {highlight}\n"
        )
    else:
        summary_input = str(plan)[:500]

    prompt = (
        "Write a concise summary for a travel plan. "
        "Include the rough total budget, the main places to visit, and one highlight or tip.\n\n"
        + summary_input
    )
    return falcon_chat(prompt)



# Combined Endpoint: Plan + Summary

@app.post("/generate_plan_with_summary")
def generate_plan_with_summary(travel_data: TravelRequest):
    """Generate a travel plan and its summary in one API call"""
    formatted_prompt = TRAVEL_PROMPT_TEMPLATE.format(
        destination=travel_data.destination,
        duration=travel_data.duration,
        budget=travel_data.budget,
        travel_style=travel_data.travel_style,
        interests=", ".join(travel_data.interests) if travel_data.interests else "General sightseeing",
        accommodation=travel_data.accommodation,
        transportation=travel_data.transportation,
        special_requests=travel_data.special_requests
    )

    try:
        # Generate plan from Gemini
        response = gemini_model.generate_content(formatted_prompt)
        response_text = response.text.strip()

                
        if response_text.startswith("```json"):
            response_text = response_text[7:]  
        elif response_text.startswith("```"):
            response_text = response_text[3:]  
        if response_text.endswith("```"):
            response_text = response_text[:-3]  
        response_text = response_text.strip()


        # Parse JSON safely
        try:
            travel_plan = json.loads(response_text)
        except json.JSONDecodeError:
            fixed_text = response_text.replace(",}", "}").replace(",]", "]")
            travel_plan = json.loads(fixed_text)

        # Summarize using Falcon
        summary = summarize_plan(travel_plan)

        return {
            "plan": travel_plan,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "GlobeTrotter API running"}
