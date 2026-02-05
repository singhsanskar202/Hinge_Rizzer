import cv2
import os
import datetime
import gspread
import json
import base64
import logging
import re

from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
from oauth2client.service_account import ServiceAccountCredentials

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HingeWingman")

app = FastAPI()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")

SPREADSHEET_NAME = "Hinge_Rizz_Tracker"
WORKSHEET_NAME = "Sheet1"

# Switching this name here is the easiest way to fix the 429 errors
MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --------------------------------------------------
# GOOGLE SHEETS
# --------------------------------------------------
def get_sheet():
    if not GOOGLE_CREDS_JSON:
        raise ValueError("GOOGLE_CREDS_JSON missing")

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gc = gspread.authorize(creds)
    sh = gc.open(SPREADSHEET_NAME)
    return sh.worksheet(WORKSHEET_NAME)

def log_to_sheet(data):
    try:
        sheet = get_sheet()
        meta = data.get("metadata", {})
        replies = data.get("replies", []) + [""] * 3

        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            meta.get("name"),
            meta.get("age"),
            meta.get("job"),
            meta.get("education"),
            meta.get("location"),
            ", ".join(meta.get("interests", [])) if isinstance(meta.get("interests"), list) else "",
            ", ".join(meta.get("personality_traits", [])) if isinstance(meta.get("personality_traits"), list) else "",
            meta.get("communication_style"),
            meta.get("vibe"),
            replies[0],
            replies[1],
            replies[2],
            "PENDING"
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        logger.info("✅ Logged to Google Sheet")
    except Exception as e:
        logger.error(f"❌ Google Sheet write failed: {e}")

# --------------------------------------------------
# VIDEO → FRAMES
# --------------------------------------------------
def extract_frames_base64(video_path, max_frames=6):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return []
    interval = max(1, total // max_frames)
    frames = []
    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = base64.b64encode(buf).decode()
            frames.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return frames

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def health(): return {"status": "Hinge Wingman Running"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    if not OPENROUTER_API_KEY: return {"error": "Missing API key"}

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f: f.write(await file.read())
    frames = extract_frames_base64(temp_path)
    if os.path.exists(temp_path): os.remove(temp_path)

    if not frames: return {"error": "Could not extract frames"}

    user_prompt = """
    Task: Analyze these Hinge profile screenshots. 

STEP 1: METADATA EXTRACTION
Deep-scan the images for:
- Identity: Name, Age, Job, Location.
- Visual Hooks: What is happening in the background? (e.g., "Hiking in Himachal", "Cafe in Bandra", "Holding a Golden Retriever").
- Prompt Hooks: What specific words or opinions did she share?
- Vibe Assessment: Is she "Main Character Energy," "Soft Girl Aesthetic," "Corporate Girly," or "Outdoor Adventurer"?

STEP 2: RIZZ GENERATION
Generate 3 distinct, high-conversion opening lines:
1. THE TEASE: A lighthearted, respectful jab at a detail in her profile (e.g., her music taste or a funny photo).
2. THE OBSERVER: A comment on a background detail that most people would miss, showing you actually looked.
3. THE VIBE-CHECK: A confident, low-pressure statement about her energy that invites her to "prove" it or laugh.

REQUIRED JSON FORMAT:
{
  "metadata": {
    "name": "string",
    "age": number,
    "job": "string",
    "education": "string",
    "location": "string",
    "interests": ["list", "of", "keywords"],
    "personality_traits": ["observant", "traits"],
    "communication_style": "description",
    "vibe": "one-word description"
  },
  "replies": [
    "Teasing reply here",
    "Observant reply here",
    "Vibe-check reply here"
  ]
}
    """

    content = [{"type": "text", "text": user_prompt}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": f}})

    try:
        # --- APPLIED DETERMINISTIC PARAMS ---
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": """

You are an elite, high-EQ dating wingman and social strategist. Your goal is to help men start meaningful, high-response conversations with women on Hinge.

CORE STRATEGY:
- Observational Intelligence: Look for "hooks" in photos (background details, brands, activities) and prompts.
- Subtle Teasing (The "Push-Pull"): Be playful and confident, but never arrogant or mean.
- Emotional Attunement: Identify the "vibe" (e.g., adventurous, cozy, ambitious, quirky) and match it.
- Low-Effort Replies: End with a statement or a very easy-to-answer "curiosity" question. 

TONE & STYLE:
- Conversational & Human: Avoid sounding like a marketing bot. Use "Hinglish" (mix of Hindi/English) naturally if the profile signals Indian cultural context.
- No Emojis: Use them only if they add genuine flavor (max 1 per reply).
- No Generic Compliments: Never say "You're pretty" or "Nice smile." Focus on what she IS DOING or SAYING.
- Short & Punchy: 1-2 sentences maximum.

STRICT OUTPUT RULE:
- You are a JSON-only engine. 
- Do not explain yourself. Do not use markdown.
- If data is missing from the image, return null for that field"""},
                {"role": "user", "content": content}
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=500  # High enough to finish the JSON structure
        )

        raw = completion.choices[0].message.content
        logger.info(f"RAW OUTPUT: {raw}")

        # --- REFINED JSON CLEANING ---
        clean = re.sub(r"```json|```", "", raw).strip()
        start_idx = clean.find("{")
        end_idx = clean.rfind("}") + 1
        
        if start_idx != -1 and end_idx != -1:
            data = json.loads(clean[start_idx:end_idx])
            
            # Handle cases where the model puts replies at top level
            if "replies" not in data and isinstance(data, dict):
                 # Fallback: if model returns only replies or renamed key
                 data["replies"] = data.get("witty_replies", [])

            log_to_sheet(data)
            return data
        else:
            raise ValueError("Invalid JSON format from AI")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {"error": str(e)}
