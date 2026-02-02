import cv2
import numpy as np
import os
import datetime
import gspread
import json
from PIL import Image
import io
import base64
from oauth2client.service_account import ServiceAccountCredentials
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import logging
import re

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HingeWingman")

app = FastAPI()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")
SHEET_NAME = "Hinge_Rizz_Tracker"

# Let's switch to the Llama Vision model. 
# It is generally better at following "JSON Only" instructions than Molmo.
MODEL_NAME = "allenai/molmo-2-8b:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- HELPER: LOG TO SHEET ---
def log_to_sheet(data_dict):
    if not GOOGLE_CREDS_JSON: return
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        g_client = gspread.authorize(creds)
        sheet = g_client.open(SHEET_NAME).sheet1
        
        meta = data_dict.get('metadata', {})
        replies = data_dict.get('replies', [])
        replies = replies + [""] * (3 - len(replies))
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp,
            meta.get('name', 'Unknown'),
            meta.get('age', 'Unknown'),
            meta.get('job', 'Unknown'),
            meta.get('location', 'Unknown'),
            str(meta.get('interests', [])),
            replies[0],
            replies[1],
            replies[2],
            "PENDING"
        ]
        sheet.append_row(row)
        logger.info(f"Logged to sheet: {meta.get('name')}")
    except Exception as e:
        # Just log error but don't crash app
        logger.error(f"Sheet logging failed: {str(e)}")

# --- HELPER: EXTRACT FRAMES ---
def extract_frames_base64(video_path, max_frames=7):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return []
    
    interval = max(1, total_frames // max_frames)
    frames_b64 = []
    current_frame = 0
    
    while current_frame < total_frames and len(frames_b64) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            # Resize to 512x512 for speed/reliability
            frame = cv2.resize(frame, (512, 512)) 
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(f"data:image/jpeg;base64,{b64_str}")
        current_frame += interval
    cap.release()
    return frames_b64

@app.get("/")
def home():
    return {"status": "Wingman Active"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(None)):
    if not file: return {"reply": "Error: No file received."}
    if not OPENROUTER_API_KEY: return {"reply": "Error: Missing API Key."}

    # 1. Process Video
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    frames = extract_frames_base64(temp_filename)
    if os.path.exists(temp_filename): os.remove(temp_filename)

    if not frames: return {"reply": "Error: Video was empty or unreadable."}

    # 2. Call AI
    content_payload = [{"type": "text", "text": "Analyze the dating profile video frames provided.

Step 1: Extract structured metadata about the profile owner using ONLY visible or clearly implied information.
Step 2: Generate exactly 3 Hinge “like comments” that feel natural and reply-worthy.

Metadata schema:
{
  "name": string | null,
  "age": number | null,
  "job": string | null,
  "location": string | null,
  "interests": array of strings,
  "vibe": string
}

Reply generation rules:
1. Each reply must be based on a DIFFERENT observation
2. At least one reply must spark curiosity
3. At least one reply must include light, respectful teasing
4. Avoid compliments that could apply to anyone
5. No emojis unless it feels extremely natural (max 1 per reply)
6. Replies must sound like something a real person would send

Output JSON format:
{
  "metadata": { ... },
  "replies": [
    "reply one",
    "reply two",
    "reply three"
  ]
}
"}]
    for b64_url in frames:
        content_payload.append({"type": "image_url", "image_url": {"url": b64_url}})

    logger.info(f"Sending to model: {MODEL_NAME}")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an emotionally intelligent dating assistant specialized in Hinge profiles.

CRITICAL RULES:
- Output ONLY valid JSON. No markdown. No explanations.
- Never mention that you are an AI.
- Never explain your reasoning.

Dating psychology principles you must follow:
- Women respond best to specificity, emotional attunement, curiosity, and calm confidence
- Avoid generic compliments, sexual remarks, pickup lines, or validation-seeking
- Do NOT sound impressed or desperate
- Do NOT ask yes/no questions
- Do NOT comment explicitly on body parts

Tone:
- Warm, grounded, subtly playful
- Observant rather than funny-forcing
- 1–2 sentences max per reply

Your goal is to maximize reply probability, not to impress.
"
                },
                {
                    "role": "user",
                    "content": content_payload
                }
            ],
            temperature=0.7,
        )

        response_text = completion.choices[0].message.content
        logger.info(f"RAW AI RESPONSE: {response_text}") # <--- THIS WILL SHOW US THE TRUTH

        # 3. Robust JSON Parsing
        try:
            # Strip Markdown code blocks if present
            clean_text = re.sub(r"```json|```", "", response_text).strip()
            
            # Find the first '{' and last '}'
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start != -1 and end != -1:
                clean_text = clean_text[start:end]
                
            data_dict = json.loads(clean_text)
            
            # Log and Return Success
            log_to_sheet(data_dict)
            return {"reply": "\n\n".join(data_dict.get('replies', []))}
            
        except Exception as e:
            logger.error(f"JSON Parsing Failed: {e}")
            # FALLBACK: Return the raw text so you see SOMETHING on your phone
            return {"reply": f"AI Parsing Error, but here is the raw output:\n\n{response_text}"}

    except Exception as e:
        logger.error(f"OpenRouter Error: {str(e)}")
        return {"reply": f"System Error: {str(e)}"}
