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

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HingeWingman")

app = FastAPI()

# --- CONFIGURATION ---
# We use the generic 'openai' client but point it to OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")
SHEET_NAME = "Hinge_Rizz_Tracker"

# The model you want to use. You can change this string anytime!
# Recommended free vision models on OpenRouter:
# "google/gemini-2.0-flash-exp:free" (Very smart, vision native)
# "google/gemini-flash-1.5-8b"
# "meta-llama/llama-3.2-11b-vision-instruct:free"
MODEL_NAME = "google/gemma-3-27b-it:free" 

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
        logger.error(f"Sheet logging failed: {e}")

# --- HELPER: EXTRACT FRAMES ---
def extract_frames_base64(video_path, max_frames=5):
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
            # Resize to save tokens (OpenRouter free tier has limits)
            frame = cv2.resize(frame, (512, 512)) 
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(f"data:image/jpeg;base64,{b64_str}")
        current_frame += interval
    cap.release()
    return frames_b64

@app.get("/")
def home():
    return {"status": "OpenRouter Wingman is Active"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(None)):
    if not file:
        return {"reply": "Error: No file received. Check Shortcut key is 'file'."}
    
    if not OPENROUTER_API_KEY:
        return {"reply": "Error: OpenRouter API Key missing."}

    # 1. Save and Process Video
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    frames = extract_frames_base64(temp_filename)
    if os.path.exists(temp_filename): os.remove(temp_filename)

    if not frames:
        return {"reply": "Error: Could not extract frames."}

    # 2. Build OpenRouter Payload
    # We construct a list of image messages
    content_payload = [
        {"type": "text", "text": "Analyze this dating profile video. Extract metadata (Name, Age, Job, Location, Interests) and write 3 witty, high-status replies based on visual hooks. Return ONLY JSON."}
    ]
    
    for b64_url in frames:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": b64_url}
        })

    logger.info(f"Sending request to OpenRouter model: {MODEL_NAME}")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON-only dating assistant. Output format: {\"metadata\": {\"name\": \"...\", \"age\": \"...\", \"job\": \"...\", \"location\": \"...\", \"interests\": []}, \"replies\": [\"...\", \"...\", \"...\"]}"
                },
                {
                    "role": "user",
                    "content": content_payload
                }
            ],
            temperature=0.8,
            # 'headers' allows us to tell OpenRouter who we are (optional but polite)
            extra_headers={
                "HTTP-Referer": "https://render.com",
                "X-Title": "Hinge Wingman"
            }
        )

        response_text = completion.choices[0].message.content
        logger.info("Received response from OpenRouter.")
        
        # 3. Clean and Parse JSON
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        try:
            data_dict = json.loads(clean_text)
        except:
            # Simple fallback if model chats too much
            import ast
            # Try to find the first { and last }
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start != -1 and end != -1:
                data_dict = json.loads(clean_text[start:end])
            else:
                return {"reply": clean_text} # Return raw text if JSON fails

        # 4. Log and Return
        log_to_sheet(data_dict)
        return {"reply": "\n\n".join(data_dict.get('replies', []))}

    except Exception as e:
        logger.error(f"OpenRouter Error: {str(e)}")
        return {"reply": f"AI Error: {str(e)}"}
