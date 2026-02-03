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

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")

SPREADSHEET_NAME = "Hinge_Rizz_Tracker"
WORKSHEET_NAME = "Sheet1"

MODEL_NAME = "allenai/molmo-2-8b:free"

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

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

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
            ", ".join(meta.get("interests", [])),
            ", ".join(meta.get("personality_traits", [])),
            meta.get("communication_style"),
            meta.get("vibe"),
            replies[0],
            replies[1],
            replies[2],
            "PENDING"
        ]

        if len(row) != 14:
            raise ValueError("Row column mismatch")

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

    if total <= 0:
        return []

    interval = max(1, total // max_frames)
    frames = []
    pos = 0

    while pos < total and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = base64.b64encode(buf).decode()
            frames.append(f"data:image/jpeg;base64,{b64}")
        pos += interval

    cap.release()
    return frames

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "Hinge Wingman Running"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    if not OPENROUTER_API_KEY:
        return {"error": "Missing OpenRouter API key"}

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    frames = extract_frames_base64(temp_path)
    os.remove(temp_path)

    if not frames:
        return {"error": "Could not extract frames"}

    # --------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------
    user_prompt = """
You are analyzing screenshots of a FEMALE Hinge profile.

Extract metadata using:
- Visible text
- UI labels
- IMPLIED signals (photos, prompts, context)

If unsure, return null. Do NOT hallucinate.

Metadata schema:
{
  "name": string | null,
  "age": number | null,
  "job": string | null,
  "education": string | null,
  "location": string | null,
  "interests": array of strings,
  "personality_traits": array of strings,
  "communication_style": string,
  "vibe": string
}

Then generate EXACTLY 3 Hinge like-comments.

Reply rules:
- Each reply must reference a DIFFERENT observation
- Hinglish allowed
- Subtle poetic / shayari tone (conversational)
- One curious, one lightly teasing, one calm-confident
- No generic compliments
- No emojis
- No yes/no questions
- No body comments
- 1–2 sentences max

Return ONLY valid JSON:
{
  "metadata": {...},
  "replies": ["", "", ""]
}
"""

    content = [{"type": "text", "text": user_prompt}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": f}})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a high-EQ dating assistant for MEN replying to FEMALE Hinge profiles.

You write messages that feel:
- human
- confident
- observant
- emotionally intelligent

CRITICAL RULES:
- Output ONLY valid JSON
- No markdown or explanations
- Never mention AI
- Never sound impressed or desperate

Dating principles:
- Specificity beats compliments
- Calm confidence beats humor forcing
- Observational > validating

Before finalizing each reply, rewrite it mentally to sound like something a real person would casually type.
Your goal is to maximize reply probability, not to impress.
"""
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.7
        )

        raw = completion.choices[0].message.content
        logger.info(f"RAW OUTPUT: {raw}")

        # ---------------- CLEAN JSON ----------------
        clean = re.sub(r"```json|```", "", raw).strip()
        clean = clean[clean.find("{"): clean.rfind("}") + 1]
        data = json.loads(clean)

        # Safety normalization
        replies = data.get("replies", [])
        data["replies"] = replies[:3] + [""] * (3 - len(replies))

        log_to_sheet(data)

        return data

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {"error": str(e)}
