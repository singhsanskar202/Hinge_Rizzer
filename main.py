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
SHEET_NAME = "Hinge_Rizz_Tracker"

# ✅ Vision model that follows JSON well
MODEL_NAME = "allenai/molmo-2-8b:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --------------------------------------------------
# GOOGLE SHEETS
# --------------------------------------------------
def get_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1
    return sheet


def log_to_sheet(data):
    if not GOOGLE_CREDS_JSON:
        logger.warning("Google creds missing, skipping sheet logging")
        return

    try:
        sheet = get_sheet()

        meta = data.get("metadata", {})
        replies = data.get("replies", [])
        replies += [""] * (3 - len(replies))

        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            meta.get("name"),
            meta.get("age"),
            meta.get("job"),
            meta.get("location"),
            ", ".join(meta.get("interests", [])),
            meta.get("vibe"),
            replies[0],
            replies[1],
            replies[2],
            "PENDING"
        ]

        sheet.append_row(row, value_input_option="USER_ENTERED")
        logger.info("Logged row to Google Sheet")

    except Exception as e:
        logger.error(f"Google Sheet error: {e}")


# --------------------------------------------------
# VIDEO → FRAMES
# --------------------------------------------------
def extract_frames_base64(video_path, max_frames=7):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return []

    interval = max(1, total // max_frames)
    frames = []
    frame_id = 0

    while frame_id < total and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = base64.b64encode(buffer).decode()
            frames.append(f"data:image/jpeg;base64,{b64}")
        frame_id += interval

    cap.release()
    return frames


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def home():
    return {"status": "Wingman Active"}


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    if not OPENROUTER_API_KEY:
        return {"error": "Missing OpenRouter API key"}

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    frames = extract_frames_base64(temp_path)
    os.remove(temp_path)

    if not frames:
        return {"error": "Could not read video"}

    # ---------------- PROMPT ----------------
    user_text = """
Analyze the dating profile video frames.

1. Extract metadata using only visible or clearly implied information.
2. Generate exactly 3 Hinge like-comments.

Metadata schema:
{
  "name": string | null,
  "age": number | null,
  "job": string | null,
  "location": string | null,
  "interests": array of strings,
  "vibe": string
}

Rules for replies:
- Each reply must reference a different observation
- One must spark curiosity
- One must use light teasing
- Avoid generic compliments
- No yes/no questions
- 1–2 sentences max

Return ONLY valid JSON:
{
  "metadata": {...},
  "replies": ["", "", ""]
}
"""

    content = [{"type": "text", "text": user_text}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": f}})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an emotionally intelligent dating assistant specialized in Hinge profiles.

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

"""
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.75,
        )

        raw = completion.choices[0].message.content
        logger.info(f"RAW MODEL OUTPUT: {raw}")

        # -------- JSON PARSING --------
        clean = re.sub(r"```json|```", "", raw).strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        clean = clean[start:end]

        data = json.loads(clean)

        log_to_sheet(data)

        return {
            "replies": data.get("replies", []),
            "metadata": data.get("metadata", {})
        }

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return {"error": str(e)}
