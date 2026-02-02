import cv2
import numpy as np
import os
import datetime
import gspread
import json
import google.generativeai as genai
from PIL import Image
import io
from oauth2client.service_account import ServiceAccountCredentials
from fastapi import FastAPI, UploadFile, File
import logging

# --- LOGGING SETUP ---
# This ensures logs show up in your Render dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HingeWingman")

app = FastAPI()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")
SHEET_NAME = "Hinge_Rizz_Tracker"

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("Gemini API Key is missing!")

# --- HELPER: LOG TO SHEET ---
def log_to_sheet(data_dict):
    """
    Logs metadata + prompts to Google Sheets.
    Expects data_dict to have keys: 'metadata' (dict) and 'replies' (list)
    """
    if not GOOGLE_CREDS_JSON: 
        logger.warning("Google Creds missing. Skipping sheet log.")
        return

    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1
        
        # Extract Data
        meta = data_dict.get('metadata', {})
        replies = data_dict.get('replies', [])
        
        # Ensure we have 3 replies to fill columns
        replies = replies + [""] * (3 - len(replies))
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # New Row Format:
        # [Timestamp, Name, Age, Job, Location, Interests, Opt1, Opt2, Opt3, Status]
        row = [
            timestamp,
            meta.get('name', 'Unknown'),
            meta.get('age', 'Unknown'),
            meta.get('job', 'Unknown'),
            meta.get('location', 'Unknown'),
            ", ".join(meta.get('interests', [])), # Convert list to string
            replies[0],
            replies[1],
            replies[2],
            "PENDING"
        ]
        
        sheet.append_row(row)
        logger.info(f"Successfully logged profile: {meta.get('name', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Logging to sheet failed: {e}")

# --- HELPER: EXTRACT FRAMES ---
def extract_frames_as_pil(video_path, max_frames=6):
    logger.info(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0: 
        logger.error("Video has 0 frames.")
        return []
    
    interval = max(1, total_frames // max_frames)
    pil_images = []
    current_frame = 0
    
    while current_frame < total_frames and len(pil_images) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_images.append(pil_img)
        current_frame += interval
            
    cap.release()
    logger.info(f"Extracted {len(pil_images)} frames.")
    return pil_images

@app.get("/")
def home():
    return {"status": "Gemini Wingman is Active"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    logger.info("Received video upload request.")
    
    if not GEMINI_API_KEY:
        return {"reply": "Error: Server missing Gemini API Key."}

    # 1. Save video temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Extract frames
    frames = extract_frames_as_pil(temp_filename)
    
    # 3. Cleanup
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    if not frames:
        return {"reply": "Error: Could not process video."}

    # 4. Prepare Prompt for Gemini
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt_text = """
    You are a witty, Gen-Z dating expert. I am sending frames from a Hinge profile video.
    
    Task 1 (Extraction):
    Analyze the frames to identify the user's Name, Age, Job, Location, and key Interests. 
    If a field is not visible, return "Unknown".
    
    Task 2 (Generation):
    Generate 3 distinct, high-status, witty opening lines based on the visual hooks or text prompts found.
    
    Format:
    Return ONLY a valid JSON object. No Markdown.
    {
        "metadata": {
            "name": "String",
            "age": "String",
            "job": "String",
            "location": "String",
            "interests": ["String", "String"]
        },
        "replies": ["Reply 1", "Reply 2", "Reply 3"]
    }
    """
    
    input_content = [prompt_text] + frames

    try:
        logger.info("Sending request to Gemini...")
        response = model.generate_content(input_content)
        content_text = response.text
        
        # Clean JSON
        clean_text = content_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        try:
            data_dict = json.loads(clean_text)
        except json.JSONDecodeError:
            # Fallback for slight formatting errors
            import ast
            data_dict = ast.literal_eval(clean_text)

        logger.info("Gemini response parsed successfully.")

        # 5. Log to Sheets (Async-ish)
        log_to_sheet(data_dict)
        
        # 6. Return ONLY text to iPhone
        # We join them with newlines so the shortcut displays them cleanly
        return {"reply": "\n\n".join(data_dict.get('replies', []))}
        
    except Exception as e:
        logger.error(f"Error processing: {str(e)}")
        return {"reply": f"Error: {str(e)}"}
