import cv2
import numpy as np
import base64
import requests
import os
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from fastapi import FastAPI, UploadFile, File
import json

app = FastAPI()

# --- CONFIGURATION ---
GROK_API_KEY = os.environ.get("GROK_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON") # We will paste the JSON key here
SHEET_NAME = "Hinge_Rizz_Tracker"

# --- HELPER: LOG TO SHEET ---
def log_to_sheet(prompt_options):
    try:
        # Authenticate with Google
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open Sheet
        sheet = client.open(SHEET_NAME).sheet1
        
        # Prepare Row: [Date, Option 1, Option 2, Option 3, Status]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, prompt_options[0], prompt_options[1], prompt_options[2], "PENDING"]
        
        sheet.append_row(row)
    except Exception as e:
        print(f"Logging failed: {e}")

# --- HELPER: EXTRACT FRAMES ---
def extract_frames(video_path, max_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return []
    interval = max(1, total_frames // max_frames)
    extracted_frames = []
    current_frame = 0
    while current_frame < total_frames and len(extracted_frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            extracted_frames.append(b64_frame)
        current_frame += interval
    cap.release()
    return extracted_frames

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # 1. Process Video
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())
    
    frames_b64 = extract_frames(temp_filename)
    if os.path.exists(temp_filename): os.remove(temp_filename)

    # 2. Call Grok
    content_list = [{"type": "text", "text": "Analyze this video. Give me exactly 3 short, witty replies as a JSON list. Example: [\"Reply 1\", \"Reply 2\", \"Reply 3\"]"}]
    for b64_img in frames_b64:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

    payload = {
        "messages": [
            {"role": "system", "content": "You are a dating expert. Return ONLY a valid JSON list of 3 strings."},
            {"role": "user", "content": content_list}
        ],
        "model": "grok-vision-beta",
        "temperature": 0.85
    }
    
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    response = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers)
    
    try:
        content_text = response.json()['choices'][0]['message']['content']
        # Clean the response to ensure it's a list
        import ast
        # Attempt to parse the list from the string
        start_index = content_text.find('[')
        end_index = content_text.rfind(']') + 1
        clean_list = ast.literal_eval(content_text[start_index:end_index])
        
        # 3. LOG DATA TO GOOGLE SHEETS
        log_to_sheet(clean_list)
        
        return {"reply": "\n\n".join(clean_list)} # Return text for the Shortcut to display
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
