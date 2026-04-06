from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from app.services.speech_service import transcribe_audio

router = APIRouter()

@router.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    # 1. Save the incoming audio blob temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Transcribe using the service
        text = transcribe_audio(temp_path)
        return {"text": text}
    finally:
        # 3. Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)