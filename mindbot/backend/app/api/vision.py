from fastapi import APIRouter, UploadFile, File, Form
from app.services.vision_service import analyze_image_from_base64
import base64

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...), prompt: str = Form("What is in this image?")):
    # 1. Read the uploaded file
    image_bytes = await file.read()
    
    # 2. Encode to Base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 3. Call our new vision service
    analysis = analyze_image_from_base64(image_base64, prompt, file.content_type)
    
    return {"analysis": analysis}