import base64
import os
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings, UPLOAD_DIR
from app.db.session import Base, engine, get_db
from app.db.models import Conversation, Message, Document
from app.utils.file_utils import (
    save_upload_file,
    is_allowed_document,
    is_allowed_audio,
    is_allowed_image,
    extract_text_from_pdf,
    extract_text_from_plain_file,
    file_extension,
)
from app.utils.chunking import chunk_text
from app.services.moderation_service import moderate_text
# Note: Ensure vision_completion in llm_service uses "llama-3.2-90b-vision-instruct"
from app.services.llm_service import chat_completion, SYSTEM_PROMPT, vision_completion
# Note: Ensure transcribe_audio in speech_service uses "whisper-large-v3-turbo"
from app.services.speech_service import transcribe_audio
from app.services.rag_service import add_document_chunks, retrieve_context

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    conversation_id: int
    reply: str
    context_used: bool = False

class UploadResponse(BaseModel):
    success: bool
    filename: str
    detail: str

# --- Helper Functions ---
def get_or_create_conversation(db: Session, conversation_id: Optional[int]) -> Conversation:
    if conversation_id:
        convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if convo:
            return convo
    convo = Conversation(title="New Chat")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo

def get_recent_messages(db: Session, conversation_id: int, limit: int = 10):
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(msgs))

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    convo = get_or_create_conversation(db, request.conversation_id)
    db.add(Message(conversation_id=convo.id, role="user", content=user_text))
    db.commit()

    context = retrieve_context(user_text, top_k=settings.top_k_docs)
    context_used = bool(context.strip())
    recent_messages = get_recent_messages(db, convo.id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_used:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    
    for msg in recent_messages:
        messages.append({"role": msg.role, "content": msg.content})

    reply = chat_completion(messages)
    db.add(Message(conversation_id=convo.id, role="assistant", content=reply))
    db.commit()

    return ChatResponse(conversation_id=convo.id, reply=reply, context_used=context_used)

@app.post("/upload/document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_allowed_document(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported document type")
    
    saved_path = await save_upload_file(file, UPLOAD_DIR)
    ext = file_extension(file.filename)
    text = extract_text_from_pdf(saved_path) if ext == ".pdf" else extract_text_from_plain_file(saved_path)

    chunks = chunk_text(text)
    add_document_chunks(chunks, source=file.filename)
    
    db.add(Document(filename=file.filename, filepath=str(saved_path), filetype=ext, status="indexed"))
    db.commit()
    return UploadResponse(success=True, filename=file.filename, detail=f"Indexed {len(chunks)} chunks")

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...), prompt: str = Form("Analyze this image.")):
    if not is_allowed_image(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported image type")
    
    saved_path = await save_upload_file(file, UPLOAD_DIR)
    with open(saved_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # vision_completion should now use "llama-3.2-90b-vision-instruct"
    result = vision_completion(prompt, image_base64, file.content_type or "image/png")
    return {"success": True, "analysis": result}

@app.post("/speech-to-text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    """Handles the 'Start Recording' blob or Audio Upload for transcription."""
    temp_path = UPLOAD_DIR / f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # transcribe_audio should now use "whisper-large-v3-turbo"
        transcript = transcribe_audio(str(temp_path))
        return {"success": True, "transcript": transcript}
    finally:
        if temp_path.exists():
            os.remove(temp_path)