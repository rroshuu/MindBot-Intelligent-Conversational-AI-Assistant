from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.llm_service import chat_completion
from app.services.rag_service import retrieve_context

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    use_context: bool = True

@router.post("/")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.messages[-1].content
        context = ""
        if request.use_context:
            context = retrieve_context(user_query)
        
        if context:
            request.messages[-1].content = f"Context:\n{context}\n\nQuestion: {user_query}"

        msg_dicts = [m.model_dump() for m in request.messages]
        ai_response = chat_completion(msg_dicts)
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))