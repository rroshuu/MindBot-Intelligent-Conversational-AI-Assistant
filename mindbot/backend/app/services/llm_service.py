import os
from groq import Groq
from app.core.config import settings

# Initialize Groq Client
client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are MindBot, an intelligent conversational assistant.
Answer the user's question accurately based on the provided PDF context."""

def chat_completion(messages: list, temperature: float = 0.3) -> str:
    # Model: Llama 3.1 8B (Fast, High Quota, Free)
    MODEL_NAME = "llama-3.1-8b-instant"

    try:
        # Prepend the system prompt to the conversation
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {str(e)}"

def vision_completion(user_text: str, image_base64: str, mime_type: str = "image/png") -> str:
    # Model: Llama 3.2 11B Vision
    MODEL_NAME = "llama-3.2-11b-vision-instruct"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                        }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Vision Error: {str(e)}"