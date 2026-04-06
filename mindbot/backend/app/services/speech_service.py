import os
from groq import Groq
from app.core.config import settings

# Initialize Groq client using your existing config
client = Groq(api_key=settings.groq_api_key)

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file (mp3, wav, etc.) to text using Groq Whisper.
    """
    if not settings.groq_api_key:
        return "Error: Groq API key is missing for transcription."

    try:
        with open(file_path, "rb") as audio_file:
            # Using whisper-large-v3 for the highest accuracy
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
            return transcription
    except Exception as e:
        return f"Transcription Error: {str(e)}"

def text_to_speech_placeholder(text: str):
    """
    If you want the bot to speak back, you'd put that logic here.
    For now, we focus on getting the 'Start Recording' input working.
    """
    pass