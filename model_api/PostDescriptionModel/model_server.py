from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
from mistralai import Mistral
from fastapi.responses import FileResponse
from gtts import gTTS
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allow Angular app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY", "j57g1pobCtAZIPkC9jzy9OEcvf0crHb2")  # Secure in production
MODEL_ID = "mistral-large-latest"
ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
client = Mistral(api_key=API_KEY)
# Pydantic model for request body
class InputData(BaseModel):
    user_description: str

@app.post("/PostDescreption")
async def generate_posts(data: InputData, request: Request):
    logger.debug(f"Received request: {await request.json()}")

    prompt = f"""
    Given the following user-provided product description: '{data.user_description}', generate promotional content optimized for my fine-tuned model. Produce the following two outputs:

    1. **Facebook Post Description**: A detailed, engaging caption for a Facebook post. Start with a hook, emphasize the product's unique benefit, include a call to action, and add 1-3 relevant hashtags. Make it conversational and suited for Facebook's audience.

    2. **Instagram Post Description**: A short, punchy caption for an Instagram post. Highlight the product's appeal with a strong hook, include a call to action, and use 1-3 trendy hashtags. Keep it concise and visual-focused.

    Format the response as:
    **Facebook Post Description:** [generated text]
    **Instagram Post Description:** [generated text]
    """

    try:
        chat_response = client.chat.complete(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
        )
        content = chat_response.choices[0].message.content
        logger.debug(f"Mistral API response: {content}")

        # Generate audio from content
        tts = gTTS(text=content, lang='en')
        filename = f"post_audio_{uuid.uuid4()}.mp3"
        filepath = f"./{filename}"
        tts.save(filepath)

        return {
            "content": content,
            "audio_file": f"http://localhost:8001/audio/{filename}"
        }

    except Exception as e:
        logger.error(f"Error calling Mistral API: {str(e)}")
        return {"error": f"Failed to process request: {str(e)}", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)