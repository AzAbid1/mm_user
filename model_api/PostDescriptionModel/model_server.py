from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allow your Angular app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS and POST
    allow_headers=["*"],  # Allow all headers
)
# If using a .env file:
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY", "your-hardcoded-key")  # Secure in production

MODEL_ID = "ft:open-mistral-7b:b078f810:20250410:c7eee20e"
ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Pydantic model to parse request body
class InputData(BaseModel):
    user_description: str

@app.post("/PostDescreption")
async def generate_posts(data: InputData):
    prompt = f"""
    Given the following user-provided product description: '{data.user_description}', generate promotional content optimized for my fine-tuned model. Produce the following two outputs:

    1. **Facebook Post Description**: A detailed, engaging caption for a Facebook post. Start with a hook, emphasize the product's unique benefit, include a call to action, and add 1-3 relevant hashtags. Make it conversational and suited for Facebook's audience.

    2. **Instagram Post Description**: A short, punchy caption for an Instagram post. Highlight the product's appeal with a strong hook, include a call to action, and use 1-3 trendy hashtags. Keep it concise and visual-focused.

    Format the response as:
    **Facebook Post Description:** [generated text]
    **Instagram Post Description:** [generated text]
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        return {"output": content}
    else:
        return {"error": response.text}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
