from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import urllib.parse
from io import BytesIO
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allow your Angular app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS and POST
    allow_headers=["*"],  # Allow all headers
)

# Pydantic model to validate frontend input
class ImageGenRequest(BaseModel):
    product_name: str
    product_desc: str
    text_model: str = "openai"
    image_model: str = "Flux"
    width: str = "512"
    height: str = "512"
    seed: str | None = None

@app.post("/generate-image")
async def generate_image(data: ImageGenRequest):
    system_directives = (
        "You are TTIPEIG, the Ultimate Text-to-Image Prompt Enhancer and Image Generator. "
        "Your task is to craft a vivid, detailed prompt that instructs an image API to create a scene "
        "into which a product can be seamlessly inserted."
    )
    pre_prompt = f"{system_directives} Product Name: {data.product_name}. Description: {data.product_desc}."
    enhance_url = (
        "https://text.pollinations.ai/" + urllib.parse.quote(pre_prompt) +
        f"?model={data.text_model}"
    )
    print(f"Enhancing prompt via text API (model={data.text_model})...")
    try:
        resp = requests.get(enhance_url)
        resp.raise_for_status()
        enhanced_prompt = resp.text
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing prompt: {e}")

    

    params = {
        'model': data.image_model,
        'width': data.width,
        'height': data.height,
        'seed': data.seed,
        'nologo': 'true',
        'private': 'false',
        'enhance': 'true',
        'safe': 'false'
    }
    query = '&'.join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items() if v is not None)
    image_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(enhanced_prompt)}?{query}"
    print("Generating background scene:", image_url)
    try:
        # Add a delay to ensure loading animation is visible
        await asyncio.sleep(5)  # Simulate processing time
        img_resp = requests.get(image_url)
        img_resp.raise_for_status()
        background = Image.open(BytesIO(img_resp.content)).convert("RGB")
        
        # Convert image to base64
        buffered = BytesIO()
        background.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"message": "Image generated successfullydsd", "image": f"data:image/png;base64,{img_str}"}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)