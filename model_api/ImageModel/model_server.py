from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import urllib.parse
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Pydantic model to validate frontend input
class ImageGenRequest(BaseModel):
    product_name: str
    product_desc: str
    text_model: str = "openai"
    image_model: str = "Flux"
    width: str  = "512"
    height: str  = "512"
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

    print("Enhanced prompt:")
    print(enhanced_prompt)

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
        img_resp = requests.get(image_url)
        img_resp.raise_for_status()
        background = Image.open(BytesIO(img_resp.content)).convert("RGB")
        output_path = os.path.join(os.getcwd(), "background_generated.png")
        
        return {"message": "Image generated successfully", "file_path": output_path}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")