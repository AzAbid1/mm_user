from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from marketing_tool_improved_v5 import recommend_post_format, generate_embedding
import chromadb
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import sys
import json

# Set stdout encoding to UTF-8 to handle Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Social Media Recommendation Model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allow your Angular app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Chroma
chroma_client = chromadb.Client()
collection = chroma_client.get_collection("recommendations")

class RecommendationInput(BaseModel):
    product: str
    category: str
    tone: str = "auto"
    platform: str = "instagram"
    emotion: str = "joie"
    base_price: Optional[float] = None
    date: Optional[str] = None
    lang: str = "français"

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

@app.post("/recommend")
async def get_recommendations(input: RecommendationInput):
    try:
        # Validate product and category
        if not input.product or input.product.strip() == "":
            raise HTTPException(status_code=400, detail="Product name cannot be empty")
        if not input.category or input.category.strip() == "":
            raise HTTPException(status_code=400, detail="Category cannot be empty")

        date = pd.Timestamp(input.date) if input.date else pd.Timestamp.now()
        result = recommend_post_format(
            product=input.product.strip(),
            category=input.category.strip(),
            tone=input.tone,
            platform=input.platform,
            emotion=input.emotion,
            base_price=input.base_price,
            date=date,
            lang=input.lang
        )
        if "error" in result:
            # Log error as JSON to avoid encoding issues
            error_log = json.dumps({"error": result["error"]}, ensure_ascii=False)
            print(f"Error from recommend_post_format: {error_log}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Convert NumPy types to Python types
        serialized_result = convert_numpy_types(result)
        # Log result as JSON to handle Unicode characters
        result_log = json.dumps(serialized_result, ensure_ascii=False)
        print(f"Serialized result: {result_log}")
        return {"recommendations": serialized_result}
    except Exception as e:
        # Log exception as string to avoid encoding issues
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/search_recommendations")
async def search_recommendations(query: str, top_k: int = 5):
    try:
        query_embedding = generate_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents"]
        )
        serialized_results = convert_numpy_types({
            "recommendations": [
                {
                    "id": results["ids"][0][i],
                    "description": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                for i in range(len(results["ids"][0]))
            ]
        })
        # Log result as JSON
        result_log = json.dumps(serialized_results, ensure_ascii=False)
        print(f"Search result: {result_log}")
        return serialized_results
    except Exception as e:
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)  # Using port 8005 for recommendation model