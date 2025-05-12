from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from marketing_tool_improved_v5 import recommend_post_format, generate_embedding
import chromadb
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/recommend")
async def get_recommendations(input: RecommendationInput):
    try:
        date = pd.Timestamp(input.date) if input.date else pd.Timestamp.now()
        result = recommend_post_format(
            product=input.product,
            category=input.category,
            tone=input.tone,
            platform=input.platform,
            emotion=input.emotion,
            base_price=input.base_price,
            date=date,
            lang=input.lang
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"recommendations": result}
    except Exception as e:
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
        return {
            "recommendations": [
                {
                    "id": results["ids"][0][i],
                    "description": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                for i in range(len(results["ids"][0]))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)  # Using port 8005 for recommendation model