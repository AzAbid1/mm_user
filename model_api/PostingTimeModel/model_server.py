from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
from enum import Enum

app = FastAPI(title="Social Media Posting Time Predictor")

# Define platform enum for input validation
class Platform(str, Enum):
    facebook = "facebook"
    instagram = "instagram"

# Input model for the POST request
class PredictionInput(BaseModel):
    platform: Platform
    product: str
    tone: str

# Category keyword mapping
category_keywords = {
    'Cosmetics': ['lipstick', 'collagen', 'fragrance', 'body splash', 'makeup', 'cream', 'lotion', 'perfume', 'hair product', 'skincare', 'supplement', 'beauty', 'cosmetic', 'moisturizer', 'serum'],
    'Fashion': ['sweater', 'clothing', 'nike', 'cardigan', 'dress', 'shirt', 'jacket', 'shoes', 'accessories', 'jeans', 'fleece', 'apparel', 'sneakers', 'outfit', 'fashion'],
    'Technology': ['smartwatch', 'iphone', 'smartphone', 'oppo', 'montre connectee', 'phone', 'pc', 'gamer', 'cooler master', 'flatpack', 'phone line', 'telecom', 'mobile', 'network', 'computer', 'laptop', 'desktop'],
    'Food': ['recipe', 'dish', 'ingredient', 'food', 'meal', 'snack', 'beverage', 'dessert', 'cooking', 'cuisine', 'cookie', 'popcorn', 'healthy', 'pizza', 'restaurant', 'menu']
}

# Preferred days and rule-based hours per category
preferred_days = {
    'Cosmetics': ['Monday', 'Tuesday', 'Wednesday'],
    'Food': ['Tuesday', 'Wednesday', 'Thursday'],
    'Fashion': ['Wednesday', 'Sunday'],
    'Technology': ['Tuesday', 'Friday']
}

rule_based_hours = {
    'Cosmetics': [12, 14, 15],
    'Food': [13, 16, 17],
    'Fashion': [16, 18],
    'Technology': [14, 15]
}

# Assign category based on product
def assign_category(product: str) -> str:
    product = str(product).lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in product:
                return category
    return 'Cosmetics'  # Default to Cosmetics if no match

# Prediction function
def predict_posting_times(platform: str, product: str, tone: str, start_date: datetime = datetime(2025, 4, 28)):
    # Load model and encoders based on platform
    model_file = 'posting_time_model_fb_v11.pkl' if platform == 'facebook' else 'posting_time_model_v11.pkl'
    encoders_file = 'encoders_fb_v11.pkl' if platform == 'facebook' else 'encoders_v11.pkl'
    
    if not (os.path.exists(model_file) and os.path.exists(encoders_file)):
        raise HTTPException(status_code=500, detail=f"Model or encoders not found for {platform}.")
    
    model = joblib.load(model_file)
    encoders = joblib.load(encoders_file)

    # Load dataset for product lookup
    try:
        df = pd.read_csv('deep_analyzed_insta_fb_11.csv', parse_dates=['datetime'])
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset 'deep_analyzed_insta_fb_11.csv' not found.")

    # Determine category
    product_lower = product.lower()
    if product_lower in df['product'].str.lower().values:
        category = df[df['product'].str.lower() == product_lower]['category'].iloc[0]
        product_data = df[df['product'].str.lower() == product_lower]
    else:
        category = assign_category(product)
        product_data = df[df['category'] == category]

    # Encode inputs
    try:
        category_encoded = encoders['category'].transform([category])[0]
    except ValueError:
        category = 'Cosmetics'
        category_encoded = encoders['category'].transform([category])[0]

    tone_encoded = encoders['tone'].transform([tone])[0] if tone in encoders['tone'].classes_ else encoders['tone'].transform(['enthusiastic'])[0]

    # Filter viable hours (top 50% engagement)
    engagement_by_hour = product_data.groupby('hour')['engagement'].mean()
    viable_hours = engagement_by_hour[engagement_by_hour >= engagement_by_hour.quantile(0.5)].index.tolist()
    viable_hours = [h for h in viable_hours if 8 <= h <= 22]
    if not viable_hours:
        viable_hours = list(range(8, 23))

    # Add rule-based hours
    expected_hours = rule_based_hours.get(category, [12, 14, 15])
    for hour in expected_hours:
        if hour not in viable_hours:
            viable_hours.append(hour)

    # Get preferred days
    days = preferred_days.get(category, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Generate predictions
    predictions = []
    for day in days:
        day_encoded = encoders['day_name'].transform([day])[0]
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        for hour in viable_hours:
            hour_cat = pd.cut([hour], bins=[0, 11, 17, 23], labels=['morning', 'afternoon', 'evening'], include_lowest=True)[0]
            hour_cat_encoded = encoders['hour_category'].transform([hour_cat])[0]
            input_data = pd.DataFrame({
                'category': [category_encoded],
                'tone': [tone_encoded],
                'hour': [hour],
                'day_name': [day_encoded],
                'has_top_hashtag': [1],
                'is_weekend': [is_weekend],
                'hour_category': [hour_cat_encoded]
            })
            prob = model.predict_proba(input_data)[0][1]
            prob = min(prob * 1.5, 0.95) if hour in expected_hours else min(prob * 1.2, 0.90)
            predictions.append((day, hour, prob))

    # Sort and get top prediction
    predictions.sort(key=lambda x: x[2], reverse=True)
    top_times = predictions[:5]

    # Check alignment with expected hours
    predicted_hours = [h for _, h, _ in top_times]
    matching_hours = sum(1 for h in predicted_hours if h in expected_hours)
    if matching_hours < 2 or len(product_data) < 5:
        top_times = [(day, hour, 0.95) for day in days for hour in expected_hours][:5]

    # Generate posting times
    posting_times = []
    for i in range(7):
        date = start_date + timedelta(days=i)
        day_name = date.strftime('%A')
        if day_name in days:
            for day, hour, prob in top_times:
                if day == day_name:
                    posting_times.append((date.strftime('%Y-%m-%d'), day_name, hour, prob))
                    break  # Take only the first matching time for the day

    # Return the first posting time's date and hour
    if posting_times:
        date, _, hour, _ = posting_times[0]
        return f"{date} {hour}:00"
    else:
        raise HTTPException(status_code=500, detail="No posting times predicted.")

# POST endpoint for prediction
@app.post("/predict_posting_time/")
async def predict(input_data: PredictionInput):
    try:
        result = predict_posting_times(
            platform=input_data.platform,
            product=input_data.product,
            tone=input_data.tone
        )
        return {"posting_time": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)