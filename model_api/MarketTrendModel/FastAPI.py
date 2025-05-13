# -*- coding: utf-8 -*-
"""xgboost_fastapi.py

FastAPI application to serve 128-month retail sales forecasts using pre-trained XGBoost models.
Modified to handle running in environments with an active event loop (e.g., Jupyter, Colab) and fix model loading issues.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import nest_asyncio
import uvicorn

# Apply nest_asyncio to allow running in environments with an active event loop
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(
    title="XGBoost Retail Sales Forecast API",
    description="API for forecasting retail sales using pre-trained XGBoost models."
)

# Define input schema
class ForecastRequest(BaseModel):
    sector: str  # e.g., 'Computer_Telecom', 'Food_Stores', 'Cosmetics', 'Clothing'
    historical_data: list[float]  # List of historical index values (at least 12 months)

# Define response schema
class ForecastResponse(BaseModel):
    sector: str
    forecast: list[float]
    dates: list[str]

# Helper function to generate future dates
def generate_future_dates(start_date, periods, freq='M'):
    """
    Generate future dates for forecasting.
    
    Args:
        start_date (pd.Timestamp): Last date in historical data.
        periods (int): Number of periods to forecast.
        freq (str): Frequency of dates (default: 'M' for monthly).
    
    Returns:
        list: List of date strings in 'YYYY-MM-DD' format.
    """
    date_range = pd.date_range(start=start_date + pd.offsets.MonthEnd(1), periods=periods, freq=freq)
    return [date.strftime('%Y-%m-%d') for date in date_range]

# Forecast endpoint
@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Generate a 128-month forecast for the specified sector using XGBoost.
    
    Args:
        request (ForecastRequest): JSON payload with sector and historical data.
    
    Returns:
        ForecastResponse: JSON with sector, forecast values, and future dates.
    """
    try:
        # Normalize sector name to lowercase with underscores for consistency
        sector_normalized = request.sector.lower().replace(' ', '_')
        # Define valid sectors (lowercase with underscores to match model file names)
        valid_sectors = ['computer_telecom', 'food_stores', 'cosmetics', 'clothing']
        if sector_normalized not in valid_sectors:
            raise HTTPException(status_code=400, detail=f"Sector must be one of {valid_sectors}")

        # Validate historical data (at least 12 months for lags)
        if len(request.historical_data) < 12:
            raise HTTPException(status_code=400, detail="At least 12 months of historical data are required")

        # Construct model path
        model_dir = "Downloads/models"
        model_path = os.path.join(model_dir, f"xgboost_{sector_normalized}.joblib")
        print(f"Current working directory: {os.getcwd()}")  # Debug current directory
        print(f"Checking model path: {model_path}")  # Debug model path
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model for {request.sector} not found at {model_path}")

        # Load the XGBoost model
        model_xgb = joblib.load(model_path)

        # Prepare input data with lagged features
        historical_data = np.array(request.historical_data)
        last_values = historical_data[-12:][::-1]  # Last 12 values in reverse order (lag_1 to lag_12)

        # Forecast iteratively
        forecast_periods = 128
        xgb_forecast = []
        current_values = last_values.copy()
        for _ in range(forecast_periods):
            pred = model_xgb.predict(current_values.reshape(1, -1))[0]
            xgb_forecast.append(pred)
            current_values = np.roll(current_values, -1)
            current_values[-1] = pred

        # Generate future dates (using current date as reference)
        last_date = pd.Timestamp.now().normalize()
        future_dates = generate_future_dates(last_date, forecast_periods)

        return ForecastResponse(
            sector=request.sector,
            forecast=xgb_forecast,  # Removed .tolist() since xgb_forecast is already a list
            dates=future_dates
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)