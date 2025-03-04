from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/phishing_model.pkl")

@app.post("/predict")
def predict(url_features: dict):
    df = pd.DataFrame([url_features])
    prediction = model.predict(df)
    return {"is_phishing": bool(prediction[0])}
