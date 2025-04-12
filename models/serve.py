from fastapi import FastAPI
from pydantic import BaseModel
from model import LSTM_Model

app = FastAPI()
model = LSTM_Model(num_time_steps=30)

@app.on_event("startup")
def load_model():
    try:
        model.load_model("model/lstm_model.keras")
    except Exception as e:
        print("Model failed to load:", e)

class PredictRequest(BaseModel):
    symbol: str

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        pred = model.predict(request.symbol)
        return {"symbol": request.symbol, "prediction": pred, "movement": "Up" if pred > 0.5 else "Down"}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/ping")
def ping():
    return {"status": "ok"}
