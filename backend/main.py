import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open('models/anxiety_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")

def _normalize_key(key: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(key).lower())

@app.post("/predict")
async def predict(data: dict):
    try:
        if "symptoms" in data:
            incoming_list = data["symptoms"]
            user_inputs = {symptom: 1 for symptom in incoming_list}
        else:
            user_inputs = data

        print(f"Received data: {user_inputs}")

        full_input = pd.DataFrame(0, index=[0], columns=feature_columns)
        normalized_to_column = {_normalize_key(col): col for col in feature_columns}

        activated_count = 0
        for feature, value in user_inputs.items():
            norm_feature = _normalize_key(feature)
            if norm_feature in normalized_to_column:
                column_name = normalized_to_column[norm_feature]
                full_input[column_name] = value
                activated_count += 1
                print(f"Activated feature: {column_name}")
            else:
                print(f"Feature not found: {feature}")

        print(f"Total activated features: {activated_count}")

        probability = model.predict_proba(full_input)[0][1]
        
        thresholds = [0.2, 0.4, 0.5, 0.7, 0.9]
        analysis = []
        for t in thresholds:
            is_anxiety = bool(probability >= t)
            analysis.append({
                "threshold": t,
                "is_anxiety": is_anxiety,
                "prediction": "Anxiety Detected" if is_anxiety else "Not Anxiety"
            })

        return {
            "probability": float(probability),
            "target_disease": "anxiety",
            "threshold_analysis": analysis,
            "disclaimer": "This is a preliminary analysis. Please consult a professional."
        }

    except Exception as e:
        print(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)