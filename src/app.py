from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from src.explain import RiskExplainer

app = FastAPI(
    title="Finanacial Risk Scoring Engine",
    description="Real-Time transaction risk scoring & AML Explainability API",
    version="1.0.0"
)

# Load pipeline and explainer on startup
PIPELINE_PATH = "models/risk_pipeline.joblib"
pipeline = joblib.load(PIPELINE_PATH)
explainer = RiskExplainer(PIPELINE_PATH)

class TransactionPayload(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "risk-scoring-engine"}

@app.post("/predict")
def score_transaction(tx: TransactionPayload):
    try:
        tx_dict = tx.model_dump()
        df = pd.DataFrame([tx_dict])

        # Predict probability
        risk_probability = float(pipeline.predict_proba(df)[0, 1])

        # Thresold = 0.5
        decision = "FLAGGED" if risk_probability >= 0.5 else "APPROVED"

        # Explainability audit
        top_risk_factors = explainer.explain_transaction(tx_dict, top_k=3) if decision == "FLAGGED" else []

        return {
            "risk_score": round(risk_probability, 4),
            "decision": decision,
            "requires_manual_review": decision == "FLAGGED",
            "top_risk_factors": top_risk_factors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)