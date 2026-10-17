# Real-Time Financial Risk Scoring & AML Detection Engine

An end-to-end Machine Learning pipeline and low-latency microservice designed for real-time transaction fraud scoring and regulatory audit compliance.

## Key Features
- **Cost-Sensitive ML Modeling**: Trained on heavily imbalanced streaming transaction data using Scikit-Learn pipelines and LightGBM with automated class-weight balancing (`scale_pos_weight`).
- **Explainable AI (XAI)**: Integrated TreeSHAP to compute local feature attribution and extract top risk-driving factors for flagged transactions.
- **MLOps Tracking**: Managed experiment runs, PR-AUC / ROC-AUC metrics, and model artifacts with MLflow.
- **Low-Latency Serving**: Containerized FastAPI microservice delivering sub-millisecond inference and explainability outputs.

## Architecture & Tech Stack
- **Languages & Frameworks**: Python, Scikit-Learn, LightGBM, SHAP, FastAPI, MLflow, Docker
- **Evaluation Metrics**: Precision-Recall AUC (PR-AUC), ROC-AUC, F1-Score

## Quickstart

### 1. Run Data Generation & Train Pipeline
```bash
python src/generate_data.py
python src/pipeline.py
```

### 2. Launch Local API
```bash
uvicorn src.app:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 3. Docker Deployment
```bash
docker build -t risk-scoring-engine .
docker run -p 8000:8000 risk-scoring-engine
```