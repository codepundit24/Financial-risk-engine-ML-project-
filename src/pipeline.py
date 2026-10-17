import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

def train_model(data_path: str, model_save_path: str = "models/risk_pipeline.joblib"):
    print("Loading data...")
    # kaggle credit card fraud dataset format: Time, V1-V28, Amount, Class
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train-Test split with stratification for severe imbalance
    X_train, X_test, y_train, y_test, = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Severe class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Dataset Imbalance Ratio: {scale_pos_weight:.2f}")

    # Feature transformation pipeline
    # Amount and Time to be scale , while other features are already PCA normalized 
    scale_features = ["Amount", "Time"] if "Time" in X.columns else ["Amount"]
    passthrough_features = [col for col in X.columns if col not in scale_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", RobustScaler(), scale_features),
            ("passthrough", "passthrough", passthrough_features),
        ]
    )

    # Core LightGBM Model with Scikit-Learn Pipeline
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # MLflow tracking
    mlflow.set_experiment("financial_risk_scoring")
    with mlflow.start_run():
        print("Training pipeline...")
        full_pipeline.fit(X_train, y_train)

        # Predictions
        y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # PR-AUC evaluation is necessary for imbalance data
        pr_auc = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Log to MLflow
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            serialization_format="pickle"
        )

        # Save pipeline locally for FastAPI serving
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(full_pipeline, model_save_path)
        print(f"Pipeline saved to {model_save_path}")

if __name__ == "__main__":
    train_model("data/creditcard.csv")