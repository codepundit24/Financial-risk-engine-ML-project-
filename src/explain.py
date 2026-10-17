import joblib
import numpy as np
import pandas as pd
import shap

class RiskExplainer:
    def __init__(self, model_pipeline_path: str = "models/risk_pipeline.joblib"):
        # Load trained scikit-learn pipeline
        self.pipeline = joblib.load(model_pipeline_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["classifier"]

        # initialize TreeSHAP explainer for LightGBM
        self.explainer = shap.TreeExplainer(self.model)

    def explain_transaction(self, raw_input_dict: dict, top_k: int = 3):
        # Convert dictionary to single-row DataFrame
        df = pd.DataFrame([raw_input_dict])

        # Transformer input through scikit-learn preprocessor
        transformed_data = self.preprocessor.transform(df)

        # calculate SHAP Values
        shap_values = self.explainer.shap_values(transformed_data)

        # Handle binary classification output format
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            values = shap_values[0, :, 1]
        else:
            values = shap_values[0]

        # Extract features names from preprocessor
        feature_names = self.preprocessor.get_feature_names_out()

        # Find top risk-increasing drivers (+ve SHAP values)
        ranked_indices = np.argsort(values)[::-1][:top_k]

        top_reasons = []
        for idx in ranked_indices:
            top_reasons.append({
                "feature": str(feature_names[idx]).replace("scaler__", "").replace("passthrough__", ""),
                "shap_impact": round(float(values[idx]), 4)
            })
        return top_reasons


if __name__ == "__main__":
    # Quick sanity check
    sample_tx = {"Time": 1000, "Amount": 75000}
    for i in range(1, 16):
        sample_tx[f"V{i}"] = 2.8

    explainer = RiskExplainer()
    reasons = self_test = explainer.explain_transaction(sample_tx)
    print("Top Risk Factors: ", reasons) 