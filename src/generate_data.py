import os
import numpy as np
import pandas as pd

def generate_synthetic_transactions(output_path: str = "data/creditcard.csv", n_samples: int = 15000):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating { n_samples} banking transactions...")

    # 0.3% severe fraund imbalance simulate 
    n_fraud = int(n_samples * 0.003)
    n_normal = n_samples - n_fraud

    # Normal transactions: typical amounts (100 to 10000)
    normal_amounts = np.random.exponential(scale=1500, size=n_normal) + 50

    # Fraud transactions like anomalous amounts 
    fraud_amounts = np.concatenate([
        np.random.uniform(25000, 150000, size=n_fraud // 2),
        np.random.uniform(5, 50, size=n_fraud - (n_fraud // 2))
    ])

    amounts = np. concatenate([normal_amounts, fraud_amounts])
    labels = np.array([0] * n_normal + [1] * n_fraud)
    times = np.random.uniform(0., 172800, size=n_samples)

    # PCA-like latent behavioral features (V1 to V15 for quick modeling)
    # Normal customers cluster around mean 0
    features_normal = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, 15))

    # fraud transactions show anomaly shifts on specific axes
    features_fraud = np.random.normal(loc=2.5, scale=1.8, size=(n_fraud, 15))
    features = np.vstack([features_normal, features_fraud])

    # Create DataFrame
    data_dict = {"Time": times}
    for i in range(15):
        data_dict[f"V{i+1}"] = features[:, i]

    data_dict["Amount"] = amounts
    data_dict["Class"] = labels

    df = pd.DataFrame(data_dict)

    # Shuffle dataset
    df = df.sample(frac = 1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Data saved successfully at '{output_path}'!")
    print(f"Normal: {(df['Class'] == 0).sum()} | Fraud: {(df['Class'] == 1).sum()} (Imbalance: {(df['Class'].mean() * 100):.2f}%)")


if __name__ == "__main__":
    generate_synthetic_transactions()