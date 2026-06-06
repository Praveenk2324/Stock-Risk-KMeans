from sklearn.cluster import KMeans
import os
import pandas as pd
import joblib

PROCESSED_PATH = r"data\processed\kmeans_ready.csv"
MODEL_PATH = r"models\kmeans_model.pkl" 
RESULTS_PATH = r"data\processed\clustered_stocks.csv"

def train_kmeans(df, k=3):
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(df)

    labels = kmeans.labels_

    return kmeans, labels

def main():
    print("Starting Model Training...")

    df_ready = pd.read_csv(PROCESSED_PATH, index_col=0)

    model, labels = train_kmeans(df_ready, k=3)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"K-Means Model saved to {MODEL_PATH}")

    df_ready['cluster_Profile'] = labels

    df_ready.to_csv(RESULTS_PATH, index=True)
    print(f"Clustered dataset saved to {RESULTS_PATH}\n")

    print("Stocks per cluster:")
    print(df_ready['cluster_Profile'].value_counts())

if __name__ == "__main__":
    main()