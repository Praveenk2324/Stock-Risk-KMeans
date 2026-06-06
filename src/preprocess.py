import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

RAW_FILE_PATH = r"data\raw\Nifty_50_1y_dataset.csv"
PROCESSED_PATH = r"data\processed\kmeans_ready.csv"
SCALER_PATH = r"models\scaler.pkl"

def clean_and_scale(df, scaler_path):
    close_prices = df.xs(key='Close', axis=1, level=1).copy()
    close_prices.drop(columns=['LTIM.NS', 'TATAMOTORS.NS'], inplace=True)
    close_prices = close_prices.ffill().dropna(axis=0)
    daily_returns = close_prices.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    annual_volatility = daily_returns.std() * np.sqrt(252)

    clustering_data = pd.DataFrame({
        'Return': annual_returns,
        'Volatility': annual_volatility
    })

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data)

    k_means_ready = pd.DataFrame(
        scaled_features,
        index=clustering_data.index,
        columns=clustering_data.columns
    )

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f'Scaler saved to {scaler_path}')

    return k_means_ready

def main():
    print("Starting Cleaning...")
    
    df = pd.read_csv(RAW_FILE_PATH,  header=[0,1], index_col=0)

    df_clean = clean_and_scale(df, SCALER_PATH)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned dataset saved to {PROCESSED_PATH}\n")

if __name__=="__main__":
    main()