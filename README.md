# Financial K-Means Clustering: Nifty 50 Stocks

This project applies **K-Means Clustering** to segment the **Nifty 50** companies based on their financial performance over the past year. By analyzing the relationship between **Risk (Annualized Volatility)** and **Reward (Annualized Return)**, the model groups stocks into distinct clusters with similar profiles.

> **Note:** LTIMindtree (`LTIM.NS`) and Tata Motors (`TATAMOTORS.NS`) have been excluded. 

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Results & Visualization](#results--visualization)

## Project Overview

The objective of this project is to use unsupervised machine learning (K-Means) to discover hidden patterns in the stock market. Investors can use these clusters to diversify their portfolios by identifying which stocks behave similarly in terms of volatility and returns.

## Dataset

- **Source:** 1 year of daily historical data for Nifty 50 companies.
- **Raw Data:** Located in `data/raw/Nifty_50_1y_dataset.csv`.
- **Exclusions:** `LTIMindtree` and `Tata Motors` are dropped from the dataset to refine the clustering.

## Methodology

The machine learning pipeline is orchestrated using **DVC (Data Version Control)** and consists of three main stages:

1. **Preprocessing (`src/preprocess.py`):**
   - Loads the raw daily closing prices.
   - Drops `LTIM.NS` and `TATAMOTORS.NS`.
   - Calculates **Annualized Returns** (Reward) and **Annualized Volatility** (Risk) based on daily percentage changes.
   - Standardizes the features using `StandardScaler` to ensure the K-Means algorithm computes distances accurately.
   - Exports `kmeans_ready.csv` and the fitted `scaler.pkl`.

2. **Model Training (`src/train.py`):**
   - Applies the **K-Means** clustering algorithm with `k=3` (3 distinct clusters).
   - Assigns a cluster profile to each stock.
   - Saves the trained model as `kmeans_model.pkl` and outputs the labeled dataset to `clustered_stocks.csv`.

3. **Visualization (`src/visualize.py`):**
   - Generates an interactive scatter plot using **Plotly**.
   - Visualizes the stocks based on their Scaled Volatility (X-axis) vs Scaled Return (Y-axis), colored by their assigned cluster.
   - Exports the plot to `reports/cluster_visualization.html`.

## Project Structure

```text
Financial_KMeans/
├── data/
│   ├── raw/                 # Original 1-year Nifty 50 dataset
│   └── processed/           # Scaled features and clustered outputs
├── models/                  # Saved models (scaler.pkl, kmeans_model.pkl)
├── Notebooks/               # Jupyter notebooks for EDA and data fetching
├── reports/                 # Generated visualizations (HTML plots)
├── src/                     # Source code for the pipeline
│   ├── preprocess.py        # Data cleaning and feature engineering
│   ├── train.py             # K-Means model training
│   └── visualize.py         # Plotly visualization script
├── dvc.yaml                 # DVC pipeline configuration
└── README.md                # Project documentation
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Financial_KMeans
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   *(Ensure you have `pandas`, `numpy`, `scikit-learn`, `plotly`, `joblib`, and `dvc` installed)*
   ```bash
   pip install pandas numpy scikit-learn plotly joblib dvc
   ```

## Running the Pipeline

This project uses DVC to manage the machine learning pipeline. You can reproduce the entire workflow with a single command:

```bash
dvc repro
```

This command will automatically execute `preprocess.py`, `train.py`, and `visualize.py` in the correct order based on their dependencies defined in `dvc.yaml`.

Alternatively, you can run the scripts manually:
```bash
python src/preprocess.py
python src/train.py
python src/visualize.py
```

## Results & Visualization

After running the pipeline, open the generated HTML file in your web browser to explore the clusters interactively:

```text
reports/cluster_visualization.html
```

Hover over the data points to see individual stock tickers and their exact risk/reward metrics. 
*(Cluster characteristics depend on the specific 1-year data window).*
