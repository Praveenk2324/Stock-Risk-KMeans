import pandas as pd
import plotly.express as px
import os

# Define your file paths
RESULTS_PATH = r"data/processed/clustered_stocks.csv"
PLOT_SAVE_PATH = r"reports/cluster_visualization.html"

def main():
    print("Generating Cluster Visualization...")
    
    # 1. Load the clustered data
    # We use index_col=0 to ensure the company tickers (e.g., RELIANCE.NS) remain the index
    df = pd.read_csv(RESULTS_PATH, index_col=0)
    
    # Ensure Cluster_Profile is treated as a category (so we get distinct colors, not a gradient)
    df['cluster_Profile'] = df['cluster_Profile'].astype(str)

    # 2. Create the interactive scatter plot
    fig = px.scatter(
        df,
        x="Volatility",
        y="Return",
        color="cluster_Profile",
        hover_name=df.index,  # This is the magic line! It shows the ticker when you hover.
        title="Nifty 50 Stock Clusters: Risk vs. Reward",
        labels={
            "Volatility": "Annualized Volatility (Risk - Scaled)",
            "Return": "Annualized Return (Reward - Scaled)",
            "cluster_Profile": "Assigned Cluster"
        },
        template="plotly_dark", # A sleek dark mode theme
        size_max=10
    )

    # Increase the dot size slightly for better visibility
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))

    # 3. Save the HTML file
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
    fig.write_html(PLOT_SAVE_PATH)
    print(f"Interactive plot saved to: {PLOT_SAVE_PATH}")
    
    # 4. Open the plot in your default web browser
    fig.show()

if __name__ == "__main__":
    main()