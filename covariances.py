import numpy as np
import pandas as pd

# Load price data
price_data = np.loadtxt("priceSlice_test.txt").T  # shape (50, T)

# Compute log returns
log_prices = np.log(price_data)
log_returns = np.diff(log_prices, axis=1)

# Compute correlation matrix
corr_matrix = np.corrcoef(log_returns)

# Set self-correlations to NaN
np.fill_diagonal(corr_matrix, np.nan)

# Format as DataFrame
corr_df = pd.DataFrame(
    corr_matrix,
    index=[f"Inst_{i}" for i in range(50)],
    columns=[f"Inst_{i}" for i in range(50)]
)

# Save to CSV
corr_df.to_csv("correlation_of_returns.csv")

# Optional: print a sample
print(corr_df.iloc[:5, :5])