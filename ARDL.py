import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL

# Load the price data
prices = np.loadtxt("priceSlice_test.txt")
prices = prices.T  # Ensure shape is (50, nt)

# Calculate log returns
log_prices = np.log(prices)
returns = np.diff(log_prices, axis=1)  # shape (50, n_days - 1)

n_inst, n_days = returns.shape

# Initialize matrices to store ARDL coefficients and p-values
ardl_coefs = np.full((n_inst, n_inst), np.nan)
ardl_pvals = np.full((n_inst, n_inst), np.nan)

# Loop through all (regressor, target) pairs
for target_idx in range(n_inst):
    for regressor_idx in range(n_inst):
        if regressor_idx == target_idx:
            continue  # Skip self-regression

        y = returns[target_idx]  # Target series (log returns)
        x = returns[regressor_idx].reshape(-1, 1)  # Regressor must be 2D

        # ARDL(1,1) with intercept, lag 1 of y, and lag 1 of x (like R dynlm)
        model = ARDL(endog=y, lags=1, exog=x, order={0: [1]}, causal=True, trend="c")
        result = model.fit()

        # First param is intercept, second is y.L1, third is x.L1
        coef = result.params[2]
        pval = result.pvalues[2]

        ardl_coefs[regressor_idx, target_idx] = coef
        ardl_pvals[regressor_idx, target_idx] = pval

# Convert to DataFrame
coef_df = pd.DataFrame(ardl_coefs, columns=[f"Target_{i}" for i in range(n_inst)], index=[f"Reg_{i}" for i in range(n_inst)])
pval_df = pd.DataFrame(ardl_pvals, columns=[f"Target_{i}" for i in range(n_inst)], index=[f"Reg_{i}" for i in range(n_inst)])

# Save to CSV
coef_df.to_csv("ardl_coefficients_log_returns.csv")
pval_df.to_csv("ardl_pvalues_log_returns.csv")

print("ARDL results on log returns saved to ardl_coefficients_log_returns.csv and ardl_pvalues_log_returns.csv")
