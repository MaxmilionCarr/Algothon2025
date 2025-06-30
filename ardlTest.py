import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL

# Constants
POSLIMIT = 1000
COMMRATE = 0.0005
LOOKBACK = 100
THRESHOLD = 0.000  # for position cutoff

# Load and prepare data
price_data = np.loadtxt("priceSlice_test.txt").T  # shape (50, T)
N_INST, N_DAYS = price_data.shape
log_prices = np.log(price_data)
returns = np.diff(log_prices, axis=1)  # shape (50, T-1)

# Storage
results = []

for inst in range(N_INST):
    y_ret = returns[inst]
    pnl_series = []
    total_error = 0
    count = 0

    for t in range(LOOKBACK + 1, N_DAYS - 1):
        # Prepare endogenous and exogenous
        y_window = y_ret[t - LOOKBACK - 1:t - 1]
        X_window = returns[:, t - LOOKBACK - 1:t - 1].T  # shape (LOOKBACK, 50)

        if y_window.shape[0] != LOOKBACK or X_window.shape != (LOOKBACK, 50):
            continue

        # ARDL(1,...,1) with intercept
        order = {i: [1] for i in range(N_INST)}
        model = ARDL(endog=y_window, lags=1, exog=X_window, order=order, causal=True, trend="c")
        result = model.fit()

        # Prediction
        last_y = y_ret[t - 1]
        last_X = returns[:, t - 1]
        alpha = result.params[0]
        phi = result.params[1]
        beta = result.params[2:]

        predicted_ret = alpha + phi * last_y + np.dot(beta, last_X)
        actual_ret = y_ret[t]
        error = predicted_ret - actual_ret
        total_error += error ** 2

        # PnL logic
        price_t = price_data[inst, t]
        price_tp1 = price_data[inst, t + 1]

        if predicted_ret > THRESHOLD:
            pos = POSLIMIT / price_t
        elif predicted_ret < -THRESHOLD:
            pos = -POSLIMIT / price_t
        else:
            pos = 0

        pnl = pos * (price_tp1 - price_t) - COMMRATE * abs(pos) * price_t
        pnl_series.append(pnl)
        count += 1

    if count > 0:
        mean_pnl = np.mean(pnl_series)
        std_pnl = np.std(pnl_series)
        score = mean_pnl - 0.1 * std_pnl
        mse = total_error / count
        results.append({
            "instrument": inst,
            "total_pnl": sum(pnl_series),
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "score": score,
            "mse": mse
        })
        print(f"Inst {inst:2d} | Score: {score:.4f} | MSE: {mse:.6f} | PnL: {sum(pnl_series):.2f}")

# Save results
df = pd.DataFrame(results)
df.to_csv("ardl_strategy2_results.csv", index=False)
print("Saved results to ardl_strategy2_results.csv")