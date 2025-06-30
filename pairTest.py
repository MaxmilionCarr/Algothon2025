import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL
from itertools import product

# Load price data
price_data = np.loadtxt("priceSlice_test.txt").T # shape (50, T)
N_INST, N_DAYS = price_data.shape
POSLIMIT = 1000
COMMRATE = 0.0005
LOOKBACK = 50

# Store results
all_results = []

for target in range(N_INST):
    for regressor in range(N_INST):
        if target == regressor:
            continue

        y_log = np.log(price_data[target])
        x_log = np.log(price_data[regressor])
        y_ret = np.diff(y_log)
        x_ret = np.diff(x_log)

        min_len = min(len(y_ret), len(x_ret))
        if min_len <= LOOKBACK + 1:
            continue

        pnl_series = []
        total_error = 0
        count = 0

        for t in range(LOOKBACK + 1, N_DAYS - 1):
            y = y_ret[t - LOOKBACK - 1:t - 1]
            x = x_ret[t - LOOKBACK - 1:t - 1].reshape(-1, 1)

            model = ARDL(endog=y, lags=1, exog=x, order={0: [1]}, causal=True, trend="c")
            result = model.fit()

            alpha = result.params[0]
            phi = result.params[1]
            beta = result.params[2]

            last_y_ret = y_ret[t - 1]
            last_x_ret = x_ret[t - 1]

            predicted_y_ret = alpha + phi * last_y_ret + beta * last_x_ret
            true_y_ret = y_ret[t]
            prediction_error = predicted_y_ret - true_y_ret

            current_price = price_data[target, t + 1] if t + 1 < price_data.shape[1] else price_data[target, t]
            previous_price = price_data[target, t]
            price_diff = current_price - previous_price

            position = POSLIMIT * np.sign(predicted_y_ret) / previous_price
            pnl = position * price_diff - COMMRATE * abs(position) * previous_price

            pnl_series.append(pnl)
            total_error += prediction_error ** 2
            count += 1

        if count > 0:
            mean_pnl = np.mean(pnl_series)
            std_pnl = np.std(pnl_series)
            score = mean_pnl - 0.1 * std_pnl
            mse = total_error / count
            all_results.append({
                "target": target,
                "regressor": regressor,
                "total_pnl": sum(pnl_series),
                "mse": mse,
                "score": score
            })
            print(f"Target: {target}, Regressor: {regressor}, Total PnL: {sum(pnl_series):.2f}, MSE: {mse:.4f}, Score: {score:.4f}")

# Save to DataFrame
results_df = pd.DataFrame(all_results)
results_df.to_csv("pairwise_ardl_summary.csv", index=False)
print("Saved total PnL, MSE, and competition score to pairwise_ardl_summary.csv")
