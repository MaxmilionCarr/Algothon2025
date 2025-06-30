import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL

# Load price data
price_data = np.loadtxt("priceSlice_test.txt").T[:, :350]  # shape (50, T)
N_INST, N_DAYS = price_data.shape
POSLIMIT = 1000
COMMRATE = 0.0005
LOOKBACK = 50

# Store results
all_results = []

for target in range(N_INST):
    y_log = np.log(price_data[target])
    y_ret = np.diff(y_log)

    log_prices = np.log(price_data)
    x_ret = np.diff(log_prices, axis=1).T  # shape (T-1, 50)

    if y_ret.shape[0] <= LOOKBACK + 1:
        continue

    pnl_series = []
    total_error = 0
    count = 0

    for t in range(LOOKBACK + 1, N_DAYS - 1):
        y = y_ret[t - LOOKBACK - 1:t - 1]
        X = x_ret[t - LOOKBACK - 1:t - 1]  # shape: LOOKBACK x 50

        order = {i: [1] for i in range(N_INST)}  # ARDL(1,...,1)
        model = ARDL(endog=y, lags=1, exog=X, order=order, causal=True, trend="c")
        result = model.fit()

        alpha = result.params[0]
        phi = result.params[1]
        beta = result.params[2:]

        last_y_ret = y_ret[t - 1]
        last_x_ret = x_ret[t - 1]

        predicted_y_ret = alpha + phi * last_y_ret + np.dot(beta, last_x_ret)
        true_y_ret = y_ret[t]
        prediction_error = predicted_y_ret - true_y_ret

        current_price = price_data[target, t + 1] if t + 1 < N_DAYS else price_data[target, t]
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
            "total_pnl": sum(pnl_series),
            "mse": mse,
            "score": score
        })
        print(f"Target: {target}, Total PnL: {sum(pnl_series):.2f}, MSE: {mse:.4f}, Score: {score:.4f}")

# Save to DataFrame
results_df = pd.DataFrame(all_results)
results_df.to_csv("full_ardl_all_predictors.csv", index=False)
print("Saved results to full_ardl_all_predictors.csv")