import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL

# Load price data
price_data = np.loadtxt("priceSlice_test.txt").T  # shape (50, T)
N_INST, N_DAYS = price_data.shape
POSLIMIT = 1000
COMMRATE = 0.0005
LOOKBACK = 100
threshold = 0.000  # same threshold as in strategy_2

results = []

for target in range(N_INST):
    log_prices = np.log(price_data)
    returns = np.diff(log_prices, axis=1)  # shape (50, T-1)
    y_ret = returns[target]

    pnl_series = []
    total_error = 0
    count = 0

    for t in range(LOOKBACK + 1, N_DAYS - 1):
        y = y_ret[t - LOOKBACK - 1:t - 1]
        X = returns[:, t - LOOKBACK - 1:t - 1].T  # shape (LOOKBACK, 50)

        if len(y) != LOOKBACK or X.shape != (LOOKBACK, 50):
            continue

        order = {i: [1] for i in range(N_INST)}  # ARDL(1,...,1)
        model = ARDL(endog=y, lags=1, exog=X, order=order, causal=True, trend="c")
        result = model.fit()

        alpha = result.params[0]
        phi = result.params[1]
        beta = result.params[2:]

        last_y_ret = y_ret[t - 1]
        last_x_ret = returns[:, t - 1]

        predicted_y_ret = alpha + phi * last_y_ret + np.dot(beta, last_x_ret)
        actual_y_ret = y_ret[t]
        error = predicted_y_ret - actual_y_ret
        total_error += error ** 2

        # Trade logic (same as strategy_2)
        current_price = price_data[target, t]
        next_price = price_data[target, t + 1]

        if predicted_y_ret > threshold:
            pos = POSLIMIT / current_price
        elif predicted_y_ret < -threshold:
            pos = -POSLIMIT / current_price
        else:
            pos = 0

        pnl = pos * (next_price - current_price) - COMMRATE * abs(pos) * current_price
        pnl_series.append(pnl)
        count += 1

    if count > 0:
        mean_pnl = np.mean(pnl_series)
        std_pnl = np.std(pnl_series)
        score = mean_pnl - 0.1 * std_pnl
        mse = total_error / count
        results.append({
            "target": target,
            "total_pnl": sum(pnl_series),
            "mse": mse,
            "score": score
        })
        print(f"Target {target}: Score={score:.4f}, MSE={mse:.5f}, Total PnL={sum(pnl_series):.2f}")

# Save results
df = pd.DataFrame(results)
df.to_csv("strategy2_test_results.csv", index=False)
print("Saved to strategy2_test_results.csv")