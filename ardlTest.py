import numpy as np
import pandas as pd
from statsmodels.tsa.api import ARDL

# === Parameters ===
PRICE_FILE = "priceSlice_test.txt"
LOOKBACK = 100
POSLIMIT = 1000
COMMRATE = 0.0005

# === Load and preprocess ===
prices = np.loadtxt(PRICE_FILE).T
n_inst, n_days = prices.shape
log_prices = np.log(prices)
returns = np.diff(log_prices, axis=1)

results = []

for inst in range(n_inst):
    y = returns[inst]
    cash = 0
    position = 0
    pnl_list = []
    total_error = 0
    count = 0

    for t in range(LOOKBACK + 1, n_days - 1):
        y_train = y[t - LOOKBACK - 1:t - 1]
        X_train = returns[:, t - LOOKBACK - 1:t - 1].T

        if y_train.shape[0] != LOOKBACK or X_train.shape != (LOOKBACK, n_inst):
            continue

        order = {i: [1] for i in range(n_inst)}
        model = ARDL(endog=y_train, lags=1, exog=X_train, order=order, trend="c", causal=True)
        result = model.fit()

        y_lag = y[t - 1]
        x_lag = returns[:, t - 1]
        alpha = result.params[0]
        phi = result.params[1]
        beta = result.params[2:]

        predicted_ret = alpha + phi * y_lag + np.dot(beta, x_lag)
        actual_ret = y[t]

        # Update PnL
        current_price = prices[inst, t]
        next_price = prices[inst, t + 1]
        target_position = POSLIMIT * np.sign(predicted_ret) / current_price
        delta_position = target_position - position
        trade_value = abs(delta_position) * current_price
        commission = trade_value * COMMRATE

        pnl = target_position * (next_price - current_price) - commission
        pnl_list.append(pnl)

        cash += pnl
        position = target_position
        total_error += (predicted_ret - actual_ret) ** 2
        count += 1

    if count > 0:
        avg_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list)
        score = avg_pnl - 0.1 * std_pnl
        mse = total_error / count
        results.append({
            "instrument": inst,
            "score": score,
            "mse": mse,
            "total_pnl": np.sum(pnl_list)
        })
        print(f"Inst {inst:2d}: Score={score:.4f}, MSE={mse:.6f}, TotalPnL={np.sum(pnl_list):.2f}")

# === Save to CSV ===
df = pd.DataFrame(results)
df.to_csv("strategy2_test_results_FIXED.csv", index=False)
print("Saved to strategy2_test_results_FIXED.csv")