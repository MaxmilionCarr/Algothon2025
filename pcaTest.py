import numpy as np
import pandas as pd

# Load price data from actual file
price_data = pd.read_csv("priceSlice_test.txt", delim_whitespace=True, header=None)
prcAll = price_data.values.T  # Shape should be (50, 750)

# Constants
LOOKBACK = 20
POSLIMIT = 10000
threshold_multiplier = 2
N_COMPONENTS = 5
commRate = 0.0005

n_inst, n_days = prcAll.shape
positions = np.zeros((n_inst, n_days))
cash = np.zeros(n_inst)
value = np.zeros(n_inst)
PnL_list = [[] for _ in range(n_inst)]

# Backtest loop
for t in range(LOOKBACK, n_days - 1):
    window = prcAll[:, t - LOOKBACK:t]
    log_prices = np.log(window + 1e-8)
    log_prices -= np.mean(log_prices, axis=1, keepdims=True)

    cov = np.cov(log_prices)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:N_COMPONENTS]]

    proj = eigvecs.T @ log_prices
    reconstructed = eigvecs @ proj
    residuals = log_prices - reconstructed
    last_resid = residuals[:, -1]
    thresholds = np.std(residuals, axis=1) * threshold_multiplier
    prices_today = prcAll[:, t]
    prices_next = prcAll[:, t + 1]

    for inst in range(n_inst):
        if last_resid[inst] > thresholds[inst]:
            new_pos = int(-POSLIMIT / prices_today[inst])
        elif last_resid[inst] < -thresholds[inst]:
            new_pos = int(POSLIMIT / prices_today[inst])
        else:
            new_pos = positions[inst, t]  # hold current position

        delta_pos = new_pos - positions[inst, t]
        dvolume = abs(delta_pos) * prices_today[inst]
        comm = dvolume * commRate

        cash[inst] -= delta_pos * prices_today[inst] + comm
        positions[inst, t + 1] = new_pos
        value[inst] = cash[inst] + new_pos * prices_next[inst]
        PnL_list[inst].append(value[inst])

# Final PnL results
results = []
for inst in range(n_inst):
    pnl = PnL_list[inst]
    total_pnl = pnl[-1] if pnl else 0
    mean_pnl = np.mean(np.diff(pnl)) if len(pnl) > 1 else 0
    std_pnl = np.std(np.diff(pnl)) if len(pnl) > 1 else 0
    score = mean_pnl - 0.1 * std_pnl
    results.append((inst, total_pnl, score))

df_results = pd.DataFrame(results, columns=["Instrument", "TotalPnL", "Score"])
df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)

print(df_results.to_string(index=False))
print("\\nAverage TotalPnL: {:.2f}".format(df_results['TotalPnL'].mean()))
print("Average Score: {:.4f}".format(df_results['Score'].mean()))