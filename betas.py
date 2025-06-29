from eval import loadPrices
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pricesFile="./prices.txt"
price_array = loadPrices(pricesFile)


def compute_betas(prc, ):
    n_inst, n_days = prc.shape
    log_prc = np.log(prc)
    betas = np.zeros(n_inst)

    if n_days < window + 1:
        return betas  # fallback: assume 0 beta

    market_returns = np.mean(np.diff(log_prc[:, -window-1:], axis=1), axis=0)

    for i in range(n_inst):
        stock_returns = np.diff(log_prc[i, -window-1:])
        cov = np.cov(stock_returns, market_returns)[0, 1]
        var_market = np.var(market_returns)
        betas[i] = cov / (var_market + 1e-8)

    return betas



fig, ax = plt.subplots(figsize=(14, 6))
for i in range(45, 51):
    ax.plot(price_array[i, -750:], label=f"Inst {i}", alpha=0.6)
ax.set_title("Prices of All 50 Instruments Over Last 200 Days")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2, fontsize='small')
plt.tight_layout()
plt.show()