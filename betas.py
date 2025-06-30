import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eval import loadPrices

def compute_betas(prices: np.ndarray, current_day: int) -> np.ndarray:
    n_inst, n_days = prices.shape
    assert current_day <= n_days, "Invalid current_day for beta computation."

    log_prices = np.log(prices[:, :current_day])
    returns = np.diff(log_prices, axis=1)
    market_returns = returns.mean(axis=0)

    betas = []
    for i in range(n_inst):
        stock_returns = returns[i]
        cov = np.cov(stock_returns, market_returns)[0, 1]
        var = np.var(market_returns)
        beta = cov / (var + 1e-8)
        betas.append(beta)

    return np.array(betas)

def plot_beta_bins(prices: np.ndarray, betas: np.ndarray, current_day: int, bins: list = None):
    if bins is None:
        bins = [-np.inf, -1.5, -0.5, 0, 0.5, 1.0, 1.5, np.inf]

    bin_labels = [f"Bin{i+1}" for i in range(len(bins) - 1)]
    beta_bins = pd.cut(betas, bins=bins, labels=bin_labels)
    grouped = {label: [] for label in bin_labels}
    for i, label in enumerate(beta_bins):
        grouped[label].append(i)

    days = np.arange(current_day)
    market_line = prices[:, :current_day].mean(axis=0)

    for label in bin_labels:
        fig, ax = plt.subplots(figsize=(12, 4))
        for i in grouped[label]:
            ax.plot(days, prices[i, :current_day], alpha=0.6, label=f"Inst {i}")
        ax.plot(days, market_line, label="Market Avg", color='black', linewidth=2.5, linestyle='--', zorder=10)
        ax.set_title(f"{label} (β ∈ [{bins[bin_labels.index(label)]:.2f}, {bins[bin_labels.index(label)+1]:.2f}])")
        ax.set_ylabel("Price")
        ax.set_xlabel("Day")
        ax.grid(True)
        ax.legend(loc='upper left', fontsize="x-small", ncol=4)
        plt.tight_layout()
        plt.show()



pricesFile="./prices.txt"
price_array = loadPrices(pricesFile)

betas = compute_betas(price_array, current_day=50)
plot_beta_bins(price_array, betas, current_day=50)