from eval import loadPrices
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pricesFile="./prices.txt"
price_array = loadPrices(pricesFile)

def plot_instrument_groups_with_market(prices: np.ndarray, group_size: int = 15, days: int = 750):
    """
    Plots instruments in groups, including a market average line.
    """
    n_inst, n_days = prices.shape
    assert days <= n_days, "Requested more days than available"

    # Compute market average (mean across all instruments)
    market_mean = prices[:, -days:].mean(axis=0)

    for start in range(0, n_inst, group_size):
        end = min(start + group_size, n_inst)

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot instruments
        for i in range(start, end):
            ax.plot(prices[i, -days:], label=f"Inst {i}", alpha=0.6)

        # Plot market line
        ax.plot(market_mean, label="Market Avg", color='black', linewidth=2.5, linestyle='--', zorder=10)

        ax.set_title(f"Prices of Instruments {start}–{end - 1} (Last {days} Days)")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2, fontsize='small')
        plt.tight_layout()
        plt.show()

plot_instrument_groups_with_market(price_array, group_size=15, days=750)