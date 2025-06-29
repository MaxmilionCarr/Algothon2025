from eval import loadPrices
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pricesFile="./prices.txt"
price_array = loadPrices(pricesFile)

period = 200 # Period in days





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