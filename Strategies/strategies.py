import numpy as np

NINST = 50

### Different Strategies ###

# struct prc -> price so far {
# tuple shape -> (number of instruments, number of days of data available up to current)
# float dtype -> (daily adjusted close price)
# }

# prc_t prc -> price of the instrument on day i


### Scalable Measures ###

# int lookback -> measures how many days back to calculate returns
# int scale -> position sizing constant / aggressiveness of our strat

### General Terms ###

# float signal -> pos/buy, neg/sel, zero/no buy

### Strategies ###

# Short term momentum

def short_term_momentum(prc, lookback, scale):
    if prc.shape[1] < lookback + 1:
        return np.zeros(NINST)
    ret = np.log(prc[:, -1] / prc[:, -lookback - 1])
    norm = np.linalg.norm(ret)
    signal = ret / norm if norm > 0 else ret
    return (scale * signal / prc[:, -1])

# Normalized momentum
def normalized_momentum(prc, lookback, scale):
    if prc.shape[1] < lookback + 1:
        return np.zeros(NINST)
    ret = np.log(prc[:, -1] / prc[:, -lookback - 1])
    signal = (ret - np.mean(ret)) / (np.std(ret) + 1e-8)
    return (scale * signal / prc[:, -1])

# Mean Reversion of Z-Score
def mean_reversion(prc, window, scale):
    if prc.shape[1] < window:
        return np.zeros(NINST)
    mean = np.mean(prc[:, -window:], axis=1)
    std = np.std(prc[:, -window:], axis=1) + 1e-8
    z = (prc[:, -1] - mean) / std
    return (-scale * z / prc[:, -1])

# SMA
def sma_crossover(prc, short, long, scale):
    if prc.shape[1] < long:
        return np.zeros(NINST)
    sma_s = np.mean(prc[:, -short:], axis=1)
    sma_l = np.mean(prc[:, -long:], axis=1)
    signal = np.sign(sma_s - sma_l)
    return (scale * signal)

# Volatility Breakout 
def volatility_breakout(prc, vol_window, scale, threshold):
    if prc.shape[1] < vol_window + 1:
        return np.zeros(NINST)
    ret = np.log(prc[:, -1] / prc[:, -2])
    vol = np.std(np.log(prc[:, -vol_window:] / prc[:, -vol_window-1:-1]), axis=1) + 1e-8
    signal = (ret > threshold * vol).astype(float) - (ret < -threshold * vol).astype(float)
    return (scale * signal)

# Pairs Mean Reversion 
def pairs_mean_reversion(prc, a, b, lookback, scale):
    if prc.shape[1] < lookback:
        return np.zeros(NINST)
    spread = prc[a, -lookback:] - prc[b, -lookback:]
    mean = np.mean(spread)
    std = np.std(spread) + 1e-8
    z = (spread[-1] - mean) / std
    pos = np.zeros(NINST)
    pos[a] = -scale * z
    pos[b] = scale * z
    return pos

