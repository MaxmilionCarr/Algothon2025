import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

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
        return np.zeros(nInst)
    ret = np.log(prc[:, -1] / prc[:, -lookback - 1])
    norm = np.linalg.norm(ret)
    signal = ret / norm if norm > 0 else ret
    return (scale * signal / prc[:, -1])

# Normalized momentum
def normalized_momentum(prc, lookback, scale):
    if prc.shape[1] < lookback + 1:
        return np.zeros(nInst)
    ret = np.log(prc[:, -1] / prc[:, -lookback - 1])
    signal = (ret - np.mean(ret)) / (np.std(ret) + 1e-8)
    return (scale * signal / prc[:, -1])

# Mean Reversion of Z-Score
def mean_reversion(prc, window, scale):
    if prc.shape[1] < window:
        return np.zeros(nInst)
    mean = np.mean(prc[:, -window:], axis=1)
    std = np.std(prc[:, -window:], axis=1) + 1e-8
    z = (prc[:, -1] - mean) / std
    return (-scale * z / prc[:, -1])

# SMA
def sma_crossover(prc, short, long, scale):
    if prc.shape[1] < long:
        return np.zeros(nInst)
    sma_s = np.mean(prc[:, -short:], axis=1)
    sma_l = np.mean(prc[:, -long:], axis=1)
    signal = np.sign(sma_s - sma_l)
    return (scale * signal)

# Volatility Breakout 
def volatility_breakout(prc, vol_window, scale, threshold):
    if prc.shape[1] < vol_window + 1:
        return np.zeros(nInst)
    ret = np.log(prc[:, -1] / prc[:, -2])
    vol = np.std(np.log(prc[:, -vol_window:] / prc[:, -vol_window-1:-1]), axis=1) + 1e-8
    signal = (ret > threshold * vol).astype(float) - (ret < -threshold * vol).astype(float)
    return (scale * signal)

# Pairs Mean Reversion 
def pairs_mean_reversion(prc, a, b, lookback, scale):
    if prc.shape[1] < lookback:
        return np.zeros(nInst)
    spread = prc[a, -lookback:] - prc[b, -lookback:]
    mean = np.mean(spread)
    std = np.std(spread) + 1e-8
    z = (spread[-1] - mean) / std
    pos = np.zeros(nInst)
    pos[a] = -scale * z
    pos[b] = scale * z
    return pos


def getMyPosition(prcSoFar : np.ndarray) -> np.ndarray:
    global currentPos
    pos = np.zeros(nInst)

    ## Define inputs ##
    lookback = 10 # Amount of days before current to compute momentum >= 1
    scale = 2000 # Aggressiveness of the strategy >= 1
    window = 10 # Rolling window for mean and std >= 1
    short = 5 # Short-term moving average window 1<x<=20
    long = 30 # Long-term moving average window 30<x<=60
    vol_window = 5 #days for volatility estimation >=1
    threshold = 1 # number of stds movement that must exceed to trigger entry >= 0
    a = 0 # first of 2 instrument to trade 0<=x<=49
    b = 49 # second of 2 instrument to trade 0<=x<=49


    # Combine multiple strategies
    pos += short_term_momentum(prcSoFar, lookback, scale=scale)
    #pos += normalized_momentum(prcSoFar, lookback, scale=scale)
    #pos += mean_reversion(prcSoFar, window, scale=scale)
    #pos += sma_crossover(prcSoFar, short, long, scale)
    #pos += volatility_breakout(prcSoFar, vol_window, scale, threshold)
    #pos += pairs_mean_reversion(prcSoFar, a, b, lookback, scale)

    # Apply position limits
    curPrices = prcSoFar[:, -1]
    posLimits = (10000 / curPrices).astype(int)
    newPos = np.clip(pos, -posLimits, posLimits)

    currentPos = newPos.astype(int)
    return currentPos
