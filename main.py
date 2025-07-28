"""
Implements a trend following and volatility filtered trading strategy.

Created by team 'Fremen' for the UNSW FinTechSoc x Susquehanna Algothon 2025.

Logic:
- Dual trend: a “fast” and “slow” trend based on avergae raw price diffs across all instruments.
- Volatility filter: skip instruments whose recent log-return vol exceeds a multiple of market average.
- Per-instrument multipliers allow inversion or exclusion.
"""

# === Import Modules === 
import numpy as np

# === Global Constants ===
COMMRATE = 0.0005
POSLIMIT = 10000
N_INST = 50

currentPos = np.zeros(N_INST)
nDays = 0
trend = 0
trendSlow = 0
historical_break_scores = np.zeros(N_INST)

TREND_LENGTH = 8
VOL_WINDOW = 26
VOL_MULTIPLIER = 1.8
multiplier = {
    0: 1,
    1: 1,
    2: 0,
    3: 1,
    4: 1,
    5: 0,
    6: -1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 0,
    16: 1,
    17: 1,
    18: 0,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
    32: 1,
    33: 1,
    34: 1,
    35: 1,
    36: 0,
    37: 1,
    38: 1,
    39: 1,
    40: 0,
    41: 1,
    42: -1,
    43: 1,
    44: 1,
    45: 1,
    46: 1,
    47: 1,
    48: 1,
    49: -1,
}

def getMyPosition(prcSoFar):
    """
    Compute optimal positions for all instruments given price history.

    Parameters:
        prcSoFar (np.ndarray): price history with shape (N_INST, nDays)

    Returns:
        np.ndarray: integer position vector (size N_INST)
    """
    global currentPos, nDays, trend, trendSlow

    _, nDays = prcSoFar.shape

    if nDays < max(TREND_LENGTH+1, VOL_WINDOW):
        return currentPos
    
    trend = getTrend(prcSoFar,TREND_LENGTH)
    trendSlow = getTrend(prcSoFar,TREND_LENGTH+1)

    volList = []

    for inst in range(N_INST):
        volList.append(getVolatility(prcSoFar[inst, :nDays]))
    avgVol = np.mean(volList)
    volThreshold = avgVol * VOL_MULTIPLIER  

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst, volThreshold))

    return currentPos

def getVolatility(prices):
    """
    Compute standard deviation of log-returns over the last VOL_WINDOW days.

    Parameters:
        prices (np.ndarray): 1D array of past prices

    Returns:
        float: volatility
    """
    logReturns = np.diff(np.log(prices[-VOL_WINDOW:]))
    return np.std(logReturns)

def getTrend(prcSoFar,trendLength):
    """
    Compute average raw price momentum across all instruments.

    Parameters:
        prcSoFar    (np.ndarray): price history (N_INST, nDays)
        trendLength (int)       : lookback window

    Returns:
        float: mean of mean(price diffs) over all instruments
    """
    slopes = []
    for inst in range(N_INST):
        prices = prcSoFar[inst, nDays - trendLength: nDays]
        slope = np.diff(prices)
        slopes.append(slope)
    return np.mean(slopes)

def getPos(prcSoFar, inst, volThreshold):
    """
    Compute a single instrument's target position.

    Parameters:
        prcSoFar      (np.ndarray): price history (N_INST, nDays)
        inst          (int)       : instrument index
        volThreshold  (float)     : maximum volatility allowed

    Returns:
        int: target number of units (signed)
    """
    currentPrice = prcSoFar[inst, -1]
    prevPos = currentPos[inst]
    maxPos = POSLIMIT / currentPrice

    mult = multiplier[inst]

    prices = prcSoFar[inst, :nDays]
    vol = getVolatility(prices)

    if vol > volThreshold:
        return 0

    if np.sign(trend) != np.sign(trendSlow):
        return int(np.sign(prevPos) * maxPos)
    else:
        signal = np.sign(trend)

    return maxPos * signal * mult