'''
Implements a ...

Created by team 'Fremen' for the UNSW FinTechSoc x Susquehanna Algothon 2025
'''

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

TREND_LENGTH = 8
VOL_WINDOW = 25
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
    global currentPos, nDays, trend, trendSlow

    _, nDays = prcSoFar.shape

    if nDays < max(TREND_LENGTH, VOL_WINDOW):
        return currentPos
    
    trend = getTrend(prcSoFar,TREND_LENGTH)
    trendSlow = getTrend(prcSoFar,TREND_LENGTH+1)

    # Calculate the average volatility of the market
    volList = []
    for inst in range(N_INST):
        volList.append(getVolatility(prcSoFar[inst, :nDays]))
    avgVol = np.mean(volList)

    # Apply a multiplier to dynamically scale the threshold of max volatility
    volThreshold = avgVol * VOL_MULTIPLIER  

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst, volThreshold))

    return currentPos

def getVolatility(prices):
    '''
    Calculates rolling volatility over the last VOL_WINDOW days

    Takes:
        prices: Array of history prices for an instrument
    
    Returns:
        float: Stanfard deviation of log return over the window
    '''

    logReturns = np.diff(np.log(prices[-VOL_WINDOW-1:]))
    return np.std(logReturns)

def getTrend(prcSoFar,trendLength):

    slopes = []
    for inst in range(N_INST):
        prices = prcSoFar[inst, nDays - trendLength: nDays]
        slope = np.diff(prices)
        slopes.append(slope)
    return np.mean(slopes)

def getPos(prcSoFar, inst, volThreshold):

    currentPrice = prcSoFar[inst, -1]
    prevPos = currentPos[inst]
    maxPos = POSLIMIT / currentPrice

    mult = multiplier[inst]

    prices = prcSoFar[inst, :nDays]

    # Fetch the individual volatility of the instrument
    vol = getVolatility(prices)

    # If a instrument is more volatile than the market and multiplier
    # reduce the position to 0
    if vol > volThreshold:
        return 0

    if np.sign(trend) != np.sign(trendSlow):
        return int(np.sign(prevPos) * maxPos)
    else:
        signal = np.sign(trend)

    return maxPos * signal * mult