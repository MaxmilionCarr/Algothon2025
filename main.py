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
trendFast = 0
historical_break_scores = np.zeros(N_INST)

TREND_LENGTH = 9
VOL_WINDOW = 24
VOL_MULTIPLIER = 1.71
multiplier = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
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
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 1,
    41: 1,
    42: 1,
    43: 1,
    44: 1,
    45: 1,
    46: 1,
    47: 1,
    48: 1,
    49: 1,
}

def getMyPosition(prcSoFar):
    global currentPos, nDays, trend, trendFast, trendSlow

    _, nDays = prcSoFar.shape

    if nDays < max(TREND_LENGTH, VOL_WINDOW):
        return currentPos
    
    trend = getTrend(prcSoFar,TREND_LENGTH)
    trendFast = getTrend(prcSoFar,TREND_LENGTH-1)

    volList = []

    for inst in range(N_INST):
        volList.append(getVolatility(prcSoFar[inst, :nDays]))
    avgVol = np.mean(volList)
    volThreshold = avgVol * VOL_MULTIPLIER  

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst, volThreshold))

    return currentPos

def getVolatility(prices):

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

    if np.sign(trend) != np.sign(trendFast):
        return 0
    else:
        signal = np.sign(trend)

    prices = prcSoFar[inst, :nDays]
    vol = getVolatility(prices)

    if vol > volThreshold:
        return int(np.sign(prevPos) * maxPos)

    return maxPos * signal * mult