'''
Implements a trend-following trading strategy

Created by team 'Fremen' for the UNSW FinTechSoc x Susquehanna Algothon 2025
'''
# === Import Modules === 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

# === Global Constants ===
COMMRATE = 0.0005               # Commission rate per trade (5bps)
POSLIMIT = 10000                # Maximum position value per instrument ($)
N_INST = 50                     # Number of instruments

# === Global State ===
currentPos = np.zeros(N_INST)   # Current position vector
nDays = 0                       # Current day index
trend = []                      # Average trend across all instruments
historical_break_scores = np.zeros(N_INST)

# === Strategy Parameters ===
TREND_LENGTH = 10               # Number of previous prices used to calculate trend 
THRESH_SCORE = 0.85             # Threshold for excluding instruments with unstable price behavior (based on trend break history)
VOL_WINDOW = 25                 # Number of days used in calculating rolling volatility for each instrument 
TREND_BREAK_WINDOW = 10         # Number of days used in calculating rolling trend break for each instrument 
VOL_MULTIPLIER = 1.547          # Multiplier applied to average market volatility to define a dynamic exclusion threshold
ALPHA = 0.35                    # Smoothing factor for trend breaks

TREND_BREAK_WINDOW = TREND_LENGTH        #  <----- OPTIONAL

exclude = {
    0: False,
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
    6: False,
    7: False,
    8: False,
    9: False,
    10: False,
    11: False,
    12: False,
    13: False,
    14: False,
    15: False,
    16: False,
    17: False,
    18: False,
    19: False,
    20: False,
    21: False,
    22: False,
    23: False,
    24: False,
    25: False,
    26: False,
    27: False,
    28: False,
    29: False,
    30: False,
    31: False,
    32: False,
    33: False,
    34: False,
    35: False,
    36: False,
    37: False,
    38: False,
    39: False,
    40: False,
    41: False,
    42: False,
    43: False,
    44: False,
    45: False,
    46: False,
    47: False,
    48: False,
    49: False,
}

def getMyPosition(prcSoFar):
    """
    Provide optimal positions (long/short up to $10k value) based on historical prices of 50 assets.

    Parameters:
        prcSoFar (np.array): Array of historical prices with shape (N_INST, ndays)

    Returns:
        np.array: 1D array of 50 desired positions
    """
    global currentPos, nDays, trend

    _, nDays = prcSoFar.shape  # Get current day index

    if nDays < max(TREND_LENGTH, VOL_WINDOW + 1):
        return currentPos  # Not enough data

    trend.append(getTrend(prcSoFar))

    # === Dynamic volatility threshold ===

    # Calculate the total volatility of all instruments
    vol_list = [
        compute_volatility(prcSoFar[j, :nDays])
        for j in range(N_INST)
        if nDays >= VOL_WINDOW + 1
    ]
    avg_vol = np.mean(vol_list)
    # Applies a multiplier to the average market volatility, any stock with
    # a volatility higher than this will be tossed
    vol_threshold = avg_vol * VOL_MULTIPLIER  

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst, vol_threshold))

    return currentPos

def update_historical_break(inst, trend_break):
    '''
    Applies a smoothing to the value of the trend break based on how the
    instrument performed in the past 
    Higher values prioritize recent trend breaks while lower values prioritize
    historical trend breaks
    '''
    global historical_break_scores
    historical_break_scores[inst] = (
        ALPHA * trend_break + (1 - ALPHA) * historical_break_scores[inst]
    )

def compute_volatility(prices):
    '''
    Computes the volatility of an instrument over a certain window
    '''
    if len(prices) < VOL_WINDOW + 1:
        return 0.0
    log_returns = np.diff(np.log(prices[-VOL_WINDOW-1:]))
    return np.std(log_returns)

def compute_trend_break(prices):
    '''
    Applies a value to the difference between the actual value
    of an instrument to the expectation
    '''
    if len(prices) < TREND_BREAK_WINDOW:
        return 0.0
    x = np.arange(TREND_BREAK_WINDOW)
    y = np.log(prices[-TREND_BREAK_WINDOW:])
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    diffs = y - trend_line
    return np.mean(np.abs(diffs)) / (np.std(y) + 1e-8)

def getTrend(prcSoFar):
    slopes = []
    for j in range(N_INST):
        p = prcSoFar[j, nDays - TREND_LENGTH: nDays + 1]
        slope = np.polyfit(np.arange(len(p)), np.log(p), 1)[0]
        slope = np.diff(p)
        slopes.append(slope)
    return np.mean(slopes)

def getPos(prcSoFar, inst, vol_threshold):
    """
    Compute optimal position for a single instrument with volatility and trend-break screening.

    Parameters:
        prcSoFar (np.array): Array of historical prices
        inst (int): Instrument index
        vol_threshold (float): Dynamic upper limit for volatility

    Returns:
        int: Desired position
    """
    global currentPos, nDays

    if exclude[inst]:
        return 0          
        return int(np.sign(prev_pos) * min(abs(prev_pos), max_pos))     # <------- OPTIONAL

    current_price = prcSoFar[inst, -1]
    prev_pos = currentPos[inst]
    max_pos = POSLIMIT / current_price

    prices = prcSoFar[inst, :nDays]

    # === Skip volatile or unstable instruments ===
    vol = compute_volatility(prices)
    trend_break = compute_trend_break(prices)

    update_historical_break(inst, trend_break)

    # Two checks to ensure highly volatile stocks (Based on market volatility) aren't traded
    # Or instruments with a trend difference (scaled based on the alpha smoothing value) higher 
    # than the threshold "trend break" score will not be traded
    if vol > vol_threshold or historical_break_scores[inst] > THRESH_SCORE:
        return int(np.sign(prev_pos) * min(abs(prev_pos), max_pos))
        return 0                                                        # <------- OPTIONAL
    
    current_price = prcSoFar[inst, -1]
    max_pos = POSLIMIT / current_price

    return max_pos * np.sign(trend[-1])
