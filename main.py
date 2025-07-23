'''
Implements a logistic-regression-based trading strategy

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
trend = 0.0                     # Average trend across all instruments
historical_break_scores = np.zeros(N_INST)



# === Strategy Parameters ===
LOOKBACK = 4                    # Number of previous prices to fit the logistic regression model to
TREND_LENGTH = 3                # Number of previous prices used to calculate trend
THRESH_SCORE = 1                # Threshold for excluding instruments with unstable price behavior (based on trend break history)
VOL_WINDOW = 20                 # Number of days used in calculating rolling volatility for each instrument
TREND_WINDOW = 20               # Number of days used in calculating rolling trend break for each instrument
VOL_MULTIPLIER = 1.55           # Multiplier applied to average market volatility to define a dynamic exclusion threshold
ALPHA = 0.4                     # Smoothing factor for exponentially weighted moving average of trend breaks

# === Best Parameters ===
# LOOKBACK = 4                    
# TREND_LENGTH = 3                
# THRESH_SCORE = 1                
# VOL_WINDOW = 20               
# TREND_WINDOW = 20               
# VOL_MULTIPLIER = 1.55           
# ALPHA = 0.4  
# Score (Last 500 Days): 23.17   
# Score (Middle 500 Days): -4.59
# Score (First 500 Days): 16.72       

def update_historical_break(inst, trend_break, alpha=ALPHA):
    """
    Exponential moving average of trend break score for instrument.
    """
    global historical_break_scores
    historical_break_scores[inst] = (
        alpha * trend_break + (1 - alpha) * historical_break_scores[inst]
    )

def compute_volatility(prices, window=VOL_WINDOW):
    if len(prices) < window + 1:
        return 0.0
    log_returns = np.diff(np.log(prices[-window-1:]))
    return np.std(log_returns)

def compute_trend_break(prices, window=TREND_WINDOW):
    if len(prices) < window:
        return 0.0
    x = np.arange(window)
    y = np.log(prices[-window:])
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    diffs = y - trend_line
    return np.mean(np.abs(diffs)) / (np.std(y) + 1e-8)

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

    if nDays < max(TREND_LENGTH, LOOKBACK, VOL_WINDOW + 1):
        return currentPos  # Not enough data

    trend = getTrend(prcSoFar)  # Market trend vector

    # --- Dynamic volatility threshold ---
    vol_list = [
        compute_volatility(prcSoFar[j, :nDays])
        for j in range(N_INST)
        if nDays >= VOL_WINDOW + 1
    ]
    avg_vol = np.mean(vol_list)
    vol_threshold = avg_vol * VOL_MULTIPLIER  # Dynamic limit

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst, vol_threshold))

    return currentPos

def getTrend(prcSoFar):
    """
    Compute the current average of all instruments' trends.
    The trend is defined as the slope of a linear fit to the log prices of an instrument.

    Parameters:
        prcSoFar (np.array): Array of historical prices with shape (N_INST, ndays)

    Returns:
        float: Average trend accross instruments
    """
    # Iterate over each insturment, and in a vector, store the slope of a linear fit to the 
    # log of the most recent n prices, as specified by TREND_LENGTH
    global slope
    slopes = []
    for j in range(N_INST):
        p = prcSoFar[j, nDays - TREND_LENGTH : nDays + 1]
        try:
            slope = np.polyfit(np.arange(len(p)), np.log(p), 1)[0]
            slopes.append(slope)
        except np.linalg.LinAlgError:
            pass # Skip instruments with invalid slopes

    trend = np.mean(slopes) # Calculate the average of all instruments' slopes

    return trend

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

    current_price = prcSoFar[inst, -1]
    prev_pos = currentPos[inst]
    max_pos = POSLIMIT / current_price

    prices = prcSoFar[inst, :nDays]

    # --- Pre-checks: skip volatile or unstable instruments ---
    vol = compute_volatility(prices)
    trend_break = compute_trend_break(prices)

    update_historical_break(inst, trend_break)  # Update EMA of break score
    if vol > vol_threshold or historical_break_scores[inst] > THRESH_SCORE:
        return int(np.sign(prev_pos) * min(abs(prev_pos), max_pos))

    # === Build Y and X ===
    Y, X = [], []
    for i in range(nDays - 1 - LOOKBACK, nDays - 1):
        change = prcSoFar[inst, i + 1] - prcSoFar[inst, i]
        if change > COMMRATE:
            Y.append(1)
        elif change < -COMMRATE:
            Y.append(-1)
        else:
            Y.append(0)
        X.append([prcSoFar[j, i] for j in range(N_INST)])

    X = np.array(X)
    Y = np.array(Y)

    if len(np.unique(Y)) == 1:
        return int(np.sign(prev_pos) * min(abs(prev_pos), max_pos))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_scaled, Y)

    x_today = np.array([prcSoFar[j, nDays - 1] for j in range(N_INST)]).reshape(1, -1)
    x_today = scaler.transform(x_today)

    probs = model.predict_proba(x_today)[0]
    p_buy, p_sell = probs[1], probs[0]

    signal = 0
    if p_buy > 0.5 and trend > 0:
        signal = 1
    elif p_sell > 0.5 and trend < 0:
        signal = -1

    target_pos = max_pos * signal

    return int(np.sign(prev_pos) * min(abs(prev_pos), max_pos)) if signal == 0 else int(target_pos)
