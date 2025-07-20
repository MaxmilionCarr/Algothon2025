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

# === Strategy Parameters ===
LOOKBACK = 3                    # Number of previous prices to fit the logistic regression model to
TREND_LENGTH = 2                # Number of previous prices used to calculate trend
MIN_PROB = 0.5                  # Minimum modelled probability required for a trade (note: <0.5 will bias towards buy trades)

def getMyPosition(prcSoFar):
    """
    Provide optimal positions (long/short up to $10k value) based on historical prices of 50 assets.

    Parameters:
        prcSoFar (np.array): Array of historical prices with shape (N_INST, ndays)

    Returns:
        np.array: 1D array of 50 desired positions
    """
    global currentPos, nDays, trend

    _, nDays = prcSoFar.shape # Get current day index

    if nDays < max(TREND_LENGTH,LOOKBACK):
        return currentPos # Return empty vector if not enough prices exist for calculations

    trend = getTrend(prcSoFar) # Get current trend across market once per day

    # Iterate over each instrument and compute the optimal position
    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst))

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
    slopes = []
    for j in range(N_INST):
        p = prcSoFar[j, nDays - TREND_LENGTH : nDays]
        try:
            slope = np.polyfit(np.arange(len(p)), np.log(p), 1)[0]
            slopes.append(slope)
        except np.linalg.LinAlgError:
            pass # Skip instruments with invalid slopes

    trend = np.mean(slopes) # Calculate the average of all instruments' slopes

    return trend

def getPos(prcSoFar, inst):
    """
    Compute optimal position for a single instrument.

    Parameters:
        prcSoFar (np.array): Array of historical prices with shape (N_INST, ndays)
        inst (int): Current instrument index

    Returns:
        int: Optimal position size (negative for short positions)
    """
    global currentPos, nDays

    current_price = prcSoFar[inst, -1]
    prev_pos = currentPos[inst]
    max_pos = POSLIMIT / current_price

    # === Build Y vector and X array ===

    Y = [] # Stores historical outcomes of target intrument
    X = [] # Stores historical prices of all 50 instruments

    # Iterate over a number of most recent prices, as specified by LOOKBACK
    for i in range(nDays - 1 - LOOKBACK, nDays - 1):

        # Model will be trained based on previous price change of instrument
        change = (prcSoFar[inst, i + 1]) - (prcSoFar[inst, i])
        if change > COMMRATE:
            Y.append(1)             # Positive change is outcome 1
        elif change < -COMMRATE:
            Y.append(-1)            # Negative change is outcome -1
        else:
            Y.append(0)             # Absolute change < commrate is outcome 0

        row = [prcSoFar[j, i] for j in range(N_INST)]
        X.append(row) # Add the prices for every other instrument to X

    X = np.array(X)
    Y = np.array(Y)

    # If only one outcome in training data, hold current position
    if len(np.unique(Y)) == 1:
        return int(np.sign(prev_pos) * max_pos) # Scale down to $10k size if needed

    # === Scale & Fit  Model ===

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(solver='lbfgs', max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_scaled, Y)

    # === Predict Next Outcome Using Current Prices ===

    # Get current prices of other instruments
    x_today = np.array([prcSoFar[j, nDays - 1] for j in range(N_INST)]).reshape(1, -1)
    x_today = scaler.transform(x_today)

    # Predict one out-of-sample outcome
    probs = model.predict_proba(x_today)[0]

    try:
        p_buy = probs[model.classes_ == 1]   # Probability of positive change
    except:
        p_buy = 0
    
    try:
        p_sell = probs[model.classes_ == -1] # Probability of negative change
    except:
        p_sell = 0

    # If predicted change is against the trend, signal is the previous position
    signal = 0
    if p_buy > MIN_PROB and trend > 0:
        signal = 1
    elif p_sell > MIN_PROB and trend < 0:
        signal = -1
    elif np.sign(prev_pos) == np.sign(trend):
        signal = np.sign(prev_pos)
    else:
        signal = 0

    target_pos = max_pos * signal # Take maximum position in signal direction

    return int(target_pos)