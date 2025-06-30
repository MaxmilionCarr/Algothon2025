import numpy as np
from collections import Counter
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Global constants
START = 0     # <-- Start trading after 10 days
COMMRATE = 0.0005  # <-- Commission rate
POSLIMIT = 10000   # <-- Dollar position limit
N_INST = 50        # <-- Number of instruments

# Global variables
currentPos = np.zeros(N_INST)

assignments = {
    0: 0,  1: 1,  2: 0,  3: 3,  4: 1,  5: 3,  6: 1,  7: 1,  8: 1,  9: 1,
    10: 1, 11: 0, 12: 0, 13: 0, 14: 1, 15: 2, 16: 1, 17: 0, 18: 2, 19: 0,
    20: 3, 21: 1, 22: 1, 23: 1, 24: 2, 25: 1, 26: 2, 27: 1, 28: 1, 29: 2,
    30: 1, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 0, 37: 3, 38: 3, 39: 0,
    40: 0, 41: 3, 42: 1, 43: 1, 44: 1, 45: 2, 46: 0, 47: 0, 48: 2, 49: 2
}

nDays = None

# Predictor relationships from ARDL (predict target using regressor)
predictors = {
    0: 31, 1: 34, 2: 21, 3: 4, 4: 19, 5: 28, 6: 3, 7: 26, 8: 29, 9: 40,
    10: 6, 11: 2, 12: 8, 13: 14, 14: 19, 15: 34, 16: 27, 17: 27, 18: 22, 19: 42,
    20: 47, 21: 36, 22: 27, 23: 4, 24: 14, 25: 11, 26: 34, 27: 9, 28: 24, 29: 20,
    30: 46, 31: 32, 32: 17, 33: 13, 34: 19, 35: 26, 36: 32, 37: 26, 38: 27, 39: 12,
    40: 34, 41: 34, 42: 31, 43: 22, 44: 41, 45: 41, 46: 36, 47: 5, 48: 8, 49: 20
}

# Strategy definitions
def strategy_0(prcSoFar, inst):
    return 0

def strategy_1(prcSoFar, inst):

    LOOKBACK = 50

    if inst not in predictors or prcSoFar.shape[1] < LOOKBACK + 2:
        return currentPos[inst]

    x_idx = predictors[inst]
    y_log = np.log(prcSoFar[inst])
    x_log = np.log(prcSoFar[x_idx])
    y_ret = np.diff(y_log)
    x_ret = np.diff(x_log)

    y = y_ret[-(LOOKBACK + 1):-1]
    x = x_ret[-(LOOKBACK + 1):-1].reshape(-1, 1)

    model = ARDL(endog=y, lags=1, exog=x, order={0: [1]}, causal=True, trend="c")
    result = model.fit()

    alpha, phi, beta = result.params[:3]
    conf_std = np.std(result.resid)

    last_y_ret = y_ret[-1]
    last_x_ret = x_ret[-1]

    current_price = prcSoFar[inst, -1]

    # Predict next return and price
    predicted_y_next = alpha + phi * last_y_ret + beta * last_x_ret
    predicted_p_next = current_price * np.exp(predicted_y_next)

    threshold = 0.0000

    if predicted_y_next > 0:
        if predicted_y_next > threshold:
            return int(POSLIMIT / current_price)
        elif currentPos[inst] > 0:
            return currentPos[inst]
        else:
            return 0
    elif predicted_y_next < 0:
        if abs(predicted_y_next) > threshold:
            return int(-POSLIMIT / current_price)
        elif currentPos[inst] < 0:
            return currentPos[inst]
        else:
            return 0
    else:
        return currentPos[inst]

def strategy_2(prcSoFar, inst):
    LOOKBACK = 100
    N_COMPONENTS = 1
    threshold_multiplier = 2

    if prcSoFar.shape[1] < LOOKBACK:
        return currentPos[inst]

    window = prcSoFar[:, -LOOKBACK:]
    log_prices = np.log(window + 1e-8)
    log_prices -= np.mean(log_prices, axis=1, keepdims=True)

    # PCA decomposition
    cov = np.cov(log_prices)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:N_COMPONENTS]]

    proj = eigvecs.T @ log_prices
    reconstructed = eigvecs @ proj
    residuals = log_prices - reconstructed

    last_resid = residuals[inst, -1]
    resid_std = np.std(residuals[inst]) * threshold_multiplier
    current_price = prcSoFar[inst, -1]

    if last_resid > resid_std:
        return int(-POSLIMIT / current_price)
    elif last_resid < -resid_std:
        return int(POSLIMIT / current_price)
    else:
        return currentPos[inst]


def strategy_3(prcSoFar, inst):
    global currentPos

    LOOKBACK = 10
    COMM = COMMRATE
    SHARPE_THRESHOLD = 2

    if prcSoFar.shape[1] < LOOKBACK + 2:
        return currentPos[inst]

    log_prices = np.log(prcSoFar[inst])
    returns = np.diff(log_prices)[-LOOKBACK:]

    model = SARIMAX(
        returns,
        order=(1, 0, 0),
        trend='c',
        initialization='approximate_diffuse'
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.filter(model.start_params)

    predicted_return = result.forecast()[0]
    resid_std = np.std(result.resid[-LOOKBACK:]) + 1e-9
    signal_strength = predicted_return / resid_std

    current_price = prcSoFar[inst, -1]
    position_size = int(POSLIMIT / current_price)

    if abs(predicted_return) < 2 * COMM or abs(signal_strength) < SHARPE_THRESHOLD:
        return currentPos[inst]

    return position_size if signal_strength > 0 else -position_size

def strategy_4(prcSoFar, inst):
    return 0

def strategy_5(prcSoFar, inst):
    return 0

# Strategy mappings
strategy_functions = {
    0: strategy_0,
    1: strategy_1,
    2: strategy_2,
    3: strategy_3,
    4: strategy_4,
    5: strategy_5,
}

# Main function
def getMyPosition(prcSoFar):
    global assignments, currentPos, N_INST, nDays

    N_INST, nDays = prcSoFar.shape

    if nDays < START:
        return currentPos

    for inst in range(N_INST):
        strat_num = assignments[inst]
        strat_func = strategy_functions[strat_num]
        desired_position = strat_func(prcSoFar, inst)
        currentPos[inst] = int(desired_position)

    return currentPos
