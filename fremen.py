import numpy as np
from collections import Counter
from statsmodels.tsa.api import ARDL

# Global constants
START = 0     # <-- Start trading after 10 days
COMMRATE = 0.0005  # <-- Commission rate
POSLIMIT = 1000   # <-- Dollar position limit
N_INST = 50        # <-- Number of instruments

# Global variables
currentPos = np.zeros(N_INST)
# assignments = {
#     0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1,
#     10: 1, 11: 0, 12: 1, 13: 0, 14: 1, 15: 0, 16: 1, 17: 1, 18: 0, 19: 0,
#     20: 0, 21: 1, 22: 1, 23: 1, 24: 0, 25: 1, 26: 0, 27: 1, 28: 1, 29: 0,
#     30: 1, 31: 0, 32: 1, 33: 1, 34: 0, 35: 0, 36: 0, 37: 1, 38: 0, 39: 0,
#     40: 1, 41: 0, 42: 1, 43: 1, 44: 1, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0
# }
assignments = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
    10: 0, 11: 0, 12: 0, 13: 2, 14: 2, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
    20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0,
    30: 0, 31: 0, 32: 0, 33: 2, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0,
    40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 2, 47: 0, 48: 0, 49: 0
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

    threshold = 0.000

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
    if prcSoFar.shape[1] < LOOKBACK + 2:
        return currentPos[inst]  # not enough data

    log_prices = np.log(prcSoFar[:, -LOOKBACK - 1:])
    log_returns = np.diff(log_prices, axis=1)  # shape: (50, LOOKBACK)

    y = log_returns[inst][1:]                     # endogenous
    X = log_returns[:, :-1].T                     # exogenous: shape (LOOKBACK, 50)

    order = {i: [1] for i in range(X.shape[1])}
    model = ARDL(endog=y, lags=1, exog=X, order=order, causal=True, trend="c")
    result = model.fit()

    y_lag = log_returns[inst][-1]
    x_lag = log_returns[:, -1]

    intercept = result.params[0]
    phi = result.params[1]
    betas = result.params[2:]

    predicted_ret = intercept + phi * y_lag + np.dot(betas, x_lag)

    # position logic: match test file
    price_now = prcSoFar[inst, -1]
    position = POSLIMIT * np.sign(predicted_ret) / price_now

    return int(position)

def strategy_3(prcSoFar, inst):
    return 0

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
