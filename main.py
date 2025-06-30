import numpy as np
from collections import Counter
from statsmodels.tsa.api import ARDL

# Global constants
START = 10     # <-- Start trading after 10 days
COMMRATE = 0.0005  # <-- Commission rate
POSLIMIT = 1000   # <-- Dollar position limit
N_INST = 50        # <-- Number of instruments

# Global variables
currentPos = np.zeros(N_INST)
assignments = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
    10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
    20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0,
    30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0,
    40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0
}
nDays = None

# Predictor relationships from ARDL (predict target using regressor)
predictors = {
    0: 25, 1: 0, 2: 27, 3: 46, 4: 19, 5: 32, 6: 35, 7: 6, 8: 35, 9: 14,
    10: 39, 11: 29, 12: 41, 13: 16, 14: 47, 15: 37, 16: 18, 17: 35, 18: 35, 19: 27,
    20: 44, 21: 12, 22: 38, 23: 39, 24: 9, 25: 23, 26: 45, 27: 6, 28: 2, 29: 6,
    30: 23, 31: 30, 32: 36, 33: 35, 34: 13, 35: 0, 36: 32, 37: 19, 38: 16, 39: 10,
    40: 39, 41: 7, 42: 47, 43: 36, 44: 41, 45: 44, 46: 25, 47: 35, 48: 38, 49: 48
}

# Strategy definitions
def strategy_0(prcSoFar, inst):
    return 0

def strategy_1(prcSoFar, inst):
    if inst not in predictors or prcSoFar.shape[1] < 5:
        return 0

    x_idx = predictors[inst]

    # Calculate log returns
    y_log = np.log(prcSoFar[inst])
    x_log = np.log(prcSoFar[x_idx])
    y_ret = np.diff(y_log)
    x_ret = np.diff(x_log)

    # Limit lookback window
    LOOKBACK = 20
    y_ret = y_ret[-LOOKBACK:]
    x_ret = x_ret[-LOOKBACK:]
    x_log = np.log(prcSoFar[x_idx])
    y_ret = np.diff(y_log)
    x_ret = np.diff(x_log)

    if len(y_ret) < 3:
        return 0

    # Align returns: y[2:] ~ y[1:-1] + x[1:-1]
    y = y_ret[2:]
    y_lag = y_ret[1:-1]
    x_lag = x_ret[1:-1]

    model = ARDL(endog=y, lags=0, exog=np.column_stack((y_lag, x_lag)), trend="c")
    result = model.fit()

    alpha = result.params[0]
    phi = result.params[1]   # y lag coefficient
    beta = result.params[2]  # x lag coefficient

    last_y_ret = y_ret[-1]
    last_x_ret = x_ret[-1]

    # Predict y_{t+1} using y_t and x_t
    predicted_y_ret = alpha + phi * last_y_ret + beta * last_x_ret

    current_price = prcSoFar[inst, -1]
    predicted_price = current_price * np.exp(predicted_y_ret)
    fair_value_diff = predicted_price - current_price

    # Proposed target position
    target_pos = POSLIMIT * np.sign(fair_value_diff) / current_price
    current_pos = currentPos[inst]

    

    return int(target_pos)

def strategy_2(prcSoFar, inst):
    return 0

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
