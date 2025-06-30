import numpy as np
from collections import Counter
from statsmodels.tsa.api import ARDL

# Global constants
START = 50     # <-- Start trading after 10 days
COMMRATE = 0.0005  # <-- Commission rate
POSLIMIT = 10000   # <-- Dollar position limit
N_INST = 50        # <-- Number of instruments

# Global variables
currentPos = np.zeros(N_INST)
assignments = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
    10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
    20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
    30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1,
    40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1
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
    if inst not in predictors or prcSoFar.shape[1] < 5:
        return currentPos[inst]

    x_idx = predictors[inst]
    y_log = np.log(prcSoFar[inst])
    x_log = np.log(prcSoFar[x_idx])
    y_ret = np.diff(y_log)
    x_ret = np.diff(x_log)

    LOOKBACK = 50
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

    if predicted_y_next > 0:
        return int(POSLIMIT / current_price)
    elif predicted_y_next < 0:
        return int(-POSLIMIT / current_price)
    else:
        return currentPos[inst]

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
