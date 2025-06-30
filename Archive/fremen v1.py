import numpy as np
import statsmodels.api as sm
from collections import Counter

# Global constants
REASSIGN = 20      # <-- Reassign strategies every 20 days
LOOKBACK = 60       # <-- Lookback window for evaluating strategies

# Global variables
nInst = 50
currentPos = np.zeros(nInst)
assignments = None
nDays = None
mean_price_lookback = None
std_price_lookback = None
cross_section_z = None
volatility = None
corr_matrix = None
ols_models = {}
commrate = 0.0005
dlrPosLimit = 10000

def strategy_0(prcSoFar, inst): # No Strategy (zero position size)
    return 0

def strategy_1(prcSoFar, inst): # Mean Reversion (Individual)
    #print("Executing strategy 1 for instrument:", inst)
    current_price = prcSoFar[inst, -1]
    mean = mean_price_lookback[inst]
    std = std_price_lookback[inst]

    max_position = dlrPosLimit // current_price

    threshold = 0.5

    if current_price < mean - threshold * std:
        return max_position
    elif current_price > mean + threshold * std:
        return -max_position
    else:
        return currentPos[inst] 

def strategy_2(prcSoFar, inst): # Cross-Sectional Mean Reversion
    #print("Executing strategy 2 for instrument:", inst)
    z = cross_section_z[inst]
    current_price = prcSoFar[inst, -1]

    max_position = dlrPosLimit // current_price

    threshold = 0.1

    if z < -threshold:
        return max_position
    elif z > threshold:
        return -max_position
    else:
        return currentPos[inst] 

def strategy_3(prcSoFar, inst): # Volatility Filtered Reversion
    #print("Executing strategy 3 for instrument:", inst)
    if volatility[inst] > 0.02:
        return currentPos[inst] 

    current_price = prcSoFar[inst, -1]
    mean = mean_price_lookback[inst]
    std = std_price_lookback[inst]
    max_position = dlrPosLimit // current_price

    threshold = 0.5

    if current_price < mean - threshold * std:
        return max_position
    elif current_price > mean + threshold * std:
        return -max_position
    else:
        return currentPos[inst] 

def strategy_4(prcSoFar, inst): 
    #print("Executing strategy 4 for instrument:", inst)
    beta, residual = ols_models[inst]
    current_resid = residual[-1]
    resid_std = np.std(residual)

    if resid_std == 0:
        return currentPos[inst]  # Maintain current position

    z_score = current_resid / resid_std
    current_price = prcSoFar[inst, -1]
    max_position = dlrPosLimit // current_price

    threshold = 0.1

    if z_score > threshold:
        return -max_position
    elif z_score < -threshold:
        return max_position
    else:
        return currentPos[inst] 

def test_0(prcSoFar, inst):
    return 0

def test_1(prcSoFar, inst):
    window = min(LOOKBACK, nDays)
    prc = prcSoFar[inst, -window:]
    ret = prc[1:] - prc[:-1]

    positions = []
    for t in range(1, window):
        pos = strategy_1(prcSoFar[:, -(window - t):], inst)
        positions.append(pos)

    positions = np.array(positions)
    prev_positions = np.insert(positions[:-1], 0, currentPos[inst])
    prices = prc[1:]
    pnl = positions * ret - commrate * np.abs(positions - prev_positions) * prices

    score = np.mean(pnl) - 0.1 * np.std(pnl)
    return score

def test_2(prcSoFar, inst):
    window = min(LOOKBACK, nDays)
    prc = prcSoFar[inst, -window:]
    ret = prc[1:] - prc[:-1]

    positions = []
    for t in range(1, window):
        pos = strategy_2(prcSoFar[:, -(window - t):], inst)
        positions.append(pos)

    positions = np.array(positions)
    prev_positions = np.insert(positions[:-1], 0, currentPos[inst])
    prices = prc[1:]
    pnl = positions * ret - commrate * np.abs(positions - prev_positions) * prices

    score = np.mean(pnl) - 0.1 * np.std(pnl)
    return score

def test_3(prcSoFar, inst):
    window = min(LOOKBACK, nDays)
    prc = prcSoFar[inst, -window:]
    ret = prc[1:] - prc[:-1]

    positions = []
    for t in range(1, window):
        pos = strategy_3(prcSoFar[:, -(window - t):], inst)
        positions.append(pos)

    positions = np.array(positions)
    prev_positions = np.insert(positions[:-1], 0, currentPos[inst])
    prices = prc[1:]
    pnl = positions * ret - commrate * np.abs(positions - prev_positions) * prices

    score = np.mean(pnl) - 0.1 * np.std(pnl)
    return score

def test_4(prcSoFar, inst):
    window = min(LOOKBACK, nDays)
    prc = prcSoFar[inst, -window:]
    ret = prc[1:] - prc[:-1]

    positions = []
    for t in range(1, window):
        pos = strategy_4(prcSoFar[:, -(window - t):], inst)
        positions.append(pos)

    positions = np.array(positions)
    prev_positions = np.insert(positions[:-1], 0, currentPos[inst])
    prices = prc[1:]
    pnl = positions * ret - commrate * np.abs(positions - prev_positions) * prices

    score = np.mean(pnl) - 0.1 * np.std(pnl)
    return score

# Mappings
strategy_functions = {
    0: strategy_0,
    1: strategy_1,
    2: strategy_2,
    3: strategy_3,
    4: strategy_4,
}

test_functions = {
    0: test_0,
    1: test_1,
    2: test_2,
    3: test_3,
    4: test_4,
}

# Assignment function
def assign_strategies(prcSoFar):
    global nInst, nDays, assignments
    global mean_price_lookback, std_price_lookback, cross_section_z
    global volatility, corr_matrix, ols_models

    prc_window = prcSoFar[:, -min(LOOKBACK, nDays):]
    assignments = {}

    # Mean and std per instrument
    mean_price_lookback = np.mean(prc_window, axis=1)
    std_price_lookback = np.std(prc_window, axis=1)

    # Cross-sectional z-scores at latest day
    latest_prices = prc_window[:, -1]
    cross_section_z = (latest_prices - np.mean(latest_prices)) / np.std(latest_prices)

    # Volatility of returns
    returns = prc_window[:, 1:] / prc_window[:, :-1] - 1
    volatility = np.std(returns, axis=1)

    # Correlation matrix
    corr_matrix = np.corrcoef(prc_window)

    # Pairwise OLS regressions
    ols_models = {}
    for i in range(nInst):
        best_corr = -1
        best_j = None
        for j in range(nInst):
            if i == j:
                continue
            corr = np.corrcoef(prc_window[i], prc_window[j])[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_j = j
        if best_j is not None:
            x = prc_window[best_j]
            y = prc_window[i]
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            beta = model.params[1]
            residual = y - model.predict(x)
            ols_models[i] = (beta, residual)

    for inst in range(nInst):
        scores = []
        for strat_num in range(0, 5):
            scores.append(test_functions[strat_num](prcSoFar, inst))
        assignments[inst] = np.argmax(scores)

    # Count strategy assignments
    assignment_counts = Counter(assignments.values())
    print("Strategy assignment counts:", assignment_counts)

# Main function
def getMyPosition(prcSoFar):
    global assignments, currentPos, nInst, nDays

    nInst, nDays = prcSoFar.shape

    if nDays < REASSIGN:
        return currentPos

    if (nDays - 1) % REASSIGN == 0 or assignments is None:
        assign_strategies(prcSoFar)

    for inst in range(nInst):
        strat_num = assignments[inst]
        strat_func = strategy_functions[strat_num]
        desired_position = strat_func(prcSoFar, inst)
        currentPos[inst] = int(desired_position)

    return currentPos