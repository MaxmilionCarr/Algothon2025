import numpy as np
import statsmodels.api as sm
from collections import Counter
from statsmodels.tsa.api import ARDL
from sklearn.linear_model import LinearRegression


# Global constants
REASSIGN = 20      # <-- Reassign strategies every 20 days
LOOKBACK = 20      # <-- Lookback window for evaluating strategies

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
dlrPosLimit = 1000
ardl_models = {}
PRED_THRESHOLD = 0.0

def strategy_0(prcSoFar, inst): # No Strategy (zero position size)
    return 0

def strategy_1(prcSoFar, inst): #
    return 0

def strategy_2(prcSoFar, inst): #
    return 0

def strategy_3(prcSoFar, inst): #
    return 0

def strategy_4(prcSoFar, inst): #
    return 0

    
def strategy_5(prcSoFar, inst): #
    return 0

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

def test_5(prcSoFar, inst):
    window = min(LOOKBACK, nDays)
    prc = prcSoFar[inst, -window:]
    ret = prc[1:] - prc[:-1]

    positions = []
    for t in range(1, window):
        pos = strategy_5(prcSoFar[:, -(window - t):], inst)
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
    5: strategy_5,
}

test_functions = {
    0: test_0,
    1: test_1,
    2: test_2,
    3: test_3,
    4: test_4,
    5: test_5,
}

# Assignment function
def assign_strategies(prcSoFar):
    global nInst, nDays, assignments
    global mean_price_lookback, std_price_lookback, cross_section_z
    global volatility, corr_matrix, ols_models

    prc_window = prcSoFar[:, -min(LOOKBACK, nDays):]
    assignments = {}

    for inst in range(nInst):
        scores = []
        for strat_num in range(0, 6):
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