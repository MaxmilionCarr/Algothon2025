import numpy as np
import statsmodels.api as sm
import itertools
from sklearn.linear_model import Ridge
import time

COMMRATE = 0.0005
POSLIMIT = 10000
N_INST = 50
currentPos = np.zeros(N_INST)
nDays = 0

def getMyPosition(prcSoFar):
    global currentPos, N_INST, nDays

    N_INST, nDays = prcSoFar.shape

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst))

    return currentPos

def getPos(prcSoFar, inst):
    global currentPos, POSLIMIT, nDays, candidates, BACKTEST_WINDOW, MIN_SCORE_THRESH

    if nDays <= BACKTEST_WINDOW + 1:
        return 0

    current_price = prcSoFar[inst, -1]
    prev_pos = currentPos[inst]
    max_pos = POSLIMIT / current_price

    best_score = -np.inf
    best_strategy = None
    best_params = None

    for strategy_fn, param_grid in candidates:
        param_names = list(param_grid.keys())
        param_combos = list(itertools.product(*[param_grid[k] for k in param_names]))

        for combo in param_combos:
            params = dict(zip(param_names, combo))
            score, _ = backtest(prcSoFar, strategy_fn, params, BACKTEST_WINDOW, inst)

            if score > best_score:
                best_score = score
                best_strategy = strategy_fn
                best_params = params

    if best_score < MIN_SCORE_THRESH:
        return 0

    best_params['self'] = inst
    best_signal = best_strategy(prcSoFar, nDays - 1, best_params)
    best_signal = np.clip(best_signal, -1, 1)
    target_pos = (POSLIMIT / current_price) * best_signal

    if abs(best_signal) > 1e-6:
        if abs(target_pos - prev_pos) > 1e-6:
            return target_pos
        else:
            return np.sign(prev_pos) * min(abs(prev_pos), max_pos)
    else:
        return 0
    
def backtest(prcSoFar, strategy_fn, params, backtest_window, inst):
    global POSLIMIT, COMMRATE, nDays
    params['self'] = inst

    daily_returns = []
    last_position = 0
    for t in range(nDays - backtest_window - 1, nDays - 1):
        price = prcSoFar[inst, t]
        signal = strategy_fn(prcSoFar, t, params)
        signal = np.clip(signal, -1, 1)
        position = (POSLIMIT / price) * signal

        next_ret = prcSoFar[inst, t + 1] - prcSoFar[inst, t]
        pl = position * next_ret

        if np.sign(position) != np.sign(last_position):
            if last_position != 0:
                pl -= COMMRATE * abs(last_position) * prcSoFar[inst, t]
            if position != 0:
                pl -= COMMRATE * abs(position) * prcSoFar[inst, t]

        daily_returns.append(pl)
        last_position = position

    pl_arr = np.array(daily_returns)
    if len(pl_arr) == 0:
        return -np.inf, 0

    mean_pl = np.mean(pl_arr)
    std_pl = np.std(pl_arr)
    score = mean_pl - 0.1 * std_pl

    # === Robustness check: Subwindow score stability ===
    subwindow = backtest_window // 4
    scores = []

    for i in range(4):
        start = i * subwindow
        end = (i + 1) * subwindow
        sub_pl = pl_arr[start:end]
        if len(sub_pl) == 0:
            continue
        sub_mean = np.mean(sub_pl)
        sub_std = np.std(sub_pl)
        if (sub_mean - 0.1 * sub_std) < 0:
            return -np.inf, 0
        
    latest_signal = strategy_fn(prcSoFar, nDays - 1, params)
    latest_signal = np.clip(latest_signal, -1, 1)

    return score, latest_signal

######################################################################

def strategy_1(prcSoFar, t, params):
    inst = params['self']
    partner = params['partner']
    lookback = params['lookback']
    lags_y = params['lags_y']
    lags_x = params['lags_x']
    adjr2_thresh = params['adjr2_thresh']
    magnitude_thresh = params['magnitude_thresh']

    if partner == inst or t < lookback + max(lags_y, lags_x):
        return 0

    y_full = prcSoFar[inst, t - lookback - max(lags_y, lags_x): t]
    x_full = prcSoFar[partner, t - lookback - max(lags_y, lags_x): t]

    if len(y_full) < lookback or len(x_full) < lookback:
        return 0

    y_target = y_full[max(lags_y, lags_x):]
    y_lags = np.column_stack([y_full[max(lags_y, lags_x) - i:-i] for i in range(1, lags_y + 1)])
    x_lags = np.column_stack([x_full[max(lags_y, lags_x) - i:-i] for i in range(1, lags_x + 1)])
    X = np.column_stack([y_lags, x_lags])
    X = sm.add_constant(X)

    model = sm.OLS(y_target, X).fit()
    if model.rsquared_adj < adjr2_thresh:
        return 0

    y_recent = np.array([prcSoFar[inst, t - i] for i in range(1, lags_y + 1)])
    x_recent = np.array([prcSoFar[partner, t - i] for i in range(1, lags_x + 1)])
    features = np.concatenate([y_recent, x_recent])
    features = np.insert(features, 0, 1)

    forecast = model.predict([features])[0]
    delta = forecast - prcSoFar[inst, t]

    return np.sign(delta)

# === CANDIDATES ===
candidates = [
    (strategy_1, {
        'lookback': [20],
        'lags_y': [2],
        'lags_x': [2],
        'adjr2_thresh': [0.95],
        'magnitude_thresh': [0],
        'partner': list(range(N_INST))
    })
]

BACKTEST_WINDOW = 20
MIN_SCORE_THRESH = 10