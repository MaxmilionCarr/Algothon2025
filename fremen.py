import numpy as np
import statsmodels.api as sm
import itertools
from sklearn.linear_model import Ridge
import time
from sklearn.metrics import r2_score

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

    #print(best_strategy)         #         <------- print strategy

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

    # === Robustness check: Repeat full backtest window backward in time ===
    for i in range(N_CHECKS):
        end_day = nDays - 1 - i * backtest_window
        start_day = end_day - backtest_window
        if start_day < 0:
            continue

        pl_check = []
        last_pos = 0
        for t_check in range(start_day, end_day):
            price = prcSoFar[inst, t_check]
            params['self'] = inst
            signal = strategy_fn(prcSoFar, t_check, params)
            signal = np.clip(signal, -1, 1)
            position = (POSLIMIT / price) * signal

            next_ret = prcSoFar[inst, t_check + 1] - prcSoFar[inst, t_check]
            pl = position * next_ret

            if np.sign(position) != np.sign(last_pos):
                if last_pos != 0:
                    pl -= COMMRATE * abs(last_pos) * prcSoFar[inst, t_check]
                if position != 0:
                    pl -= COMMRATE * abs(position) * prcSoFar[inst, t_check]

            pl_check.append(pl)
            last_pos = position

        pl_check = np.array(pl_check)
        if len(pl_check) == 0:
            continue
        mean_check = np.mean(pl_check)
        std_check = np.std(pl_check)
        if (mean_check - 0.1 * std_check) < 0:
            return -np.inf, 0

    # Final signal for current day
    latest_signal = strategy_fn(prcSoFar, nDays - 1, params)
    latest_signal = np.clip(latest_signal, -1, 1)

    return score, latest_signal

######################################################################

def strategy_1(prcSoFar, t, params): # -8.6 most frequently traded
    return 0
    lookback = params['lookback']
    x_lags = params['x_lags']
    r2_thresh = params['r2_thresh']
    inst = params['self']

    if t < lookback + x_lags:
        return 0

    y = prcSoFar[inst, t - lookback + 1 : t + 1]

    X = []
    for lag in range(1, x_lags + 1):
        X.append(prcSoFar[inst, t - lookback + 1 - lag : t + 1 - lag])
    X = np.vstack(X).T

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    if model.rsquared < r2_thresh:
        return 0

    x_next = [prcSoFar[inst, t - lag + 1] for lag in range(1, x_lags + 1)]
    x_next = np.array([1.0] + x_next)
    y_pred = model.predict(x_next)[0]

    delta = y_pred - prcSoFar[inst, t]
    if delta > 0:
        return 1
    elif delta < 0:
        return -1
    else:
        return 0

def strategy_2(prcSoFar, t, params): # -2.08
    #return 0
    lookback = params['lookback']
    x_lags = params['x_lags']
    r2_thresh = params['r2_thresh']
    alpha = params['alpha']
    inst = params['self']

    if t < lookback + x_lags:
        return 0

    y = prcSoFar[inst, t - lookback + 1 : t + 1]

    X = []
    for lag in range(1, x_lags + 1):
        X.append(prcSoFar[inst, t - lookback + 1 - lag : t + 1 - lag])
    X = np.vstack(X).T  # shape = (lookback, x_lags)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)

    y_fit = model.predict(X)
    r2 = r2_score(y, y_fit)
    if r2 < r2_thresh:
        return 0

    x_next = [prcSoFar[inst, t - lag + 1] for lag in range(1, x_lags + 1)]
    x_next = np.array(x_next).reshape(1, -1)
    y_pred = model.predict(x_next)[0]

    delta = y_pred - prcSoFar[inst, t]
    if delta > 0:
        return 1
    elif delta < 0:
        return -1
    else:
        return 0

def strategy_3(prcSoFar, t, params): # -4.64 way too slow
    return 0

    lookback = params['lookback']
    r2_thresh = params['r2_thresh']
    inst = params['self']
    N_INST = prcSoFar.shape[0]

    if t < lookback + 1:
        return 0

    y = prcSoFar[inst, t - lookback + 1 : t + 1]                     # current instrument
    y_lag = prcSoFar[inst, t - lookback : t]                         # lagged self

    best_r2 = -np.inf
    best_peer = None

    for j in range(N_INST):
        if j == inst:
            continue
        x_lag = prcSoFar[j, t - lookback : t]

        X = np.column_stack([y_lag, x_lag])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared

        if r2 > best_r2:
            best_r2 = r2
            best_peer = j

    if best_r2 < r2_thresh:
        return 0

    # Refit ARDL(1,1) with best peer
    x_best_lag = prcSoFar[best_peer, t - lookback : t]
    X_final = np.column_stack([y_lag, x_best_lag])
    X_final = sm.add_constant(X_final)
    model_final = sm.OLS(y, X_final).fit()

    final_r2 = model_final.rsquared
    if final_r2 < r2_thresh:
        return 0

    # Forecast y_{t+1}
    y_last = prcSoFar[inst, t]
    x_last = prcSoFar[best_peer, t]
    X_next = np.array([1.0, y_last, x_last])
    y_pred = model_final.predict(X_next)[0]

    delta = y_pred - y_last
    if delta > 0:
        return 1
    elif delta < 0:
        return -1
    else:
        return 0

def strategy_4(prcSoFar, t, params): # -4.9 very fast -- add more candidates
    #return 0

    lookback = params['lookback']
    slope_thresh = params['slope_thresh']
    r2_thresh = params['r2_thresh']
    inst = params['self']

    if t < lookback:
        return 0

    y = prcSoFar[inst, t - lookback + 1 : t + 1]
    x = np.arange(lookback)

    # Fit linear trend: y = a + b*t
    b, a = np.polyfit(x, y, 1)  # slope b, intercept a
    y_pred = a + b * x

    # Compute R² manually
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    if abs(b) < slope_thresh or r2 < r2_thresh:
        return 0
    return 1 if b > 0 else -1

# === CANDIDATES ===

candidates = [
    (strategy_1, {
        'lookback': [5],
        'x_lags': [1,2,3],
        'r2_thresh': [0.95]
    }),
    (strategy_2, {
        'lookback': [5],
        'x_lags': [1,2,3],
        'r2_thresh': [0.95],
        'alpha': [1]
    }),
    (strategy_3, {
        'lookback': [5],
        'r2_thresh': [0.95]
    }),
    (strategy_4, {
        'lookback': [5],
        'slope_thresh': [0],
        'r2_thresh': [0.95]
    })
]

BACKTEST_WINDOW = 5
MIN_SCORE_THRESH = 10
N_CHECKS = 5