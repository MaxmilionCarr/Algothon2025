import numpy as np
import itertools

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

    price_series = prcSoFar[inst, :]
    current_price = price_series[-1]
    prev_pos = currentPos[inst]
    max_pos = POSLIMIT / current_price

    best_score = -np.inf
    best_signal = 0

    for strategy_fn, param_grid in candidates:
        param_names = list(param_grid.keys())
        param_combos = list(itertools.product(*[param_grid[k] for k in param_names]))

        for combo in param_combos:
            params = dict(zip(param_names, combo))
            score, signal = backtest(price_series, strategy_fn, params, BACKTEST_WINDOW)

            if score > best_score:
                best_score = score
                best_signal = signal

    if best_score < MIN_SCORE_THRESH:
        return 0

    best_signal = np.clip(best_signal, -1, 1)
    target_pos = (POSLIMIT / current_price) * best_signal

    if abs(best_signal) > 1e-6:
        if abs(target_pos - prev_pos) > 1e-6:
            return target_pos  # update position
        else:
            return np.sign(prev_pos) * min(abs(prev_pos), max_pos)  # hold or resize if above max size
    else:
        return 0  # close
    
def backtest(price_series, strategy_fn, params, backtest_window):
    global POSLIMIT, COMMRATE, nDays

    daily_returns = []
    last_position = 0

    for t in range(nDays - backtest_window - 1, nDays - 1):
        price = price_series[t]
        signal = strategy_fn(price_series, t, params)
        signal = np.clip(signal, -1, 1)
        position = (POSLIMIT / price) * signal

        next_ret = price_series[t + 1] - price_series[t]
        pl = position * next_ret

        # === Commission logic ===
        if np.sign(position) != np.sign(last_position):
            if last_position != 0:
                # Exit commission
                pl -= COMMRATE * abs(last_position) * price
            if position != 0:
                # Entry commission
                pl -= COMMRATE * abs(position) * price
        # If holding, no commission

        daily_returns.append(pl)
        last_position = position

    pl_arr = np.array(daily_returns)
    if len(pl_arr) == 0:
        return -np.inf, 0

    mean_pl = np.mean(pl_arr)
    std_pl = np.std(pl_arr)
    score = mean_pl - 0.1 * std_pl

    latest_signal = strategy_fn(price_series, nDays - 1, params)
    return score, latest_signal
    
def strategy_1(prices, t, params):
    return 0

candidates = [
        (strategy_1, {})
    ]

BACKTEST_WINDOW = 50
MIN_SCORE_THRESH = 0