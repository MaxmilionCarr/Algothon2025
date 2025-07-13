import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LogisticRegression
from scipy.stats import skew

COMMRATE = 0.0005
POSLIMIT = 10000
N_INST = 50
currentPos = np.zeros(N_INST)
nDays = 0
best_strategy_cache = {}  # inst -> (last_eval_day, strategy_fn, params)

def getMyPosition(prcSoFar):
    global currentPos, N_INST, nDays

    N_INST, nDays = prcSoFar.shape

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst))
    #print(currentPos)
    return currentPos

def getPos(prcSoFar, inst):
    global currentPos, POSLIMIT, nDays, candidates, BACKTEST_WINDOW, MIN_SCORE_THRESH, best_strategy_cache

    if nDays <= BACKTEST_WINDOW + 1:
        return 0

    current_price = prcSoFar[inst, -1]
    prev_pos = currentPos[inst]
    max_pos = POSLIMIT / current_price

    last_eval_day, best_strategy, best_params = best_strategy_cache.get(inst, (-BACKTEST_WINDOW, None, None))

    if nDays - 1 >= last_eval_day + BACKTEST_WINDOW:
        best_score = -np.inf

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
            best_strategy_cache[inst] = (nDays - 1, None, None)
            return 0

        best_strategy_cache[inst] = (nDays - 1, best_strategy, best_params)

    if best_strategy is None:
        return 0

    best_params['self'] = inst
    best_signal = best_strategy(prcSoFar, nDays - 1, best_params)
    best_signal = np.clip(best_signal, -1, 1)
    target_pos = (POSLIMIT / current_price) * best_signal

    #print(best_strategy)         #         <------- print strategy

    if abs(best_signal) > 1e-6:
        if np.sign(target_pos) != np.sign(prev_pos):
            return target_pos
        else:
            return np.sign(prev_pos) * min(abs(prev_pos), max_pos)
    else:
        return 0
    
def backtest(prcSoFar, strategy_fn, params, backtest_window, inst):
    global POSLIMIT, COMMRATE, nDays
    params['self'] = inst

    price_series = prcSoFar[inst]
    start_main = nDays - backtest_window - 1
    end_main = nDays - 1

    pl_list = []
    last_pos = 0

    for t in range(start_main, end_main):
        price = price_series[t]
        signal = np.clip(strategy_fn(prcSoFar, t, params), -1, 1)
        position = (POSLIMIT / price) * signal

        next_ret = price_series[t + 1] - price_series[t]
        pl = position * next_ret

        if np.sign(position) != np.sign(last_pos):
            if last_pos != 0:
                pl -= COMMRATE * abs(last_pos) * price
            if position != 0:
                pl -= COMMRATE * abs(position) * price

        pl_list.append(pl)
        last_pos = position

    if not pl_list:
        return -np.inf, 0

    pl_arr = np.array(pl_list)
    mean_pl = pl_arr.mean()
    std_pl = pl_arr.std()
    score = mean_pl - 0.1 * std_pl

    # === Optional robustness checks ===
    if N_CHECKS > 0:
        for i in range(N_CHECKS):
            end_day = nDays - 1 - i * backtest_window
            start_day = end_day - backtest_window
            if start_day < 0:
                continue

            check_pl = []
            last_pos = 0
            for t in range(start_day, end_day):
                price = price_series[t]
                signal = np.clip(strategy_fn(prcSoFar, t, params), -1, 1)
                position = (POSLIMIT / price) * signal
                next_ret = price_series[t + 1] - price_series[t]
                pl = position * next_ret

                if np.sign(position) != np.sign(last_pos):
                    if last_pos != 0:
                        pl -= COMMRATE * abs(last_pos) * price
                    if position != 0:
                        pl -= COMMRATE * abs(position) * price

                check_pl.append(pl)
                last_pos = position

            check_pl = np.array(check_pl)
            if check_pl.size == 0:
                continue
            if (check_pl.mean() - 0.1 * check_pl.std()) < MIN_SCORE_THRESH: #     <----- score
                return -np.inf, 0
            # if check_pl.mean() < MIN_SCORE_THRESH:
            #     return -np.inf, 0

    latest_signal = np.clip(strategy_fn(prcSoFar, nDays - 1, params), -1, 1)
    return score, latest_signal

######################################################################

from sklearn.linear_model import LogisticRegression
import numpy as np

def strategy_1(prcSoFar, t, params):
    inst = params['self']
    lookback = params['lookback']
    window = params['window']
    n_partners = params['n_partners']
    feature_set = params['feature_set']
    use_prices = params['use_prices']
    prob_thresh = params['prob_thresh']

    if t < lookback + window:
        return 0

    y_prices = prcSoFar[inst, t - lookback - window + 1 : t + 1]

    # === Partner Selection ===
    top_partners = []
    if n_partners > 0:
        y_logr = np.diff(np.log(y_prices))
        partner_scores = []
        for j in range(prcSoFar.shape[0]):
            if j == inst:
                continue
            p = prcSoFar[j, t - lookback - window + 1 : t + 1]
            r = np.diff(np.log(p))
            if len(r) == len(y_logr):
                r_lagged = r[:-1]
                y_trimmed = y_logr[1:]
                if len(r_lagged) == len(y_trimmed):
                    c = np.corrcoef(y_trimmed, r_lagged)[0, 1]
                    partner_scores.append((j, abs(c)))
        partner_scores.sort(key=lambda x: -x[1])
        top_partners = [j for j, _ in partner_scores[:n_partners]]

    # === Dataset Construction ===
    X, Y = [], []
    for i in range(len(y_prices) - lookback - window + 1, len(y_prices) - window):
        start = i
        end = i + window + 1
        if end > len(y_prices):
            break

        y_seg = y_prices[start:end]
        volatility = np.std(np.diff(np.log(y_seg)))
        feat = extract_features(y_seg, feature_set, window)

        if use_prices:
            feat.extend(y_seg[:-1].tolist())  # Exclude last price to align with y_i

        for j in top_partners:
            p_seg = prcSoFar[j, t - lookback - window + 1 : t + 1]
            p_window = p_seg[start:end]
            p_feat = extract_features(p_window, feature_set, window)
            feat.extend(p_feat)
            if use_prices:
                feat.extend(p_window[:-1].tolist())

        r_next = np.log(y_prices[end - 1]) - np.log(y_prices[end - 2])
        if r_next > 0: #5*COMMRATE:
            y_i = 1
        elif r_next < 0: #5*-COMMRATE:
            y_i = -1
        else:
            y_i = 0

        if abs(r_next) > MAX_RET or volatility > MAX_VOL: #    <--- volatility guard
            y_i = 0

        if len(np.unique(Y)) == 1:
            if len(Y) >= 10:  # Or some tunable threshold
                return int(Y[0])  # Consistently only one class, trust it
            else:
                return 0  # Not enough data to trust

        X.append(feat)
        Y.append(y_i)

    X = np.array(X)
    Y = np.array(Y)

    model = LogisticRegression(solver='liblinear')
    model.fit(X, Y)

    # === Predict for Today ===
    x_today = extract_features(y_prices, feature_set, window)
    if use_prices:
        x_today.extend(y_prices[-window - 1:-1].tolist())

    for j in top_partners:
        p_prices = prcSoFar[j, t - window : t + 1]
        p_feat = extract_features(p_prices, feature_set, window)
        x_today.extend(p_feat)
        if use_prices:
            x_today.extend(p_prices[:-1].tolist())

    x_today = np.array(x_today).reshape(1, -1)
    probs = model.predict_proba(x_today)[0]
    labels = model.classes_
    prob_map = dict(zip(labels, probs))

    # === Use probability threshold ===
    p_buy = prob_map.get(1, 0)
    p_sell = prob_map.get(-1, 0)

    if p_buy > prob_thresh:
        return p_buy
    elif p_sell > prob_thresh:
        return -p_sell
    return 0

def extract_features(prices, feature_set, window):
    features = []
    log_returns = np.diff(np.log(prices))

    # print('Feature set:', feature_set)
    
    for feat in feature_set:
        if feat == 'mean_return':
            features.append(np.mean(log_returns))
        
        elif feat == 'volatility':
            features.append(np.std(log_returns))
        
        elif feat == 'skewness':
            features.append(skew(log_returns))
        
        elif feat == 'price_position':
            rel_pos = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
            features.append(rel_pos)
        
        elif feat == 'log_slope':
            slope = np.polyfit(np.arange(len(prices)), np.log(prices), 1)[0]
            features.append(slope)
        
        elif feat == 'autocorr':
            if len(log_returns) >= 2:
                ac = np.corrcoef(log_returns[1:], log_returns[:-1])[0, 1]
                features.append(ac)
            else:
                features.append(0.0)  # Safe default for short inputs
        
        else:
            raise ValueError(f"Unsupported feature: {feat}")
    
    return features


# === CANDIDATES ===

candidates = [
    (strategy_1, {
        'lookback': [20],
        'window': [5],
        'prob_thresh': [0.5],
        'n_partners': [0,1],
        'feature_set': [[
            'mean_return',      # Mean log return
            'volatility',       # Std dev of log return
            'skewness',         # Skew of log return
            'price_position',   # Current price in window range
            'log_slope',        # Linear slope of log(price)
            'autocorr',         # Lag-1 autocorrelation of log return
        ]],
        'use_prices': [False]
    })
]

BACKTEST_WINDOW = 5
MIN_SCORE_THRESH = 0
N_CHECKS = 5

MAX_RET = 100000000
MAX_VOL = 0.005