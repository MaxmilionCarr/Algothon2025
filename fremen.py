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
best_strategy_cache = {}
cached_features = {}

def get_most_volatile_instruments(prices: np.ndarray, window=20, top_k=5):
    returns = np.diff(np.log(prices[:, -window:]), axis=1)
    vol = np.std(returns, axis=1)
    return np.argsort(vol)[-top_k:]

def getMyPosition(prcSoFar):
    global currentPos, N_INST, nDays

    N_INST, nDays = prcSoFar.shape

    preCalcs(prcSoFar, nDays)

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst))
    # Volatile instrument logic with backtested parameter selection
    top_volatile = get_most_volatile_instruments(prcSoFar, window=20, top_k=5)

    for inst in top_volatile:
        best_score = -np.inf
        best_signal = 0

        for strategy_fn, param_grid in candidates:
            if strategy_fn.__name__ != "strategy_volatility":
                continue

            param_names = list(param_grid.keys())
            combos = list(itertools.product(*[param_grid[k] for k in param_names]))

            for combo in combos:
                params = dict(zip(param_names, combo))
                params["self"] = inst

                if nDays < params["vol_window"] + 1:
                    continue

                score, signal = backtest(prcSoFar, strategy_fn, params, BACKTEST_WINDOW, inst)

                if score > best_score:
                    best_score = score
                    best_signal = signal

        if abs(best_signal) > 1e-6:
            current_price = prcSoFar[inst, -1]
            volatile_target_pos = (POSLIMIT / current_price) * best_signal
            currentPos[inst] += int(volatile_target_pos)

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

    if abs(best_signal) > 1e-6:
        if np.sign(target_pos) != np.sign(prev_pos):
            return target_pos
        elif abs(target_pos) < abs(np.sign(prev_pos) * min(abs(prev_pos), max_pos)):
            return np.sign(prev_pos) * min(abs(prev_pos), max_pos)
        else: 
            return target_pos
    else:
        return np.sign(prev_pos) * min(abs(prev_pos), max_pos)

def preCalcs(prcSoFar, t):
    global cached_features, candidates
    if t in cached_features:
        return
    
    N = prcSoFar.shape[0]

    cached_features[t] = {}

    # Collect all unique trend lengths from global candidates
    trend_lengths = sorted(set(
        trend_len
        for (_, param_grid) in candidates
        for trend_len in param_grid.get('trend_length', [])
    ))

    for trend_len in trend_lengths:
        slopes = []
        for j in range(N):
            p = prcSoFar[j, t - trend_len + 1 : t + 1]

            # Guard against invalid values
            if len(p) < 2 or np.any(p <= 0) or np.any(np.isnan(p)) or np.allclose(p, p[0]):
                slope = 0.0
            else:
                try:
                    slope = np.polyfit(np.arange(len(p)), np.log(p), 1)[0]
                except np.linalg.LinAlgError:
                    slope = 0.0

            slopes.append(slope)
        cached_features[t][f'avg_market_trend_{trend_len}'] = np.mean(slopes)

def backtest(prcSoFar, strategy_fn, params, backtest_window, inst):
    global POSLIMIT, COMMRATE, nDays
    params['self'] = inst

    price_series = prcSoFar[inst]
    start_main = nDays - backtest_window - 1
    end_main = nDays - 1

    pl_list = []
    last_pos = 0

    for t in range(start_main, end_main):
        preCalcs(prcSoFar, t) 
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

    if N_CHECKS > 0:
        for i in range(N_CHECKS):
            end_day = nDays - 1 - i * backtest_window
            start_day = end_day - backtest_window
            if start_day < 0:
                continue

            check_pl = []
            last_pos = 0
            for t in range(start_day, end_day):
                preCalcs(prcSoFar, t) 
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
            if (check_pl.mean() - 0.1 * check_pl.std()) < MIN_SCORE_THRESH:
                return -np.inf, 0

    preCalcs(prcSoFar, nDays - 1)
    latest_signal = np.clip(strategy_fn(prcSoFar, nDays - 1, params), -1, 1)
    return score, latest_signal

def extract_features(prices, feature_set):
    features = []
    log_returns = np.diff(np.log(prices))

    for feat in feature_set:  

        if feat == 'skewness':
            features.append(skew(log_returns))
        
        elif feat == 'price_position':
            rel_pos = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
            features.append(rel_pos)
        
        elif feat == 'autocorr':
            if len(log_returns) >= 2:
                ac = np.corrcoef(log_returns[1:], log_returns[:-1])[0, 1]
                features.append(ac)
            else:
                features.append(0.0)

    return features

def strategy_volatility(prcSoFar, t, params):
    inst = params["self"]
    window = params["vol_window"]
    ret_thresh = params["return_threshold"]

    prices = prcSoFar[inst, t - window + 1 : t + 1]
    if len(prices) < window or np.any(prices <= 0):
        return 0

    log_returns = np.diff(np.log(prices))
    momentum = np.sum(log_returns)

    if abs(momentum) < ret_thresh:
        return 0

    return np.clip(momentum / ret_thresh, -1, 1)


def strategy_1(prcSoFar, t, params):

    inst = params['self']
    lookback = params['lookback']
    window = params['window']
    n_partners = params['n_partners']
    feature_set = params['feature_set']
    use_prices = params['use_prices']
    prob_thresh = params['prob_thresh']
    trend_length = params['trend_length']
    avg_trend = cached_features[nDays][f'avg_market_trend_{trend_length}']

    if t < lookback + window:
        return 0

    y_prices = prcSoFar[inst, t - lookback - window + 1 : t + 1]

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

    X, Y = [], []
    for i in range(len(y_prices) - lookback - window + 1, len(y_prices) - window):
        start = i
        end = i + window + 1
        if end > len(y_prices):
            break

        y_seg = y_prices[start:end]
        feat = extract_features(y_seg, feature_set)

        if use_prices:
            feat.extend(y_seg[:-1].tolist())

        for j in top_partners:
            p_seg = prcSoFar[j, t - lookback - window + 1 : t + 1]
            p_window = p_seg[start:end]
            p_feat = extract_features(p_window, feature_set)
            feat.extend(p_feat)
            if use_prices:
                feat.extend(p_window[:-1].tolist())

        r_next = np.log(y_prices[end - 1]) - np.log(y_prices[end - 2])
        if r_next > COMMRATE:
            y_i = 1
        elif r_next < -COMMRATE:
            y_i = -1
        else:
            y_i = 0

        X.append(feat)
        Y.append(y_i)

    X = np.array(X)
    Y = np.array(Y)

    if len(np.unique(Y)) == 1:
        if len(Y) >= 10 and 1 == 0:
            return int(Y[0])
        else:
            return 0

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, Y)

    x_today = extract_features(y_prices, feature_set)
    if use_prices:
        x_today.extend(y_prices[-window - 1:-1].tolist())

    for j in top_partners:
        p_prices = prcSoFar[j, t - window : t + 1]
        p_feat = extract_features(p_prices, feature_set)
        x_today.extend(p_feat)
        if use_prices:
            x_today.extend(p_prices[:-1].tolist())

    x_today = np.array(x_today).reshape(1, -1)
    probs = model.predict_proba(x_today)[0]
    labels = model.classes_
    prob_map = dict(zip(labels, probs))

    p_buy = prob_map.get(1, 0)
    p_sell = prob_map.get(-1, 0)

    if avg_trend > 0:
        p_sell = 0
    elif avg_trend < 0:
        p_buy = 0

    if p_buy > prob_thresh:
        return p_buy
    elif p_sell > prob_thresh:
        return -p_sell
    return 0

candidates = [
    (strategy_1, {
        'lookback': [20],
        'window': [5],
        'prob_thresh': [0.5],
        'n_partners': [0],
        'feature_set': [[
            'skewness',
            'price_position',
            'autocorr',
        ]],
        'use_prices': [False],
        'trend_length': [3, 10]
    }),
    (strategy_volatility, {
    'vol_window': [10, 15, 20],
    'return_threshold': [0.01, 0.015, 0.02]
    }),
    ]

# Best so far
#    (strategy_volatility, {
#    'vol_window': [5, 10, 15, 20],
#    'return_threshold': [0.01, 0.015, 0.02]
#    }),


# Best so far 
# BACKTEST_WINDOW = 6
BACKTEST_WINDOW = 6
MIN_SCORE_THRESH = -np.inf
N_CHECKS = 0