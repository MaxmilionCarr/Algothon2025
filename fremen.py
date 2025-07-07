import numpy as np
from collections import Counter
from statsmodels.tsa.api import ARDL

# Market Class
class Market:
    '''
        .compute_market_returns - Fetches average market returns to a specific day
        .update_prices - updates the prices in the market object (Maybe take out later)
    '''

    def __init__(self, prices: np.ndarray):
        self.prices = prices  # shape (n_inst, n_days)
        self.market_returns_cache: dict[int, np.ndarray] = {}

    def compute_market_returns(self, day: int) -> np.ndarray:
        if day not in self.market_returns_cache:
            log_prices = np.log(self.prices[:, :day])
            returns = np.diff(log_prices, axis=1)
            self.market_returns_cache[day] = returns.mean(axis=0)
        return self.market_returns_cache[day]
    
    def update_prices(self, new_prices: np.ndarray) -> np.ndarray:
        self.prices = new_prices.copy()

# Instrument Class
class Instrument:
    '''
        .compute_beta - Computes beta for stcok at a certain day and caches it in history (returns the computed beta)
        .update - Updates the cached prices to include a certain day
        .get_price - Fetches the price at a certain day
        .get_beta - Fetches beta at a certain day
        .get_beta_array - Fetches a nparray of all beta history
        .clear_beta - Clears beta history
    '''
    
    def __init__(self, inst_id: int, inst_prices: np.ndarray):
        self.inst_id = inst_id
        self.prices = inst_prices  # shape: (n_days,)
        self.n_days = inst_prices.shape[0]
        self.beta_history: dict[int, float] = {}  # day -> beta
        self.strategy = None ## Could classify the type of strat used here

    def compute_beta(self, current_day: int, market_returns: np.ndarray, lookback: int | None = None) -> float:
        assert current_day <= self.n_days, "Invalid day"
        assert len(market_returns) == current_day - 1, "Market return shape mismatch"

        if lookback is not None and lookback < current_day:
            start_idx = current_day - lookback
            log_prices = np.log(self.prices[start_idx:current_day])
            returns = np.diff(log_prices)
            mr = market_returns[-lookback:]
        else:
            log_prices = np.log(self.prices[:current_day])
            returns = np.diff(log_prices)
            mr = market_returns 

        cov = np.cov(returns, mr)[0, 1]
        var = np.var(mr)
        beta = cov / (var + 1e-8)

        self.beta_history[current_day] = beta
        return beta
    
    def update_prices(self, new_prices: np.ndarray) -> None:
        self.prices = new_prices
        self.n_days = new_prices.shape[0]

    def get_price(self, day: int) -> float | None:
        return self.prices[day] if 0 <= day < self.n_days else None

    def get_beta(self, day: int) -> float | None:
        return self.beta_history.get(day)
    
    def clear_history(self):
        self.beta_history.clear()

def get_beta_array(day: int, instruments: list) -> np.ndarray:
    return np.array([inst.get_beta(day) for inst in instruments])

# Global constants
START = 50     # <-- Start trading after 10 days
COMMRATE = 0.0005  # <-- Commission rate
POSLIMIT = 10000   # <-- Dollar position limit
N_INST = 50        # <-- Number of instruments
instruments = None
market = None

# Global variables
currentPos = np.zeros(N_INST)

nDays = None

# Strategy definitions
def strategy_0(prcSoFar, inst):
    return 0

def strategy_1(inst_idx: int, predictors: dict[int, int]) -> int:
    if inst_idx not in predictors:
        return currentPos[inst_idx]

    pred_idx = predictors[inst_idx]
    y_prices = instruments[inst_idx].prices
    x_prices = instruments[pred_idx].prices

    if len(y_prices) < 60 or len(x_prices) < 60:
        return currentPos[inst_idx]

    y_log = np.log(y_prices)
    x_log = np.log(x_prices)
    y_ret = np.diff(y_log)
    x_ret = np.diff(x_log)

    LOOKBACK = 60
    MIN_CORRELATION = 0.3
    CONFIDENCE_THRESHOLD = 0.5

    # Check correlation
    corr = np.corrcoef(y_ret[-LOOKBACK:], x_ret[-LOOKBACK:])[0, 1]
    if corr < MIN_CORRELATION:
        return 0

    # Fit ARDL
    y = y_ret[-(LOOKBACK + 1):-1]
    x = x_ret[-(LOOKBACK + 1):-1].reshape(-1, 1)
    model = ARDL(endog=y, lags=1, exog=x, order={0: [1]}, causal=True, trend="c")
    result = model.fit()

    alpha, phi, beta = result.params[:3]
    last_y_ret = y_ret[-1]
    last_x_ret = x_ret[-1]
    predicted_y_next = alpha + phi * last_y_ret + beta * last_x_ret

    # Confidence filter
    confidence = np.std(y_ret)
    if abs(predicted_y_next) < CONFIDENCE_THRESHOLD * confidence:
        return 0

    current_price = y_prices[-1]
    if predicted_y_next > 0:
        position_size = int((abs(predicted_y_next) / confidence) * POSLIMIT / current_price)
        return min(position_size, POSLIMIT / current_price)
    elif predicted_y_next < 0:
        position_size = int((abs(predicted_y_next) / confidence) * POSLIMIT / current_price)
        return max(position_size, -POSLIMIT / current_price)
    else:
        return 0


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

    global instruments, market

    N_INST, nDays = prcSoFar.shape

    if nDays < START:
        return currentPos


    # Set up market for each day
    if market is None:
        market = Market(prices=prcSoFar.copy())
    else:
        market.update_prices(prcSoFar)
    
    market_returns = market.compute_market_returns(nDays)

    # Set up instruments for each day
    if instruments is None:
        instruments = [Instrument(i, prcSoFar[i].copy()) for i in range(N_INST)]
        for inst in instruments:
            inst.compute_beta(nDays, market_returns)
    else:
        for i in range(N_INST):
            instruments[i].update_prices(prcSoFar[i])
            instruments[i].compute_beta(nDays, market_returns)

    # Trading time

    # Ideas
    # 1.) Low beta instruments are traded on ARDL
    # 2.) Medium instruments are traded on momentum strategies
    # 3.) Medium-High beta instruments ??? Maybe not

    assignments = {}
    pos = np.zeros(N_INST)
    predictors = {}
    betas = get_beta_array(nDays, instruments)


    for i, inst in enumerate(instruments):
        beta = inst.get_beta(nDays)

        if beta is None:
            assignments[i] = 0
            currentPos[i] = 0
            continue
   
        elif 0.5 < abs(beta) < 1.3:
            assignments[i] = 1
            diffs = np.square(betas - beta)
            diffs[i] = np.inf

            closest_index = int(np.argmin(diffs))
            beta_diff = abs(betas[closest_index] - beta)
            
            if beta_diff <= 0.15:
                predictors[i] = closest_index
                currentPos[i] = strategy_1(i, predictors)
            else:
                assignments[i] = 0
                currentPos[i] = 0



        #strat_num = assignments[i]
        #strat_func = strategy_functions[strat_num]
        #desired_position = strat_func(prcSoFar, i, predictors)
        #currentPos[i] = int(desired_position)
    
    traded = len([x for x in currentPos if x != 0])
    print(f"Number of instruments traded: {traded} / 50")
    return currentPos
