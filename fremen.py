import numpy as np
from strategies import short_term_momentum, normalized_momentum, mean_reversion, sma_crossover, volatility_breakout, pairs_mean_reversion

class Market:
    def __init__(self, prices: np.ndarray):
        self.prices = prices  # shape (n_inst, n_days)
        self.market_returns_cache: dict[int, np.ndarray] = {}

    def compute_market_returns(self, day: int) -> np.ndarray:
        if day not in self.market_returns_cache:
            log_prices = np.log(self.prices[:, :day])
            returns = np.diff(log_prices, axis=1)
            self.market_returns_cache[day] = returns.mean(axis=0)
        return self.market_returns_cache[day]

class Instrument:
    def __init__(self, inst_id: int, inst_prices: np.ndarray):
        self.inst_id = inst_id
        self.prices = inst_prices  # shape: (n_days,)
        self.n_days = inst_prices.shape[0]
        self.beta_history: dict[int, float] = {}  # day -> beta
        self.strategy = None ## Could classify the type of strat used here

    def compute_beta(self, current_day: int, market_returns: np.ndarray) -> float:
        assert current_day <= self.n_days, "Invalid day"
        log_prices = np.log(self.prices[:current_day])
        returns = np.diff(log_prices)

        cov = np.cov(returns, market_returns)[0, 1]
        var = np.var(market_returns)
        beta = cov / (var + 1e-8)

        self.beta_history[current_day] = beta
        return beta
    
    def update_prices(self, new_prices: np.ndarray):
        self.prices = new_prices
        self.n_days = new_prices.shape[0]

    def get_price(self, day: int) -> float | None:
        return self.prices[day + 1]

    def get_beta(self, day: int) -> float | None:
        return self.beta_history.get(day)

    def clear_history(self):
        self.beta_history.clear()


NINST = 50
instruments = None
currentPos = np.zeros(NINST)

def getMyPosition(prcSoFar : np.ndarray) -> np.ndarray:
    global currentPos
    global instruments

    if instruments is None:
        instruments = [
            Instrument(inst_id=x, inst_prices=prcSoFar[x].copy()) 
            for x in range(prcSoFar.shape[0])
        ]
    else:
        for i in range(len(instruments)):
            instruments[i].update_prices(prcSoFar[i])
            
    pos = np.zeros(NINST)

    ## Define inputs ##
    lookback = 10 # Amount of days before current to compute momentum >= 1
    scale = 2000 # Aggressiveness of the strategy >= 1
    window = 10 # Rolling window for mean and std >= 1
    short = 5 # Short-term moving average window 1<x<=20
    long = 30 # Long-term moving average window 30<x<=60
    vol_window = 5 #days for volatility estimation >=1
    threshold = 1 # number of stds movement that must exceed to trigger entry >= 0
    a = 0 # first of 2 instrument to trade 0<=x<=49
    b = 49 # second of 2 instrument to trade 0<=x<=49


    # Combine multiple strategies
    pos += short_term_momentum(prcSoFar, lookback, scale=scale)
    #pos += normalized_momentum(prcSoFar, lookback, scale=scale)
    #pos += mean_reversion(prcSoFar, window, scale=scale)
    #pos += sma_crossover(prcSoFar, short, long, scale)
    #pos += volatility_breakout(prcSoFar, vol_window, scale, threshold)
    #pos += pairs_mean_reversion(prcSoFar, a, b, lookback, scale)

    # Apply position limits
    curPrices = prcSoFar[:, -1]
    posLimits = (10000 / curPrices).astype(int)
    newPos = np.clip(pos, -posLimits, posLimits)

    currentPos = newPos.astype(int)
    return currentPos
