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
    
    def update_prices(self, new_prices: np.ndarray):
        self.prices = new_prices.copy()

class Instrument:
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
    
    def update(self, new_prices: np.ndarray, market_returns: np.ndarray) -> None:
        self.prices = new_prices
        self.n_days = new_prices.shape[0]

        if self.n_days >= 2:
            self.compute_beta(current_day=self.n_days, market_returns=market_returns)

    def get_price(self, day: int) -> float | None:
        return self.prices[day] if 0 <= day < self.n_days else None

    def get_beta(self, day: int) -> float | None:
        return self.beta_history.get(day)

    def get_beta_array(day: int) -> np.ndarray:
        return np.array([inst.get_beta(day) for inst in instruments])
    
    def clear_history(self):
        self.beta_history.clear()

NINST = 50
instruments = None
market = None
currentPos = np.zeros(NINST)

def getMyPosition(prcSoFar : np.ndarray) -> np.ndarray:
    global currentPos
    global instruments
    global market
    pos = np.zeros(NINST)

    # Set up Market object
    if market is None:
        market = Market(prices=prcSoFar.copy())
    else:
        market.update_prices(prcSoFar)
    
    # Get market returns at today
    market_returns = market.compute_market_returns(prcSoFar.shape[1])

    # Set up Instrument list object
    if instruments is None:
        instruments = [
            Instrument(inst_id=x, inst_prices=prcSoFar[x].copy()) 
            for x in range(NINST)
        ]
        for i in range(NINST):
            instruments[i].compute_beta(current_day=prcSoFar.shape[1], market_returns=market_returns)
    else:
        for i in range(NINST):
            instruments[i].update(prcSoFar[i], market_returns)
            

    # Ideas on how to use beta to assign trading strategies #
    # 1.) Closely beta related should be hedged with each other
    # 2.) Position reductions for high beta stocks and position increases for low beta stocks
    # 3.) Consistent beta increases could be funnelled with cash (Volatility Momentum)
    # 4.) Rotate a designated position amount to high beta and low beta stocks based on the market regime

    # Apply position limits
    curPrices = prcSoFar[:, -1]
    posLimits = (10000 / curPrices).astype(int)
    newPos = np.clip(pos, -posLimits, posLimits)

    currentPos = newPos.astype(int)
    return currentPos
