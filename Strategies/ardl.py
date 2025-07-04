from statsmodels.tsa.api import ARDL
import numpy as np

predictors = {
    0: 31, 1: 34, 2: 21, 3: 4, 4: 19, 5: 28, 6: 3, 7: 26, 8: 29, 9: 40,
    10: 6, 11: 2, 12: 8, 13: 14, 14: 19, 15: 34, 16: 27, 17: 27, 18: 22, 19: 42,
    20: 47, 21: 36, 22: 27, 23: 4, 24: 14, 25: 11, 26: 34, 27: 9, 28: 24, 29: 20,
    30: 46, 31: 32, 32: 17, 33: 13, 34: 19, 35: 26, 36: 32, 37: 26, 38: 27, 39: 12,
    40: 34, 41: 34, 42: 31, 43: 22, 44: 41, 45: 41, 46: 36, 47: 5, 48: 8, 49: 20
}

def ardl(prcSoFar: np.ndarray, inst: int, predictors: dict[int, int], pos_limit: int = 10000) -> int:

    if inst not in predictors or prcSoFar.shape[1] < 55:  # ensure 50 lookback + 5 buffer
        return 0

    LOOKBACK = 50
    x_idx = predictors[inst]
    
    try:
        y_log = np.log(prcSoFar[inst])
        x_log = np.log(prcSoFar[x_idx])
        y_ret = np.diff(y_log)
        x_ret = np.diff(x_log)

        y = y_ret[-(LOOKBACK + 1):-1]
        x = x_ret[-(LOOKBACK + 1):-1].reshape(-1, 1)

        model = ARDL(endog=y, lags=1, exog=x, order={0: [1]}, causal=True, trend="c")
        result = model.fit()

        alpha, phi, beta = result.params[:3]
        last_y_ret = y_ret[-1]
        last_x_ret = x_ret[-1]
        predicted_y_next = alpha + phi * last_y_ret + beta * last_x_ret

        current_price = prcSoFar[inst, -1]
        max_shares = int(pos_limit / current_price)

        if predicted_y_next > 0:
            return max_shares
        elif predicted_y_next < 0:
            return -max_shares
        else:
            return 0

    except Exception:
        return 0  # fallback on errors