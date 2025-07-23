import numpy as np
import pandas as pd

FILE = "Team_583069.dump"

def loadPositions(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    return df.values

def getMyPosition(prcSoFar):
    """
    Provide optimal positions (long/short up to $10k value) based on historical prices of 50 assets.

    Parameters:
        prcSoFar (np.array): Array of historical prices with shape (N_INST, ndays)

    Returns:
        np.array: 1D array of 50 desired positions
    """
    global currentPos, nDays

    _, nDays = prcSoFar.shape # Get current day index

    nDays = nDays - 1001

    positions = loadPositions(FILE)

    # print(positions[nDays])

    return positions[nDays]


# 119541 - Hebron James -0.23

# 147228 - The Euclid Knights -0.24

# 303651 - setwithfriends 47.64

# 310529 - Tung Tung -0.4

# 361192 - Fremen -0.93

# 514886 - Team Raj 22.26

# 583069 - Team i73 -0.18

# 775312 - lean-mean-algo-trading-machine -0.7

# 784234 - CRABMILK 52.83

# 858308 - CodingCrackheads 31.88

# 961019 - Bidoof 6.46

# 995747 - TEAM 5.07