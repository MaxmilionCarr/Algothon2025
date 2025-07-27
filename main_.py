# === Import Modules === 
import numpy as np

# === Global Constants ===
COMMRATE = 0.0005
POSLIMIT = 10000
N_INST = 50

# === Global State ===
currentPos = np.zeros(N_INST)
nDays = 0
trend = []   # Stores one scalar market trend per day

# === Strategy Parameters ===
TREND_LENGTH = 10

def getMyPosition(prcSoFar):
    global currentPos, nDays, trend

    _, nDays = prcSoFar.shape

    if nDays < TREND_LENGTH:
        return currentPos

    trend.append(getTrend(prcSoFar))

    if nDays < TREND_LENGTH:
        return currentPos

    for inst in range(N_INST):
        currentPos[inst] = int(getPos(prcSoFar, inst))

    # === Append currentPos to file with aligned formatting ===
    with open("positions.dump", "a") as f:
        row_str = "".join(f"{int(pos):5d}" for pos in currentPos)
        f.write(row_str + "\n")

    return currentPos

def getTrend(prcSoFar):
    slopes = []
    for j in range(N_INST):
        p = prcSoFar[j, nDays - TREND_LENGTH: nDays + 1]
        slope = np.polyfit(np.arange(len(p)), np.log(p), 1)[0]
        slope = np.diff(p)
        slopes.append(slope)
    return np.mean(slopes)

def getPos(prcSoFar, inst):
    global currentPos, nDays, trend

    current_price = prcSoFar[inst, -1]
    max_pos = POSLIMIT / current_price

    return max_pos * np.sign(trend[-1])
