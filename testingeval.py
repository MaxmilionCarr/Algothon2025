#!/usr/bin/env python

import numpy as np
import pandas as pd
from main import getMyPosition as getPosition
from main import reset_state
import time

start_time = time.time()

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, startDay, endDay):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []

    for t in range(startDay, endDay + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]
        if t < endDay:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        if t > startDay:
            todayPLL.append(todayPL)

    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = np.sqrt(249) * plmu / plstd if plstd > 0 else 0
    ret = value / totDVolume if totDVolume > 0 else 0
    return (plmu, ret, plstd, annSharpe, totDVolume)

# === Sliding 500-day windows ===
WINDOW = 500
scores = []

for start in range(1, nt - WINDOW + 2):  # +2 because range is exclusive at the end
    reset_state()
    end = start + WINDOW - 1
    print(f"\nRunning sliding window: Days {start} to {end}")
    (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll, start, end)
    score = meanpl - 0.1 * plstd
    scores.append(score)

    print("=====")
    print("mean(PL): %.1lf" % meanpl)
    print("return: %.5lf" % ret)
    print("StdDev(PL): %.2lf" % plstd)
    print("annSharpe(PL): %.2lf " % sharpe)
    print("totDvolume: %.0lf " % dvol)
    print("Score: %.2lf" % score)

# === Overall Summary ===
print("\n==== FINAL SUMMARY ====")
print(f"Average Score: {np.mean(scores):.2f}")
print(f"Score Std Dev: {np.std(scores):.2f}")
print(f"Best Score: {np.max(scores):.2f} at window {np.argmax(scores) + 1}")
print(f"Worst Score: {np.min(scores):.2f} at window {np.argmin(scores) + 1}")

print("--- %.2f seconds ---" % (time.time() - start_time))
