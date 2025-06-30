
import numpy as np
from statsmodels.tsa.api import AutoReg

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):

    global currentPos
    (nins, nt) = prcSoFar.shape

    print(nt)

    if nt < 100:
        return np.zeros(nins)

    pred = np.zeros(nins)
    sum_sq = np.zeros(nins)

    n = 50

    for k in range(n):
        for i in range(nins):
            endo = prcSoFar[i][:nt-n+1+k]
            model = AutoReg(endo,5,'n',False)
            ardl_model = model.fit()
            predict = ardl_model.forecast(1)
            pred[i] = predict[0]
            if np.sum(np.square(ardl_model.pvalues[0])) > 0:
                sum_sq[i] = 1/np.sum(np.square(ardl_model.resid[0]))
            else: sum_sq[i] = 0
        thresh = np.percentile(sum_sq,99)
        for i in range(nins):
            if sum_sq[i] >= thresh and thresh >= 0:
                if pred[i] - prcSoFar[i][nt-1] > prcSoFar[i][nt-1]*0.01:
                    currentPos[i] = 1000/prcSoFar[i][0]
                elif pred[i] - prcSoFar[i][nt-1] < prcSoFar[i][nt-1]*-0.01:
                    currentPos[i] = -1000/prcSoFar[i][0]
            
    return currentPos
