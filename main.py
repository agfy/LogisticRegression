import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

perceptron = pd.read_csv('data-logistic.csv', header=None)

w1_new = w2_new = w1 = w2 = 0.0
k = 0.1
c = 0.0
l = len(perceptron)

for j in range(10000):
    sum1 = sum2 = 0.0

    for i in range(l):
        y = perceptron[0][i]
        x1 = perceptron[1][i]
        x2 = perceptron[2][i]
        div = (1 + np.exp(-1 * y * (w1 * x1 + w2 * x2)))

        sum1 += y * x1 * (1 - 1 / div) - k * c * w1
        sum2 += y * x2 * (1 - 1 / div) - k * c * w2

    w1_new = w1 + k * sum1 / l
    w2_new = w2 + k * sum2 / l

    diff = np.sqrt((w1-w1_new)**2 + (w2-w2_new)**2)
    if diff < 0.00001:
        break

    w1 = w1_new
    w2 = w2_new

a = []
for i in range(l):
    a.append(1 / (1 + np.exp(-w1 * perceptron[1][i] - w2 * perceptron[2][i])))

result = roc_auc_score(perceptron[0], a)
