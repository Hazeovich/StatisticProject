import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

count = 1000
alpha, beta = 3, 3
x = np.linspace(0, 1, count)
a = stats.beta(alpha, beta).pdf(x)

plt.figure()
plt.plot(x, a)
plt.show()
