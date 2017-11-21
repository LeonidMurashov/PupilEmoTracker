import numpy as np
import matplotlib.pyplot as plt
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

a = list(range(10))
a[5] = 0

print(a)
print(moving_average(a))

plt.plot(range(10), a)
plt.show()

plt.plot(range(8), moving_average(a))
plt.show()
