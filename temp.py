import numpy as np
from scipy import special

a = np.load("DCCZ.npy")
print(a)
# for i in range(4):
#     for j in range(4):
#         print("%.4f" % np.abs(a[i][j]), end='_')
#         print("%.4f" % np.angle(a[i][j]), end=',')
#     print()