import numpy as np
import matplotlib.pyplot as plt

n = 10

A = np.zeros((n,n))

for i in range(n):
    A[:,i] = i/n

plt.figure()
plt.imshow(A, cmap="gray")
plt.savefig("shades.png")

