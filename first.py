import numpy as np
import matplotlib.pyplot as plt

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    s = np.random.randint(1, 7, size=n) + np.random.randint(1, 7, size=n)
    h, h2 = np.histogram(s, bins=range(2, 14))

    plt.bar(h2[:-1], h / n)
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Frequency")
    plt.title(f"Dice Roll Distribution (n={n})")
    plt.show()
