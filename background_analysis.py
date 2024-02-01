import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

background_counts = np.loadtxt('Bkgrnd_Data_20240130.txt', unpack=True, skiprows=2, usecols=1)

plt.hist(background_counts,density=True, bins='auto')
plt.title('Histogram of Background Counts')
plt.xlabel('Counts')
plt.ylabel('Frequency')

mu, std = stats.norm.fit(background_counts)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f"Fit: μ={mu:.2f}, σ={std:.2f}")

plt.legend()


background_value = mu
background_uncertainty = std / np.sqrt(len(background_counts))

print(background_value, background_uncertainty)
plt.show()
