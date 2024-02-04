import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("Radioactivity.csv", delimiter=',', unpack=True)
    x = []
    y = []
    y_uncertainty = []
    for lst in data:
        x.append(lst[0])
        y.append(np.mean(lst[1:]))
        y_uncertainty.append(np.std(lst[1:])/np.sqrt(len(lst)-1))
    print(x, y, y_uncertainty)
    plt.errorbar(x, y, yerr=y_uncertainty, label="Average Count at Each Voltage")
    plt.title("Average Count of at Each Voltage")
    plt.show()
