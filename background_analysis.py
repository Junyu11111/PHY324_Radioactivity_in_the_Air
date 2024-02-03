import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit


def exponential_decay(t, N0, lambda_decay):
    return N0 * np.exp(-lambda_decay * t)


def exponential_decay_fixed(t, N0):
    return +N0 * np.exp(-0.000018 * t)


if __name__ == "__main__":
    # background
    background_counts = np.loadtxt('20240202_Bkgrnd_20s_25min.txt', unpack=True, skiprows=2, usecols=1)

    plt.hist(background_counts, density=True, bins='auto')
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
    background_uncertainty = std

    # model
    plt.figure("data")
    time, data_counts = np.loadtxt('20240202_main_data_20s_4hour.txt', unpack=True, skiprows=2)
    time = time * 20

    data_counts -= background_value

    start_index = 500

    initial_guesses = (100, 1e-3)
    popt, pcov = curve_fit(exponential_decay, time[start_index:], data_counts[start_index:], p0=initial_guesses)
    plt.plot(time, exponential_decay(time, *popt),label="model best fit starting time index {}".format(start_index*20))
    plt.plot(time, data_counts - exponential_decay(time, *popt), label="data-model best fit")
    print(popt)

    initial_guesses = (100)
    popt, pcov = curve_fit(exponential_decay_fixed, time[start_index:], data_counts[start_index:], p0=initial_guesses)
    plt.plot(time, exponential_decay_fixed(time, *popt), label="model Pb212 half life")
    plt.plot(time, data_counts, label="data-model Pb212 half life")
    plt.legend()
    print(popt)

    print(background_value, background_uncertainty)
    plt.show()
