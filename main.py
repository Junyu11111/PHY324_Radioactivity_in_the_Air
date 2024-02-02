import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from background_analysis import *


def decay_model(t, A_B_0, R):
    term1 = 1 + np.exp(-lambda_B * t) * (epsilon_C / epsilon_B) * lambda_C / (
                lambda_C - lambda_B)
    term2 = (epsilon_C / epsilon_B) * (
                R - (lambda_C / (lambda_C - lambda_B)) * np.exp(-lambda_C * t))
    return A_B_0 * (term1 + term2)


def decay_model_complex(t, A_B_0, R, N0, lambda_decay):
    return decay_model(t, A_B_0, R) + exponential_decay(t, N0, lambda_decay)


if __name__ == "__main__":
    epsilon_B = 0.80
    epsilon_C = 0.95
    half_life_B = 19.9 * 60
    half_life_C = 26.8 * 60
    lambda_B = np.log(2) / half_life_B
    lambda_C = np.log(2) / half_life_C
    sample_size = 20  # s/sample

    time, counts = np.loadtxt('Main_Data_20240201.txt', unpack=True, skiprows=2)

    initial_guesses = [200, 1, 200, 1e-3]

    time = time * sample_size
    limited_range = [15 * 3, -1]
    # time = time[limited_range[0]: limited_range[1]]
    # counts = counts[limited_range[0]: limited_range[1]]
    lower_bounds = (40, -10, 40, 1e-6)
    upper_bounds = (250, 3, 250, 1e-1)

    # uncertainties
    mu, std = 11, 3
    counts_uncertainty = np.sqrt(counts) + std
    counts -= mu
    popt, pcov = curve_fit(decay_model_complex, time, counts,
                           sigma=counts_uncertainty, absolute_sigma=True,
                           p0=initial_guesses, maxfev=50000,
                           bounds=(lower_bounds, upper_bounds))
    print(popt)

    plt.figure(figsize=(10, 6))
    plt.errorbar(time, counts, yerr=counts_uncertainty, label='Data', fmt='none', ecolor="grey")
    plt.plot(time, decay_model_complex(time, *popt), 'r-', label='Model Fit')
    background_decay = exponential_decay(time, popt[-2], popt[-1])
    plt.plot(time, counts - background_decay)
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.legend()
    plt.show()
