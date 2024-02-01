import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def decay_model(t, A_B_0, R):
    term1 = (1 + np.exp(-lambda_B * t) * (epsilon_C / epsilon_B) * lambda_C / (lambda_C - lambda_B))
    term2 = (epsilon_C / epsilon_B) * (R - (lambda_C / (lambda_C - lambda_B)) * np.exp(-lambda_C * t))
    return A_B_0 * (term1 + term2)


if __name__ == "__main__":
    epsilon_B = 0.80
    epsilon_C = 0.95
    half_life_B = 19.9 * 60
    half_life_C = 26.8 * 60
    lambda_B = np.log(2)/half_life_B
    lambda_C = np.log(2)/half_life_C
    sample_size = 20  # s/sample

    time, counts = np.loadtxt('Main_Data_20240130.txt', unpack=True, skiprows=2)
    time = time * sample_size
    initial_guesses = [1, 1]

    popt, pcov = curve_fit(decay_model, time, counts, p0=initial_guesses)
    A_B_0, R = popt

    plt.figure(figsize=(10, 6))
    plt.plot(time, counts, 'b.', label='Data')
    plt.plot(time, decay_model(time, *popt), 'r-', label='Model Fit')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.legend()
    plt.show()
