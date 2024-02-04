import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

from background_analysis import *
from PHY324_Module import *


def decay_model(t, A_B_0, R):
    term1 = (1 + np.exp(-lambda_B * t) * (epsilon_C / epsilon_B) * lambda_C / (lambda_C - lambda_B))
    term2 = (epsilon_C / epsilon_B) * (R - (lambda_C / (lambda_C - lambda_B)) * np.exp(-lambda_C * t))
    return A_B_0 * (term1 + term2)


def decay_model_shift(t, A_B_0, R, t0):
    t_copy = t + t0
    term1 = (1 + np.exp(-lambda_B * t_copy) * (epsilon_C / epsilon_B) * lambda_C / (lambda_C - lambda_B))
    term2 = (epsilon_C / epsilon_B) * (R - (lambda_C / (lambda_C - lambda_B)) * np.exp(-lambda_C * t_copy))
    return A_B_0 * (term1 + term2)


def decay_model_complex(t, A_B_0, R, N0, lambda_decay):
    return decay_model(t, A_B_0, R) + exponential_decay(t, N0, lambda_decay)


def calculate_A_b_c(A_obs_b, A_obs_b_err, R, R_err):
    A_B, A_B_err = multiply_with_errors(((A_obs_b, A_obs_b_err), (1 / (geo_eff * epsilon_B), 0)))
    A_C, A_C_err = multiply_with_errors(((A_B, A_B_err), (R, R_err)))
    return (A_B, A_B_err), (A_C, A_C_err)


def calculate_n_b(A_B, A_B_err):
    N_b, N_b_err = multiply_with_errors(((A_B, A_B_err), (1 / lambda_B, 0)))
    r = lambda_B / lambda_A
    k = 13
    k_err = 1
    t = 1 * 60 * 60
    term1 = (r + 1) * (1 - np.exp(-lambda_B * t)) / lambda_B
    term2 = (r / (lambda_B - lambda_A)) * (np.exp(-lambda_B * t) - np.exp(-lambda_A * t))
    n_b, n_b_err = divide_with_errors((N_b, N_b_err), multiply_with_errors(((k, k_err), (term1+term2,0))))
    return multiply_with_errors(((n_b, n_b_err), (lambda_B / lambda_rn, 0)))


def plot_data_range(data_dict, name):
    count = 0
    ax = plt.gca()
    ax.axes.get_yaxis().set_visible(False)
    for k in data_dict:
        plt.errorbar(data_dict[k][0], count, xerr=data_dict[k][1], label="{} from {}".format(name, k), marker="x",
                     capsize=5,
                     capthick=1)
        count += 1
    plt.legend(fontsize="14", fancybox=True, framealpha=0.3)


if __name__ == "__main__":
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)

    geo_eff = 0.5
    epsilon_B = 0.80
    epsilon_C = 0.95
    half_life_A = 3.1 * 60
    half_life_B = 19.9 * 60
    half_life_C = 26.8 * 60
    half_life_rn = 3.8 * 24 * 60 * 60
    lambda_A = np.log(2) / half_life_A
    lambda_B = np.log(2) / half_life_B
    lambda_C = np.log(2) / half_life_C
    lambda_rn = np.log(2) / half_life_rn
    sample_size = 20  # s/sample
    result = {}
    r_dict = {}

    # background
    background_counts = np.loadtxt('20240202_Bkgrnd_20s_25min.txt', unpack=True, skiprows=2, usecols=1)

    plt.figure("Background", figsize=(10, 6))
    plt.hist(background_counts, density=True, bins='auto')
    plt.title('Histogram of Background Counts')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')

    mu, std = stats.norm.fit(background_counts)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f"Normal Distribution Fit: μ={mu:.2f}, σ={std:.2f}")
    plt.legend()
    plt.savefig("Graphs/Background.png", dpi=300)

    # main data
    time, counts = np.loadtxt('20240202_main_data_20s_4hour.txt', unpack=True, skiprows=2)
    time = time[45:] - time[45]  # remove data from 0-15 min
    counts = counts[45:]
    counts_uncertainty = np.sqrt(counts + std) / sample_size
    counts -= mu
    counts = counts / sample_size

    # Composite model
    initial_guesses = [20, 1, 1, 1e-5]
    # initial_guesses = [200, 1]

    time = time * sample_size
    limited_range = [15 * 3, -1]
    # time = time[limited_range[0]: limited_range[1]]
    # counts = counts[limited_range[0]: limited_range[1]]
    lower_bounds = (0.5, -10, 0.1, 1e-6)
    upper_bounds = (100, 3, 200, 1e-2)

    popt, pcov = curve_fit(decay_model_complex, time, counts,
                           sigma=counts_uncertainty, absolute_sigma=True,
                           p0=initial_guesses, maxfev=50000,
                           bounds=(lower_bounds, upper_bounds))
    print("Composite Model:", popt, np.sqrt(np.diag(pcov)))

    plt.figure("Composite Model", figsize=(10, 6))
    plt.title("Observed Beta Decay Over Time and Bestfit Composite Model")
    plt.errorbar(time, counts, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    exp = decay_model_complex(time, *popt)
    plt.plot(time, exp, 'r-', label='Model for $A_B(t)+A_C(t)+A_{background}(t)$', alpha=0.7)
    chisq = np.sum(((counts - exp) / counts_uncertainty) ** 2)
    dof = (len(counts) - 4)
    print("composite model chi^2, chi^2 prob", chisq / dof, 1 - chi2.cdf(chisq, dof))
    background_decay = exponential_decay(time, popt[-2], popt[-1])
    plt.errorbar(time, counts - background_decay, yerr=counts_uncertainty, ls='', lw=1, marker='o', mfc='darkorange',
                 mec="darkorange",
                 markersize=2.5, capsize=1.5, capthick=0.5,
                 label="{}".format("Data Subtracting Constant Background and $A_{background}(t)$"), zorder=1,
                 ecolor="tan",
                 alpha=0.3)
    exp_decay = decay_model(time, popt[0], popt[1])
    plt.plot(time, decay_model(time, popt[0], popt[1]), "g-", label=r"Model for $A_B(t)+A_C(t)$", alpha=0.7)
    chisq = np.sum(((counts - background_decay - exp_decay) / counts_uncertainty) ** 2)
    dof = (len(counts) - 2)
    print("composite model chi^2, chi^2 prob", chisq / dof, 1 - chi2.cdf(chisq, dof))
    plt.plot(time, background_decay, "k-", label=r"Model for $A_{background}(t)$", alpha=0.7)
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.legend()
    plt.savefig("Graphs/Composite Model.png", dpi=300)
    errors = np.sqrt(np.diag(pcov))
    A_b, A_c = calculate_A_b_c(popt[0], errors[0], popt[1], pcov[1])
    print(A_b, A_c)
    print("concentration=", calculate_n_b(*A_b))
    r_dict["Composite Model"] = (popt[1], errors[1])

    # Composite residual
    plt.figure("Composite Model Residual", figsize=(10, 6))
    plt.errorbar(time, counts - exp, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.title('Residual Plot Of the Bestfit Composite Model')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Residual Count Rate')
    plt.legend()
    plt.savefig("Graphs/Composite Model Residual.png", dpi=300)

    # Separate model:
    start_index = int(8000 / 20)  # time index /sample size
    initial_guesses = (100, 1e-3)
    plt.figure("Separate Model", figsize=(10, 6))
    plt.title("Observed Beta Decay Over Time and Separate Bestfit Composite Model")
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.errorbar(time, counts, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    popt, pcov = curve_fit(exponential_decay, time[start_index:], counts[start_index:], p0=initial_guesses)
    background_decay = exponential_decay(time, *popt)
    plt.errorbar(time, counts - background_decay, yerr=counts_uncertainty, ls='', lw=1, marker='o', mfc='darkorange',
                 mec="darkorange",
                 markersize=2.5, capsize=1.5, capthick=0.5,
                 label="{}".format("Data Subtracting Constant Background and $A_{background}(t)$"), zorder=1,
                 ecolor="tan",
                 alpha=0.3)
    print("Separate Model Background:", popt, np.sqrt(np.diag(pcov)))
    initial_guesses = [200, 1]
    popt, pcov = curve_fit(decay_model, time, counts - background_decay,
                           sigma=counts_uncertainty, absolute_sigma=True,
                           p0=initial_guesses, maxfev=50000)
    print("Separate Model A_B + A_C:", popt, np.sqrt(np.diag(pcov)))
    exp_bc = decay_model(time, *popt)
    exp = exp_bc + background_decay
    plt.plot(time, exp, 'r-', label='Model for $A_B(t)+A_C(t)+A_{background}(t)$', alpha=0.7)
    plt.plot(time, decay_model(time, popt[0], popt[1]), "g-", label=r"Model for $A_B(t)+A_C(t)$", alpha=0.7)
    plt.plot(time, background_decay, "k-", label=r"Model for $A_{background}(t)$", alpha=0.7)
    plt.legend()
    chisq = np.sum(((counts - exp) / counts_uncertainty) ** 2)
    dof = (len(counts) - 2)
    print("Separate model chi^2, chi^2 prob", chisq / dof, 1 - chi2.cdf(chisq, dof))
    plt.savefig("Graphs/Separate Model.png", dpi=300)

    errors = np.sqrt(np.diag(pcov))
    A_b, A_c = calculate_A_b_c(popt[0], errors[0], popt[1], pcov[1])
    print(A_b, A_c)
    print("concentration=", calculate_n_b(*A_b))
    # r_dict["Separate fit Composite Model"] = (popt[1], errors[1])

    # Separate residual
    plt.figure("Separate Model Residual", figsize=(10, 6))
    plt.errorbar(time, counts - exp, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.title('Residual Plot Of the Separate Bestfit Composite Model')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.legend()
    plt.savefig("Graphs/Separate Model Residual.png", dpi=300)

    # Constant Background
    plt.figure("Constant Background Model", figsize=(10, 6))
    plt.title("Observed Beta Decay Over Time and Bestfit Decay Model")
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')

    initial_guesses = [200, 1]
    popt, pcov = curve_fit(decay_model, time, counts,
                           sigma=counts_uncertainty, absolute_sigma=True,
                           p0=initial_guesses, maxfev=50000)
    plt.errorbar(time, counts, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    print("Constant Back Ground:", popt, np.sqrt(np.diag(pcov)))
    exp = decay_model(time, *popt)
    plt.plot(time, exp, 'r-', label='Model for $A_B(t)+A_C(t)$', alpha=0.7)
    chisq = np.sum(((counts - exp) / counts_uncertainty) ** 2)
    dof = (len(counts) - 2)
    print("Constant Background model chi^2, chi^2 prob", chisq / dof, 1 - chi2.cdf(chisq, dof))
    plt.savefig("Graphs/Constant Background Model.png", dpi=300)

    errors = np.sqrt(np.diag(pcov))
    A_b, _ = calculate_A_b_c(popt[0], errors[0], popt[1], pcov[1])
    print(A_b)
    print("concentration=", calculate_n_b(*A_b))
    # r_dict["Constant Background Model"] = (popt[1], errors[1])

    # Constant background residual
    plt.figure("Constant Background Model Residual", figsize=(10, 6))
    plt.errorbar(time, counts - exp, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.title('Residual Plot Of the Constant Background Model')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.legend()
    plt.savefig("Graphs/Constant Background Model Residual.png", dpi=300)

    # Shifted Model
    plt.figure("Shifted Model", figsize=(10, 6))
    plt.title("Observed Beta Decay Over Time and Bestfit Shifted Decay Model")
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    initial_guesses = [33, 1, 2000]
    upper_bounds = [np.inf, 10, np.inf]
    lower_bounds = [10, -10, -2000]

    popt, pcov = curve_fit(decay_model_shift, time, counts,
                           sigma=counts_uncertainty, absolute_sigma=True,
                           p0=initial_guesses, maxfev=50000, bounds=(lower_bounds, upper_bounds))
    print("time:", time[0], time[-1])
    print("Shift:", popt, np.sqrt(np.diag(pcov)))
    exp_shift = decay_model_shift(time, *popt)
    plt.plot(time, exp_shift, 'r-', label='Model for $A_B(t+t_0)+A_C(t+t_0)$', alpha=0.7)
    plt.errorbar(time, counts, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='', lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    plt.legend()
    chisq = np.sum(((counts - exp_shift) / counts_uncertainty) ** 2)
    dof = (len(counts) - 3)
    print("Shifted model chi^2, chi^2 prob", chisq / dof, 1 - chi2.cdf(chisq, dof))
    plt.savefig("Graphs/Shifted Model.png", dpi=300)

    errors = np.sqrt(np.diag(pcov))
    A_b, A_c = calculate_A_b_c(popt[0], errors[0], popt[1], pcov[1])
    print(A_b, A_c)
    print("concentration=", calculate_n_b(*A_b))
    r_dict["Shifted Model"] = (popt[1], errors[1])

    # Shifted residual
    plt.figure("Shifted Model Residual", figsize=(10, 6))
    plt.errorbar(time, counts - exp_shift, yerr=counts_uncertainty, label='Data Subtracting Constant Background', ls='',
                 lw=1,
                 marker='o', mfc='navy', mec="navy",
                 markersize=2.5, capsize=1.5, capthick=0.5, zorder=0, ecolor="royalblue", alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.title('Residual Plot Of the Bestfit Shifted Decay Model')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Count Rate')
    plt.savefig("Graphs/Shifted Model Residual.png", dpi=300)
    plt.legend()

    plt.figure("log")
    counts += mu / sample_size
    plt.plot(time, np.log(counts), "bo")

    plt.figure("Range Graph for R", figsize=(10, 6))
    plt.title("R Value from Each Model")
    plot_data_range(r_dict, "R value")
    plt.xlabel("R")
    plt.savefig("Graphs/R Values")

    plt.show()
