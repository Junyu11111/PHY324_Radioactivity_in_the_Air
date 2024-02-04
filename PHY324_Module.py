import numpy as np


def multiply_with_errors(values_with_uncertainties):
    """
    values_with_uncertainties:[[value, uncertainty]]
    """
    product = 1.0
    squared_relative_error_sum = 0.0

    for value_with_uncertainty in values_with_uncertainties:
        value, uncertainty = value_with_uncertainty
        product *= value
        squared_relative_error_sum += (uncertainty / value) ** 2

    total_uncertainty = product * np.sqrt(squared_relative_error_sum)

    return product, total_uncertainty


def divide_with_errors(value1_with_uncertainty, value2_with_uncertainty):
    value1, delta1 = value1_with_uncertainty
    value2, delta2 = value2_with_uncertainty

    quotient = value1 / value2

    relative_error1 = delta1 / value1
    relative_error2 = delta2 / value2
    total_relative_error = np.sqrt(relative_error1 ** 2 + relative_error2 ** 2)
    quotient_error = quotient * total_relative_error

    return quotient, quotient_error


def sum_with_errors(values_with_uncertainties: list[list]):
    sum_result = 0.0
    squared_errors_sum = 0.0

    for value_uncertainty_list in values_with_uncertainties:
        value, uncertainty = value_uncertainty_list
        sum_result += value
        squared_errors_sum += uncertainty ** 2

    sum_error = np.sqrt(squared_errors_sum)

    return sum_result, sum_error


def power_with_error(base_with_uncertainty, exponent):
    base, base_uncertainty = base_with_uncertainty

    power_result = base ** exponent
    power_error = power_result * abs(exponent) * (base_uncertainty / base)

    return power_result, power_error


