# Imports
import numpy as np


def get_fwhm_positions(left_array, right_array):
    return (left_array + right_array) / 2


def get_fwhm_position_errors(left_errors, right_errors):
    return 1 / 2 * np.sqrt(left_errors ** 2 + right_errors ** 2)


def get_asymmetry(dip_positions):
    return np.array([2 * dip_pos[1] - dip_pos[2] - dip_pos[0] for dip_pos in dip_positions])


def get_asymmetry_errors(dip_position_errors):
    root_term = [(2 * dip_pos_err[1]) ** 2 + dip_pos_err[2] ** 2 + dip_pos_err[0] ** 2 for
                 dip_pos_err in dip_position_errors]

    return np.sqrt(np.array(root_term))


def get_gamma(frequencies, B):
    return 2 * np.pi * frequencies / B


def get_gamma_uncertainties(frequencies, B, delta_frequencies, delta_B):
    root_term = delta_frequencies ** 2 + (frequencies * delta_B / B) ** 2

    return 2 * np.pi / B * np.sqrt(root_term)
