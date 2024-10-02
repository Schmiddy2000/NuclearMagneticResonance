import numpy as np
from matplotlib import pyplot as plt
nf_5 = np.array([125, 434.3, 737.2, 1046.8])
nf_6 = np.array([125, 434.2, 737.3, 1047])
nf_7 = np.array([107, 417, 719.3, 1030.8])
nf_8 = np.array([89, 401, 701.8, 1013])
nf_9 = np.array([52.8, 359, 665, 971])
nf_10 = np.array([41, 348.8, 653, 961.2])
nf_11 = np.array([35, 341.2, 648, 954.5])
nf_12 = np.array([26, 332.2, 639.1, 945])
nf_13 = np.array([22, 329.2, 635, 941.2])
arrays = [nf_5, nf_6, nf_7, nf_8, nf_9, nf_10, nf_11, nf_12, nf_13]
frequency = np.array([19.4568, 19.4447, 19.4400, 19.4166, 19.4043, 19.3982, 19.3894, 19.3802, 19.3749])
# Funktion zur Berechnung der Abstände der Abstände
def calculate_abs_differences_sum(arr):
    first_differences = np.diff(arr)
    second_differences = np.diff(first_differences)
    return np.sum(np.abs(second_differences))

# Calculate the absolute sums of second differences for each array
abs_second_differences_sums = [calculate_abs_differences_sum(arr) for arr in arrays]
print(len(frequency), len(abs_second_differences_sums))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(frequency, abs_second_differences_sums, marker='o', linestyle='-', color='blue')

# Adding labels and title
plt.xlabel("Frequency (MHz)", fontsize=14)
plt.ylabel("Sum of Second Differences", fontsize=14)
plt.title("Sum of Second Differences vs Frequency", fontsize=16)

# Grid and plot
plt.grid(True)
plt.tight_layout()
plt.show()