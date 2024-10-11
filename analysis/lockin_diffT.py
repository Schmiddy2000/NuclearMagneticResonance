import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from data.lockin_analysis import get_delta_gamma

# --- Data Arrays ---
delta_x0 = np.array([
    14.66977569943515,
    20.595493236009133,
    28.159563733640653,
    -38.094024530603576,
    -96.11714740725733,
    -65.81745617497467,
    -24.35934414947897,
    221.53604073078725,
    208.89932697056395,
    400.82052664996297,
    96.25727493981975,
    -30.53227962042922,
    -39.600694487414785,
    -35.05061208387622,
    -17.759288363603247,
    -18.409125485020695,
    -12.948681398892589,
    -13.453189253991042,
    -11.841789476345411,
    -16.056281436216068,
    -47.59757452407712,
    -100.54538262682252,
    38.18038109018471,
    76.7583915249092
])

delta_x0_err = np.array([
    19.5126596429879,
    36.06296669504264,
    108.3595793911669,
    10.76694244225714,
    53.290596721220005,
    49.53479768133792,
    17.590936938387685,
    50.32907180995178,
    53.19117544852634,
    35.90517079278487,
    31.249872986056037,
    26.83941246884079,
    101.59283147727473,
    66.91383584861775,
    14.805372885000786,
    48.33536937864743,
    7.573629834919619,
    8.066747244654373,
    59.647910144039514,
    35.314613066440316,
    37.755805987331385,
    29.50067626825329,
    18.09888435871548,
    25.777399926597898
])

F_values = np.array([
    18.5491,  # NewFile38
    18.549,   # NewFile39
    18.5417,  # NewFile40
    18.5342,  # NewFile41
    18.5646,  # NewFile42
    18.5646,  # NewFile43
    18.5492,  # NewFile44
    18.5304,  # NewFile45
    18.5304,  # NewFile46
    18.5171,  # NewFile47
    18.5301,  # NewFile49
    18.5468,  # NewFile50
    18.5473,  # NewFile51
    18.5471,  # NewFile52
    18.5471,  # NewFile53
    18.5471,  # NewFile54
    18.5471,  # NewFile55
    18.5471,  # NewFile56
    18.5471,  # NewFile57
    18.5468,  # NewFile59
    18.5544,  # NewFile60
    18.5653,  # NewFile61
    18.5347,  # NewFile62
    18.5248   # NewFile63
])

F_errors = np.array([
    0.0001,  # NewFile38
    0.0001,  # NewFile39
    0.0002,  # NewFile40
    0.0002,  # NewFile41
    0.0001,  # NewFile42
    0.0001,  # NewFile43
    0.0001,  # NewFile44
    0.0003,  # NewFile45
    0.0003,  # NewFile46
    0.0002,  # NewFile47
    0.0002,  # NewFile49
    0.0002,  # NewFile50
    0.0001,  # NewFile51
    0.0002,  # NewFile52
    0.0002,  # NewFile53
    0.0002,  # NewFile54
    0.0005,  # NewFile55
    0.0005,  # NewFile56
    0.0005,  # NewFile57
    0.0002,  # NewFile59
    0.0002,  # NewFile60
    0.0002,  # NewFile61
    0.0002,  # NewFile62
    0.0002   # NewFile63
])

# Period duration in seconds for each measurement
T_values = np.array([
    100,  # NewFile38
    300,  # NewFile39
    300,  # NewFile40
    300,  # NewFile41
    300,  # NewFile42
    100,  # NewFile43
    100,  # NewFile44
    100,  # NewFile45
    100,  # NewFile46
    100,  # NewFile47
    100,  # NewFile49
    100,  # NewFile50
     30,  # NewFile51
     30,  # NewFile52
     30,  # NewFile53
     30,  # NewFile54
     30,  # NewFile55
     30,  # NewFile56
     30,  # NewFile57
     30,  # NewFile59
     30,  # NewFile60
     30,  # NewFile61
     30,  # NewFile62
     30   # NewFile63
])
B_values = np.array([
    432,
    432,
    432,
    432,
    432,
    436,
    436,
    436,
    436,
    436,
    436,
    436,
    435,
    435,
    435,
    435,
    435,
    435,
    435,
    435,
    435,
    435,
    435,
    435
]) * 1e-3  # Umrechnung in Tesla

B_errors = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4
]) * 1e-3  # Umrechnung in Tesla

def get_gamma(F_values, B_values):
    return (2*np.pi *F_values/B_values)
delta_gamma = get_delta_gamma(F_values, F_errors, B_values, B_errors)

# --- Group data by period duration ---
# Create boolean indices for each period duration
indices_T100 = (T_values == 100)
indices_T300 = (T_values == 300)
indices_T30 = (T_values == 30)

# Extract data for T = 100 seconds
F_values_T100 = F_values[indices_T100]
delta_x0_T100 = delta_x0[indices_T100]
delta_x0_err_T100 = delta_x0_err[indices_T100]
F_errors_T100 = F_errors[indices_T100]

# Extract data for T = 300 seconds
F_values_T300 = F_values[indices_T300]
delta_x0_T300 = delta_x0[indices_T300]
delta_x0_err_T300 = delta_x0_err[indices_T300]
F_errors_T300 = F_errors[indices_T300]

# Extract data for T = 30 seconds
F_values_T30 = F_values[indices_T30]
delta_x0_T30 = delta_x0[indices_T30]
delta_x0_err_T30 = delta_x0_err[indices_T30]
F_errors_T30 = F_errors[indices_T30]

# --- Define the linear model ---
def linear_func(x, m, b):
    return m * x + b

# --- Perform linear regression for each period duration ---
from scipy.optimize import curve_fit

# Fit for T = 100 seconds
popt_T100, pcov_T100 = curve_fit(
    linear_func,
    F_values_T100,
    delta_x0_T100,
    sigma=delta_x0_err_T100,
    absolute_sigma=True
)
slope_T100, intercept_T100 = popt_T100
slope_err_T100, intercept_err_T100 = np.sqrt(np.diag(pcov_T100))

# Fit for T = 300 seconds
popt_T300, pcov_T300 = curve_fit(
    linear_func,
    F_values_T300,
    delta_x0_T300,
    sigma=delta_x0_err_T300,
    absolute_sigma=True
)
slope_T300, intercept_T300 = popt_T300
slope_err_T300, intercept_err_T300 = np.sqrt(np.diag(pcov_T300))

# Fit for T = 30 seconds
popt_T30, pcov_T30 = curve_fit(
    linear_func,
    F_values_T30,
    delta_x0_T30,
    sigma=delta_x0_err_T30,
    absolute_sigma=True
)
slope_T30, intercept_T30 = popt_T30
slope_err_T30, intercept_err_T30 = np.sqrt(np.diag(pcov_T30))

# --- Calculate zeros (x-intercepts) and their uncertainties ---
def calculate_zero_intercept(slope, intercept, slope_err, intercept_err, cov):
    """
    Calculates the x-intercept (-intercept/slope) and its uncertainty.
    """
    x0 = -intercept / slope
    # Error propagation formula
    x0_err = np.sqrt(
        (intercept_err / slope)**2 +
        (intercept * slope_err / slope**2)**2 -
        2 * (intercept / slope**3) * cov
    )
    return x0, x0_err

# Calculate zeros for each period duration
x0_T100, x0_err_T100 = calculate_zero_intercept(
    slope_T100, intercept_T100, slope_err_T100, intercept_err_T100, pcov_T100[0,1]
)
x0_T300, x0_err_T300 = calculate_zero_intercept(
    slope_T300, intercept_T300, slope_err_T300, intercept_err_T300, pcov_T300[0,1]
)
x0_T30, x0_err_T30 = calculate_zero_intercept(
    slope_T30, intercept_T30, slope_err_T30, intercept_err_T30, pcov_T30[0,1]
)

# --- Output the results ---
print(f"T = 100 s:")
print(f"Slope: {slope_T100:.4f} ± {slope_err_T100:.4f}")
print(f"Intercept: {intercept_T100:.2f} ± {intercept_err_T100:.2f}")
print(f"Zero (x-intercept): {x0_T100:.4f} ± {x0_err_T100:.4f} MHz\n")

print(f"T = 300 s:")
print(f"Slope: {slope_T300:.4f} ± {slope_err_T300:.4f}")
print(f"Intercept: {intercept_T300:.2f} ± {intercept_err_T300:.2f}")
print(f"Zero (x-intercept): {x0_T300:.4f} ± {x0_err_T300:.4f} MHz\n")

print(f"T = 30 s:")
print(f"Slope: {slope_T30:.4f} ± {slope_err_T30:.4f}")
print(f"Intercept: {intercept_T30:.2f} ± {intercept_err_T30:.2f}")
print(f"Zero (x-intercept): {x0_T30:.4f} ± {x0_err_T30:.4f} MHz\n")

# --- Plotting the data, fits, and zeros with uncertainties ---
plt.figure(figsize=(10, 6))

# Data and fit for T = 100 seconds
plt.errorbar(
    F_values_T100,
    delta_x0_T100,
    yerr=delta_x0_err_T100,
    xerr=F_errors_T100,
    fmt='o', color='darkblue',
    label='T = 100 s',
    capsize=5
)
F_fit_T100 = np.linspace(min(F_values_T100), max(F_values_T100), 100)
plt.plot(
    F_fit_T100,
    linear_func(F_fit_T100, *popt_T100),
    label='Fit T = 100 s', color='skyblue'
)
# Vertical span for zero crossing uncertainty
plt.axvspan(
    x0_T100 - x0_err_T100,
    x0_T100 + x0_err_T100,
    color='blue',
    alpha=0.2,
    label=f'Zero T=100 s: {x0_T100:.2f} ± {x0_err_T100:.2f} MHz'
)
plt.axvline(x0_T100, color='blue')
# Data and fit for T = 30 seconds
plt.errorbar(
    F_values_T30,
    delta_x0_T30,
    yerr=delta_x0_err_T30,
    xerr=F_errors_T30,
    fmt='^', color='red',
    label='T = 30 s',
    capsize=5
)
F_fit_T30 = np.linspace(min(F_values_T30), max(F_values_T30), 100)
plt.plot(
    F_fit_T30,
    linear_func(F_fit_T30, *popt_T30),
    label='Fit T = 30 s', color='orange'
)
plt.axvline(x0_T30, color='darkred')
# Vertical span for zero crossing uncertainty
plt.axvspan(
    x0_T30 - x0_err_T30,
    x0_T30 + x0_err_T30,
    color='darkred',
    alpha=0.2,
    label=f'Zero T=30 s: {x0_T30:.2f} ± {x0_err_T30:.2f} MHz'
)

# --- Customize the plot ---
plt.xlabel('Frequency F (MHz)')
plt.ylabel('Difference of x-axis intercepts (CH2 - CH1)')
plt.title('Data for Different Period Durations with Zero Crossings')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lockin_final.png', dpi=200)
plt.show()
print(get_gamma(x0_T30, np.mean(B_values)), get_delta_gamma(x0_T30, x0_err_T30, np.mean(B_values), np.mean(B_errors)))

print(get_gamma(x0_T100, np.mean(B_values)), get_delta_gamma(x0_T100, x0_err_T100, np.mean(B_values), np.mean(B_errors)))