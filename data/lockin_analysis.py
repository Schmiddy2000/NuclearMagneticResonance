import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Ihre vorhandenen Datenarrays
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
    18.5491,
    18.549,
    18.5417,
    18.5342,
    18.5646,
    18.5646,
    18.5492,
    18.5304,
    18.5304,
    18.5171,
    18.5301,
    18.5468,
    18.5473,
    18.5471,
    18.5471,
    18.5471,
    18.5471,
    18.5471,
    18.5471,
    18.5468,
    18.5544,
    18.5653,
    18.5347,
    18.5248
])

F_errors = np.array([
    0.0001,
    0.0001,
    0.0002,
    0.0002,
    0.0001,
    0.0001,
    0.0001,
    0.0003,
    0.0003,
    0.0002,
    0.0002,
    0.0002,
    0.0001,
    0.0002,
    0.0002,
    0.0002,
    0.0005,
    0.0005,
    0.0005,
    0.0002,
    0.0002,
    0.0002,
    0.0002,
    0.0002
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

def get_delta_gamma(f, df, b, db):
    dg_df = 2 * np.pi / b
    dg_db = -2 * np.pi * f / (b**2)
    delta_gamma = np.sqrt((dg_df * df)**2 + (dg_db * db)**2)
    return delta_gamma

gamma = 2 * np.pi * F_values / B_values
delta_gamma = get_delta_gamma(F_values, F_errors, B_values, B_errors)

# Definieren des linearen Modells
def linear_func(x, m, b):
    return m * x + b

# Durchführung der gewichteten linearen Regression
popt, pcov = curve_fit(
    linear_func,
    gamma,
    delta_x0,
    sigma=delta_x0_err,
    absolute_sigma=True
)

slope, intercept = popt
slope_err, intercept_err = np.sqrt(np.diag(pcov))

# Kovarianz zwischen m und b
sigma_mb = pcov[0, 1]

# Ergebnisse ausgeben
print(f"Steigung: {slope:.4f} ± {slope_err:.4f}")
print(f"Achsenschnittpunkt: {intercept:.2f} ± {intercept_err:.2f}")

# Berechnung der angepassten Gerade
gamma_fit = np.linspace(np.min(gamma), np.max(gamma), 1000)
delta_x0_fit = linear_func(gamma_fit, slope, intercept)
# Nullstelle berechnen
x0 = -intercept / slope

# Berechnung der Unsicherheit von x0
term1 = (intercept * slope_err / slope**2)**2
term2 = (intercept_err / slope)**2
term3 = -2 * (intercept * sigma_mb) / slope**3

sigma_x0_squared = term1 + term2 + term3
sigma_x0 = np.sqrt(sigma_x0_squared)

print(f"Nullstelle (x0): {x0:.4f} ± {sigma_x0:.4f} rad MHz T$^{-1}$")
# Berechnung der Standardfehler für die Konfidenzbänder
def calc_se_y(x):
    var_y = (x**2) * slope_err**2 + intercept_err**2 + 2 * x * sigma_mb
    return np.sqrt(var_y)

# Freiheitsgrade
n = len(gamma)
p = 2  # Anzahl der Parameter (Steigung und Achsenabschnitt)
dof = max(0, n - p)

# t-Wert für das gewünschte Konfidenzniveau
confidence_level = 0.95
alpha = 1.0 - confidence_level
t_val = stats.t.ppf(1.0 - alpha/2., dof)

# Standardfehler berechnen
se_y_fit = calc_se_y(gamma_fit)

# Konfidenzband berechnen
delta = t_val * se_y_fit
upper = delta_x0_fit + delta
lower = delta_x0_fit - delta

# Plot der Datenpunkte mit Fehlerbalken
plt.figure(figsize=(12, 6))
plt.errorbar(
    gamma,
    delta_x0,
    yerr=delta_x0_err,
    fmt='o',
    label='Daten',
    capsize=5
)

# Plotten der angepassten Gerade
plt.plot(gamma_fit, delta_x0_fit, 'r-', label='Linearer Fit')

# Plotten des Konfidenzbandes
plt.fill_between(gamma_fit, lower, upper, color='pink', alpha=0.3, label='95% Konfidenzband')
# Hinzufügen der vertikalen Linie für die Nullstelle
plt.axvline(x=x0, color='green', linestyle='--', label=f'Nullstelle x0 = {x0:.2f}')

# Hinzufügen des Bereichs der Unsicherheit
plt.axvspan(x0 - sigma_x0, x0 + sigma_x0, color='green', alpha=0.2, label='Unsicherheit x0')

# Anpassung des Plots
plt.xlabel('γ (rad s$^{-1}$ T$^{-1}$)')
plt.ylabel('Differenz der x-Achsenabschnitte (CH2 - CH1)')
plt.title('Gewichtete lineare Regression mit Konfidenzband')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lockin_distances.png', dpi=200)
plt.show()

