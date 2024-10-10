import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#from data.extracted_glycol_data import get_delta_gamma

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
    0.0001,  # NewFile38
    0.0001,  # NewFile39
    0.0002,  # NewFile40
    0.0002,  # NewFile41
    0.0001,  # NewFile42
    0.0001,  # NewFile43 (keine Angabe)
    0.0001,  # NewFile44
    0.0003,  # NewFile45
    0.0003,  # NewFile46
    0.0002,  # NewFile47
    0.0002,  # NewFile49
    0.0002,  # NewFile50
    0.0001,  # NewFile51 (keine Angabe)
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


B_values = np.array([
    432,    # NewFile38 (keine Angabe)
    432,    # NewFile39 (keine Angabe)
    432,       # NewFile40
    432,       # NewFile41
    432,       # NewFile42
    436,       # NewFile43
    436,       # NewFile44
    436,       # NewFile45
    436,       # NewFile46
    436,       # NewFile47
    436,       # NewFile49
    436,       # NewFile50
    435,       # NewFile51
    435,       # NewFile52
    435,       # NewFile53
    435,       # NewFile54
    435,       # NewFile55
    435,       # NewFile56
    435,       # NewFile57
    435,       # NewFile59
    435,       # NewFile60
    435,       # NewFile61
    435,       # NewFile62
    435        # NewFile63
])

# Unsicherheiten der B-Werte
B_errors = np.array([
    1,
    1,
    1,
    1,
    1,    # NewFile42
    1,         # NewFile43
    1,         # NewFile44
    1,         # NewFile45
    1,         # NewFile46
    1,         # NewFile47
    1,         # NewFile49
    1,         # NewFile50
    4,         # NewFile51
    4,         # NewFile52
    4,         # NewFile53
    4,         # NewFile54
    4,         # NewFile55
    4,         # NewFile56
    4,         # NewFile57
    4,         # NewFile59
    4,         # NewFile60
    4,         # NewFile61
    4,         # NewFile62
    4          # NewFile63

])

def get_delta_gamma(f, df, b, db):
    dg_df= 2*np.pi/b
    dg_db = -2*np.pi*f/(b**2)
    delta_gamma= np.sqrt((dg_df*df)**2 + (dg_db*db)**2)
    return delta_gamma

gamma = 2*np.pi *F_values/B_values
delta_gamma= get_delta_gamma(F_values, F_errors, B_values, B_errors)

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

# Ergebnisse ausgeben
print(f"Steigung: {slope:.4f} ± {slope_err:.4f}")
print(f"Achsenschnittpunkt: {intercept:.2f} ± {intercept_err:.2f}")

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

# Berechnung der angepassten Gerade
F_fit = np.linspace(np.min(gamma), np.max(gamma), 1000)
delta_x0_fit = linear_func(F_fit, slope, intercept)

# Plotten der angepassten Gerade
plt.plot(F_fit, delta_x0_fit, 'r-', label='Linearer Fit')

# Anpassung des Plots
plt.xlabel('Frequenz F (MHz)')
plt.ylabel('Differenz der x-Achsenabschnitte (CH2 - CH1)')
plt.title('Gewichtete lineare Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
x0 = -intercept / slope

# Kovarianz zwischen m und b
sigma_mb = pcov[0, 1]

# Berechnung der Unsicherheit von x0
term1 = (intercept * slope_err / slope**2)**2
term2 = (intercept_err / slope)**2
term3 = -2 * (intercept * sigma_mb) / slope**3

sigma_x0_squared = term1 + term2 + term3
sigma_x0 = np.sqrt(sigma_x0_squared)

print(f"Nullstelle (x0): {x0:.4f} ± {sigma_x0:.4f} MHz")