import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Tuple


def get_csv_data(file_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_path = f'C:/Users/benni/PycharmProjects/NuclearMagneticResonance/data/oscilloscope_data/NewFile{file_index}.csv'
    df = pd.read_csv(file_path)

    # Überprüfen, ob die benötigten Spalten vorhanden sind
    required_columns = ['X', 'CH3', 'CH2']
    if not all(col in df.columns for col in required_columns):
        print("Die CSV-Datei enthält nicht die benötigten Spalten.")
        return np.array([]), np.array([]), np.array([])

    channel_df = df.drop([0], axis=0)  # Entfernen der ersten Zeile, falls erforderlich
    channel_df = channel_df[['X', 'CH3', 'CH2']]
    channel_df.columns = ['x', 'ch_one', 'ch_two']
    channel_df = channel_df.astype(np.float64)

    x = channel_df['x'].to_numpy()
    ch_1 = channel_df['ch_one'].to_numpy()
    ch_2 = channel_df['ch_two'].to_numpy()

    return x, ch_1, ch_2


def perform_linear_regression(x: np.ndarray, y: np.ndarray, x_start: float, x_end: float, channel_name: str):
    # Auswahl des Bereichs
    mask = (x >= x_start) & (x <= x_end)
    x_reg = x[mask]
    y_reg = y[mask]

    # Überprüfen, ob Daten vorhanden sind
    if len(x_reg) == 0:
        print(f"Der ausgewählte Bereich enthält keine Daten für {channel_name}. Bitte überprüfen Sie 'x_start' und 'x_end'.")
        return None

    # Lineare Regression durchführen
    res = linregress(x_reg, y_reg)
    slope = res.slope
    intercept = res.intercept
    r_value = res.rvalue
    p_value = res.pvalue
    std_err = res.stderr
    intercept_stderr = res.intercept_stderr if hasattr(res, 'intercept_stderr') else compute_intercept_stderr(x_reg, std_err)

    # Berechnung des x-Achsenabschnitts und dessen Unsicherheit
    x0, x0_err = compute_x_intercept_and_error(slope, intercept, std_err, intercept_stderr)

    # Ergebnisse ausgeben
    print(f"Lineare Regression für {channel_name}:")
    print(f"Steigung: {slope} ± {std_err}")
    print(f"Achsenschnittpunkt: {intercept} ± {intercept_stderr}")
    print(f"Korrelationskoeffizient r: {r_value}")
    print(f"x-Achsenabschnitt (Nullstelle): {x0} ± {x0_err}\n")

    # Rückgabe der Ergebnisse für das Plotten und weitere Berechnungen
    return {
        'x_reg': x_reg,
        'y_reg': y_reg,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'intercept_stderr': intercept_stderr,
        'x0': x0,
        'x0_err': x0_err,
        'channel_name': channel_name
    }


def compute_intercept_stderr(x: np.ndarray, slope_stderr: float) -> float:
    n = len(x)
    x_mean = np.mean(x)
    s_xx = np.sum((x - x_mean) ** 2)
    intercept_stderr = slope_stderr * np.sqrt(np.sum(x ** 2) / (n * s_xx))
    return intercept_stderr


def compute_x_intercept_and_error(slope: float, intercept: float, slope_stderr: float, intercept_stderr: float) -> Tuple[float, float]:
    x0 = -intercept / slope
    # Fehlerfortpflanzung
    x0_err = np.sqrt(
        (intercept_stderr / slope) ** 2 +
        (intercept * slope_stderr / slope ** 2) ** 2
    )
    return x0, x0_err


# Beispielaufruf
file_index = 42  # Passen Sie die Dateinummer entsprechend an

# Geben Sie unterschiedliche x_start und x_end für CH1 und CH2 an
x_start_ch1, x_end_ch1 = (552, 598)  # Bereich für CH1
x_start_ch2, x_end_ch2 = (0, 1200)  # Bereich für CH2

# Daten abrufen
x, ch_1, ch_2 = get_csv_data(file_index)

if x.size == 0:
    print("Es wurden keine Daten geladen. Bitte überprüfen Sie den Dateipfad und die Dateiinhalte.")
else:
    # Lineare Regression für CH1 durchführen
    result_ch1 = perform_linear_regression(x, ch_1, x_start_ch1, x_end_ch1, 'CH1')

    # Lineare Regression für CH2 durchführen
    result_ch2 = perform_linear_regression(x, ch_2, x_start_ch2, x_end_ch2, 'CH2')

    # Überprüfen, ob beide Regressionen erfolgreich waren
    if result_ch1 is not None and result_ch2 is not None:
        # Beide Kanäle in einem Plot darstellen
        plt.figure(figsize=(10, 6))

        # Gesamten x-Bereich für das Plotten definieren
        x_plot = np.linspace(min(x), max(x), 1000)


        # CH2 Daten und Fit
        plt.plot(x, ch_2, 'x-', label='CH2 Daten', markersize=5)
        plt.plot(x_plot, result_ch2['slope'] * x_plot + result_ch2['intercept'], '-', label='CH2 Fit')
        plt.plot(x, ch_1, 'x-', label='CH1 Daten', markersize=5)
        plt.plot(x_plot, result_ch1['slope'] * x_plot + result_ch1['intercept'], '-', label='CH1 Fit')

        plt.xlabel('X')
        plt.xlim(520, 620)
        plt.ylim(-8, 11)
        plt.ylabel('Signal')
        plt.title('Lineare Regression von CH1 und CH2 gegen X')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Vergleich der x-Achsenabschnitte
        x0_ch1 = result_ch1['x0']
        x0_err_ch1 = result_ch1['x0_err']
        x0_ch2 = result_ch2['x0']
        x0_err_ch2 = result_ch2['x0_err']
        delta_x0 = x0_ch2 - x0_ch1
        delta_x0_err = np.sqrt(x0_err_ch1 ** 2 + x0_err_ch2 ** 2)
        print(f"Differenz der x-Achsenabschnitte (CH2 - CH1): {delta_x0} ± {delta_x0_err}")
