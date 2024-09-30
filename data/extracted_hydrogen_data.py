import numpy as np
from matplotlib import pyplot as plt

from tools import averager, get_csv_data, show_basic_csv_plot

# Ranges from 14 to 22 for hydrogen
file_index = 22

# Show the plot
show_basic_csv_plot(file_index)

# Datapoints
hydrogen_dips = np.array([[128, 435.5, 740, 1048], [128, 435.5, 740, 1048], [128, 435.5, 740, 1048],
                          [49, 358, 662, 970], [41, 348, 653.5, 961], [28, 334, 641.5, 947],
                          [43.5, 350, 656, 963.5], [67, 371.5, 679.5, 984], [122, 431, 735, 1044]])


