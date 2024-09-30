import pandas as pd
import numpy as np
from tools import averager

from typing import Tuple

from matplotlib import pyplot as plt
from fit_peaks import fit_skew_gaussian


# fit_skew_gaussian(averager(channel_df.ch_one.to_numpy()), 47, 56, True, True)

# plt.figure(figsize=(12, 5))
#
# plt.plot(x, averager(channel_df.ch_one.to_numpy()), c='y', label='CH1')
# plt.plot(x, averager(channel_df.ch_two.to_numpy())* 100, c='b', label='CH2')
#
# plt.legend()
# plt.show()
