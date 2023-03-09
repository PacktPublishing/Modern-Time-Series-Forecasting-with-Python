import warnings

import numpy as np


def calc_norm_sd(x, original):
    if (len(x) <= 2) and np.all(x == 0):
        warnings.warn(
            "Array should not be all zeroes or should atleast more than 1 datapoint. COV will be NaN"
        )
        cov = np.nan
    else:
        cov = np.std(x) / np.mean(original)
    return cov


def calc_cov(x):
    if (len(x) <= 2) and np.all(x == 0):
        warnings.warn(
            "Array should not be all zeroes or should atleast more than 1 datapoint. COV will be NaN"
        )
        cov = np.nan
    else:
        cov = np.std(x) / np.mean(x)
    return cov
