import warnings
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np
import math
try:
    import pymannkendall as mk
    MANN_KENDALL_INSTALLED = True
except ImportError:
    MANN_KENDALL_INSTALLED = False
from collections import namedtuple
from scipy.signal import argrelmax
from scipy.stats import norm
import scipy.stats as stats
# from src.transforms.target_transformations import AdditiveDifferencingTransformer, MultiplicativeDifferencingTransformer, LogTransformer, BoxCoxTransformer, YeoJohnsonTransformer, DetrendingTransformer
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import acf

def _check_convert_y(y):
    assert not np.any(np.isnan(y)), "`y` should not have any nan values"
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()
    assert y.ndim==1
    return y

def _check_stationary_adfuller(y, confidence, **kwargs):
    y = _check_convert_y(y)
    res = namedtuple("ADF_Test", ["stationary", "results"])
    result = adfuller(y, **kwargs)
    if result[1]>confidence:
        return res(False, result)
    else:
        return res(True, result)
    
def _check_stationary_kpss(y, confidence, **kwargs):
    y = _check_convert_y(y)
    res = namedtuple("KPSS_Test", ["stationary", "results"])
    result = kpss(y, **kwargs)
    if result[1]<confidence:
        return res(False, result)
    else:
        return res(True, result)

def check_unit_root(y, confidence=0.05, adf_params={}):
    adf_params['regression'] = "c"
    return _check_stationary_adfuller(y, confidence, **adf_params)

def _check_kendall_tau(y, confidence=0.05):
    y = _check_convert_y(y)
    
    tau, p_value = stats.kendalltau(y, np.arange(len(y)))
    trend=True if p_value<confidence else False
    if tau>0:
        direction="increasing"
    else:
        direction="decreasing"
    return "Kendall_Tau_Test",tau, p_value, trend, direction

#https://abhinaya-sridhar-rajaram.medium.com/mann-kendall-test-in-python-for-trend-detection-in-time-series-bfca5b55b
def _check_mann_kendall(y, confidence=0.05, seasonal_period=None, prewhiten=None):
    if not MANN_KENDALL_INSTALLED:
        raise ValueError("`pymannkendall` needs to be installed for the mann_kendal test. `pip install pymannkendall` to install")
    #https://www.tandfonline.com/doi/pdf/10.1623/hysj.52.4.611
    if prewhiten is None:
        if len(y)<50:
            prewhiten = True
        else:
            prewhiten = False
    else:
        if not prewhiten and len(y)<50:
            warnings.warn("For timeseries with < 50 samples, it is recommended to prewhiten the timeseries. Consider passing `prewhiten=True`")
        if prewhiten and len(y)>50:
            warnings.warn("For timeseries with > 50 samples, it is not recommended to prewhiten the timeseries. Consider passing `prewhiten=False`")
    y = _check_convert_y(y)
    if seasonal_period is None:
        if prewhiten:
            _res = mk.pre_whitening_modification_test(y, alpha=confidence)
        else:
            _res = mk.original_test(y, alpha=confidence)
    else:
        _res = mk.seasonal_test(y, alpha=confidence, period=seasonal_period)
    trend=True if _res.p<confidence else False
    if _res.slope>0:
        direction="increasing"
    else:
        direction="decreasing"
    return type(_res).__name__,_res.slope, _res.p, trend, direction

def check_trend(y, confidence=0.05, seasonal_period=None, mann_kendall=False, prewhiten=None):
    if mann_kendall:
        name, slope, p, trend, direction = _check_mann_kendall(y, confidence, seasonal_period, prewhiten)
    else:
        name, slope, p, trend, direction = _check_kendall_tau(y, confidence)
    det_trend_res = check_deterministic_trend(y, confidence)
    res = namedtuple(name, ["trend", "direction", "slope", "p_value", "deterministic", "deterministic_trend_results"])
    return res(trend, direction, slope, p, det_trend_res.deterministic_trend, det_trend_res)

def check_deterministic_trend(y, confidence=0.05):
    res = namedtuple("ADF_deterministic_Trend_Test", ["deterministic_trend", "adf_res", "adf_ct_res"])
    adf_res = _check_stationary_adfuller(y, confidence)
    adf_ct_res = _check_stationary_adfuller(y, confidence, regression="ct")
    if (not adf_res.stationary) and (adf_ct_res.stationary):
        deterministic_trend = True
    else:
        deterministic_trend = False
    return res(deterministic_trend, adf_res, adf_ct_res)


#https://towardsdatascience.com/heteroscedasticity-is-nothing-to-be-afraid-of-730dd3f7ca1f
def check_heteroscedastisticity(y, confidence=0.05):
    y = _check_convert_y(y)
    res = namedtuple("White_Test", ["heteroscedastic", "lm_statistic", "lm_p_value"])
    #Fitting a linear trend regression
    x = np.arange(len(y))
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    lm_stat, lm_p_value, f_stat, f_p_value = het_white(results.resid, x)
    if lm_p_value<confidence and f_p_value < confidence:
        hetero = True
    else:
        hetero = False
    return res(hetero, lm_stat, lm_p_value)


def _bartlett_formula(r: np.ndarray,
                      m: int,
                      length: int) -> float:
    """
    Computes the standard error of `r` at order `m` with respect to `length` according to Bartlett's formula.
    Parameters
    ----------
    r
        The array whose standard error is to be computed.
    m
        The order of the standard error.
    length
        The size of the underlying sample to be used.
    Returns
    -------
    float
        The standard error of `r` with order `m`.
    """

    if m == 1:
        return math.sqrt(1 / length)
    else:
        return math.sqrt((1 + 2 * sum(map(lambda x: x ** 2, r[:m - 1]))) / length)

# Adapted and generalised fomr https://github.com/unit8co/darts/blob/f0bb54ba26ffea66e199331a1e64b2bf1f92a28b/darts/utils/statistics.py#L25
def check_seasonality(y, max_lag=24, seasonal_period=None, confidence=0.05, verbose=True):
    res = namedtuple("Seasonality_Test", ["seasonal", "seasonal_periods"])
    y = _check_convert_y(y)
    if seasonal_period is not None and (seasonal_period < 2 or not isinstance(seasonal_period, int)):
        raise ValueError('seasonal_period must be an integer greater than 1.')

    if seasonal_period is not None and seasonal_period >= max_lag:
        raise ValueError('max_lag must be greater than seasonal_period.')

    n_unique = np.unique(y).shape[0]

    if n_unique == 1:  # Check for non-constant TimeSeries
        return res(False, 0)
    r = acf(y, nlags=max_lag, fft=False)  # In case user wants to check for seasonality higher than 24 steps.

    # Finds local maxima of Auto-Correlation Function
    candidates = argrelmax(r)[0]

    if len(candidates) == 0:
        if verbose:
            print('The ACF has no local maximum for m < max_lag = {}. Try larger max_lag'.format(max_lag))
        return res(False, 0)

    if seasonal_period is not None:
        # Check for local maximum when m is user defined.
        test = seasonal_period not in candidates

        if test:
            return res(False, seasonal_period)

        candidates = [seasonal_period]

    # Remove r[0], the auto-correlation at lag order 0, that introduces bias.
    r = r[1:]

    # The non-adjusted upper limit of the significance interval.
    band_upper = r.mean() + norm.ppf(1 - confidence / 2) * r.var()

    # Significance test, stops at first admissible value. The two '-1' below
    # compensate for the index change due to the restriction of the original r to r[1:].
    for candidate in candidates:
        stat = _bartlett_formula(r, candidate - 1, len(y))
        if r[candidate - 1] > stat * band_upper:
            return res(True, candidate)
    return res(False, 0)


# def check_stationarity(x, confidence=0.05, adf_params={}, kpss_params={}):
#     if "nlags" not in kpss_params:
#         kpss_params['nlags'] = "auto"
#     adf_params['regression'] = "c"
#     adf_stationary, adf_results = _check_stationary_adfuller(x, confidence, **adf_params)
#     adf_params['regression'] = "ct"
#     adf_ct_stationary, adf_ct_results = _check_stationary_adfuller(x, confidence, **adf_params)
#     kpss_params['regression'] = "c"
#     kpss_stationary, kpss_results = _check_stationary_kpss(x, confidence, **kpss_params)
#     kpss_params['regression'] = "ct"
#     kpss_ct_stationary, kpss_ct_results = _check_stationary_kpss(x, confidence, **kpss_params)
#     ret_dict ={
#         "adf": adf_stationary,
#         "adf_ct": adf_ct_stationary,
#         "kpss": kpss_stationary,
#         "kpss_ct": kpss_ct_stationary,
#         "adf_results": adf_results,
#         "kpss_results": kpss_results,
#         "adf_ct_results": adf_ct_results,
#         "kpss_ct_results": kpss_ct_results
#     }
#     if adf_stationary and kpss_stationary:
#         ret_dict['type'] = "stationary"
#     elif (not adf_stationary and adf_ct_stationary) and (not kpss_stationary and kpss_ct_stationary):
#         ret_dict['type'] = "trend-stationary"
#     else:
#         ret_dict['type'] = "non-stationary"
#     return ret_dict


# TRANSFORM_CLASSES = [LogTransformer, YeoJohnsonTransformer, BoxCoxTransformer]
# DIFFERENCING_CLASSES = [MultiplicativeDifferencingTransformer, LogDifferencingTransformer]
# def make_stationary(y, diff_gap=1, freq=None, confidence=0.05, verbose=False):
#     transforms = []
#     res = check_stationarity(y, confidence)
#     if res['type'] == "stationary":
#         if verbose:
#             print("Series already stationary")
#         return y, transforms
#     else:
#         if verbose:
#             print("Applying Differencing")
#         diff = AdditiveDifferencingTransformer(diff_gap=diff_gap)
#         diff.fit(y, freq=freq)
#         y_diff = diff.transform(y)
#         transforms.append(diff)
#         res = check_stationarity(y_diff.dropna(), confidence)
#         if res['type'] == "stationary":
#             if verbose:
#                 print("Series stationary")
#             return y_diff, transforms
#         elif res['type'] == "trend-stationary":
#             #Detrend
#             if verbose:
#                 print("Detrending")
#             detrender = DetrendingTransformer(degree=1)
#             detrender.fit(y_diff, freq=freq)
#             y_detrend = detrender.transform(y_diff)
#             res = check_stationarity(y_detrend.dropna(), confidence)
#             if res['type'] == "stationary":
#                 #append transform to list
#                 if verbose:
#                     print("Series stationary")
#                 transforms.append(detrender)
#                 return y_detrend, transforms
#         for tr in DIFFERENCING_CLASSES:
#             if verbose:
#                 print(f"Applying {tr.__name__}")
#             transform = tr(diff_gap=diff_gap)
#             transform.fit(y, freq=freq)
#             y_tr = transform.transform(y)
#             res = check_stationarity(y_tr.dropna(), confidence)
#             if res['type'] == "stationary":
#                 if verbose:
#                     print("Series stationary")
#                 transforms.append(transform)
#                 return y_tr, transforms
        
#         for tr in TRANSFORM_CLASSES:
#             if verbose:
#                 print(f"Applying {tr.__name__}")
#             transform = tr()
#             y_tr = transform.fit_transform(y)
#             res = check_stationarity(y_tr.dropna(), confidence)
#             if res['type'] == "stationary":
#                 if verbose:
#                     print("Series stationary")
#                 transforms.append(transform)
#                 return y_tr, transforms
#         # Returning default case of a BoxCoxTransformer
#         transforms.append(transform)
#         return y_tr, transforms