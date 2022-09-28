import numpy as np
from functools import partial
from src.decomposition.seasonal import _detrend

def make_stationary(x: np.ndarray, method: str="detrend", detrend_kwargs:dict={}):
    """Utility to make time series stationary

    Args:
        x (np.ndarray): The time series array to be made stationary
        method (str, optional): {"detrend","logdiff"}. Defaults to "detrend".
        detrend_kwargs (dict, optional): These kwargs will be passed on to the detrend method
    """
    if method=="detrend":
        detrend_kwargs["return_trend"] = True
        stationary, trend = _detrend(x, **detrend_kwargs)
        def inverse_transform(st, trend):
            return st+trend
        return stationary, partial(inverse_transform, trend=trend)
    elif method == "logdiff":
        stationary = np.log(x[:-1]/x[1:])
        def inverse_transform(st, x):
            _x = np.exp(st)
            return _x*x[1:]
        return stationary, partial(inverse_transform, x=x)

from darts import TimeSeries
from darts.metrics.metrics import _get_values_or_raise
from darts.metrics import metrics as dart_metrics
from typing import Optional, Tuple, Union, Sequence, Callable, cast
from src.utils.data_utils import is_datetime_dtypes
import pandas as pd

def _remove_nan_union(array_a: np.ndarray,
                      array_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the two inputs arrays where all elements are deleted that have an index that corresponds to
    a NaN value in either of the two input arrays.
    """

    isnan_mask = np.logical_or(np.isnan(array_a), np.isnan(array_b))
    return np.delete(array_a, isnan_mask), np.delete(array_b, isnan_mask)

def forecast_bias(actual_series: Union[TimeSeries, Sequence[TimeSeries], np.ndarray],
        pred_series: Union[TimeSeries, Sequence[TimeSeries], np.ndarray],
        intersect: bool = True,
        *,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False) -> Union[float, np.ndarray]:
    """ Forecast Bias (FB).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` or `Sequence[TimeSeries]` of actual values.
    pred_series
        The `TimeSeries` or `Sequence[TimeSeries]` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate `TimeSeries` instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        `Sequence[TimeSeries]`. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
        passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        The Forecast Bias (OPE)
    """
    assert type(actual_series) is type(pred_series), "actual_series and pred_series should be of same type."
    if isinstance(actual_series, np.ndarray):
        y_true, y_pred = actual_series, pred_series
    else:
        y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    y_true_sum, y_pred_sum = np.sum(y_true), np.sum(y_pred)
    # raise_if_not(y_true_sum > 0, 'The series of actual value cannot sum to zero when computing OPE.', logger)
    return ((y_true_sum - y_pred_sum) / y_true_sum) * 100.

def cast_to_series(df):
    is_pd_dataframe = isinstance(df, pd.DataFrame)    
    if is_pd_dataframe: 
        if df.shape[1]==1:
            df = df.squeeze()
        else:
            raise ValueError("Dataframes with more than one columns cannot be converted to pd.Series")
    return df

def darts_metrics_adapter(metric_func, actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        insample: Union[TimeSeries, Sequence[TimeSeries]] = None,
        m: Optional[int] = 1,
        intersect: bool = True,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False):
    
    actual_series, pred_series = cast_to_series(actual_series), cast_to_series(pred_series)
    if insample is not None:
        insample = cast_to_series(insample)
    assert type(actual_series) is type(pred_series), f"actual_series({type(actual_series)}) and pred_series({type(pred_series)}) should be of same type."
    if insample is not None:
        assert type(actual_series) is type(insample), "actual_series and insample should be of same type."
    is_nd_array = isinstance(actual_series, np.ndarray)
    is_pd_series = isinstance(actual_series, pd.Series)
    
    if is_pd_series:
        is_datetime_index = is_datetime_dtypes(actual_series.index) and is_datetime_dtypes(pred_series.index)
        if insample is not None:
            is_datetime_index = is_datetime_index and is_datetime_dtypes(insample.index)
    else:
        is_datetime_index = False
    if metric_func.__name__ == "mase":
        if not is_datetime_index:
            raise ValueError("MASE needs pandas Series with datetime index as inputs")
    
    if is_nd_array or (is_pd_series and not is_datetime_index):
        actual_series, pred_series = TimeSeries.from_values(actual_series.values if is_pd_series else actual_series), TimeSeries.from_values(pred_series.values if is_pd_series else pred_series)
        if insample is not None:
            insample = TimeSeries.from_values(insample.values if is_pd_series else insample)

    elif is_pd_series and is_datetime_index:
        actual_series, pred_series = TimeSeries.from_series(actual_series), TimeSeries.from_series(pred_series)
        if insample is not None:
            insample = TimeSeries.from_series(insample)
    else:
        raise ValueError()
    if metric_func.__name__ == "mase":
        return metric_func(actual_series=actual_series, pred_series=pred_series, insample=insample, m=m, intersect=intersect, reduction=reduction, inter_reduction=inter_reduction, n_jobs=n_jobs, verbose=verbose)
    else:
        return metric_func(actual_series=actual_series, pred_series=pred_series, intersect=intersect, reduction=reduction, inter_reduction=inter_reduction, n_jobs=n_jobs, verbose=verbose)

def mae(actuals, predictions):
    return np.nanmean(np.abs(actuals-predictions))

def mse(actuals, predictions):
    return np.nanmean(np.power(actuals-predictions, 2))

def forecast_bias_aggregate(actuals, predictions):
    return 100*(np.nansum(predictions)-np.nansum(actuals))/np.nansum(actuals)

def rmsse(
    actual_series,
    pred_series,
    insample,
    m = 1,
    intersect = True,
    *,
    reduction = np.mean,
):

    def _multivariate_mase(
        actual_series,
        pred_series,
        insample,
        m,
        intersect,
        reduction,
    ):

        assert actual_series.width == pred_series.width, "The two TimeSeries instances must have the same width."
        
        assert actual_series.width == insample.width, "The insample TimeSeries must have the same width as the other series."
        
        assert insample.end_time() + insample.freq == pred_series.start_time(), "The pred_series must be the forecast of the insample series"

        insample_ = (
            insample.quantile_timeseries(quantile=0.5)
            if insample.is_stochastic
            else insample
        )

        value_list = []
        for i in range(actual_series.width):
            # old implementation of mase on univariate TimeSeries
            y_true, y_hat = _get_values_or_raise(
                actual_series.univariate_component(i),
                pred_series.univariate_component(i),
                intersect,
                remove_nan_union=False,
            )

            x_t = insample_.univariate_component(i).values()
            errors = np.square(y_true - y_hat)
            scale = np.mean(np.square(x_t[m:] - x_t[:-m]))
            assert not np.isclose(scale, 0), "cannot use MASE with periodical signals"
            value_list.append(np.sqrt(np.mean(errors / scale)))

        return reduction(value_list)

    if isinstance(actual_series, TimeSeries):
        assert isinstance(pred_series, TimeSeries), "Expecting pred_series to be TimeSeries"
        assert isinstance(insample, TimeSeries), "Expecting insample to be TimeSeries"
        return _multivariate_mase(
            actual_series=actual_series,
            pred_series=pred_series,
            insample=insample,
            m=m,
            intersect=intersect,
            reduction=reduction,
        )
    else:
        raise(
            ValueError(
                "Input type not supported, only TimeSeries is accepted."
            )
        )