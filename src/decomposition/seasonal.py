import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from pandas.core.nanops import nanmean as pd_nanmean
from plotly.subplots import make_subplots
from sklearn.linear_model import RidgeCV
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.seasonal import DecomposeResult


# Ported from statsmodels
def _get_pandas_wrapper(X, trim_head=None, trim_tail=None, names=None):
    index = X.index
    # TODO: allow use index labels
    if trim_head is None and trim_tail is None:
        index = index
    elif trim_tail is None:
        index = index[trim_head:]
    elif trim_head is None:
        index = index[:-trim_tail]
    else:
        index = index[trim_head:-trim_tail]
    if hasattr(X, "columns"):
        if names is None:
            names = X.columns
        return lambda x: X.__class__(x, index=index, columns=names)
    else:
        if names is None:
            names = X.name
        return lambda x: X.__class__(x, index=index, name=names)


def _maybe_get_pandas_wrapper(X, trim_head=None, trim_tail=None):
    """
    If using pandas returns a function to wrap the results, e.g., wrapper(X)
    trim is an integer for the symmetric truncation of the series in some
    filters.
    otherwise returns None
    """
    if _is_using_pandas(X, None):
        return _get_pandas_wrapper(X, trim_head, trim_tail)
    else:
        return


def _maybe_get_pandas_wrapper_freq(X, trim=None):
    if _is_using_pandas(X, None):
        index = X.index
        func = _get_pandas_wrapper(X, trim)
        freq = index.inferred_freq
        return func, freq
    else:
        return lambda x: x, None


def _detrend(x, lo_frac=0.6, lo_delta=0.01, return_trend=False):
    # use some existing pieces of statsmodels
    lowess = sm.nonparametric.lowess
    # get plain np array
    observed = np.asanyarray(x).squeeze()
    # calc trend, remove from observation
    trend = lowess(
        observed,
        [x for x in range(len(observed))],
        frac=lo_frac,
        delta=lo_delta * len(observed),
        return_sorted=False,
    )
    detrended = observed - trend
    return detrended, trend if return_trend else detrended


class DecomposeResult(DecomposeResult):
    """
    A small tweak to the standard statsmodes return object to allow interactive plotly plots
    """

    def __init__(self, observed, seasonal, trend, resid, weights=None):
        super().__init__(observed, seasonal, trend, resid, weights=weights)
        self.is_multi = isinstance(self.seasonal, dict)

    @property
    def total_seasonality(self):
        return (
            np.sum(list(self.seasonal.values()), axis=0)
            if isinstance(self.seasonal, OrderedDict)
            else self.seasonal
        )

    def plot(
        self, observed=True, seasonal=True, trend=True, resid=True, interactive=True
    ):
        """Plots the decomposition output

        Args:
            observed (bool, optional): Flag to turn off plotting the original. Defaults to True.
            seasonal (bool, optional): Flag to turn off plotting the seasonal component(s). Defaults to True.
            trend (bool, optional): Flag to turn off plotting the trend component. Defaults to True.
            resid (bool, optional): Flag to turn off plotting the residual component. Defaults to True.
            interactive (bool, optional): Flag to turn off plotly plots and revert to matplotlib. Defaults to True.

        Raises:
            ValueError: If all the compoenent flags are `False`, throws a ValueError
        """
        if interactive or self.is_multi:
            series = []
            if observed:
                series += ["Original"]
            if trend:
                series += ["Trend"]
            if seasonal:
                if self.is_multi:
                    series += list(self.seasonal.keys())
                else:
                    series += ["Seasonal"]
            if resid:
                series += ["Residual"]
            if len(series) == 0:
                raise ValueError(
                    "All component flags were off. Need atleast one of the flags turned on to plot."
                )
            fig = make_subplots(
                rows=len(series), cols=1, shared_xaxes=True, subplot_titles=series
            )
            x = self.trend.index
            row = 1
            if observed:
                fig.append_trace(
                    go.Scatter(x=x, y=self.observed, name="Original"), row=row, col=1
                )
                row += 1
            if trend:
                fig.append_trace(
                    go.Scatter(x=x, y=self.trend, name="Trend"), row=row, col=1
                )
                row += 1
            if seasonal:
                if self.is_multi:
                    for name, seasonal_component in self.seasonal.items():
                        fig.append_trace(
                            go.Scatter(x=x, y=seasonal_component, name=name),
                            row=row,
                            col=1,
                        )
                        row += 1
                else:
                    fig.append_trace(
                        go.Scatter(x=x, y=self.seasonal, name="Seasonal"),
                        row=row,
                        col=1,
                    )
                    row += 1
            if resid:
                fig.append_trace(
                    go.Scatter(x=x, y=self.resid, name="Residual"), row=row, col=1
                )
                row += 1

            fig.update_layout(
                title_text="Seasonal Decomposition",
                autosize=False,
                width=1200,
                height=700,
                title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
                titlefont={"size": 20},
                legend_title=None,
                showlegend=False,
            )
            return fig

        else:
            return super().plot(observed, seasonal, trend, resid)


# Adapted and modified from https://github.com/jrmontag/STLDecompose/blob/master/stldecompose/stl.py
class BaseDecomposition:
    def __init__(
        self,
        seasonality_period: Union[str, int] = None,
        model: str = "additive",
        lo_frac: float = 0.6,
        lo_delta: float = 0.01,
    ) -> None:
        """Base class for all the seasonal decomposition techniques, using Loess Regression for trend
        estimation. All child classes needs to implement the `_extract_seasonality` method. This implementation is modeled
        after the ``statsmodels.tsa.seasonal_decompose`` method
        but substitutes a Lowess regression for a convolution in its trend estimation.
        For more details on lo_frac and lo_delta, see:
        `statsmodels.nonparametric.smoothers_lowess.lowess()`
        Args:
            seasonality_period (int): Most significant periodicity in the observed time series, in units of
            1 observation. Ex: to accomodate strong annual periodicity within years of daily
            observations, ``seasonality_period=365``.
            model (str, optional): {"additive", "multiplicative"} Type of seasonal component. Defaults to "additive".
            lo_frac (float, optional): Fraction of data to use in fitting Lowess regression. Defaults to 0.6.
            lo_delta (float, optional): Fractional distance within which to use linear-interpolation
            instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases
            computation time. Defaults to 0.01.
        """
        self.seasonality_period = seasonality_period
        if isinstance(seasonality_period, int):
            self._seasonality_type = "period"
        elif isinstance(seasonality_period, str):
            self._seasonality_type = "string"
        else:
            self._seasonality_type = "custom"
            # warnings.warn(
            #     "Initialized without seasonality parameter. The .fit should be called with the seasonality array for it to work"
            # )
        self.model = model
        self.lo_frac = lo_frac
        self.lo_delta = lo_delta

    @abstractmethod
    def _extract_seasonality(self, detrended, **seasonality_kwargs):
        raise NotImplementedError(
            "Any inheriting class should implement method with the signature `def _extract_seasonality(self, detrended, **seasonality_kwargs)`"
        )

    def fit(
        self, df: pd.DataFrame, seasonality: np.ndarray = None, detrend: bool = True
    ) -> DecomposeResult:
        """Fit the sesonal decomposition

        Args:
            df (pd.DataFrame): Time series of observed counts. This DataFrame must be continuous (no
            gaps or missing data), and include a ``pandas.DatetimeIndex``.
            seasonality (np.ndarray, optional): Custom seasonality parameter. An array of the same size as the input series
                which has an ordinal representation of the seasonality.
                If it is an annual seasonality of daily data, the array would have a minimum value of 1 and maximum value of 365
                as it increases by one every day of the year. Defaults to None.
            detrend (bool, optional): Flag to disable detrending before seasonality estimation. Useful when we are estimating multiple seasonalities.
                Defaults to None.

        Returns:
            DecomposeResult: An object with DataFrame attributes for the
            seasonal, trend, and residual components, as well as the average seasonal cycle.
        """
        assert isinstance(
            df, (pd.DataFrame, pd.Series)
        ), "`df` should be a `pd.Dataframe` or a `pd.Series`."
        assert isinstance(
            df.index, pd.DatetimeIndex
        ), "`df` should be a dataframe with datetime index."
        if self._seasonality_type == "custom":
            if seasonality is None:
                raise ValueError(
                    "Class was initialized without seasonality parameter. `seasonality` cannot be `None`"
                )
        elif self._seasonality_type == "period":
            if len(df) < 2 * np.max(self.seasonality_period):
                raise ValueError(
                    f"{self.__class__.__name__} needs at least two cycles of the maximum seasonality period to estimate the seasonal component. Try using FourierDecomposition, which will work with shorter timeseries."
                )
        # use some existing pieces of statsmodels
        # lowess = sm.nonparametric.lowess
        _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)
        # get plain np array
        observed = np.asanyarray(df).squeeze()
        if self.model.startswith("m"):
            if np.any(observed <= 0):
                raise ValueError(
                    "Multiplicative seasonality is not appropriate "
                    "for zero and negative values"
                )
        if detrend:
            # calc trend, remove from observation
            _, trend = _detrend(
                observed, self.lo_frac, self.lo_delta, return_trend=True
            )
            if self.model == "additive":
                detrended = observed - trend
            else:
                detrended = observed / trend
        else:
            trend = None
            detrended = observed
        seasonal = self._extract_seasonality(
            detrended, date_index=df.index, seasonality=seasonality
        )
        if self.model == "additive":
            resid = detrended - seasonal
        else:
            resid = observed / seasonal / trend

        # convert the arrays back to appropriate dataframes, stuff them back into
        #  the statsmodel object
        results = list(map(_pandas_wrapper, [seasonal, trend, resid, observed]))
        dr = DecomposeResult(
            seasonal=results[0], trend=results[1], resid=results[2], observed=results[3]
        )
        return dr


class STL(BaseDecomposition):
    def __init__(
        self,
        seasonality_period: int,
        model: str = "additive",
        lo_frac: float = 0.6,
        lo_delta: float = 0.01,
    ) -> None:
        """Create a seasonal-trend (with Loess, aka "STL") decomposition of observed time series data.
        This implementation is modeled after the ``statsmodels.tsa.seasonal_decompose`` method
        but substitutes a Lowess regression for a convolution in its trend estimation.
        For more details on lo_frac and lo_delta, see:
        `statsmodels.nonparametric.smoothers_lowess.lowess()`

        Args:
            seasonality_period (int): Most significant periodicity in the observed time series, in units of
            1 observation. Ex: to accomodate strong annual periodicity within years of daily
            observations, ``seasonality_period=365``.
            model (str, optional): {"additive", "multiplicative"} Type of seasonal component. Defaults to "additive".
            lo_frac (float, optional): Fraction of data to use in fitting Lowess regression. Defaults to 0.6.
            lo_delta (float, optional): Fractional distance within which to use linear-interpolation
                instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases
                computation time. Defaults to 0.01.
        """
        super().__init__(
            seasonality_period=seasonality_period,
            model=model,
            lo_frac=lo_frac,
            lo_delta=lo_delta,
        )

    def _extract_seasonality(self, detrended, **seasonality_kwargs):
        """Extracts Seasonality from detrended data using averages"""
        if detrended.shape[0] < 2 * self.seasonality_period:
            raise ValueError(
                f"time series must have 2 complete cycles requires {2 * self.seasonality_period} "
                f"observations. time series only has {detrended.shape[0]} observation(s)"
            )
        # period must not be larger than size of series to avoid introducing NaNs
        if self.seasonality_period > len(detrended):
            warnings.warn(
                "`period` should not be less than length of series. Setting period to length of series"
            )
            period = len(detrended)
        else:
            period = self.seasonality_period
        # calc one-period seasonality, remove tiled array from detrended
        period_averages = np.array(
            [pd_nanmean(detrended[i::period]) for i in range(period)]
        )
        # 0-center the period avgs
        if self.model == "additive":
            period_averages -= np.mean(period_averages)
        else:
            period_averages /= np.mean(period_averages)
        # Saving period_averages in the object
        self.period_averages = period_averages
        seasonal = np.tile(period_averages, len(detrended) // period + 1)[
            : len(detrended)
        ]
        return seasonal


class FourierDecomposition(BaseDecomposition):

    ALLOWED_SEASONALITY = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "day_of_year",
        "dayofyear",
        "weekofyear",
        "week",
        "dayofweek",
        "day_of_week",
        "weekday",
        "quarter",
    ]

    def __init__(
        self,
        seasonality_period: str = None,
        model: str = "additive",
        lo_frac: float = 0.6,
        lo_delta: float = 0.01,
        n_fourier_terms: int = 1,
    ) -> None:
        """Create a seasonal-trend (with Loess) decomposition of observed time series data.
        This implementation is modeled after the ``statsmodels.tsa.seasonal_decompose`` method
        but substitutes a Lowess regression for a convolution in its trend estimation.
        For seasonality signals, the implementation uses fourier terms and Regularized(Ridge) Regression.
        For more details on lo_frac and lo_delta, see:
        `statsmodels.nonparametric.smoothers_lowess.lowess()`
        Args:
            seasonality_period (str): Seasonality to be extracted from the datetime index. pandas datetime properties like `week_of_day`,
                `month`, etc. can be used to specify the most prominent seasonality. If left None, need to provide the seasonality array
                while fitting. Defaults to None.
            model (str, optional): {"additive", "multiplicative"} Type of seasonal component. Defaults to "additive".
            lo_frac (float, optional): Fraction of data to use in fitting Lowess regression. Defaults to 0.6.
            lo_delta (float, optional): Fractional distance within which to use linear-interpolation
                instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases
                computation time. Defaults to 0.01.
            n_fourier_terms (int): Number of fourier terms to use to extract the seasonality. Increase this to make the seasonal pattern
                more flexible. Defaults to 1.
        """
        super().__init__(
            seasonality_period=seasonality_period,
            model=model,
            lo_frac=lo_frac,
            lo_delta=lo_delta,
        )
        if seasonality_period is not None:
            assert (
                seasonality_period in self.ALLOWED_SEASONALITY
            ), "seasonality should be one of these strings {ALLOWED_SEASONALITY} for FourierDecomposition"
        self.seasonality_period = seasonality_period
        self.n_fourier_terms = n_fourier_terms

    def _calculate_fourier_terms(self, seasonal_cycle: np.ndarray, max_cycle: int):
        """Calculates Fourier Terms given the seasonal cycle and max_cycle"""
        sin_X = np.empty((len(seasonal_cycle), self.n_fourier_terms), dtype="float64")
        cos_X = np.empty((len(seasonal_cycle), self.n_fourier_terms), dtype="float64")
        for i in range(1, self.n_fourier_terms + 1):
            sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
            cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
        return np.hstack([sin_X, cos_X])

    def _prepare_X(self, detrended, **seasonality_kwargs):
        if (
            self.seasonality_period is None
            and seasonality_kwargs.get("seasonality") is None
        ):
            raise ValueError(
                f"{type(self).__name__} was initialized with seasonality and seasonality passed to .fit was None or not an numpy array"
            )
        date_index = seasonality_kwargs.get("date_index")
        if self.seasonality_period is None:
            seasonal_cycle = seasonality_kwargs.get("seasonality")
        else:
            seasonal_cycle = (getattr(date_index, self.seasonality_period)).values
        return self._calculate_fourier_terms(
            seasonal_cycle, max_cycle=np.max(seasonal_cycle)
        )

    def _extract_seasonality(self, detrended, **seasonality_kwargs):
        """Extracts Seasonality from detrended data using fourier terms"""
        X = self._prepare_X(detrended, **seasonality_kwargs)
        self.seasonality_model = RidgeCV(normalize=True, fit_intercept=False).fit(
            X, detrended
        )
        return self.seasonality_model.predict(X)


class MultiSeasonalDecomposition:

    ALLOWED_SEASONALITY = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "day_of_year",
        "dayofyear",
        "weekofyear",
        "week",
        "dayofweek",
        "day_of_week",
        "weekday",
        "quarter",
    ]
    ALLOWABLE_SEASONAL_MODELS = {"fourier": FourierDecomposition, "averages": STL}

    def __init__(
        self,
        seasonal_model: str,
        seasonality_periods: List[Union[str, int]] = [],
        model: str = "additive",
        lo_frac: float = 0.6,
        lo_delta: float = 0.01,
        n_fourier_terms: int = 1,
    ) -> None:
        """Uses Fourier Decomposition or STL to decompose time series signals with multiple seasonalities in a step-wise approach.

        Args:
            seasonal_model (str): {"fourier", "averages"} Choice between `fourier` and `averages` as the seasonality model for decomposition
            seasonality_periods (List[Union[str, int]], optional): A list of expected seasonalities. For STL, it is a list of seasonal
                periods, and for Fourier Decomposition it is a list of strings which denotes pandas datetime properties. Defaults to [].
            model (str, optional): {"additive", "multiplicative"} Type of seasonal component. Defaults to "additive".
            lo_frac (float, optional): Fraction of data to use in fitting Lowess regression. Defaults to 0.6.
            lo_delta (float, optional): Fractional distance within which to use linear-interpolation
                instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases
                computation time. Defaults to 0.01.
            n_fourier_terms (int): Number of fourier terms to use to extract the seasonality. Increase this to make the seasonal pattern
                more flexible. Defaults to 1.
        """
        # super().__init__(model=model, lo_frac=lo_frac, lo_delta=lo_delta)
        assert (
            seasonal_model in self.ALLOWABLE_SEASONAL_MODELS.keys()
        ), f"seasonal_model should be one of {self.ALLOWABLE_SEASONAL_MODELS.keys()}"
        if isinstance(seasonality_periods, str):
            assert all(
                [s in self.ALLOWED_SEASONALITY for s in seasonality_periods]
            ), f"seasonality should be either an array or one of these strings {self.ALLOWED_SEASONALITY}"
        self.seasonality_periods = seasonality_periods
        seasonality_periods = [] if seasonality_periods is None else seasonality_periods
        self.n_seasonal_components = len(seasonality_periods)
        if self.n_seasonal_components > 0:
            if all([isinstance(s, str) for s in self.seasonality_periods]):
                self._seasonality_type = "string"
            elif all([isinstance(s, int) for s in self.seasonality_periods]):
                self._seasonality_type = "period"
                # Sorting in ascending order
                self.seasonality_periods = sorted(self.seasonality_periods)
        else:
            if seasonal_model == "averages":
                raise ValueError(
                    "For `seasonal_model='averages'`, seasonality_periods is a mandatory parameter"
                )
            else:
                self._seasonality_type = "custom"
                # warnings.warn(
                #     "Initialized without seasonality parameter. The .fit should be called with the seasonality array for it to work"
                # )

        if self.n_seasonal_components == 1:
            warnings.warn("Only single seasonality supplied.")
        self.n_fourier_terms = n_fourier_terms
        self.seasonal_model = seasonal_model
        self._seasonal_model = self.ALLOWABLE_SEASONAL_MODELS[seasonal_model]
        self.model = model
        self.lo_delta = lo_delta
        self.lo_frac = lo_frac

    def _initialize_seasonal_model(self, seasonality_period):
        params = dict(model=self.model, lo_frac=self.lo_frac, lo_delta=self.lo_delta)
        if isinstance(seasonality_period, (str, int)):
            params["seasonality_period"] = seasonality_period
        if self.seasonal_model == "fourier":
            params["n_fourier_terms"] = self.n_fourier_terms
        return self._seasonal_model(**params)

    def fit(
        self,
        df: pd.DataFrame,
        seasonality: List[np.ndarray] = None,
    ) -> DecomposeResult:
        """Fit the multi seasonal decomposition

        Args:
            df (pd.DataFrame): Time series of observed counts. This DataFrame must be continuous (no
            gaps or missing data), and include a ``pandas.DatetimeIndex``.
            seasonality (List[np.ndarray], optional): Custom seasonality parameter. A list of array of the same size as the input
                series which has an ordinal representation of the seasonality.
                If it is an annual seasonality of daily data, the array would have a minimum value of 1 and maximum value of 365
                as it increases by one every day of the year. Defaults to None.
            detrend (bool, optional): Doesn't do anything. Exist only for compatibility. Defaults to None.
        Returns:
            DecomposeResult: An object with DataFrame attributes for the
            seasonal, trend, and residual components, as well as the average seasonal cycle.
        """
        if self._seasonality_type == "custom" and (
            seasonality is None
            or all([(not isinstance(s, np.ndarray)) for s in seasonality])
        ):
            raise ValueError(
                f"{type(self).__name__} was initialized with seasonality and seasonality passed to .fit was None or not an numpy array"
            )

        _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)
        seasonal = OrderedDict()
        seasonality_iter = (
            seasonality if seasonality is not None else self.seasonality_periods
        )
        # First decomposition
        s = seasonality_iter[0]
        seasonal_model = self._initialize_seasonal_model(s)
        seasonality_key = (
            s if self._seasonality_type in ["string", "period"] else "seasonality_0"
        )
        if self._seasonality_type == "custom":
            decomposition = seasonal_model.fit(df, seasonality=seasonality[0])
        else:
            decomposition = seasonal_model.fit(df)
        observed = decomposition.observed
        trend = decomposition.trend
        seasonal[seasonality_key] = decomposition.seasonal
        _resid = decomposition.resid
        for i, s in enumerate(seasonality_iter[1:]):
            seasonal_model = self._initialize_seasonal_model(s)
            seasonality_key = (
                s
                if self._seasonality_type in ["string", "period"]
                else f"seasonality_{i+1}"
            )
            if self._seasonality_type == "custom":
                decomposition = seasonal_model.fit(
                    _resid, seasonality=seasonality[i + 1], detrend=False
                )
            else:
                decomposition = seasonal_model.fit(_resid, detrend=False)
            # decomposition = seasonal_model.fit(_resid, detrend=False)
            seasonal[seasonality_key] = decomposition.seasonal
            _resid = decomposition.resid
        # convert the arrays back to appropriate dataframes, stuff them back into
        #  the statsmodel object
        results = list(map(_pandas_wrapper, [trend, _resid, observed]))
        dr = DecomposeResult(
            seasonal=seasonal, trend=results[0], resid=results[1], observed=results[2]
        )
        return dr
