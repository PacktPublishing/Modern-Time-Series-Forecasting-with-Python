from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Type, Union
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy import optimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox, variation
from src.decomposition.seasonal import STL, FourierDecomposition
from src.transforms.stationary_utils import (
    check_heteroscedastisticity,
    check_seasonality,
    check_trend,
)
from src.utils.data_utils import is_datetime_dtypes


def check_input(y: pd.Series) -> Union[pd.Series, np.ndarray]:
    assert isinstance(
        y, (pd.Series, pd.DataFrame)
    ), "time series inputs should be a series of dataframe with a datetime index"
    if isinstance(y, pd.DataFrame):
        assert (
            len(y.columns) == 1
        ), "time series inputs should have only one column, if dataframe is given"
        # converting to series
        y = y.squeeze()
    assert is_datetime_dtypes(y.index), "timeseries inputs should have a datetime index"
    return y


def check_negative(y: pd.Series):
    if np.min(y) < 0:
        raise ValueError(
            "`y` values cannot be negative. Add sufficient offset to make it strictly positive"
        )


def check_fitted(is_fitted):
    if not is_fitted:
        raise ValueError(
            "`fit` must be called before `transform` or `inverse_transform`"
        )


class BaseDifferencingTransformer(metaclass=ABCMeta):
    def __init__(self, diff_gap: int):
        """Base Class for all the differencing transformers

        Args:
            num_diff (int): The number of timesteps to skip for the differencing operation
        """
        self.diff_gap = diff_gap
        self._is_fitted = False

    def _get_offset_series(self, y, freq: str, strict: bool = False) -> pd.Series:
        time_index = y.index
        offset_index = time_index - self.diff_gap * to_offset(freq)
        if strict and len(set(offset_index) - set(self._train_series.index)) > 0:
            raise ValueError("Diff needs previous actuals")
        else:
            offset_series = self._train_series.shift(self.diff_gap)[time_index]
        return offset_series

    @staticmethod
    @abstractmethod
    def difference_operation(series: pd.Series, offset_series: pd.Series) -> pd.Series:
        raise NotImplementedError(
            "`difference_operation` should be implemented by any inheriting class"
        )

    @staticmethod
    @abstractmethod
    def inverse_difference_operation(
        series: pd.Series, offset_series: pd.Series
    ) -> pd.Series:
        raise NotImplementedError(
            "`inverse_difference_operation` should be implemented by any inheriting class"
        )

    def _update_train_series(self, full_series):
        assert isinstance(
            full_series, (pd.Series, pd.DataFrame)
        ), "`full_series` should be a series of dataframe with a datetime index"
        self._train_series = self._train_series.append(
            full_series[~full_series.index.isin(self._train_series.index)]
        )

    def fit_transform(
        self, y: pd.Series, freq: str = None, full_series: pd.Series = None
    ) -> pd.Series:
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            freq (str, optional): Use this to either override the inferred frequency or if frequency is missing in the index. Defaults to None.
            full_series (pd.Series, optional): The full available time series because differencing transforms
                requires (n-d)th point, where n is current timestep, and d is difference gap. Providing the full series along with the datetime
                aligns the seres on time and carries out the differencing operation. Defaults to None.

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y, freq)
        return self.transform(y, full_series)

    def fit(self, y: pd.Series, freq: str = None):
        """Sets the train series and frequency as the fit process

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            freq (str, optional): Use this to either override the inferred frequency or if frequency is missing in the index. Defaults to None.

        Raises:
            ValueError: If freq is missing in index as well as parameters
        """
        y = check_input(y)
        self._train_series = y
        if y.index.freq is None and freq is None:
            raise ValueError(
                "Frequency missing in `y`. Use the `freq` parameter to indicate the frequency of the time series"
            )
        else:
            self.freq = y.index.freq if freq is None else freq
        self._is_fitted = True
        return self

    def transform(self, y: pd.Series, full_series: pd.Series = None) -> pd.Series:
        """Transforms the time series with the datetime that is aligned in `fit`

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            full_series (pd.Series, optional): The full available time series because differencing transforms
                requires (n-d)th point, where n is current timestep, and d is difference gap. Providing the full series along with the datetime
                aligns the seres on time and carries out the differencing operation. Defaults to None.

        Returns:
            pd.Series: The transformed series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        if full_series is not None:
            check_input(full_series)
            self._update_train_series(full_series)
        return self.difference_operation(y, self._get_offset_series(y, self.freq))

    def inverse_transform(
        self, y: pd.Series, full_series: pd.Series = None
    ) -> pd.Series:
        """Inverse transforms the differenced series back to the original one

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            full_series (pd.Series, optional): The full available time series because differencing transforms
                requires (n-d)th point, where n is current timestep, and d is difference gap. Providing the full series along with the datetime
                aligns the seres on time and carries out the differencing operation. Defaults to None.

        Returns:
            pd.Series: The original series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        if full_series is not None:
            check_input(full_series)
            self._update_train_series(full_series)
        return self.inverse_difference_operation(
            y, self._get_offset_series(y, self.freq, strict=True)
        )


class AdditiveDifferencingTransformer(BaseDifferencingTransformer):
    def __init__(self, diff_gap=1):
        """The additive differencing operation.
        y = y_{t} - y_{t-1}
        """
        super().__init__(diff_gap=diff_gap)

    @staticmethod
    def difference_operation(series: pd.Series, offset_series: pd.Series) -> pd.Series:
        return series - offset_series

    @staticmethod
    def inverse_difference_operation(
        series: pd.Series, offset_series: pd.Series
    ) -> pd.Series:
        return series + offset_series


class MultiplicativeDifferencingTransformer(BaseDifferencingTransformer):
    def __init__(self, diff_gap=1):
        """The multiplicative differencing operation.
        y = y_{t} / y_{t-1}
        """
        super().__init__(diff_gap=diff_gap)

    @staticmethod
    def difference_operation(series: pd.Series, offset_series: pd.Series) -> pd.Series:
        return series / offset_series

    @staticmethod
    def inverse_difference_operation(
        series: pd.Series, offset_series: pd.Series
    ) -> pd.Series:
        return series * offset_series

    def fit(self, y: pd.Series, freq: str = None):
        check_negative(y)
        return super().fit(y, freq)


class AddMTransformer:
    def __init__(self, M: float) -> None:
        """A Transformer which adds a constant value to the time series

        Args:
            M (float): The constant to be added.
        """
        self.M = abs(M)

    @staticmethod
    def check_input(y: pd.Series) -> pd.Series:
        assert isinstance(
            y, (pd.Series, pd.DataFrame)
        ), "time series inputs should be a series of dataframe with a datetime index"
        if isinstance(y, pd.DataFrame):
            assert (
                len(y.columns) == 1
            ), "time series inputs should have only one column, if dataframe is given"
            # converting to series
            y = y.squeeze()
        assert is_datetime_dtypes(
            y.index
        ), "timeseries inputs should have a datetime index"
        return y

    def fit_transform(self, y: pd.Series):
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y)
        return self.transform(y)

    def fit(self, y: pd.Series):
        """No action is being done apart from checking the input. This is a dummy method for compatibility

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
        """
        self.check_input(y)
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Applies the constant offset

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        y = self.check_input(y)
        return y + self.M

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Reverses the constant offset

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The original series
        """
        return y - self.M


class LogTransformer:
    def __init__(self, add_one: bool = True) -> None:
        """The logarithmic transformer

        Args:
            add_one (bool, optional): Flag to add one to the series before applying log
                to avoid log 0. Defaults to True.
        """
        self.add_one = add_one

    def fit_transform(self, y: pd.Series):
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y)
        return self.transform(y)

    def fit(self, y: pd.Series):
        """No action is being done apart from checking the input. This is a dummy method for compatibility

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
        """
        y = check_input(y)
        check_negative(y)
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Applies the log transform

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Raises:
            ValueError: If there are zero values and `add_one` is False

        Returns:
            pd.Series: The transformed series
        """
        y = check_input(y)
        check_negative(y)
        return np.log1p(y) if self.add_one else np.log(y)

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Reverses the log transform by applying the exponential

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Raises:
            ValueError: If there are zero values and `add_one` is False

        Returns:
            pd.Series: The original series
        """
        y = check_input(y)
        return np.expm1(y) if self.add_one else np.exp(y)


class BoxCoxTransformer:
    def __init__(
        self,
        boxcox_lambda: float = None,
        seasonal_period: int = None,
        optimization="guerrero",
        bounds: Tuple[int, int] = (-1, 2),
        add_one: bool = True,
    ) -> None:
        """Performs the Box-Cox transformation. Also finds out the optimal lambda if not given

        Args:
            boxcox_lambda (float, optional): The lambda parameter in the box-cox transformation. Defaults to None.
            seasonal_period (int, optional): Expected seasonality period. Only used in Guerrero's method of finding optimal lambda to
                split the series into homogenous sub-series. Defaults to None.
            optimization (str, optional): Sets the method used to optimize lambda if not given. Allowed values {'guerrero','loglikelihood}. Defaults to "guerrero".
            bounds (Tuple[int, int], optional): The upper and lower bound to optimize lambda. Only used in Guerrero's method. Defaults to (-1, 2).
            add_one (bool, optional): Convenience method to add one to deal with zeroes in the data. Defaults to True.

        Raises:
            ValueError: If `bounds` is not a tuple of lenght two, or it is a tuple but upper < lower
            ValueError: If the optimization is set to guerrero, but no seasonal_period is given
        """
        assert optimization in [
            "guerrero",
            "loglikelihood",
        ], "`optimization should be one of ['guerrero', 'loglikelihood']"
        self.boxcox_lambda = boxcox_lambda
        self.seasonal_period = seasonal_period
        self.optimization = optimization
        self.add_one = add_one
        # input checks on bounds
        if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[1] < bounds[0]:
            raise ValueError(
                f"`bounds` must be a tuple of length 2, and upper should be greater than lower, but found: {bounds}"
            )
        self.bounds = bounds
        if boxcox_lambda is None:
            self._do_optimize = True
            if optimization == "guerrero" and seasonal_period is None:
                raise ValueError(
                    "For Guerrero method of finding optimal lambda for box-cox transform, seasonal_period is needed."
                )
        else:
            self._do_optimize = False
        self._is_fited = False

    def fit_transform(self, y: pd.Series):
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y)
        return self.transform(y)

    def _add_one(self, y):
        if self.add_one:
            return y + 1
        else:
            return y

    def _subtract_one(self, y):
        if self.add_one:
            return y - 1
        else:
            return y

    def _optimize_lambda(self, y):
        if self.optimization == "loglikelihood":
            _, lmbda = boxcox(y)
        elif self.optimization == "guerrero":
            lmbda = self._guerrero(y, self.seasonal_period, self.bounds)
        return lmbda

    # Adapted from https://github.com/alan-turing-institute/sktime/blob/db0242f6e0230ee3a96d0a62973535d9c328c2ea/sktime/transformations/series/boxcox.py#L305
    @staticmethod
    def _guerrero(x, sp, bounds=None):
        r"""
        Returns lambda estimated by the Guerrero method [Guerrero].
        Parameters
        ----------
        x : ndarray
            Input array. Must be 1-dimensional.
        sp : integer
            Seasonal periodicity value. Must be an integer >= 2
        bounds : {None, (float, float)}, optional
            Bounds on lambda to be used in minimization.
        Returns
        -------
        lambda : float
            Lambda value that minimizes the coefficient of variation of
            variances of the time series in different periods after
            Box-Cox transformation [Guerrero].
        References
        ----------
        [Guerrero] V.M. Guerrero, "Time-series analysis supported by Power
        Transformations ", Journal of Forecasting, Vol. 12, 37-48 (1993)
        https://doi.org/10.1002/for.3980120104
        """

        if sp is None or not isinstance(sp, int) or sp < 2:
            raise ValueError(
                "Guerrero method requires an integer seasonal periodicity (sp) value >= 2."
            )

        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Data must be 1-dimensional.")

        num_obs = len(x)
        len_prefix = num_obs % sp

        x_trimmed = x[len_prefix:]
        x_mat = x_trimmed.reshape((-1, sp))
        x_mean = np.mean(x_mat, axis=1)

        # [Guerrero, Eq.(5)] uses an unbiased estimation for
        # the standard deviation
        x_std = np.std(x_mat, axis=1, ddof=1)

        def _eval_guerrero(lmb, x_std, x_mean):
            x_ratio = x_std / x_mean ** (1 - lmb)
            x_ratio_cv = variation(x_ratio)
            return x_ratio_cv

        return optimize.fminbound(
            _eval_guerrero, bounds[0], bounds[1], args=(x_std, x_mean)
        )

    def fit(self, y: pd.Series):
        """No action is being done apart from checking the input. This is a dummy method for compatibility

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
        """
        check_input(y)
        check_negative(y)
        y = self._add_one(y)
        if self._do_optimize:
            self.boxcox_lambda = self._optimize_lambda(y)
        self._is_fitted = True
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Applies the log transform

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Raises:
            ValueError: If there are zero values and `add_one` is False

        Returns:
            pd.Series: The transformed series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        check_negative(y)
        y = self._add_one(y)
        return pd.Series(boxcox(y.values, lmbda=self.boxcox_lambda), index=y.index)

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Reverses the log transform by applying the exponential

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Raises:
            ValueError: If there are zero values and `add_one` is False

        Returns:
            pd.Series: The original series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        return pd.Series(
            self._subtract_one(inv_boxcox(y.values, self.boxcox_lambda)), index=y.index
        )


class DetrendingTransformer:
    def __init__(self, degree: int = 1) -> None:
        """Detrending Transformer. Fits a trend(depending on the degree) using
        `np.polyfit` and extends the trend into the future based on dates

        Args:
            degree (int, optional): The degree of the line to be fit as trend. Defaults to 1.
        """
        self.degree = degree
        if degree > 1:
            warnings.warn("Trends with degree>1 are very strong and use with care.")
        self._is_fitted = False

    def fit_transform(self, y: pd.Series, freq: str = None):
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y, freq)
        return self.transform(y)

    def fit(self, y: pd.Series, freq: str = None):
        """Fits a polynomial line to the timeseries to extract trend

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            freq (str, optional): Use this to either override the inferred frequency or if frequency is missing in the index. Defaults to None.

        Raises:
            ValueError: If freq is missing in index as well as parameters

        """
        y = check_input(y)
        if y.index.freq is None and freq is None:
            raise ValueError(
                "Frequency missing in `y`. Use the `freq` parameter to indicate the frequency of the time series"
            )
        else:
            self.freq = y.index.freq if freq is None else freq
        x = np.arange(len(y))
        self.start_date = y.index.min()
        self.linear_params = np.polyfit(x=x, y=y, deg=self.degree)
        self._is_fitted = True
        return self

    def _get_trend(self, y: pd.Series):
        date_array = pd.date_range(self.start_date, y.index.max(), freq=self.freq)
        date_array = pd.Series(np.arange(len(date_array)), index=date_array)
        x = date_array[y.index].values
        trend = np.sum(
            [p * np.power(x, i) for i, p in enumerate(reversed(self.linear_params))],
            axis=0,
        )
        return trend

    def transform(self, y: pd.Series) -> pd.Series:
        """Calculates the trend according to the dates provided and detrends the series

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: Detrended time series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        trend = self._get_trend(y)
        return y - trend

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Calculates the trend according to the dates provided and adds back the trend to the series

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: Original time series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        trend = self._get_trend(y)
        return y + trend


class DeseasonalizingTransformer:

    ALLOWABLE_EXTRACTION_METHODS = ["period_averages", "fourier_terms"]
    DESEASONALIZERS = {"period_averages": STL, "fourier_terms": FourierDecomposition}

    def __init__(
        self,
        seasonal_period: Union[int, str],
        seasonality_extraction: str = "period_averages",
        n_fourier_terms: int = 1,
    ) -> None:
        """Deseasonalizing Transformer. uses STL or FourierDecompostion to extract seasonality
            and extends it into the future based on dates

        Args:
            seasonal_period (Union[int, str]): The period after which seasonality is expected
                to repeat for seasonality_extraction=`period_averages` and the pandas datetime property string
                for seasonality_extraction=`fourier_terms`
            seasonality_extraction (str, optional): Whether to use STL or FourierDecomposition.
                Allowable values: {"period_averages", "fourier_terms"} Defaults to "period_averages".
            n_fourier_terms (int): Number of fourier terms to use to extract the seasonality. Increase this to make the seasonal pattern
                more flexible. Defaults to 1.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        assert (
            seasonality_extraction in self.ALLOWABLE_EXTRACTION_METHODS
        ), f"`seasonality_extraction` should one of {self.ALLOWABLE_EXTRACTION_METHODS}"
        seasonality_mode = "additive"
        if seasonality_extraction == "period_averages" and isinstance(
            seasonal_period, str
        ):
            raise ValueError(
                "`seasonality_period` should be an integer for period_averages. eg. 12, 52, etc."
            )
        if seasonality_extraction == "fourier_terms" and isinstance(
            seasonal_period, int
        ):
            raise ValueError(
                "`seasonality_period` should be an string with the pandas datetime properties for fourier_terms. eg. 'week', 'hour', etc."
            )
        self.seasonal_period = seasonal_period
        self.seasonality_extraction = seasonality_extraction
        self.seasonality_mode = seasonality_mode
        # Do not detrend before extracting seasonality. Any detrending is expected to be done before
        self.detrend = False
        # Setting value for compatibility. Is not used because we don't do detrending.
        self.lo_delta = 0.1
        self.lo_frac = 0.6
        self.n_fourier_terms = n_fourier_terms
        if seasonality_extraction == "period_averages":
            self._seasonal_model = self.DESEASONALIZERS[seasonality_extraction](
                seasonal_period, seasonality_mode, self.lo_frac, self.lo_delta
            )
        else:
            self._seasonal_model = self.DESEASONALIZERS[seasonality_extraction](
                seasonal_period,
                seasonality_mode,
                self.lo_frac,
                self.lo_delta,
                n_fourier_terms=n_fourier_terms,
            )
        self._is_fitted = False

    def fit_transform(self, y: pd.Series, freq: str = None):
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y, freq=freq)
        return self.transform(y)

    def fit(self, y: pd.Series, seasonality: np.ndarray = None, freq: str = None):
        """Fits a polynomial line to the timeseries to extract trend

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            freq (str, optional): Use this to either override the inferred frequency or if frequency is missing in the index. Defaults to None.

        Raises:
            ValueError: If freq is missing in index as well as parameters

        """
        y = check_input(y)
        if y.index.freq is None and freq is None:
            raise ValueError(
                "Frequency missing in `y`. Use the `freq` parameter to indicate the frequency of the time series"
            )
        else:
            self.freq = y.index.freq if freq is None else freq
        res = self._seasonal_model.fit(y, seasonality=seasonality, detrend=self.detrend)
        if self.seasonality_extraction == "period_averages":
            self.repeating_period_average = res.seasonal[: self.seasonal_period]
        self.start_date = y.index.min()
        self._is_fitted = True
        return self

    def _get_seasonality(self, y: pd.Series, seasonality: np.ndarray):
        date_array = pd.date_range(self.start_date, y.index.max(), freq=self.freq)
        if self.seasonality_extraction == "period_averages":
            date_array = pd.Series(
                np.resize(self.repeating_period_average, len(date_array)),
                index=date_array,
            )
            seasonality = date_array[y.index].values
        else:
            X = self._seasonal_model._prepare_X(
                y, seasonality=seasonality, date_index=y.index
            )
            seasonality = self._seasonal_model.seasonality_model.predict(X)
            seasonality = pd.Series(seasonality, index=y.index)
        return seasonality

    def transform(self, y: pd.Series, seasonality: np.ndarray = None) -> pd.Series:
        """Calculates the trend according to the dates provided and detrends the series

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: Detrended time series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        seasonality = self._get_seasonality(y, seasonality)
        return y - seasonality

    def inverse_transform(
        self, y: pd.Series, seasonality: np.ndarray = None
    ) -> pd.Series:
        """Calculates the trend according to the dates provided and adds back the trend to the series

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: Original time series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        seasonality = self._get_seasonality(y, seasonality)
        return y + seasonality


class AutoStationaryTransformer:
    def __init__(
        self,
        confidence: float = 0.05,
        seasonal_period: Optional[int] = None,
        seasonality_max_lags: int = 60,
        trend_check_params: Dict = {"mann_kendall": False},
        detrender_params: Dict = {"degree": 1},
        deseasonalizer_params: Dict = {},
        box_cox_params: Dict = {"optimization": "guerrero"},
    ) -> None:
        """A Transformer which takes an automatic approach at making a series stationary by Detrending, Deseasonalizing, and/or Box-Cox Transforms

        Args:
            confidence (float, optional): The confidence level for the statistical tests. Defaults to 0.05.
            seasonal_period (Optional[int], optional): The number of periods after which the seasonality cycle repeats itself.
                If None, seasonal_period will be inferred from data. Defaults to None.
            seasonality_max_lags (int, optional): Maximum lags within which the transformer tries to identifies seasonality, in case seasonality is not provided. Defaults to 60.
            trend_check_params (Dict, optional): The parameters which are used in the statistical tests for trend. `check_trend`. Defaults to {"mann_kendall": False}.
            detrender_params (Dict, optional): The parameters passed to `DetrendingTransformer`. Defaults to {"degree":1}.
            deseasonalizer_params (Dict, optional): The parameters passed to `DeseasonalizingTransformer`. 
                seasonality_extraction is fixed as "period_averages". Defaults to {}.
            box_cox_params (Dict, optional): The parameters passed on to `BoxCoxTransformer`. Defaults to {"optimization": "guerrero"}.
        """
        self.confidence = confidence
        self._infer_seasonality = True if seasonal_period is None else False

        self.seasonal_period = seasonal_period
        self.seasonality_max_lags = seasonality_max_lags
        self.trend_check_params = trend_check_params
        self.detrender_params = detrender_params
        deseasonalizer_params["seasonality_extraction"] = "period_averages"
        self.deseasonalizer_params = deseasonalizer_params
        self.box_cox_params = box_cox_params
        self._is_fitted = False

    def fit_transform(self, y: pd.Series, freq: str = None):
        """Convenience method to do `fit` and `transform` ina single step. For detailed documentaion,
            check `fit` and `transform` independently.

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: The transformed series
        """
        self.fit(y, freq)
        return self.transform(y)

    def fit(self, y: pd.Series, freq: str = None):
        """Uses a heurstic to apply a few transformers and saves those in a list of transformers in a pipeline

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index
            freq (str, optional): Use this to either override the inferred frequency or if frequency is missing in the index. Defaults to None.

        Raises:
            ValueError: If freq is missing in index as well as parameters

        """
        y = check_input(y)
        if y.index.freq is None and freq is None:
            raise ValueError(
                "Frequency missing in `y`. Use the `freq` parameter to indicate the frequency of the time series"
            )
        else:
            self.freq = y.index.freq if freq is None else freq
        self._pipeline = []
        _min_max_lag = min(len(y) // 2 - 2 , self.seasonality_max_lags)
        n_unique = len(np.unique(y))
        if _min_max_lag>0 and n_unique>2:
            _trend_check = check_trend(y, self.confidence, **self.trend_check_params)
            self._trend_check = {k:v for k,v in _trend_check._asdict().items() if k!="deterministic_trend_results"}
            if _trend_check.trend:
                detrender = DetrendingTransformer(**self.detrender_params)
                y = detrender.fit_transform(y, freq=self.freq)
                self._pipeline.append(detrender)
            _seasonality_check = check_seasonality(
                y,
                max_lag=self.seasonality_max_lags
                if self._infer_seasonality
                else self.seasonal_period + 1,
                seasonal_period=self.seasonal_period,
                confidence=self.confidence,
                verbose=False
            )
            self._seasonality_check = _seasonality_check._asdict()
            if _seasonality_check.seasonal and self._infer_seasonality:
                self.seasonal_period = int(_seasonality_check.seasonal_periods)
            if _seasonality_check.seasonal and len(y)>2*self.seasonal_period:
                self.deseasonalizer_params["seasonal_period"] = self.seasonal_period
                deseasonalizer = DeseasonalizingTransformer(**self.deseasonalizer_params)
                y = deseasonalizer.fit_transform(y, freq=self.freq)
                self._pipeline.append(deseasonalizer)

            _hetero_check = check_heteroscedastisticity(y, self.confidence)
            self._hetero_check = _hetero_check._asdict()
            if _hetero_check.heteroscedastic:
                if y.min() < 0:
                    add_m = AddMTransformer(np.abs(y.min()) + 1)
                    y = add_m.fit_transform(y)
                    self._pipeline.append(add_m)
                # No seasonality has been identified. But we still need to split the series into sub series for Guerrero's method
                self.box_cox_params["seasonal_period"] = max(
                    len(y) // 4 if self.seasonal_period is None else self.seasonal_period, 5
                )
                box_cox_transformer = BoxCoxTransformer(**self.box_cox_params)
                y = box_cox_transformer.fit_transform(y)
                self._pipeline.append(box_cox_transformer)
        self._is_fitted = True
        # self._unit_root_check = check_unit_root(y, confidence=confidence)
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Executes the transformers in the pipeline in the same order

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: Stationary time series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        for tr in self._pipeline:
            try:
                y = tr.transform(y, freq=self.freq)
            except TypeError:
                # Box Cox Transformer doesnt have freq parameter
                y = tr.transform(y)
        return y

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Inverts the transformers in the pipeline in the reverse order

        Args:
            y (pd.Series): The time series as a panda series or dataframe with a datetime index

        Returns:
            pd.Series: Original time series
        """
        check_fitted(self._is_fitted)
        y = check_input(y)
        for tr in reversed(self._pipeline):
            y = tr.inverse_transform(y)
        return y
