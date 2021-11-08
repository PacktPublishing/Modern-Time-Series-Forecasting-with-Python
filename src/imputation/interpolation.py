import warnings

import numpy as np
import pandas as pd
from ._solver import Solver
from sklearn.utils import check_array
from statsmodels.tsa.seasonal import seasonal_decompose


class SeasonalInterpolation(Solver):
    def __init__(
        self,
        seasonal_period: int,
        decomposition_strategy: str = "additive",
        decomposition_args: dict = {},
        interpolation_strategy: str = "linear",
        interpolation_args: dict = {},
        fill_border_values: int = 0,
        min_value: int = None,
        max_value: int = None,
        verbose: bool = True,
    ):
        """Interpolates after Seasonal Decomposition

        Args:
            seasonal_period (int): The number of periods after which we expect seasonality to repeat
            decomposition_strategy (str, optional): The decomposition strategy. Either `additive` or `mulitplicative`. Defaults to "additive".
            decomposition_args (dict, optional): The arguments to be passed to `seasonal_decompose` of `statsmodels`. Defaults to {}.
            interpolation_strategy (str, optional): Strategy to interpolate the deseasonalized array.
                Options are `linear`, `quadratic`, `splie`, `polynomial`, etc.
                For full list refer to pd.Series.interpolate. Defaults to "linear".
            interpolation_args (dict, optional): The arguments to be passed to pd.Series.interpolate. Defaults to {}.
            fill_border_values (int, optional): Defines what to fill in border nulls which are not filled in by interpolate. Defaults to 0.
            min_value (int, optional): Max value. Defaults to None.
            max_value (int, optional): Min value. Defaults to None.
            verbose (bool, optional): Controls the verbosity. Defaults to True.
        """
        Solver.__init__(
            self, fill_method="zero", min_value=min_value, max_value=max_value
        )
        # Check if dec_model has a valid value:
        if decomposition_strategy not in ["multiplicative", "additive"]:
            raise ValueError(
                decomposition_strategy + " is not a supported decomposition strategy."
            )
        if (
            interpolation_strategy in ["spline", "polynomial"]
            and "order" not in interpolation_args.keys()
        ):  # ['linear', 'nearest', "zero", "quadratic", "spline", "polynomial"]
            raise ValueError(
                interpolation_strategy
                + " interpolation strategy needs an order to be sopecified in the interpolation_args."
            )
        self.interpolation_strategy = interpolation_strategy
        self.decomposition_strategy = decomposition_strategy
        extrapolate = decomposition_args["extrapolate_trend"] if "extrapolate_trend" in decomposition_args.keys() else "freq"
        decomposition_args.update(
            {"model": decomposition_strategy, "period": seasonal_period, "extrapolate_trend": extrapolate}
        )
        interpolation_args.update(
            {
                "method": interpolation_strategy,
            }
        )
        self.interpolation_args = interpolation_args
        self.decomposition_args = decomposition_args
        self.fill_border_values = fill_border_values
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`

        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = check_array(X, force_all_finite=False)
        if missing_mask.sum() == 0:
            warnings.warn(
                "[Seasonal Interpolation] Warning: provided matrix doesn't contain any missing values."
            )
            warnings.warn(
                "[Seasonal Interpolation] The algorithm will run, but will return an unchanged matrix."
            )
        X_filled = (
            pd.DataFrame(X)
            .interpolate(axis=0, **self.interpolation_args)
            .fillna(self.fill_border_values)
            .values
        )
        trends = []
        resids = []
        seasonality = []
        for col in range(X_original.shape[1]):
            decomposition = seasonal_decompose(
                X_filled[:, col], **self.decomposition_args
            )
            trends.append(decomposition.trend)
            resids.append(decomposition.resid)
            seasonality.append(decomposition.seasonal)
        trends = np.vstack(trends).T
        resids = np.vstack(resids).T
        seasonality = np.vstack(seasonality).T
        if self.decomposition_strategy == "additive":
            deseasonalized = trends + resids
        elif self.decomposition_strategy == "multiplicative":
            deseasonalized = trends * resids
        deseasonalized[missing_mask] = np.nan
        deseasonalized = (
            pd.DataFrame(deseasonalized)
            .interpolate(axis=0, **self.interpolation_args)
            .fillna(self.fill_border_values)
            .values
        )
        if self.decomposition_strategy == "additive":
            X_result = deseasonalized + seasonality
        elif self.decomposition_strategy == "multiplicative":
            X_result = deseasonalized * seasonality
        X_result = self.clip(X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def fit(self, X, y=None):
        """
        Fit the imputer on input `X`.

        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.fit not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only fit_transform is "
            "supported at this time." % (self.__class__.__name__,)
        )

    def transform(self, X, y=None):
        """
        Transform input `X`.

        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.transform not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only %s.fit_transform is "
            "supported at this time."
            % (self.__class__.__name__, self.__class__.__name__)
        )
