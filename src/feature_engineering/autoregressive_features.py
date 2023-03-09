import warnings
from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_list_like
from window_ops.rolling import (
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_std,
)

from src.utils.data_utils import _get_32_bit_dtype

ALLOWED_AGG_FUNCS = ["mean", "max", "min", "std"]
SEASONAL_ROLLING_MAP = {
    "mean": seasonal_rolling_mean,
    "min": seasonal_rolling_min,
    "max": seasonal_rolling_max,
    "std": seasonal_rolling_std,
}


"""
Different ways of creating lags and runtimes
1.
train_df['lag1']=train_df.groupby(["LCLid"])['energy_consumption'].shift(1)
723 ms ± 13.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
2.
train_df['lag1']=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: x.shift(1))
1.63 s ± 50.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
3.
from window_ops.shift import shift_array
train_df['lag1'] = train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: shift_array(x.values, 1))
1.58 s ± 27.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""


def add_lags(
    df: pd.DataFrame,
    lags: List[int],
    column: str,
    ts_id: str = None,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Lags for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        lags (List[int]): List of lags to be created
        column (str): Name of the column to be lagged
        ts_id (str, optional): Column name of Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(lags), "`lags` should be a list of all required lags"
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        )
        # Assuming just one unique time series in dataset
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df[column].shift(l).astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {f"{column}_lag_{l}": df[column].shift(l) for l in lags}
    else:
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column]
                .shift(l)
                .astype(_32_bit_dtype)
                for l in lags
            }
        else:
            col_dict = {
                f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
            }
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())
    return df, added_features


"""
Different ways of calculating rolling statistics
1.
train_df["rolling_3_mean"]=train_df.groupby(["LCLid"])['energy_consumption'].shift(1).rolling(3).mean()
1.02 s ± 11.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
2.
train_df["rolling_3_mean"]=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: x.shift(1).rolling(3).mean())
1.92 s ± 45.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
3.
train_df["rolling_3_mean"]=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: rolling_mean(x.shift(1).values, window_size=3))
1.67 s ± 17.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
4. (for multiple aggregations)
train_df.groupby(["LCLid"])['energy_consumption'].shift(1).rolling(3).agg({"rolling_3_mean": "mean", "rolling_3_std": "std"})
1.8 s ± 26.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""


def add_rolling_features(
    df: pd.DataFrame,
    rolls: List[int],
    column: str,
    agg_funcs: List[str] = ["mean", "std"],
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Add rolling statistics from the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        rolls (List[int]): Different windows over which the rolling aggregations to be done
        column (str): The column used for feature engineering
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].
        ts_id (str, optional): Unique id for a time series. Defaults to None.
        n_shift (int, optional): Number of time steps to shift before computing rolling statistics.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(
        rolls
    ), "`rolls` should be a list of all required rolling windows"
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    assert (
        len(set(agg_funcs) - set(ALLOWED_AGG_FUNCS)) == 0
    ), f"`agg_funcs` should be one of {ALLOWED_AGG_FUNCS}"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        )
        # Assuming just one unique time series in dataset
        rolling_df = pd.concat(
            [
                df[column]
                .shift(n_shift)
                .rolling(l)
                .agg({f"{column}_rolling_{l}_{agg}": agg for agg in agg_funcs})
                for l in rolls
            ],
            axis=1,
        )

    else:
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        rolling_df = pd.concat(
            [
                df.groupby(ts_id)[column]
                .shift(n_shift)
                .rolling(l)
                .agg({f"{column}_rolling_{l}_{agg}": agg for agg in agg_funcs})
                for l in rolls
            ],
            axis=1,
        )

    df = df.assign(**rolling_df.to_dict("list"))
    added_features = rolling_df.columns.tolist()
    if use_32_bit and _32_bit_dtype is not None:
        df[added_features] = df[added_features].astype(_32_bit_dtype)
    return df, added_features


def add_seasonal_rolling_features(
    df: pd.DataFrame,
    seasonal_periods: List[int],
    rolls: List[int],
    column: str,
    agg_funcs: List[str] = ["mean", "std"],
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Add seasonal rolling statistics from the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        seasonal_periods (List[int]): List of seasonal periods over which the seasonal rolling operations should be done
        rolls (List[int]): List of seasonal rolling window over which the aggregation functions will be applied
        column (str): [description]
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].. Defaults to ["mean", "std"].
        ts_id (str, optional): Unique id for a time series. Defaults to None.
        n_shift (int, optional): The number of seasonal shifts to be applied before the seasonal rolling operation.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(
        rolls
    ), "`rolls` should be a list of all required rolling windows"
    assert isinstance(
        seasonal_periods, list
    ), "`seasonal_periods` should be a list of all required seasonal cycles over which rolling statistics to be created"
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    assert (
        len(set(agg_funcs) - set(ALLOWED_AGG_FUNCS)) == 0
    ), f"`agg_funcs` should be one of {ALLOWED_AGG_FUNCS}"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    agg_funcs = {agg: SEASONAL_ROLLING_MAP[agg] for agg in agg_funcs}
    added_features = []
    for sp in seasonal_periods:
        if ts_id is None:
            warnings.warn(
                "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
            )
            # Assuming just one unique time series in dataset
            if use_32_bit and _32_bit_dtype is not None:
                col_dict = {
                    f"{column}_{sp}_seasonal_rolling_{l}_{name}": df[column]
                    .transform(
                        lambda x: agg(
                            x.shift(n_shift * sp).values,
                            season_length=sp,
                            window_size=l,
                        )
                    )
                    .astype(_32_bit_dtype)
                    for (name, agg) in agg_funcs.items()
                    for l in rolls
                }
            else:
                col_dict = {
                    f"{column}_{sp}_seasonal_rolling_{l}_{name}": df[column].transform(
                        lambda x: agg(
                            x.shift(n_shift * sp).values,
                            season_length=sp,
                            window_size=l,
                        )
                    )
                    for (name, agg) in agg_funcs.items()
                    for l in rolls
                }

        else:
            assert (
                ts_id in df.columns
            ), "`ts_id` should be a valid column in the provided dataframe"
            if use_32_bit and _32_bit_dtype is not None:
                col_dict = {
                    f"{column}_{sp}_seasonal_rolling_{l}_{name}": df.groupby(ts_id)[
                        column
                    ]
                    .transform(
                        lambda x: agg(
                            x.shift(n_shift * sp).values,
                            season_length=sp,
                            window_size=l,
                        )
                    )
                    .astype(_32_bit_dtype)
                    for (name, agg) in agg_funcs.items()
                    for l in rolls
                }
            else:
                col_dict = {
                    f"{column}_{sp}_seasonal_rolling_{l}_{name}": df.groupby(ts_id)[
                        column
                    ].transform(
                        lambda x: agg(
                            x.shift(n_shift * sp).values,
                            season_length=sp,
                            window_size=l,
                        )
                    )
                    for (name, agg) in agg_funcs.items()
                    for l in rolls
                }
        df = df.assign(**col_dict)
        added_features += list(col_dict.keys())
    return df, added_features


"""
Different ways of calculating ewma
1.
train_df["ewma_alpha_0.9"]=train_df.groupby(["LCLid"])['energy_consumption'].shift(1).ewm(alpha=0.9).mean()
868 ms ± 13.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
2.
train_df["ewma_alpha_0.9"]=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: x.shift(1).ewm(alpha=0.9).mean())
1.95 s ± 85 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
3.
from window_ops.ewm import ewm_mean
train_df["ewma_alpha_0.9"]=train_df.groupby(["LCLid"])['energy_consumption'].transform(lambda x: ewm_mean(x.shift(1).values, alpha=0.9))
1.9 s ± 53 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""


def add_ewma(
    df: pd.DataFrame,
    column: str,
    alphas: List[float] = [0.5],
    spans: List[float] = None,
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Exponentially Weighted Average for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        column (str): Name of the column to be lagged
        alphas (List[float]): List of alphas (smoothing parameters) using which ewmas are be created
        spans (List[float]): List of spans using which ewmas are be created. When we refer to a 60 period EWMA, span is 60.
            alpha = 2/(1+span). If span is given, we ignore alpha.
        ts_id (str, optional): Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        n_shift (int, optional): Number of time steps to shift before computing ewma.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    if spans is not None:
        assert isinstance(
            spans, list
        ), "`spans` should be a list of all required period spans"
        use_spans = True
    if alphas is not None:
        assert isinstance(
            alphas, list
        ), "`alphas` should be a list of all required smoothing parameters"
    if spans is None and alphas is None:
        raise ValueError(
            "Either `alpha` or `spans` should be provided for the function to"
        )
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        )
        # Assuming just one unique time series in dataset
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                .astype(_32_bit_dtype)
                for param in (spans if use_spans else alphas)
            }
        else:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                for param in (spans if use_spans else alphas)
            }
    else:
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df.groupby(
                    [ts_id]
                )[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                .astype(_32_bit_dtype)
                for param in (spans if use_spans else alphas)
            }
        else:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df.groupby(
                    [ts_id]
                )[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                for param in (spans if use_spans else alphas)
            }
    df = df.assign(**col_dict)
    return df, list(col_dict.keys())
