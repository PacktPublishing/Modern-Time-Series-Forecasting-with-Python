import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


# adapted from gluonts
def time_features_from_frequency_str(freq_str: str) -> List[str]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """

    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.MonthBegin: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.MonthEnd: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.Week: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
        ],
        offsets.Day: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.BusinessDay: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.Hour: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
        ],
        offsets.Minute: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
            "Minute",
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return feature

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y, YS   - yearly
            alias: A
        M, MS   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)


# adapted from fastai
@classmethod
def make_date(df: pd.DataFrame, date_field: str):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
    return df


# adapted from fastai
def add_temporal_features(
    df: pd.DataFrame,
    field_name: str,
    frequency: str,
    add_elapsed: bool = True,
    prefix: str = None,
    drop: bool = True,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds columns relevant to a date in the column `field_name` of `df`.

    Args:
        df (pd.DataFrame): Dataframe to which the features need to be added
        field_name (str): The date column which should be encoded using temporal features
        frequency (str): The frequency of the date column so that only relevant features are added.
            If frequency is "Weekly", then temporal features like hour, minutes, etc. doesn't make sense.
        add_elapsed (bool, optional): Add time elapsed as a monotonically increasing function. Defaults to True.
        prefix (str, optional): Prefix to the newly created columns. If left None, will use the field name. Defaults to None.
        drop (bool, optional): Flag to drop the data column after feature creation. Defaults to True.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    field = df[field_name]
    prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
    attr = time_features_from_frequency_str(frequency)
    _32_bit_dtype = "int32"
    added_features = []
    for n in attr:
        if n == "Week":
            continue
        df[prefix + n] = (
            getattr(field.dt, n.lower()).astype(_32_bit_dtype)
            if use_32_bit
            else getattr(field.dt, n.lower())
        )
        added_features.append(prefix + n)
    # Pandas removed `dt.week` in v1.1.10
    if "Week" in attr:
        week = (
            field.dt.isocalendar().week
            if hasattr(field.dt, "isocalendar")
            else field.dt.week
        )
        df.insert(
            3, prefix + "Week", week.astype(_32_bit_dtype) if use_32_bit else week
        )
        added_features.append(prefix + "Week")
    if add_elapsed:
        mask = ~field.isna()
        df[prefix + "Elapsed"] = np.where(
            mask, field.values.astype(np.int64) // 10**9, None
        )
        if use_32_bit:
            if df[prefix + "Elapsed"].isnull().sum() == 0:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("int32")
            else:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("float32")
        added_features.append(prefix + "Elapsed")
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df, added_features


def _calculate_fourier_terms(
    seasonal_cycle: np.ndarray, max_cycle: int, n_fourier_terms: int
):
    """Calculates Fourier Terms given the seasonal cycle and max_cycle"""
    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])


def add_fourier_features(
    df: pd.DataFrame,
    column_to_encode: str,
    max_value: Optional[int] = None,
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds Fourier Terms for the specified seasonal cycle column, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cyycles which has to be encoded
        column_to_encode (str): The column name which has the seasonal cycle
        max_value (int): The maximum value the seasonal cycle can attain. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None

    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert (
        column_to_encode in df.columns
    ), "`column_to_encode` should be a valid column name in the dataframe"
    assert is_numeric_dtype(
        df[column_to_encode]
    ), "`column_to_encode` should have numeric values."
    if max_value is None:
        max_value = df[column_to_encode].max()
        raise warnings.warn(
            "Inferring max cycle as {} from the data. This may not be accuracte if data is less than a single seasonal cycle."
        )
    fourier_features = _calculate_fourier_terms(
        df[column_to_encode].astype(int).values,
        max_cycle=max_value,
        n_fourier_terms=n_fourier_terms,
    )
    feature_names = [
        f"{column_to_encode}_sin_{i}" for i in range(1, n_fourier_terms + 1)
    ] + [f"{column_to_encode}_cos_{i}" for i in range(1, n_fourier_terms + 1)]
    df[feature_names] = fourier_features
    if use_32_bit:
        df[feature_names] = df[feature_names].astype("float32")
    return df, feature_names


def bulk_add_fourier_features(
    df: pd.DataFrame,
    columns_to_encode: List[str],
    max_values: List[int],
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds Fourier Terms for all the specified seasonal cycle columns, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cyycles which has to be encoded
        columns_to_encode (List[str]): The column names which has the seasonal cycle
        max_values (List[int]): The list of maximum values the seasonal cycles can attain in the
            same order as the columns to encode. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None

    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert len(columns_to_encode) == len(
        max_values
    ), "`columns_to_encode` and `max_values` should be of same length."
    added_features = []
    for column_to_encode, max_value in zip(columns_to_encode, max_values):
        df, features = add_fourier_features(
            df,
            column_to_encode,
            max_value,
            n_fourier_terms=n_fourier_terms,
            use_32_bit=use_32_bit,
        )
        added_features += features
    return df, added_features
