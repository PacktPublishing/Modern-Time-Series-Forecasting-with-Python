import warnings
from collections import defaultdict
from datetime import datetime
import distutils
import pandas as pd
from tqdm.autonotebook import tqdm
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py
# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_monash_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    distutils.util.strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(
                                    distutils.util.strtobool(line_content[1])
                                )

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val == None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def tsf_row_to_df(row, frequency):
    if frequency == "daily":
        _freq = "1D"
    elif frequency == "half_hourly":
        _freq = "30min"
    date_range = pd.date_range(
        start=row["start_timestamp"], periods=len(row["series_value"]), freq=_freq
    )
    df = pd.DataFrame(
        row["series_value"].astype(float), index=date_range, columns=["values"]
    )
    df.index.name == row["series_name"]
    return df


def write_compact_to_ts(
    df: pd.DataFrame,
    filename: str,
    static_columns: list,
    time_varying_columns: list,
    sep: str = ";",
    encoding: str = "utf-8",
    date_format: str = "%Y-%m-%d %H-%M-%S",
    chunk_size: int = 50,
):
    """Writes a dataframe in the compact form to disk

    Args:
        df (pd.DataFrame): The dataframe in compact form
        filename (str): Filename to which the dataframe should be written to
        static_columns (list): List of column names of static features
        time_varying_columns (list): List of column names of time varying columns
        sep (str, optional): Separator with which the arrays are stored in the text file. Defaults to ";".
        encoding (str, optional): encoding of the text file. Defaults to "utf-8".
        date_format (str, optional): Format in which datetime shud be written out in text file. Defaults to "%Y-%m-%d %H-%M-%S".
        chunk_size (int, optional): Chunk size while writing files to disk. Defaults to 50.

    Returns:
        None
    """
    if sep == ":":
        warnings.warn(
            "Using `:` as separator will not work well if `:` is present in the string representation of date time."
        )
    with open(filename, "w", encoding=encoding) as f:
        for c, dtype in df.dtypes.items():
            if c in static_columns:
                typ = "static"
            elif c in time_varying_columns:
                typ = "time_varying"
            f.write(f"@column {c} {dtype.name} {typ}")
            f.write("\n")
        f.write(f"@data")
        f.write("\n")

        def write_ts(x):
            l = ""
            for c in x.index:
                if isinstance(x[c], np.ndarray):
                    l += "|".join(x[c].astype(str))
                    l += sep
                elif isinstance(x[c], pd.Timestamp):
                    l += x[c].strftime(date_format)
                    l += sep
                else:
                    l += str(x[c])
                    l += sep
            l += "\n"
            return l

        [
            f.writelines([write_ts(x.loc[i]) for i in tqdm(x.index)])
            for x in tqdm(
                np.split(df, np.arange(chunk_size, len(df), chunk_size)),
                desc="Writing in Chunks...",
            )
        ]


def read_ts_to_compact(
    filename: str,
    sep: str = ";",
    encoding: str = "utf-8",
    date_format: str = "%Y-%m-%d %H-%M-%S",
) -> pd.DataFrame:
    """Reads a .ts file from disk to a dataframe in the compact form

    Args:
        filename (str): the file name to be read
        sep (str, optional): Separator which is used in the .ts file. Defaults to ";".
        encoding (str, optional): encoding of the text file. Defaults to "utf-8".
        date_format (str, optional): Format in which datetime shud be written out in text file. Defaults to "%Y-%m-%d %H-%M-%S".

    Returns:
        pd.DataFrame: The dataframe in the compact form
    """
    col_names = []
    col_types = []
    col_meta_types = []
    all_data = defaultdict(list)
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(filename, "r", encoding=encoding) as file:
        for line in tqdm(file):
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@column"):
                            if (
                                len(line_content) != 4
                            ):  # Columns have both name and type and metatype (1)
                                raise Exception("Invalid column specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                            col_meta_types.append(line_content[3])
                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )
                        found_data_tag = True
                elif not line.startswith("#"):  # Skipping comment lines
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                        full_info = line.split(sep)
                        # text split creates an empty item at the end
                        if len(full_info) - 1 != (len(col_names)):
                            raise Exception("Missing attributes/values in series.")

                        for col, typ, meta_typ, info in zip(
                            col_names, col_types, col_meta_types, full_info[:-1]
                        ):
                            if meta_typ == "static":
                                if np.issubdtype(typ, np.datetime64):
                                    all_data[col].append(
                                        pd.to_datetime(info, format=date_format)
                                    )
                                else:
                                    all_data[col].append(info)
                            elif meta_typ == "time_varying":
                                if info[0].isnumeric():
                                    all_data[col].append(
                                        np.fromstring(info, sep="|", dtype=float)
                                    )
                                else:
                                    all_data[col].append(np.array(info.split("|")))
                                # arr = np.array(info.split("|"))
                                # try:
                                #     int(info[0])
                                #     # all_data[col].append(
                                #     #     np.fromstring(info, sep="|", dtype=float)
                                #     # )
                                #     all_data[col].append(
                                #         arr.astype(float)
                                #     )
                                #     # all_data[col].append(np.array(list(map(float, info.split("|")))))
                                # except ValueError:
                                #     # all_data[col].append(
                                #     #     np.fromstring(info, sep="|", dtype=str)
                                #     # )
                                #     all_data[col].append(arr)
                                #     # all_data[col].append(np.array(info.split("|")))
    df = pd.DataFrame(all_data)
    for col, typ in zip(col_names, col_types):
        df[col] = df[col].astype(typ)
    return df


def compact_to_expanded(
    df, timeseries_col, static_cols, time_varying_cols, ts_identifier
):
    def preprocess_expanded(x):
        ### Fill missing dates with NaN ###
        # Create a date range from  start
        dr = pd.date_range(
            start=x["start_timestamp"],
            periods=len(x["energy_consumption"]),
            freq=x["frequency"],
        )
        df_columns = defaultdict(list)
        df_columns["timestamp"] = dr
        for col in [ts_identifier, timeseries_col] + static_cols + time_varying_cols:
            df_columns[col] = x[col]
        return pd.DataFrame(df_columns)

    all_series = []
    for i in tqdm(range(len(df))):
        all_series.append(preprocess_expanded(df.iloc[i]))
    df = pd.concat(all_series)
    del all_series
    return df


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """

    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError(
            "no discernible frequency found to `idx`.  Specify"
            " a frequency string with `freq`."
        )
    return idx


# block_df = read_ts_to_compact("D:\Playground\AdvancedTimeSeriesForecastingBook\Code Dev\data\london_smart_meters\preprocessed\london_smart_meters_merged_block_0-36.ts")


def reduce_memory_footprint(df):
    dtypes = df.dtypes
    object_cols = dtypes[dtypes == "object"].index.tolist()
    float_cols = dtypes[dtypes == "float64"].index.tolist()
    int_cols = dtypes[dtypes == "int64"].index.tolist()
    df[int_cols] = df[int_cols].astype("int32")
    df[object_cols] = df[object_cols].astype("category")
    df[float_cols] = df[float_cols].astype("float32")
    return df


def _get_32_bit_dtype(x):
    dtype = x.dtype
    if dtype.name.startswith("float"):
        redn_dtype = "float32"
    elif dtype.name.startswith("int"):
        redn_dtype = "int32"
    else:
        redn_dtype = None
    return redn_dtype


def replace_array_in_dataframe(df, X, keep_columns=True, keep_index=True):
    return pd.DataFrame(
        X,
        columns=df.columns if keep_columns else None,
        index=df.index if keep_index else None,
    )

def as_ndarray(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.values
    elif isinstance(y, np.ndarray):
        return y
    else:
        raise ValueError("`y` should be pd.SEries, pd.DataFrame, or np.ndarray to cast to np.ndarray")

def is_datetime_dtypes(x):
    return is_datetime(x)
