import copy
import warnings
from dataclasses import MISSING, dataclass, field
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from darts.metrics import mae, mase, mse
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler

from src.utils.general import difference_list, intersect_list
from src.utils.ts_utils import darts_metrics_adapter, forecast_bias

# from category_encoders import OneHotEncoder


@dataclass
class MissingValueConfig:

    bfill_columns: List = field(
        default_factory=list,
        metadata={"help": "Column names which should be filled using strategy=`bfill`"},
    )

    ffill_columns: List = field(
        default_factory=list,
        metadata={"help": "Column names which should be filled using strategy=`ffill`"},
    )
    zero_fill_columns: List = field(
        default_factory=list,
        metadata={"help": "Column names which should be filled using 0"},
    )

    def impute_missing_values(self, df: pd.DataFrame):
        df = df.copy()
        bfill_columns = intersect_list(df.columns, self.bfill_columns)
        df[bfill_columns] = df[bfill_columns].fillna(method="bfill")
        ffill_columns = intersect_list(df.columns, self.ffill_columns)
        df[ffill_columns] = df[ffill_columns].fillna(method="ffill")
        zero_fill_columns = intersect_list(df.columns, self.zero_fill_columns)
        df[zero_fill_columns] = df[zero_fill_columns].fillna(0)
        check = df.isnull().any()
        missing_cols = check[check].index.tolist()
        missing_numeric_cols = intersect_list(
            missing_cols, df.select_dtypes([np.number]).columns.tolist()
        )
        missing_object_cols = intersect_list(
            missing_cols, df.select_dtypes(["object"]).columns.tolist()
        )
        # Filling with mean and NA as default fillna strategy
        df[missing_numeric_cols] = df[missing_numeric_cols].fillna(
            df[missing_numeric_cols].mean()
        )
        df[missing_object_cols] = df[missing_object_cols].fillna("NA")
        return df


@dataclass
class FeatureConfig:

    date: List = field(
        default=MISSING,
        metadata={"help": "Column name of the date column"},
    )
    target: str = field(
        default=MISSING,
        metadata={"help": "Column name of the target column"},
    )

    original_target: str = field(
        default=None,
        metadata={
            "help": "Column name of the original target column in acse of transformed target. If None, it will be assigned same value as target"
        },
    )

    continuous_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields. Defaults to []"},
    )
    boolean_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the boolean fields. Defaults to []"},
    )

    index_cols: str = field(
        default_factory=list,
        metadata={
            "help": "Column names which needs to be set as index in the X and Y dataframes."
        },
    )
    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the exogenous features. Must be a subset of categorical and continuous features"
        },
    )
    feature_list: List[str] = field(init=False)

    def __post_init__(self):
        assert (
            len(self.categorical_features) + len(self.continuous_features) > 0
        ), "There should be at-least one feature defined in categorical or continuous columns"
        self.feature_list = (
            self.categorical_features + self.continuous_features + self.boolean_features
        )
        assert (
            self.target not in self.feature_list
        ), f"`target`({self.target}) should not be present in either categorical, continuous or boolean feature list"
        assert (
            self.date not in self.feature_list
        ), f"`date`({self.target}) should not be present in either categorical, continuous or boolean feature list"
        extra_exog = set(self.exogenous_features) - set(self.feature_list)
        assert (
            len(extra_exog) == 0
        ), f"These exogenous features are not present in feature list: {extra_exog}"
        intersection = (
            set(self.continuous_features)
            .intersection(self.categorical_features + self.boolean_features)
            .union(
                set(self.categorical_features).intersection(
                    self.continuous_features + self.boolean_features
                )
            )
            .union(
                set(self.boolean_features).intersection(
                    self.continuous_features + self.categorical_features
                )
            )
        )
        assert (
            len(intersection) == 0
        ), f"There should not be any overlaps between the categorical contonuous and boolean features. {intersection} are present in more than one definition"
        if self.original_target is None:
            self.original_target = self.target

    def get_X_y(
        self, df: pd.DataFrame, categorical: bool = False, exogenous: bool = False
    ):
        feature_list = copy.deepcopy(self.continuous_features)
        if categorical:
            feature_list += self.categorical_features + self.boolean_features
        if not exogenous:
            feature_list = list(set(feature_list) - set(self.exogenous_features))
        feature_list = list(set(feature_list))
        delete_index_cols = list(set(self.index_cols) - set(self.feature_list))
        (X, y, y_orig) = (
            df.loc[:, set(feature_list + self.index_cols)]
            .set_index(self.index_cols, drop=False)
            .drop(columns=delete_index_cols),
            df.loc[:, [self.target] + self.index_cols].set_index(
                self.index_cols, drop=True
            )
            if self.target in df.columns
            else None,
            df.loc[:, [self.original_target] + self.index_cols].set_index(
                self.index_cols, drop=True
            )
            if self.original_target in df.columns
            else None,
        )
        return X, y, y_orig


@dataclass
class ModelConfig:

    model: BaseEstimator = field(
        default=MISSING, metadata={"help": "Sci-kit Learn Compatible model instance"}
    )

    name: str = field(
        default=None,
        metadata={
            "help": "Name or identifier for the model. If left None, will use the string representation of the model"
        },
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "Flag whether to normalize the input or not"},
    )
    fill_missing: bool = field(
        default=True,
        metadata={"help": "Flag whether to fill missing values before fitting"},
    )
    encode_categorical: bool = field(
        default=False,
        metadata={"help": "Flag whether to encode categorical values before fitting"},
    )
    categorical_encoder: BaseEstimator = field(
        default=None,
        metadata={"help": "Categorical Encoder to be used"},
    )

    def __post_init__(self):
        assert not (
            self.encode_categorical and self.categorical_encoder is None
        ), "`categorical_encoder` cannot be None if `encode_categorical` is True"

    def clone(self):
        self.model = clone(self.model)
        return self


class MLForecast:
    def __init__(
        self,
        model_config: ModelConfig,
        feature_config: FeatureConfig,
        missing_config: MissingValueConfig = None,
        target_transformer: object = None,
    ) -> None:
        """Convenient wrapper around scikit-learn style estimators

        Args:
            model_config (ModelConfig): Instance of the ModelConfig object defining the model
            feature_config (FeatureConfig): Instance of the FeatureConfig object defining the features
            missing_config (MissingValueConfig, optional): Instance of the MissingValueConfig object
                defining how to fill missing values. Defaults to None.
            target_transformer (object, optional): Instance of target transformers from src.transforms.
                Should support `fit`, `transform`, and `inverse_transform`. It should also
                return `pd.Series` with datetime index to work without an error. Defaults to None.
        """
        self.model_config = model_config
        self.feature_config = feature_config
        self.missing_config = missing_config
        self.target_transformer = target_transformer
        self._model = clone(model_config.model)
        if self.model_config.normalize:
            self._scaler = StandardScaler()
        if self.model_config.encode_categorical:
            self._cat_encoder = self.model_config.categorical_encoder
            self._encoded_categorical_features = copy.deepcopy(
                self.feature_config.categorical_features
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        is_transformed: bool = False,
        fit_kwargs: Dict = {},
    ):
        """Handles standardization, missing value handling, and training the model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns
            y (Union[pd.Series, np.ndarray]): Dataframe, Series, or np.ndarray with the targets
            is_transformed (bool, optional): Whether the target is already transformed.
            If `True`, fit wont be transforming the target using the target_transformer
                if provided. Defaults to False.
            fit_kwargs (Dict, optional): The dictionary with keyword args to be passed to the
                fit funciton of the model. Defaults to {}.
        """
        missing_feats = difference_list(X.columns, self.feature_config.feature_list)
        if len(missing_feats) > 0:
            warnings.warn(
                f"Some features in defined in FeatureConfig is not present in the dataframe. Ignoring these features: {missing_feats}"
            )
        self._continuous_feats = intersect_list(
            self.feature_config.continuous_features, X.columns
        )
        self._categorical_feats = intersect_list(
            self.feature_config.categorical_features, X.columns
        )
        self._boolean_feats = intersect_list(
            self.feature_config.boolean_features, X.columns
        )
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            missing_cat_cols = difference_list(
                self._categorical_feats,
                self.model_config.categorical_encoder.cols,
            )
            assert (
                len(missing_cat_cols) == 0
            ), f"These categorical features are not handled by the categorical_encoder : {missing_cat_cols}"
            # In later versions of sklearn get_feature_names have been deprecated
            try:
                feature_names = self.model_config.categorical_encoder.get_feature_names()
            except AttributeError:
                # in favour of get_feature_names_out()
                feature_names = self.model_config.categorical_encoder.get_feature_names_out()
            X = self._cat_encoder.fit_transform(X, y)
            self._encoded_categorical_features = difference_list(
                feature_names,
                self.feature_config.continuous_features
                + self.feature_config.boolean_features,
            )
        else:
            self._encoded_categorical_features = []
        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.fit_transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )
        self._train_features = X.columns.tolist()
        # print(len(self._train_features))
        if not is_transformed and self.target_transformer is not None:
            y = self.target_transformer.fit_transform(y)
        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts on the given dataframe using the trained model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns. The index is passed on to the prediction series

        Returns:
            pd.Series: predictions using the model as a pandas Series with datetime index
        """
        assert len(intersect_list(self._train_features, X.columns)) == len(
            self._train_features
        ), f"All the features during training is not available while predicting: {difference_list(self._train_features, X.columns)}"
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            X = self._cat_encoder.transform(X)
        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )
        y_pred = pd.Series(
            self._model.predict(X).ravel(),
            index=X.index,
            name=f"{self.model_config.name}",
        )
        if self.target_transformer is not None:
            y_pred = self.target_transformer.inverse_transform(y_pred)
            y_pred.name = f"{self.model_config.name}"
        return y_pred

    def feature_importance(self) -> pd.DataFrame:
        """Generates the feature importance dataframe, if available. For linear
            models the coefficients are used and tree based models use the inbuilt
            feature importance. For the rest of the models, it returns an empty dataframe.

        Returns:
            pd.DataFrame: Feature Importance dataframe, sorted in descending order of its importances.
        """
        if hasattr(self._model, "coef_") or hasattr(
            self._model, "feature_importances_"
        ):
            feat_df = pd.DataFrame(
                {
                    "feature": self._train_features,
                    "importance": self._model.coef_.ravel()
                    if hasattr(self._model, "coef_")
                    else self._model.feature_importances_.ravel(),
                }
            )
            feat_df["_abs_imp"] = np.abs(feat_df.importance)
            feat_df = feat_df.sort_values("_abs_imp", ascending=False).drop(
                columns="_abs_imp"
            )
        else:
            feat_df = pd.DataFrame()
        return feat_df


def calculate_metrics(
    y: pd.Series, y_pred: pd.Series, name: str, y_train: pd.Series = None
):
    """Method to calculate the metrics given the actual and predicted series

    Args:
        y (pd.Series): Actual target with datetime index
        y_pred (pd.Series): Predictions with datetime index
        name (str): Name or identification for the model
        y_train (pd.Series, optional): Actual train target to calculate MASE with datetime index. Defaults to None.

    Returns:
        Dict: Dictionary with MAE, MSE, MASE, and Forecast Bias
    """
    return {
        "Algorithm": name,
        "MAE": darts_metrics_adapter(mae, actual_series=y, pred_series=y_pred),
        "MSE": darts_metrics_adapter(mse, actual_series=y, pred_series=y_pred),
        "MASE": darts_metrics_adapter(
            mase, actual_series=y, pred_series=y_pred, insample=y_train
        )
        if y_train is not None
        else None,
        "Forecast Bias": darts_metrics_adapter(
            forecast_bias, actual_series=y, pred_series=y_pred
        ),
    }
