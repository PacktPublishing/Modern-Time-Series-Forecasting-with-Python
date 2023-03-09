from typing import Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# PyTorch Dataset class for time series data


class TimeSeriesDataset:
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        window: int,
        horizon: int,
        n_val: Union[float, int] = 0.2,
        n_test: Union[float, int] = 0.2,
        normalize: str = "None",  # options are "none", "local", "global"
        normalize_params: Tuple[
            float, float
        ] = None,  # tuple of mean and std for pre-calculated standardization
        mode="train",  # options are "train", "val", "test"
    ):
        if isinstance(n_val, float):
            n_val = int(n_val * len(data))
        if isinstance(n_test, float):
            n_test = int(n_test * len(data))
        if isinstance(data, pd.DataFrame):
            data = data.values
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if normalize == "global" and mode != "train":
            assert (
                isinstance(normalize_params, tuple) and len(normalize_params) == 2
            ), "If using Global Normalization, in valid and test mode normalize_params argument should be a tuple of precalculated mean and std"
        self.data = data.copy()
        self.n_val = n_val
        self.n_test = n_test
        self.window = window
        self.horizon = horizon
        self.normalize = normalize
        self.mode = mode
        total_data_set_length = len(data)
        # The beginning of the data set is where 'train' starts
        # The end of the dataset is here we find the last testing data
        # We therefore start at 0
        # And end at total_data_set_length = n_samples + (n_model+1) + n_val + n_test
        # (a sample is n_model vectors for X and 1 vector for Y)
        # Final -1 is to reflect Python's 0-array convention
        self.n_samples = (
            total_data_set_length - (self.horizon + 1) - self.n_val - self.n_test
        )
        # Adjust the start of the dataset for training / val / test
        if mode == "train":
            start_index = 0
            end_index = (self.horizon + 1) + self.n_samples

        elif mode == "val":
            start_index = (self.horizon + 1) + self.n_samples - self.window
            end_index = (self.horizon + 1) + self.n_samples + self.n_val

        elif mode == "test":
            start_index = (self.horizon + 1) + self.n_samples + self.n_val - self.window
            end_index = (self.horizon + 1) + self.n_samples + self.n_val + self.n_test

        # This is the actual input on which to iterate
        self.data = data[start_index:end_index, :]
        if normalize == "global":
            if mode == "train":
                self.mean = data.mean()
                self.std = data.std()
            else:
                self.mean, self.std = normalize_params
            self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return (
            len(self.data) - self.horizon - self.window + 1
        )  # to account for zero indexing

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window, :]
        y = None
        y = self.data[idx + self.window : idx + self.window + self.horizon, :]
        if self.normalize == "local":
            x = (x - x.mean()) / x.std()
            y = (y - y.mean()) / y.std()
        return x, y


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        n_val: Union[float, int] = 0.2,
        n_test: Union[float, int] = 0.2,
        window: int = 10,
        horizon: int = 1,
        normalize: str = "none",
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data = data
        self.n_val = n_val
        self.n_test = n_test
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self._is_global = normalize == "global"

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = TimeSeriesDataset(
                data=self.data,
                window=self.window,
                horizon=self.horizon,
                n_val=self.n_val,
                n_test=self.n_test,
                normalize=self.normalize,
                normalize_params=None,
                mode="train",
            )
            self.val = TimeSeriesDataset(
                data=self.data,
                window=self.window,
                horizon=self.horizon,
                n_val=self.n_val,
                n_test=self.n_test,
                normalize=self.normalize,
                normalize_params=(self.train.mean, self.train.std)
                if self._is_global
                else None,
                mode="val",
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = TimeSeriesDataset(
                data=self.data,
                window=self.window,
                horizon=self.horizon,
                n_val=self.n_val,
                n_test=self.n_test,
                normalize=self.normalize,
                normalize_params=(self.train.mean, self.train.std)
                if self._is_global
                else None,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# data = np.arange(100).reshape(100, 1)
# data = np.vstack([np.arange(100), np.arange(200,300)]).T
# ds = TimeSeriesDataset(data, window=5, horizon=5, normalize="none")
# dl = DataLoader(ds, batch_size=1, shuffle=True)

# for i, (x, y) in enumerate(dl):
#     # print(x.shape, y.shape)
#     print(x)
#     print("`"*10)
#     print(y)
#     print ("#"*150)
#     if i>5:
#         break


# tr = np.arange(50)
# val = np.arange(100, 120)
# test = np.arange(520, 550)
# # data = np.vstack([np.arange(100), np.arange(200,300)]).T
# data = np.concatenate([tr, val, test])
# dm = TimeSeriesDataModule(
#     data, n_val=len(val), n_test=len(test), normalize="none", batch_size=1
# )
# dm.setup()
# dl = dm.test_dataloader()

# for i, (x, y) in enumerate(dl):
#     # print(x.shape, y.shape)
#     print(x)
#     print("`" * 10)
#     print(y)
#     print("#" * 150)
#     if i > 5:
#         break

################################################################################
##
# Dataset class
##
# class TimeSeriesDataset(Dataset):
#     def __init__(self, global_state: GlobalState, data, mode="train",
#                  debug=False):
#         super(TimeSeriesDataset, self).__init__()

#         self.global_state = global_state
#         self.data_type = mode

#         if self.global_state.debug and (
#                 self.__class__.__name__ not in self.global_state.skip_debug):
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Creating dataloader for data set: {mode}")

#         # In debug mode, only use about 2 epoch of input
#         # TODO refactor to use exactly 2 epoch instead of 700 dates.
#         if self.global_state.debug and (
#                 self.__class__.__name__ not in self.global_state.skip_debug):
#             total_data_set_length = min(global_state.dataset_size, data.size(0))
#         else:
#             total_data_set_length = data.size(0)

#         # The beginning of the data set is where 'train' starts
#         # The end of the dataset is here we find the last testing data
#         # We therefore start at 0
#         # And end at total_data_set_length = n_samples + (n_model+1) + n_val + n_test
#         # (a sample is n_model vectors for X and 1 vector for Y)
#         # Final -1 is to reflect Python's 0-array convention
#         self.n_samples = total_data_set_length - \
#                          (global_state.n_model + 1) - \
#                          global_state.n_val - \
#                          global_state.n_test - \
#                          1

#         # Adjust the start of the dataset for training / val / test
#         if mode == "train":
#             start_index = 0
#             end_index = (global_state.n_model + 1) + self.n_samples

#         elif mode == "val":
#             start_index = self.n_samples
#             end_index = (global_state.n_model + 1) + self.n_samples + \
#                         global_state.n_val

#         elif mode == "test":
#             start_index = self.n_samples + global_state.n_val
#             end_index = (global_state.n_model + 1) + self.n_samples + \
#                         global_state.n_val + \
#                         global_state.n_test

#         # This is the actual input on which to iterate
#         self.data = data[start_index:end_index, :]

#         if self.global_state.debug and (
#                 self.__class__.__name__ not in self.global_state.skip_debug):
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Dataset {self.data_type} - Start index: {start_index}")
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Dataset {self.data_type} - End index: {end_index}")
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Dataset {self.data_type} - data: {self.data.size()}")
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Dataset {self.data_type} - data set iterator"
#                     f" length: {self.data.size()[0]}")
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Dataset {self.data_type} - calculated"
#                     f" n_samples: {self.n_samples}")

#         # d_series is the depth of a series (how many input points per dates)
#         # n_series is the number of series (how many dates)
#         self.n_series, self.d_series = data.size()

#     def __getitem__(self, index):
#         # An item is a tuple of:
#         #   - a transformer_model input being, say, 60 dates of time series
#         #   -  the following date as expected output
#         # if self.global_state.debug and (
#         #         self.__class__.__name__ not in self.global_state.skip_debug):
#             # logging(f"{self.__class__.__name__}, "
#             #         f"{inspect.currentframe().f_code.co_name}: "
#             #         f" {self.data_type} \t item  no.: {index}")
#             # logging(f"{self.__class__.__name__}, "
#             #         f"{inspect.currentframe().f_code.co_name}: "
#             #         f"       x: from {index} to {index + self.global_state.n_model}")
#             # logging(f"{self.__class__.__name__}, "
#             #         f"{inspect.currentframe().f_code.co_name}: "
#             #         f"       y: at {index + self.global_state.n_model}")

#         return (self.data[index: index + self.global_state.n_model, :],
#                 self.data[index + self.global_state.n_model, :])

#     def __len__(self):
#         """
#         Total number of samples in the dataset
#         """
#         if self.global_state.debug and (
#                 self.__class__.__name__ not in self.global_state.skip_debug):
#             logging(f"{self.__class__.__name__}, "
#                     f"{inspect.currentframe().f_code.co_name}: "
#                     f" Call to __len__() on {self.data_type} returning"
#                     f" self.data.size()[0] - (self.global_state.n_model + 1) ="
#                     f" {self.data.size()[0] - (self.global_state.n_model + 1)}")
#         return self.data.size()[0] - (self.global_state.n_model + 1)
