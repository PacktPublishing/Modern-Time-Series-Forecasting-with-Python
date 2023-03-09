import random
import warnings

import numpy as np
from darts import TimeSeries

from src.utils.ts_utils import _remove_nan_union


def sse(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    return np.sum((y_true - y_pred) ** 2)


def block_shuffle(x, num_blocks):
    sh_array = np.array_split(x, num_blocks)
    random.shuffle(sh_array)
    return np.concatenate(sh_array)


def _backtest(model, x, backtesting_start, n_folds):
    history_len = int(len(x) * backtesting_start)
    train_x = x[:history_len]
    test_x = x[history_len:]
    blocks = np.array_split(test_x, n_folds)
    metric_l = []
    for i, block in enumerate(blocks):
        x_ = TimeSeries.from_values(train_x)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            model.fit(x_)
        y_pred = model.predict(len(block))
        metric_l.append(sse(block, np.squeeze(y_pred.data_array().values)))
        if i < len(blocks) - 1:
            train_x = np.concatenate([train_x, block])
    return np.mean(metric_l) if len(metric_l) > 1 else metric_l[0]


def kaboudan_metric(x, model, block_size=5, backtesting_start=0.5, n_folds=1):
    sse_before = _backtest(model, x, backtesting_start, n_folds)
    x_shuffled = block_shuffle(x, num_blocks=len(x) // block_size)
    sse_after = _backtest(model, x_shuffled, backtesting_start, n_folds)
    return 1 - (sse_before / sse_after)


def modified_kaboudan_metric(x, model, block_size=5, backtesting_start=0.5, n_folds=1):
    sse_before = _backtest(model, x, backtesting_start, n_folds)
    x_shuffled = block_shuffle(x, num_blocks=len(x) // block_size)
    sse_after = _backtest(model, x_shuffled, backtesting_start, n_folds)
    return np.clip(1 - np.sqrt(sse_before / sse_after), 0, None)
