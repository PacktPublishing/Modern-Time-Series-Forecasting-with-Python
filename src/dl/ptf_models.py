"""Base Model"""
from abc import ABCMeta, abstractmethod
from pytorch_forecasting.models import BaseModel
from dataclasses import dataclass, field
from typing import Callable, Dict
import torch
import torch.nn as nn


class SingleStepRNN(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
    ):
        super().__init__()
        if rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("Invalid RNN type")
        self.fc = nn.Linear(hidden_size, 1)
    
    @abstractmethod
    def forward(self, x: Dict):
        raise NotImplementedError()


class SingleStepRNNModel(BaseModel):
    def __init__(self, network_callable: Callable, model_params: Dict, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters({"network_callable": network_callable, "model_params": model_params})
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = network_callable(**model_params)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prediction = self.network(x)
        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        # We need to return a dictionary that at least contains the prediction
        # The parameter can be directly forwarded from the input.
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)


# from pytorch_forecasting.models.nbeats.sub_modules import NBEATSGenericBlock, NBEATSSeasonalBlock, NBEATSTrendBlock
# from pytorch_forecasting.models import NBeats
# from pytorch_forecasting.data.encoders import NaNLabelEncoder
# from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
# from typing import Dict, List

# class NBeatsx(NBeats):
#     def __init__(
#         self,
#         stack_types: List[str] = ["trend", "seasonality"],
#         num_blocks=[3, 3],
#         num_block_layers=[3, 3],
#         widths=[32, 512],
#         sharing: List[int] = [True, True],
#         expansion_coefficient_lengths: List[int] = [3, 7],
#         prediction_length: int = 1,
#         context_length: int = 1,
#         dropout: float = 0.1,
#         learning_rate: float = 1e-2,
#         log_interval: int = -1,
#         log_gradient_flow: bool = False,
#         log_val_interval: int = None,
#         weight_decay: float = 1e-3,
#         loss: MultiHorizonMetric = None,
#         reduce_on_plateau_patience: int = 1000,
#         backcast_loss_ratio: float = 0.0,
#         logging_metrics: nn.ModuleList = None,
#         **kwargs,
#     ):
#         """
#         Initialize NBeats Model - use its :py:meth:`~from_dataset` method if possible.
#         Based on the article
#         `N-BEATS: Neural basis expansion analysis for interpretable time series
#         forecasting <http://arxiv.org/abs/1905.10437>`_. The network has (if used as ensemble) outperformed all
#         other methods
#         including ensembles of traditional statical methods in the M4 competition. The M4 competition is arguably
#         the most
#         important benchmark for univariate time series forecasting.
#         The :py:class:`~pytorch_forecasting.models.nhits.NHiTS` network has recently shown to consistently outperform
#         N-BEATS.
#         Args:
#             stack_types: One of the following values: “generic”, “seasonality" or “trend". A list of strings
#                 of length 1 or ‘num_stacks’. Default and recommended value
#                 for generic mode: [“generic”] Recommended value for interpretable mode: [“trend”,”seasonality”]
#             num_blocks: The number of blocks per stack. A list of ints of length 1 or ‘num_stacks’.
#                 Default and recommended value for generic mode: [1] Recommended value for interpretable mode: [3]
#             num_block_layers: Number of fully connected layers with ReLu activation per block. A list of ints of length
#                 1 or ‘num_stacks’.
#                 Default and recommended value for generic mode: [4] Recommended value for interpretable mode: [4]
#             width: Widths of the fully connected layers with ReLu activation in the blocks.
#                 A list of ints of length 1 or ‘num_stacks’. Default and recommended value for generic mode: [512]
#                 Recommended value for interpretable mode: [256, 2048]
#             sharing: Whether the weights are shared with the other blocks per stack.
#                 A list of ints of length 1 or ‘num_stacks’. Default and recommended value for generic mode: [False]
#                 Recommended value for interpretable mode: [True]
#             expansion_coefficient_length: If the type is “G” (generic), then the length of the expansion
#                 coefficient.
#                 If type is “T” (trend), then it corresponds to the degree of the polynomial. If the type is “S”
#                 (seasonal) then this is the minimum period allowed, e.g. 2 for changes every timestep.
#                 A list of ints of length 1 or ‘num_stacks’. Default value for generic mode: [32] Recommended value for
#                 interpretable mode: [3]
#             prediction_length: Length of the prediction. Also known as 'horizon'.
#             context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
#                 Should be between 1-10 times the prediction length.
#             backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
#                 A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
#                 forecast lengths). Defaults to 0.0, i.e. no weight.
#             loss: loss to optimize. Defaults to MASE().
#             log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
#                 failures
#             reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
#             logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
#                 Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
#             **kwargs: additional arguments to :py:class:`~BaseModel`.
#         """
#         if logging_metrics is None:
#             logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
#         if loss is None:
#             loss = MASE()
#         self.save_hyperparameters()
#         super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

#         # setup stacks
#         self.net_blocks = nn.ModuleList()
#         for stack_id, stack_type in enumerate(stack_types):
#             for _ in range(num_blocks[stack_id]):
#                 if stack_type == "generic":
#                     net_block = NBEATSGenericBlock(
#                         units=self.hparams.widths[stack_id],
#                         thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
#                         num_block_layers=self.hparams.num_block_layers[stack_id],
#                         backcast_length=context_length,
#                         forecast_length=prediction_length,
#                         dropout=self.hparams.dropout,
#                     )
#                 elif stack_type == "seasonality":
#                     net_block = NBEATSSeasonalBlock(
#                         units=self.hparams.widths[stack_id],
#                         num_block_layers=self.hparams.num_block_layers[stack_id],
#                         backcast_length=context_length,
#                         forecast_length=prediction_length,
#                         min_period=self.hparams.expansion_coefficient_lengths[stack_id],
#                         dropout=self.hparams.dropout,
#                     )
#                 elif stack_type == "trend":
#                     net_block = NBEATSTrendBlock(
#                         units=self.hparams.widths[stack_id],
#                         thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
#                         num_block_layers=self.hparams.num_block_layers[stack_id],
#                         backcast_length=context_length,
#                         forecast_length=prediction_length,
#                         dropout=self.hparams.dropout,
#                     )
#                 else:
#                     raise ValueError(f"Unknown stack type {stack_type}")

#                 self.net_blocks.append(net_block)

#     def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         Pass forward of network.
#         Args:
#             x (Dict[str, torch.Tensor]): input from dataloader generated from
#                 :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.
#         Returns:
#             Dict[str, torch.Tensor]: output of model
#         """
#         target = x["encoder_cont"][..., 0]

#         timesteps = self.hparams.context_length + self.hparams.prediction_length
#         generic_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
#         trend_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
#         seasonal_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
#         forecast = torch.zeros(
#             (target.size(0), self.hparams.prediction_length), dtype=torch.float32, device=self.device
#         )

#         backcast = target  # initialize backcast
#         for i, block in enumerate(self.net_blocks):
#             # evaluate block
#             backcast_block, forecast_block = block(backcast)

#             # add for interpretation
#             full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
#             if isinstance(block, NBEATSTrendBlock):
#                 trend_forecast.append(full)
#             elif isinstance(block, NBEATSSeasonalBlock):
#                 seasonal_forecast.append(full)
#             else:
#                 generic_forecast.append(full)

#             # update backcast and forecast
#             backcast = (
#                 backcast - backcast_block
#             )  # do not use backcast -= backcast_block as this signifies an inline operation
#             forecast = forecast + forecast_block

#         return self.to_network_output(
#             prediction=self.transform_output(forecast, target_scale=x["target_scale"]),
#             backcast=self.transform_output(prediction=target - backcast, target_scale=x["target_scale"]),
#             trend=self.transform_output(torch.stack(trend_forecast, dim=0).sum(0), target_scale=x["target_scale"]),
#             seasonality=self.transform_output(
#                 torch.stack(seasonal_forecast, dim=0).sum(0), target_scale=x["target_scale"]
#             ),
#             generic=self.transform_output(torch.stack(generic_forecast, dim=0).sum(0), target_scale=x["target_scale"]),
#         )
