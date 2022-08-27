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
