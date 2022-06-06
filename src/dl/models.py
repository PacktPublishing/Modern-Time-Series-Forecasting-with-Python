"""Base Model"""
from dataclasses import dataclass, field
import logging
from abc import ABCMeta, abstractmethod
from random import choices
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
import enum


@dataclass
class SingleStepRNNConfig:
    """Configuration for RNN"""

    rnn_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    learning_rate: float = field(default=1e-3)
    optimizer_params: Dict = field(default_factory=dict)
    lr_scheduler: Optional[str] = field(default=None)
    lr_scheduler_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.rnn_type = self.rnn_type.upper()
        assert self.rnn_type in [
            "LSTM",
            "GRU",
            "RNN",
        ], f"{self.rnn_type} is not a valid RNN type"


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _build_network(self):
        pass

    def _setup_loss(self):
        self.loss = nn.MSELoss()

    def _setup_metrics(self):
        self.metrics = [torchmetrics.functional.mean_absolute_error]
        self.metrics_name = ["MAE"]

    def calculate_loss(self, y_hat, y, tag):
        computed_loss = self.loss(y_hat, y)
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid") or (tag == "test"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )
        return computed_loss

    def calculate_metrics(self, y, y_hat, tag):
        metrics = []
        for metric, metric_str in zip(self.metrics, self.metrics_name):
            avg_metric = metric(y_hat, y)
            self.log(
                f"{tag}_{metric_str}",
                avg_metric,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True,
            )
        return metrics

    @abstractmethod
    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        pass

    @abstractmethod
    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        pass

    def training_step(self, batch, batch_idx):
        y_hat, y = self.forward(batch)
        loss = self.calculate_loss(y_hat, y, tag="train")
        _ = self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, y = self.forward(batch)
            _ = self.calculate_loss(y_hat, y, tag="valid")
            _ = self.calculate_metrics(y, y_hat, tag="valid")
        return y_hat, y

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, y = self.forward(batch)
            _ = self.calculate_loss(y_hat, y, tag="test")
            _ = self.calculate_metrics(y, y_hat, tag="test")
        return y_hat, y

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            pred = self.predict(batch)
        return pred

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            **self.hparams.optimizer_params,
        )
        if self.hparams.lr_scheduler is not None:
            try:
                self._lr_scheduler = getattr(
                    torch.optim.lr_scheduler, self.hparams.lr_scheduler
                )
            except AttributeError as e:
                print(
                    f"{self.hparams.lr_scheduler} is not a valid learning rate sheduler defined in the torch.optim.lr_scheduler module"
                )
                raise e
            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                }
            else:
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                    "monitor": self.hparams.lr_scheduler_monitor_metric,
                }
        else:
            return opt


class SingleStepRNNModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__(config)

    def _build_network(self):
        if self.hparams.rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.hparams.input_size,
                hidden_size=self.hparams.hidden_size,
                num_layers=self.hparams.num_layers,
                batch_first=True,
            )
        elif self.hparams.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.hparams.input_size,
                hidden_size=self.hparams.hidden_size,
                num_layers=self.hparams.num_layers,
                batch_first=True,
            )
        elif self.hparams.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.hparams.input_size,
                hidden_size=self.hparams.hidden_size,
                num_layers=self.hparams.num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("Invalid RNN type")
        self.fc = nn.Linear(self.hparams.hidden_size, 1)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        assert y.size(1) == 1, "y should have only a single timestep"
        # x --> (batch_size, seq_len, input_size), y--> (batch_size, seq_len, 1)
        x, _ = self.rnn(x)  # --> (batch_size, seq_len, hidden_size)
        x = self.fc(x)  # --> (batch_size, seq_len, 1)
        # shifting the input by one and concatenating with the output to get the target
        y = torch.cat([x[:, 1:, :], y], dim=1)  # --> (batch_size, seq_len, 1)
        return x, y

    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        with torch.no_grad():
            y_hat, _ = self.forward(batch)
        if ret_model_output:
            return y_hat
        else:
            return y_hat[:, -1, :]


@dataclass
class RNNConfig:
    """Configuration for RNN"""
    input_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool


@dataclass
class Seq2SeqConfig:
    """Configuration for RNN"""

    encoder_type: str
    decoder_type: str
    encoder_params: Dict[str, Any]
    decoder_params: Dict[str, Any]
    decoder_use_all_hidden: bool = True
    teacher_forcing_ratio: float = 0.0
    learning_rate: float = field(default=1e-3)
    optimizer_params: Dict = field(default_factory=dict)
    lr_scheduler: Optional[str] = field(default=None)
    lr_scheduler_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.encoder_type = self.encoder_type.upper()
        self.decoder_type = self.decoder_type.upper()
        if isinstance(self.encoder_params, RNNConfig):
            self.encoder_params = self.encoder_params.__dict__
        if isinstance(self.decoder_params, RNNConfig):
            self.decoder_params = self.decoder_params.__dict__
        assert self.encoder_type in [
            "LSTM",
            "GRU",
            "RNN",
        ], f"{self.encoder_type} is not a valid RNN type"
        assert self.decoder_type in [
            "LSTM",
            "GRU",
            "RNN",
            "FC",
        ], f"{self.decoder_type} is not a valid RNN type"
        if self.decoder_type=="FC":
            assert "window_size" in self.decoder_params, "window_size is required for FC decoder"
            assert "horizon" in self.decoder_params, "horizon is required for FC decoder"

class Seq2SeqModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__(config)

    def _build_network(self):
        bi_directional_multiplier = 2 if self.hparams.encoder_params["bidirectional"] else 1
        if self.hparams.encoder_type == "RNN":
            self.encoder = nn.RNN(
                **self.hparams.encoder_params,
                batch_first=True,
            )
        elif self.hparams.encoder_type == "LSTM":
            self.encoder = nn.LSTM(
                **self.hparams.encoder_params,
                batch_first=True,
            )
        elif self.hparams.encoder_type == "GRU":
            self.encoder = nn.GRU(
                **self.hparams.encoder_params,
                batch_first=True,
            )
        else:
            raise ValueError("Invalid RNN type")
        if self.hparams.decoder_type == "RNN":
            self.decoder = nn.RNN(
                **self.hparams.decoder_params,
                batch_first=True,
            )
        elif self.hparams.decoder_type == "LSTM":
            self.decoder = nn.LSTM(
                **self.hparams.decoder_params,
                batch_first=True,
            )
        elif self.hparams.decoder_type == "GRU":
            self.decoder = nn.GRU(
                **self.hparams.decoder_params,
                batch_first=True,
            )
        elif self.hparams.decoder_type == "FC":
            if self.hparams.decoder_use_all_hidden:
                self.decoder = nn.Linear(
                    self.hparams.encoder_params.hidden_size * bi_directional_multiplier * self.hparams.decoder_params.window_size, self.hparams.decoder_params.horizon
                )
            else:
                self.decoder = nn.Linear(
                    self.hparams.encoder_params.hidden_size*bi_directional_multiplier, self.hparams.decoder_params.horizon
                )
        if self.hparams.decoder_type!="FC":
            self.fc = nn.Linear(self.hparams.decoder_params.hidden_size*bi_directional_multiplier, 1)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        o, h = self.encoder(x) # --> (batch_size, seq_len, hidden_size) , (num_layers, batch_size, hidden_size) (hidden size*2 and num_layers*2 if bidirectional)
        if self.hparams.decoder_type == "FC":
            if self.hparams.decoder_use_all_hidden:
                y_hat = self.decoder(o.reshape(o.size(0), -1)).unsqueeze(-1)
            else:
                y_hat = self.decoder(o[:,-1,:]).unsqueeze(-1)
                
        else:
            # Loop to generate target
            y_hat = torch.zeros_like(y, device=y.device)
            dec_input = x[:, -1:, :]
            for i in range(y.size(1)):
                out, h = self.decoder(dec_input, h)
                out = self.fc(out)
                y_hat[:, i, :] = out.squeeze(1)
                #decide if we are going to use teacher forcing or not
                teacher_force = random.random() < self.hparams.teacher_forcing_ratio
                if teacher_force:
                    dec_input = y[:, i, :].unsqueeze(1)
                else:
                    dec_input = out
        return y_hat, y


        # # x --> (batch_size, seq_len, input_size), y--> (batch_size, seq_len, 1)
        # x, _ = self.rnn(x)  # --> (batch_size, seq_len, hidden_size)
        # x = self.fc(x)  # --> (batch_size, seq_len, 1)
        # # shifting the input by one and concatenating with the output to get the target
        # y = torch.cat([x[:, 1:, :], y], dim=1)  # --> (batch_size, seq_len, 1)
        # return x, y

    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        with torch.no_grad():
            y_hat, _ = self.forward(batch)
        return y_hat