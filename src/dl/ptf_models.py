"""Base Model"""
from pytorch_forecasting.models import BaseModel
from dataclasses import dataclass, field
import logging
from random import choices
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.dl.attention import (
    GeneralAttention,
    AdditiveAttention,
    DotProductAttention,
    ConcatAttention,
)

# from attention import (
#     GeneralAttention,
#     AdditiveAttention,
#     DotProductAttention,
#     ConcatAttention,
# )

ATTENTION_TYPES = {
    "dot": DotProductAttention,
    "scaled_dot": DotProductAttention,
    "general": GeneralAttention,
    "additive": AdditiveAttention,
    "concat": ConcatAttention,
}


@dataclass
class SingleStepRNNConfig:
    """Configuration for RNN"""

    rnn_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool

    def __post_init__(self):
        self.rnn_type = self.rnn_type.upper()
        assert self.rnn_type in [
            "LSTM",
            "GRU",
            "RNN",
        ], f"{self.rnn_type} is not a valid RNN type"


class SingleStepRNN(nn.Module):
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
        if self.decoder_type == "FC":
            assert (
                "window_size" in self.decoder_params
            ), "window_size is required for FC decoder"
            assert (
                "horizon" in self.decoder_params
            ), "horizon is required for FC decoder"


class Seq2SeqModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__(config)

    def _build_network(self):
        enc_bi_directional_multiplier = (
            2 if self.hparams.encoder_params["bidirectional"] else 1
        )
        if self.hparams.decoder_type != "FC":
            dec_bi_directional_multiplier = (
                2 if self.hparams.decoder_params["bidirectional"] else 1
            )
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
                    self.hparams.encoder_params.hidden_size
                    * enc_bi_directional_multiplier
                    * self.hparams.decoder_params.window_size,
                    self.hparams.decoder_params.horizon,
                )
            else:
                self.decoder = nn.Linear(
                    self.hparams.encoder_params.hidden_size
                    * enc_bi_directional_multiplier,
                    self.hparams.decoder_params.horizon,
                )
        if self.hparams.decoder_type != "FC":
            self.fc = nn.Linear(
                self.hparams.decoder_params.hidden_size * dec_bi_directional_multiplier,
                1,
            )

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        o, h = self.encoder(
            x
        )  # --> (batch_size, seq_len, hidden_size) , (num_layers, batch_size, hidden_size) (hidden size*2 and num_layers*2 if bidirectional)
        if self.hparams.decoder_type == "FC":
            if self.hparams.decoder_use_all_hidden:
                y_hat = self.decoder(o.reshape(o.size(0), -1)).unsqueeze(-1)
            else:
                y_hat = self.decoder(o[:, -1, :]).unsqueeze(-1)

        else:
            # Loop to generate target
            y_hat = torch.zeros_like(y, device=y.device)
            dec_input = x[:, -1:, :]
            for i in range(y.size(1)):
                out, h = self.decoder(dec_input, h)
                out = self.fc(out)
                y_hat[:, i, :] = out.squeeze(1)
                # decide if we are going to use teacher forcing or not
                teacher_force = random.random() < self.hparams.teacher_forcing_ratio
                if teacher_force:
                    dec_input = y[:, i, :].unsqueeze(1)
                else:
                    dec_input = out
        return y_hat, y

    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        with torch.no_grad():
            y_hat, _ = self.forward(batch)
        return y_hat


@dataclass
class Seq2SeqwAttnConfig:
    """Configuration for RNN"""

    encoder_type: str
    decoder_type: str
    encoder_params: Dict[str, Any]
    decoder_params: Dict[str, Any]
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
        ], f"{self.decoder_type} is not a valid RNN type"
        effective_encoder_hidden_size = self.encoder_params["hidden_size"] * (
            2 if self.encoder_params["bidirectional"] else 1
        )
        assert (
            self.decoder_params["input_size"]
            == self.encoder_params["input_size"] + effective_encoder_hidden_size
        ), (
            f"Encoder input size {self.encoder_params['input_size']} + "
            f"Encoder hidden size (*2 if bi directional) {effective_encoder_hidden_size} != "
            f"Decoder input size {self.decoder_params['input_size']}"
        )


class Seq2SeqwAttnModel(BaseModel):
    def __init__(
        self,
        attention_type: str,
        config: DictConfig,
        **kwargs,
    ):
        assert (
            attention_type.lower() in ATTENTION_TYPES.keys()
        ), f"{attention_type} is not a valid attention type"
        self.attention_type = attention_type.lower()
        super().__init__(config)

    def _build_network(self):
        enc_bi_directional_multiplier = (
            2 if self.hparams.encoder_params["bidirectional"] else 1
        )
        dec_bi_directional_multiplier = (
            2 if self.hparams.decoder_params["bidirectional"] else 1
        )
        self.attention = ATTENTION_TYPES[self.attention_type]
        attn_params = dict()
        if issubclass(self.attention, ConcatAttention) or issubclass(
            self.attention, DotProductAttention
        ):
            assert (
                self.hparams.encoder_params["hidden_size"]
                * enc_bi_directional_multiplier
                == self.hparams.decoder_params["hidden_size"]
                * dec_bi_directional_multiplier
            ), "Hidden size*D, where D=2 for bi directional and 1 otherwise, of encoder and decoder must be equal"
            attn_params = {
                "hidden_dim": self.hparams.encoder_params.hidden_size
                * enc_bi_directional_multiplier
            }
            if issubclass(self.attention, DotProductAttention):
                if "scale" in self.attention_type:
                    attn_params["scaled"] = True
                else:
                    attn_params["scaled"] = False
        else:
            attn_params = {
                "encoder_dim": self.hparams.encoder_params.hidden_size
                * enc_bi_directional_multiplier,
                "decoder_dim": self.hparams.decoder_params.hidden_size
                * dec_bi_directional_multiplier,
            }
        if "scale" in self.attention_type:
            if issubclass(self.attention, DotProductAttention):
                attn_params["scaled"] = True
        self.attention = self.attention(**attn_params)

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
        self.fc = nn.Linear(
            self.hparams.decoder_params.hidden_size * dec_bi_directional_multiplier, 1
        )

    def _get_top_layer_hidden_state(self, hidden_state):
        if self.hparams.encoder_type == "LSTM":
            hidden_state, _ = hidden_state
        if self.hparams.encoder_params["bidirectional"]:
            # concatenating the forward and backward hidden states
            return torch.cat((hidden_state[-1, :, :], hidden_state[-2, :, :]), dim=-1)
        else:
            return hidden_state[-1, :, :]

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        o, h = self.encoder(
            x
        )  # --> (batch_size, seq_len, hidden_size) , (num_layers, batch_size, hidden_size) (hidden size*2 and num_layers*2 if bidirectional)
        # Loop to generate target
        y_hat = torch.zeros_like(y, device=y.device)
        dec_input = x[:, -1:, :]
        for i in range(y.size(1)):
            top_h = self._get_top_layer_hidden_state(
                h
            )  # --> (batch_size, hidden_size*2 if bidirectional)
            context = self.attention(
                top_h.unsqueeze(1), o
            )  # --> (batch_size, hidden_size)
            dec_input = torch.cat((dec_input, context.unsqueeze(1)), dim=-1)
            out, h = self.decoder(dec_input, h)
            out = self.fc(out)
            y_hat[:, i, :] = out.squeeze(1)
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.hparams.teacher_forcing_ratio
            if teacher_force:
                dec_input = y[:, i, :].unsqueeze(1)
            else:
                dec_input = out
        return y_hat, y

    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        with torch.no_grad():
            y_hat, _ = self.forward(batch)
        return y_hat


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout=0.0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


@dataclass
class TransformerConfig:
    """Configuration for Transformer"""

    input_size: int
    d_model: int
    n_heads: int
    n_layers: int
    ff_multiplier: int = 4
    activation: str = "relu"  #'gelu'
    multi_step_horizon: int = 1
    dropout: float = 0.0
    learning_rate: float = field(default=1e-3)
    optimizer_params: Dict = field(default_factory=dict)
    lr_scheduler: Optional[str] = field(default=None)
    lr_scheduler_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.activation in [
            "relu",
            "gelu",
        ], "Invalid activation. Should be relu or gelu"


class TransformerModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__(config)

    def _build_network(self):
        self.input_projection = nn.Linear(
            self.hparams.input_size, self.hparams.d_model, bias=False
        )
        self.pos_encoder = PositionalEncoding(self.hparams.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.d_model,
            nhead=self.hparams.n_heads,
            dropout=self.hparams.dropout,
            dim_feedforward=self.hparams.d_model * self.hparams.ff_multiplier,
            activation=self.hparams.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.hparams.n_layers
        )
        # self.decoder = nn.Linear(self.hparams.d_model, self.hparams.multi_step_horizon)
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.d_model, 100),
            nn.ReLU(),
            nn.Linear(100, self.hparams.multi_step_horizon),
        )
        self._src_mask = None

    def _generate_square_subsequent_mask(self, sz, reset_mask=False):
        if self._src_mask is None or reset_mask:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self._src_mask = mask
        return self._src_mask

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        mask = self._generate_square_subsequent_mask(x.shape[1]).to(x.device)
        # Projecting input dimension to d_model
        x_ = self.input_projection(x)
        # Adding positional encoding
        x_ = self.pos_encoder(x_)
        # Encoding the input
        x_ = self.transformer_encoder(x_, mask)
        # Decoding the input
        y_hat = self.decoder(x_)
        # constructing a shifted by one target so that all the outputs from the decoder can be trained
        # also unfolding so that at each position we can train all H horizon forecasts
        y = torch.cat([x[:, 1:, :], y], dim=1).squeeze(-1).unfold(1, y.size(1), 1)
        return y_hat, y

    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        with torch.no_grad():
            y_hat, _ = self.forward(batch)
            # We only need the last position prediction in prediction task
            y_hat = y_hat[:, -1, :].unsqueeze(1)
        return y_hat


# h=5
# x = torch.rand(256, 48, 1)
# y = torch.rand(256, h, 1)
# tr_config = TransformerConfig(
#     input_size=1,
#     d_model=64,
#     n_heads=8,
#     n_layers=6,
#     ff_multiplier=4,
#     activation="relu",
#     multistep_ff=False,
#     multi_step_horizon=h,
#     teacher_forcing_ratio=0.5,
# )
# model = TransformerModel(tr_config)
# y_hat, y_ = model.forward((x, y))
# print(y_hat.shape)
# print(y_.shape)

# encoder_config = RNNConfig(
#     input_size=1,
#     hidden_size=128,
#     num_layers=3,
#     bidirectional=True,
# ).__dict__
# decoder_config = RNNConfig(
#     input_size=1 + 128*2,
#     hidden_size=128,
#     num_layers=3,
#     bidirectional=True,
# ).__dict__
# rnn2rnn_config = Seq2SeqwAttnConfig(
#     encoder_type="LSTM",
#     decoder_type="LSTM",
#     encoder_params=encoder_config,
#     decoder_params=decoder_config,
#     learning_rate=1e-3,
# )

# for at in ATTENTION_TYPES.keys():
#     print(at)
#     model = Seq2SeqwAttnModel(at, rnn2rnn_config)
#     yhat, y = model((x, y))
#     print(yhat.shape)
