"""Base Model"""
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict

import torch
import torch.nn as nn
from pytorch_forecasting.models import BaseModel

from src.dl.autoformer import AutoFormer
from src.dl.informer import Informer


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
        self.save_hyperparameters(
            {"network_callable": network_callable, "model_params": model_params}
        )
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


class AutoformerModel(BaseModel):
    def __init__(
        self,
        seq_len,
        label_len,
        pred_len,
        moving_avg,
        enc_in,
        dec_in,
        d_model,
        cardinality,
        dropout,
        factor,
        n_heads,
        d_ff,
        activation,
        e_layers,
        c_out,
        d_layers,
        output_attention=False,
        **kwargs
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = AutoFormer(
            seq_len,
            label_len,
            pred_len,
            moving_avg,
            enc_in,
            dec_in,
            d_model,
            cardinality,
            dropout,
            factor,
            n_heads,
            d_ff,
            activation,
            e_layers,
            c_out,
            d_layers,
            output_attention,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s_begin = 0
        s_end = s_begin + self.hparams.seq_len
        r_begin = s_end - self.hparams.label_len
        r_end = r_begin + self.hparams.label_len + self.hparams.pred_len

        Y = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)[
            :, :, -1
        ].unsqueeze(-1)
        X = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)

        batch_x = Y[:, s_begin:s_end, :]
        batch_y = Y[:, r_begin:r_end, :]
        batch_x_mark = X[:, s_begin:s_end, :]
        batch_y_mark = X[:, r_begin:r_end, :]

        dec_inp = torch.zeros_like(batch_y[:, -self.hparams.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.hparams.label_len, :], dec_inp], dim=1)

        prediction = self.network(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.hparams.output_attention:
            prediction = prediction[0]
        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        # We need to return a dictionary that at least contains the prediction
        # The parameter can be directly forwarded from the input.
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        assert (
            len(dataset.time_varying_known_reals) == 0
        ), "Informer Model does not allow time_varying_reals"
        assert (
            len(dataset.static_reals) == 0
        ), "Informer Model does not allow static_reals"
        if "seq_len" not in kwargs.keys():
            kwargs["seq_len"] = dataset.max_encoder_length
        if "pred_len" not in kwargs.keys():
            kwargs["pred_len"] = dataset.max_prediction_length
        if "enc_in" not in kwargs.keys():
            kwargs["enc_in"] = len(dataset.target_names)
        if "dec_in" not in kwargs.keys():
            kwargs["dec_in"] = len(dataset.target_names)
        if "c_out" not in kwargs.keys():
            kwargs["c_out"] = len(dataset.target_names)
        if "cardinality" not in kwargs.keys():
            kwargs["cardinality"] = [
                len(dataset.categorical_encoders[c].classes_)
                for c in dataset.categoricals
            ]
        return super().from_dataset(dataset, **kwargs)


class InformerModel(BaseModel):
    def __init__(
        self,
        seq_len,
        label_len,
        pred_len,
        distil,
        enc_in,
        dec_in,
        d_model,
        cardinality,
        dropout,
        factor,
        n_heads,
        d_ff,
        activation,
        e_layers,
        c_out,
        d_layers,
        output_attention=False,
        **kwargs
    ):
        # if "output_attention" not in model_params:
        #     model_params["output_attention"] = False
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = Informer(
            seq_len,
            label_len,
            pred_len,
            distil,
            enc_in,
            dec_in,
            d_model,
            cardinality,
            dropout,
            factor,
            n_heads,
            d_ff,
            activation,
            e_layers,
            c_out,
            d_layers,
            output_attention=output_attention,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s_begin = 0
        s_end = s_begin + self.hparams.seq_len
        r_begin = s_end - self.hparams.label_len
        r_end = r_begin + self.hparams.label_len + self.hparams.pred_len

        Y = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)[
            :, :, -1
        ].unsqueeze(-1)
        X = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)

        batch_x = Y[:, s_begin:s_end, :]
        batch_y = Y[:, r_begin:r_end, :]
        batch_x_mark = X[:, s_begin:s_end, :]
        batch_y_mark = X[:, r_begin:r_end, :]

        dec_inp = torch.zeros_like(batch_y[:, -self.hparams.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.hparams.label_len, :], dec_inp], dim=1)

        prediction = self.network(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.hparams.output_attention:
            prediction = prediction[0]
        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        # We need to return a dictionary that at least contains the prediction
        # The parameter can be directly forwarded from the input.
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        assert (
            len(dataset.time_varying_known_reals) == 0
        ), "Informer Model does not allow time_varying_reals"
        assert (
            len(dataset.static_reals) == 0
        ), "Informer Model does not allow static_reals"
        if "seq_len" not in kwargs.keys():
            kwargs["seq_len"] = dataset.max_encoder_length
        if "pred_len" not in kwargs.keys():
            kwargs["pred_len"] = dataset.max_prediction_length
        if "enc_in" not in kwargs.keys():
            kwargs["enc_in"] = len(dataset.target_names)
        if "dec_in" not in kwargs.keys():
            kwargs["dec_in"] = len(dataset.target_names)
        if "c_out" not in kwargs.keys():
            kwargs["c_out"] = len(dataset.target_names)
        if "cardinality" not in kwargs.keys():
            kwargs["cardinality"] = [
                len(dataset.categorical_encoders[c].classes_)
                for c in dataset.categoricals
            ]
        return super().from_dataset(dataset, **kwargs)
