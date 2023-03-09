import math
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module, metaclass=ABCMeta):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    @abstractmethod
    def _get_scores(
        self,
        q: torch.Tensor,  # [batch_size, decoder_dim]
        v: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ) -> torch.Tensor:  # [batch_size, seq_length]
        pass

    def forward(
        self,
        query: torch.Tensor,  # [batch_size, decoder_dim]
        values: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ):
        if query.ndim == 2:
            query = query.unsqueeze(1)
        elif query.ndim == 3:
            assert (
                query.shape[1] == 1
            ), "If `query` is a 3-D tensor should have shape [batch_size, 1, decoder_dim]"
        seq_length = values.shape[1]
        scores = self._get_scores(query, values)  # [batch_size, seq_length]
        assert scores.size(1) == seq_length
        weights = torch.nn.functional.softmax(scores, dim=-1)
        return (values * weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, encoder_dim]
        # return torch.bmm(values.transpose(1,2), weights.unsqueeze(-1)).squeeze()  # [batch_size, encoder_dim]


class DotProductAttention(Attention):
    def __init__(self, hidden_dim: int, scaled: bool = True):
        super().__init__(hidden_dim, hidden_dim)
        if scaled:
            self.scaling = math.sqrt(hidden_dim)
        else:
            self.scaling = 1.0

    def _get_scores(
        self,
        q: torch.Tensor,  # [batch_size, decoder_dim]
        v: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ) -> torch.Tensor:  # [batch_size, seq_length]
        scores = q @ v.transpose(1, 2)  # [batch_size, seq_length]
        return scores.squeeze(1) / self.scaling  # [batch_size, seq_length]


class GeneralAttention(Attention):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__(encoder_dim, decoder_dim)
        self.W = torch.nn.Parameter(
            torch.FloatTensor(self.decoder_dim, self.encoder_dim).uniform_(-0.1, 0.1)
        )

    def _get_scores(
        self,
        q: torch.Tensor,  # [batch_size, decoder_dim]
        v: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ) -> torch.Tensor:  # [batch_size, seq_length]
        scores = (q @ self.W) @ v.transpose(1, 2)  # [batch_size, seq_length]
        return scores.squeeze(1)  # [batch_size, seq_length]


class AdditiveAttention(Attention):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__(encoder_dim, decoder_dim)
        self.v = torch.nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1)
        )
        self.W_q = torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_v = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    def _get_scores(
        self,
        q: torch.Tensor,  # [batch_size, decoder_dim]
        v: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ) -> torch.Tensor:  # [batch_size, seq_length]
        q = q.repeat(1, v.size(1), 1)  # [batch_size, seq_length, decoder_dim]
        scores = self.W_q(q) + self.W_v(v)  # [batch_size, seq_length, decoder_dim]
        return torch.tanh(scores) @ self.v  # [batch_size, seq_length]


# Also referred to as Concat attention by Luong et al.
class ConcatAttention(Attention):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim, hidden_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim).uniform_(-0.1, 0.1))
        self.W = torch.nn.Linear(2 * hidden_dim, hidden_dim)

    def _get_scores(
        self,
        q: torch.Tensor,  # [batch_size, decoder_dim]
        v: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ) -> torch.Tensor:  # [batch_size, seq_length]
        q = q.repeat(1, v.size(1), 1)  # [batch_size, seq_length, decoder_dim]
        scores = self.W(
            torch.cat([q, v], dim=-1)
        )  # [batch_size, seq_length, decoder_dim]
        return torch.tanh(scores) @ self.v  # [batch_size, seq_length]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.scaling = math.sqrt(input_dim)
        self.W_q = nn.Linear(input_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attn_dim, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # DIM: q, k, v --> [batch_size, seq_length, hidden_dim]
        q = self.W_q(q)  # [batch_size, seq_length, attn_dim]
        k = self.W_k(k)  # [batch_size, seq_length, attn_dim]
        v = self.W_v(v)  # [batch_size, seq_length, attn_dim]
        attn_energies = q.bmm(k.transpose(1, 2))  # [batch_size, seq_length, seq_length]
        attn_energies = (
            attn_energies / self.scaling
        )  # [batch_size, seq_length, seq_length]
        attn_weights = F.softmax(
            attn_energies, dim=-1
        )  # [batch_size, seq_length, seq_length]
        return torch.bmm(
            attn_weights, v
        )  # v(k,q) = v * p(a(k,q)) and sum across attention dim to get [batch_size, seq_length, attn_dim]


# input_dim = 10
# attn_dim = 10
# # q = torch.randn(100, 5, input_dim)
# q = torch.randn(100, 7, input_dim)
# k = torch.randn(100, 7, input_dim)
# v = torch.randn(100, 7, input_dim)

# for mode in ["dot", "scaled_dot", "general", "concat", "additive"]:
#     attn = Attention(mode, input_dim)
#     att_v = attn(q, k, v)
#     print(att_v.shape)


# hidden_dim = 15
# seq_len = 20
# bz = 1

# q = torch.randn(bz, hidden_dim)
# v = torch.randn(bz, seq_len, hidden_dim)

# attn = ConcatAttention(hidden_dim)
# att = attn(q, v)
# print(att.shape)
# attn = DotProductAttention(hidden_dim, scaled=False)
# att = attn(q, v)
# print(att.shape)


# encoder_dim = 10
# decoder_dim = 15
# seq_len = 20
# bz = 1
# q = torch.randn(bz, decoder_dim)
# v = torch.randn(bz, seq_len, encoder_dim)
# attn = AdditiveAttention(encoder_dim, decoder_dim)
# att = attn(q, v)
# print(att.shape)
# attn = GeneralAttention(encoder_dim, decoder_dim)
# att = attn(q, v)
# print(att.shape)
