import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    max_position_embeddings: int = 256
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

# Синусоидальное позиционное кодирование. 
# Для packed batching позиции считаются отдельно 
# внутри каждой подпоследовательности.
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()

        pe = torch.zeros(max_position_embeddings, hidden_size)

        position = torch.arange(
            0,
            max_position_embeddings,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-math.log(10000.0) / hidden_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if hidden_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)
	
    #  Возвращает позиции [B, T], где внутри каждого объекта, позиции начинаются с 0.
    def make_positions(self, packed_mask: torch.Tensor) -> torch.Tensor:
        device = packed_mask.device
        batch_size, seq_len = packed_mask.shape

        positions = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.long,
            device=device,
        )

        for b in range(batch_size):
            ids = packed_mask[b].unique()

            for sid in ids:
                sid_value = int(sid.item())

                if sid_value == 0:
                    continue

                idx = torch.nonzero(
                    packed_mask[b] == sid,
                    as_tuple=False,
                ).flatten()

                positions[b, idx] = torch.arange(
                    len(idx),
                    dtype=torch.long,
                    device=device,
                )

        return positions

    def forward(self, x: torch.Tensor, packed_mask: torch.Tensor) -> torch.Tensor:
        positions = self.make_positions(packed_mask)

        if positions.max().item() >= self.pe.shape[0]:
            raise ValueError(
                "Длина подпоследовательности превышает max_position_embeddings."
            )

        pos_emb = self.pe[positions]
        return x + pos_emb


class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size должен делиться на num_attention_heads.")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
     
    # M[i,j] = (s_i == s_j) and (j <= i) and (s_i != 0)
    def build_block_causal_mask(self, packed_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = packed_mask.shape
        device = packed_mask.device

        same_sequence = packed_mask.unsqueeze(2) == packed_mask.unsqueeze(1)

        causal = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        )

        non_pad = packed_mask.unsqueeze(2) != 0

        mask = same_sequence & causal.unsqueeze(0) & non_pad

        return mask

    def forward(self, x: torch.Tensor, packed_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        mask = self.build_block_causal_mask(packed_mask)
        mask = mask.unsqueeze(1)

        scores = scores.masked_fill(~mask, -1e4)

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = attn_probs.masked_fill(~mask, 0.0)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, hidden_size)

        output = self.out_proj(context)
        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# Post-norm Transformer block:
# z1 = LayerNorm(x + Attention(x))
# z2 = LayerNorm(z1 + FFN(z1))
class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.attention = MultiHeadMaskedSelfAttention(config)
        self.ffn = FeedForward(config)

        self.ln_1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.ln_2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(self, x: torch.Tensor, packed_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x, packed_mask)
        z1 = self.ln_1(x + attn_out)

        ffn_out = self.ffn(z1)
        z2 = self.ln_2(z1 + ffn_out)

        return z2


class GPTLikeModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.position_encoding = SinusoidalPositionEncoding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        self.final_ln = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        x = self.position_encoding(x, packed_mask)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, packed_mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        return logits


# Логиты позиции i предсказывают токен i+1. Loss считается только там, где:
# M_loss_i = (s_i == s_{i+1}) and (s_i != 0)
def compute_packed_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    packed_mask: torch.Tensor,
):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    current_mask = packed_mask[:, :-1]
    next_mask = packed_mask[:, 1:]

    loss_mask = (current_mask == next_mask) & (current_mask != 0)
    loss_mask = loss_mask.float()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    loss_per_token = loss_per_token.view_as(shift_labels)

    denominator = loss_mask.sum().clamp_min(1.0)
    loss = (loss_per_token * loss_mask).sum() / denominator

    return loss, loss_mask