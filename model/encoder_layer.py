import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #1. 多头注意力
        attn_output, _ = self.mha(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        #2. 前馈神经网络
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x