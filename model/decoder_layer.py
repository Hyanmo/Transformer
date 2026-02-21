import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_outpout, tgt_mask=None, src_mask=None):
        # 1.掩码注意力
        _x, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(_x)
        x = self.norm1(x)

        # 2.交叉注意力
        _x, _ = self.cross_attn(x, enc_outpout, enc_outpout, src_mask)
        x = x + self.dropout(_x)
        x = self.norm2(x)

        # 3.前馈层
        _x = self.ffn(x)
        x = x + self.dropout(_x)
        x = self.norm3(x)

        return x