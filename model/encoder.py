import torch
import torch.nn as nn

from model.encoder_layer import EncoderLayer
from model.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        #1. 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        #2. 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        #3. 堆叠编码层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in  range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        :param x: (batch, sqe_len) token_id
        """

        #1. 词嵌入
        x = self.embedding(x) * (self.d_model ** 0.5)

        #2. 加位置编码
        x = self.pos_encoding(x)

        x = self.dropout(x)

        #3. 通过多层编码
        for layer in self.layers:
            x = layer(x, mask)

        return x