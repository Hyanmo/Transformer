import torch
import torch.nn as nn

from model.decoder_layer import DecoderLayer
from model.positional_encoding import PositionalEncoding

class Decoder(nn.Module):
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

        #1.词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        #2.位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        #3.堆叠解码层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        #4.输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """
        :param tgt: (batch, seq_len)
        :return: (batch, src_seq_len, d_model)
        """

        #1.词嵌入
        x = self.embedding(tgt) * (self.d_model ** 0.5)

        #2.加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        #3.通过多层解码层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        #输出
        out = self.fc_out(x)

        return out