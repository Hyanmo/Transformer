import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from utils.mask import generate_padding_mask, generate_subsequent_mask\

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            n_heads,
            num_layers,
            d_ff,
            max_len,
            dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            n_heads,
            num_layers,
            d_ff,
            max_len,
            dropout
        )

    def forward(self, src, tgt):
        """
        :param src: (batch, src_seq_len)
        :param tgt: (batch, tgt_seq_len)
        """

        #1.Encoder mask
        src_mask = generate_padding_mask(src, self.pad_idx).to(src.device)

        #2.Decoder mask
        tgt_padding_mask = generate_padding_mask(tgt,self.pad_idx).to(tgt.device)
        tgt_seq_len = tgt.size(1)
        tgt_sub_mask = generate_subsequent_mask(tgt_seq_len).to(tgt.device)

        tgt_mask = tgt_padding_mask * tgt_sub_mask

        #3.Encoder
        enc_output = self.encoder(src, src_mask)

        #4.Decoder
        out = self.decoder(
            tgt,
            enc_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )

        return out