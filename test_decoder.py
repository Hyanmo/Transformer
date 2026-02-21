import torch
from model.decoder import Decoder
from utils.mask import generate_padding_mask, generate_subsequent_mask

batch = 2
seq_len = 5
vocab_size = 1000
d_model = 512

tgt = torch.randint(1, vocab_size, (batch, seq_len)).cuda()
enc_out = torch.randn(batch, seq_len, d_model).cuda()

# mask
tgt_padding_mask = generate_padding_mask(tgt).cuda()
tgt_sub_mask = generate_subsequent_mask(seq_len).cuda()
combined_mask = tgt_padding_mask * tgt_sub_mask

decoder = Decoder(vocab_size).cuda()

out = decoder(
    tgt,
    enc_out,
    tgt_mask=combined_mask,
    src_mask=tgt_padding_mask
)

print(out.shape)