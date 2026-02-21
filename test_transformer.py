import torch
from model.transformer import Transformer

batch = 2
src_len = 6
tgt_len = 5
vocab_size = 1000

src = torch.randint(1, vocab_size, (batch, src_len)).cuda()
tgt = torch.randint(1, vocab_size, (batch, tgt_len)).cuda()

model = Transformer(vocab_size, vocab_size).cuda()

out = model(src, tgt)

print("FINAL:", out.shape)
