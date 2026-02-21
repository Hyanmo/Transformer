from model.encoder import Encoder
import torch

batch = 2
seq_len = 5
vocab_size = 10000

x = torch.randint(0, vocab_size, (batch, seq_len)).cuda()

encoder = Encoder(vocab_size).cuda()

out = encoder(x)

print(out.shape)
