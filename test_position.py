from model.positional_encoding import PositionalEncoding
import torch

batch = 2
seq_len = 5
d_model = 512

x = torch.zeros(batch, seq_len, d_model).cuda()

pe = PositionalEncoding(d_model).cuda()

out = pe(x)

print(out.shape)
print(out[0, 0, :10])
