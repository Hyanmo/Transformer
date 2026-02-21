from model.attention import MultiHeadAttention
import torch

batch = 2
seq_len = 5
d_model = 512

x = torch.randn(batch, seq_len, d_model).cuda()

mha = MultiHeadAttention().cuda()

out, attn = mha(x, x, x)

print(out.shape)
print(attn.shape)
