from model.feed_forward import PositionwiseFeedForward
import torch

batch = 2
seq_len = 5
d_model = 512

x = torch.randn(batch, seq_len, d_model).cuda()

ffn = PositionwiseFeedForward().cuda()

out = ffn(x)

print(out.shape)

