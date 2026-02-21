from utils.mask import generate_subsequent_mask
from utils.mask import generate_padding_mask
import torch

mask = generate_subsequent_mask(5)

x = torch.tensor([[5, 23, 0, 0]])
padding_mask = generate_padding_mask(x)

print(mask.shape)
print(mask[0,0])

print(padding_mask.shape)
print(padding_mask)