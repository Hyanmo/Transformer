import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer

device = torch.device("cuda")

batch = 32
src_len = 10
tgt_len = 10
vocab_size = 1000

model = Transformer(vocab_size, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for step in range(1000):

    # 假数据
    src = torch.randint(1, vocab_size, (batch, src_len)).to(device)
    tgt = torch.randint(1, vocab_size, (batch, tgt_len)).to(device)

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # forward
    output = model(src, tgt_input)

    # reshape
    output = output.reshape(-1, vocab_size)
    tgt_output = tgt_output.reshape(-1)

    loss = criterion(output, tgt_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("step:", step, "loss:", loss.item())