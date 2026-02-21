import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer

"""
反转任务
输入： 1 2 3 4
输出： 4 3 2 1
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 20
seq_len = 4
batch_size = 64
num_layers = 1
n_heads = 2
d_model = 128
dropout = 0.0

model = Transformer(vocab_size, vocab_size, d_model=d_model,n_heads=n_heads,num_layers=num_layers,dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1.5e-3)

for step in range(2000):

    src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)

    # ===== 反转 =====
    tgt = torch.flip(src, dims=[1])

    # teacher forcing
    tgt_input  = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    output = model(src, tgt_input)

    loss = criterion(
        output.reshape(-1, vocab_size),
        tgt_output.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        pred = output.argmax(-1)
        print("step:", step, "loss:", loss.item())
        print("src[0]: ", src[0])
        print("pred[0]:", pred[0])
        print("true[0]:", tgt_output[0])
        print()