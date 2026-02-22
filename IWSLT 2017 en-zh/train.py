import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentencepiece as spm

from utils.mask import generate_subsequent_mask, generate_padding_mask
from dataset import TranslationDataset, collate_fn
from model.transformer import Transformer  #模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.加载数据
hf_dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
train_data = hf_dataset["train"].select(range(10000))  # 小数据测试

train_dataset = TranslationDataset(train_data, "spm.model")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# 2.创建模型
vocab_size = 8000
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=128,
    n_heads=2,
    num_layers=1,
    dropout=0.0
).to(device)

# 3.定义损失
criterion = nn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=0.1
)

# 4.优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9
)

# 5.训练循环
def train():
    model.train()

    for epoch in range(10):
        total_loss = 0

        for src, tgt_input, tgt_label in train_loader:

            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_label = tgt_label.to(device)

            optimizer.zero_grad()

            output = model(src, tgt_input)

            # reshape
            output = output.view(-1, output.size(-1))
            tgt_label = tgt_label.view(-1)

            loss = criterion(output, tgt_label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

# 6.翻译器
def greedy_decode(model, sentence, sp, device, max_len=50):
    model.eval()

    pad_id = 0
    bos_id = 1
    eos_id = 2

    # 1.编码 src
    src_ids = sp.encode(sentence, out_type=int)
    src_ids = src_ids + [eos_id]

    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    # shape: [1, src_len]

    # 2.encoder 输出
    with torch.no_grad():
        enc_output = model.encoder(
            src,
            generate_padding_mask(src, pad_id).to(device)
        )

    # 3.初始化 decoder 输入
    tgt_ids = [bos_id]

    for _ in range(max_len):

        tgt = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)

        tgt_padding_mask = generate_padding_mask(tgt, pad_id).to(device)
        tgt_sub_mask = generate_subsequent_mask(tgt.size(1)).to(device)

        tgt_mask = tgt_padding_mask & tgt_sub_mask

        with torch.no_grad():
            out = model.decoder(
                tgt,
                enc_output,
                tgt_mask=tgt_mask,
                src_mask=generate_padding_mask(src, pad_id).to(device)
            )

        # 取最后一个 token
        next_token = out[:, -1, :].argmax(dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == eos_id:
            break

    # 去掉 <bos>
    return sp.decode(tgt_ids[1:])
if __name__ == "__main__":
    train()

    sp = spm.SentencePieceProcessor()
    sp.load("spm.model")

    test_sentence = "Thank you very much."

    translation = greedy_decode(model, test_sentence, sp, device)

    print("Input :", test_sentence)
    print("Output:", translation)