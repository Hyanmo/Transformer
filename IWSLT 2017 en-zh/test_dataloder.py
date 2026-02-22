from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn

dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
train_data = dataset["train"].select(range(1000))  # 先用小数据测试

ds = TranslationDataset(train_data, "spm.model")

loader = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for batch in loader:
    src, tgt_input, tgt_label = batch

    print("src shape:", src.shape)
    print("tgt_input shape:", tgt_input.shape)
    print("tgt_label shape:", tgt_label.shape)
    break