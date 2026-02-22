import torch
from numpy.random import sample
from torch.utils.data import Dataset
import sentencepiece as spm

"""
src         = 英文 token ids + <eos>
tgt_input   = <bos> + 中文 token ids
tgt_label   = 中文 token ids + <eos>
例如：
    tgt_input  = <bos> 我 喜欢 机器 学习
    tgt_label  = 我 喜欢 机器 学习 <eos>
"""

class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, sp_model_path):
        self.data = hf_dataset
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        en = sample["translation"]["en"]
        zh = sample["translation"]["zh"]

        #编码
        src_ids = self.sp.encode(en, out_type=int)
        tgt_ids = self.sp.encode(zh, out_type=int)

        #加eos
        src_ids = src_ids + [self.eos_id]

        #构造 tgt_input 和 tgt_label
        tgt_input = [self.bos_id] + tgt_ids
        tgt_label = tgt_ids + [self.eos_id]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_label, dtype=torch.long)
        )

def collate_fn(batch):
    src_batch, tgt_input_batch, tgt_label_batch = zip(*batch)

    pad_id = 0

    #找最大长度
    src_max_len = max(len(x) for x in src_batch)
    tgt_max_len = max(len(x) for x in tgt_input_batch)

    def pad_sequence(seq, max_len):
        padded = torch.full((max_len,), pad_id, dtype=torch.long)
        padded[:len(seq)] = seq
        return padded

    src_batch = torch.stack([pad_sequence(x, src_max_len) for x in src_batch])
    tgt_input_batch = torch.stack([pad_sequence(x, tgt_max_len) for x in tgt_input_batch])
    tgt_label_batch = torch.stack([pad_sequence(x, tgt_max_len) for x in tgt_label_batch])

    return src_batch, tgt_input_batch, tgt_label_batch
