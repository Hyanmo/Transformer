import torch

def generate_subsequent_mask(seq_len):
    """
    生成下三角矩阵
    :shape mask: (1, 1, seq_len, seq_len)
    """

    mask = torch.tril(torch.ones(seq_len,seq_len)).bool()

    return mask.unsqueeze(0).unsqueeze(0)

def generate_padding_mask(seq, pad_idx=0):
    """
    :param seq: (batch, seq_len)
    :return: (batch, 1, 1, seq_len)
    """

    mask = (seq != pad_idx)

    return mask.unsqueeze(1).unsqueeze(2)