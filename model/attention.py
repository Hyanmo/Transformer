import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: (batch, head, seq_len_q, d_k)
        :param K: (batch, head, seq_len_k, d_k)
        :param V: (batch, head, seq_len_k, d_v)
        :param mask: (batch, 1, seq_len_q, d_k)
        """

        d_k = Q.size(-1)

        #1. 计算Q*K转置
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        #2. 加mask（在softmax之前）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        #3. softmax
        attn = F.softmax(scores, dim=-1)

        #4. 乘V
        output = torch.matmul(attn,V)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model,d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        q_len = Q.size(1)
        k_len = K.size(1)

        #1. 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        #2. reshape 成多头
        Q = Q.view(batch_size, q_len, self.n_heads, self.d_k)
        K = K.view(batch_size, k_len, self.n_heads, self.d_k)
        V = V.view(batch_size, k_len, self.n_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        #4. 计算attention
        out, attn = self.attention(Q, K, V, mask)

        #5. 拼接多个head
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, q_len, self.d_model)

        #6, 最后线性层
        out = self.fc_out(out)

        return out, attn