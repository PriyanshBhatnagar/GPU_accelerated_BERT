import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last

import sys
sys.path.append("C:/Users/prbhatnagar/Downloads/Final_Project/build/src/pybind11_cuda_multiply/Release")
import cu_multiply_matrix
import warnings
warnings.filterwarnings('ignore')

class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None 
    dim: int = 768 
    n_layers: int = 12
    n_heads: int = 12
    dim_ff: int = 768*4 
    p_drop_hidden: float = 0.1 
    p_drop_attn: float = 0.1
    max_len: int = 512
    n_segments: int = 2

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) 
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) 

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):

        # GPU Accelerated Implementation
        data = x.view(-1, 128*768)
        data_np = data.numpy()

        q_weight = self.proj_q.weight.detach().numpy()
        k_weight = self.proj_k.weight.detach().numpy()
        v_weight = self.proj_v.weight.detach().numpy()

        q_weight_transpose = q_weight.T
        k_weight_transpose = k_weight.T
        v_weight_transpose = v_weight.T

        q = cu_multiply_matrix.multiply(data_np, q_weight_transpose)
        k = cu_multiply_matrix.multiply(data_np, k_weight_transpose)
        v = cu_multiply_matrix.multiply(data_np, v_weight_transpose)

        q_reshaped = q.reshape(-1,128, 768)
        k_reshaped = k.reshape(-1,128, 768)
        v_reshaped = v.reshape(-1,128, 768)
     
        q = torch.tensor(q_reshaped)
        k = torch.tensor(k_reshaped)
        v = torch.tensor(v_reshaped) 
        
        
        #q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
       
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h

