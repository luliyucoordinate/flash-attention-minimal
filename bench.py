import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu', 'flash_v2.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result_v2 = minimal_attn.forward_v2(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result_v2, manual_result, rtol=0, atol=1e-02))

# def my_flash(Q, K, V):
#     Q_BLOCK_SIZE = 32
#     KV_BLOCK_SIZE = 32
#     NEG_INF = -1e10  # -infinity
#     EPSILON = 1e-10
#     Q_LEN = seq_len
#     K_LEN = seq_len
#     Tr = Q_LEN // Q_BLOCK_SIZE
#     Tc = K_LEN // KV_BLOCK_SIZE
#     O = torch.zeros_like(Q).cuda()
#     l = torch.zeros(Q.shape[:-1]).cuda()[..., None]
#     m = torch.ones(Q.shape[:-1]).cuda()[..., None] * NEG_INF

#     Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
#     K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
#     V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
#     O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
#     l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
#     m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

#     # start with Q
#     for i in range(Tr):
#         Qi = Q_BLOCKS[i]
#         Oi = O_BLOCKS[i]
#         li = l_BLOCKS[i]
#         mi = m_BLOCKS[i]
        
#         for j in range(Tc):
#             #if j>i: 
#             #    continue    # ignore masked      
#             Kj = K_BLOCKS[j]
#             Vj = V_BLOCKS[j]

#             S_ij = Qi @ Kj.transpose(2,3)
#             m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
#             mi_new = torch.maximum(m_block_ij, mi)
#             P_ij_hat = torch.exp(S_ij - mi_new)
#             l_block_ij = torch.sum(P_ij_hat, dim=-1, keepdims=True) # + EPSILON
#             li_new = torch.exp(mi - mi_new) * li  + l_block_ij 
#             O_i = torch.exp(mi - mi_new) * Oi + P_ij_hat @ Vj
#             mi = mi_new
#             li = li_new
#             Oi = O_i
#         O_BLOCKS[i] = O_i / li_new # 最后做Scaled
#         # l_BLOCKS[i] = mi_new + torch.log(li_new)
#     O = torch.cat(O_BLOCKS, dim=2)
#     # l = torch.cat(l_BLOCKS, dim=2)
#     print("zhihu ", O)

# print("manual ", manual_attn(q, k, v))
# my_flash(q, k, v)
# minimal_result_v1 = minimal_attn.forward(q, k, v)
# minimal_result_v2 = minimal_attn.forward_v2(q, k, v)
# print("torch v1 ", minimal_result_v1)
# print("torch v2 ", minimal_result_v2)