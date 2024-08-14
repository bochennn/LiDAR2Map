import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.ops import scatter_nd
from mmdet3d.ops.sparse_block import replace_feature
from spconv.pytorch.pool import SparseAvgPool
from torch_scatter import scatter_add

from .scatter_former_ops import scatter_matmul_kv, scatter_matmul_qc


class AttnPillarPool(spconv.SparseModule):
    def __init__(self, dim, pillar_size=6):
        super().__init__()
        self.dim = dim
        self.pillar_size = pillar_size
        self.query_func = spconv.SparseMaxPool3d((pillar_size, 1, 1), stride=(pillar_size, 1, 1), padding=0)
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.pos_embedding = nn.Embedding(pillar_size, dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.01)

    def forward(self, x):

        src = self.query_func(x)
        _, batch_win_inds = torch.unique(x.indices[:, [0, 2, 3]], return_inverse=True, dim=0)

        scatter_indices = torch.stack([batch_win_inds, x.indices[:, 1]], -1)
        num_pillars = int(batch_win_inds.max() + 1)
        key = value = scatter_nd(
            scatter_indices,  x.features, 
            [num_pillars, self.pillar_size, self.dim]
        )
        key_padding_mask = ~scatter_nd(
            scatter_indices, 
            torch.ones_like(x.features[:, 0]),
            [num_pillars, self.pillar_size]
        ).bool()
        key = key + self.pos_embedding.weight.unsqueeze(0).repeat(num_pillars, 1, 1)

        output = self.self_attn(src.features.unsqueeze(1), key, value, key_padding_mask)[0].squeeze(1)
        src = replace_feature(src, self.norm(output + src.features))

        return src


class CWI_FFN_Layer(spconv.SparseModule):

    def __init__(self, embed_dim, indice_key=None, conv_size=13):
        super(CWI_FFN_Layer, self).__init__()

        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)

        self.conv_k = SparseAvgPool(
            3, kernel_size=3, stride=1, padding=1, indice_key=indice_key + 'k', subm=True)
        self.conv_h = SparseAvgPool(
            3, kernel_size=(1, 1, conv_size), stride=(1, 1, 1), padding=(0, 0, conv_size // 2), indice_key=indice_key + 'h', subm=True
        )
        self.conv_w = SparseAvgPool(
            3, kernel_size=(1, conv_size, 1), stride=(1, 1, 1), padding=(0, conv_size // 2, 0), indice_key=indice_key + 'w', subm=True
        )

        self.bn2 = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

        self.group_dim = embed_dim // 4

    def forward(self, src):

        src = replace_feature(src, self.bn(src.features))
        src_k = replace_feature(src, src.features[:, :self.group_dim])
        src_h = replace_feature(src, src.features[:, self.group_dim:2 * self.group_dim])
        src_w = replace_feature(src, src.features[:, 2 * self.group_dim:3 * self.group_dim])

        src_k = self.conv_k(src_k).features
        src_h = self.conv_h(src_h).features
        src_w = self.conv_w(src_w).features
        src_i = src.features[:, 3 * self.group_dim:]

        src2 = replace_feature(src, torch.cat([src_k, src_h, src_w, src_i], dim=1))        
        src = replace_feature(src, src.features + src2.features)

        src2 = replace_feature(src2, self.fc2(self.act(self.fc1(self.bn2(src.features)))))
        src = replace_feature(src, src.features + src2.features)

        return src


class SLALayer(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads    

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, offsets, counts, batch_win_inds):

        N, C = x.shape
        qkv = self.qkv(x).reshape(N, 3, C)

        q, k, v = qkv.unbind(1)
        q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])

        q = F.relu(q)
        k = F.relu(k)

        kv = scatter_matmul_kv(k, v, offsets, counts)
        s = scatter_add(k, batch_win_inds, dim=0)

        y = scatter_matmul_qc(q, kv, offsets, counts)
        z = torch.sum(s[batch_win_inds, ...] * q, -1, keepdim=True)

        y = y / (z + 1e-6)
        y = rearrange(y, "n h c -> n (h c)", h=self.num_heads)
        y = self.proj(y)

        return y


# class ScatterAttention(nn.Module):

#     def __init__(self, dim, num_heads, qkv_bias=False, indice_key=None):

#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads    

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#         # self.scale = self.head_dim ** -0.5
#         self.pool = spconv.SparseSumPool2d(3, 1, 1, indice_key=indice_key)

#     def forward(self, x, offsets, counts, batch_win_inds, 
#                 batch_win_coords, win_bev_shape, batch_size, indice_dict):

#         N, C = x.shape

#         qkv = self.qkv(x).reshape(N, 3, C)
#         q, k, v = qkv.unbind(1)
#         q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])

#         q = F.relu(q)
#         k = F.relu(k)

#         kv = scatter_matmul_kv(k, v, offsets, counts)
#         s = scatter_add(k, batch_win_inds, dim=0)
        
#         kv_tensor = spconv.SparseConvTensor(
#             kv.view(-1, self.num_heads * self.head_dim * self.head_dim), 
#             batch_win_coords, win_bev_shape, batch_size)
#         if indice_dict is not None:
#             kv_tensor.indice_dict = indice_dict
#         kv = self.pool(kv_tensor)
        
#         s_tensor = spconv.SparseConvTensor(
#             s.view(-1, self.num_heads * self.head_dim),
#             batch_win_coords, win_bev_shape, batch_size)
#         s_tensor.indice_dict = kv.indice_dict
#         s = self.pool(s_tensor)
     
#         kv = kv.features.view(-1, self.num_heads, self.head_dim, self.head_dim)
#         s = s.features.view(-1, self.num_heads, self.head_dim)

#         y = scatter_matmul_qc(q, kv, offsets, counts)
#         z = torch.sum(s[batch_win_inds, ...] * q, -1, keepdim=True)
       
#         y = y / (z + 1e-6)

#         y = rearrange(y, "n h c -> n (h c)", h=self.num_heads)
#         y = self.proj(y)

#         return y,  s_tensor.indice_dict


class ScatterFormerLayer3x(spconv.SparseModule):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, embed_dim, nhead, num_layers = 4, win_size=12, indice_key=None):
        super().__init__()
        self.nhead = nhead
        self.d_model = embed_dim
        self.num_layers = num_layers
        self.win_size = win_size

        self.norm_layers = list()
        self.attn_layers = list()
        self.cffn_layers = list()

        for _ in range(num_layers):
            self.norm_layers.append(nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01) )
            self.attn_layers.append(SLALayer(embed_dim, nhead))
            self.cffn_layers.append(CWI_FFN_Layer(embed_dim, indice_key, conv_size = win_size + 1) )

        self.norm_layers = nn.ModuleList(self.norm_layers)
        self.attn_layers = nn.ModuleList(self.attn_layers)
        self.cffn_layers = nn.ModuleList(self.cffn_layers)

    def forward(self, src):

        batch_win_coords = torch.cat([src.indices[:, :1],
            torch.div(src.indices[:, 2:], self.win_size, rounding_mode='trunc')], dim=1)
        _, batch_win_inds = torch.unique(batch_win_coords, return_inverse=True, dim=0)

        batch_win_inds, perm = torch.sort(batch_win_inds)

        src = replace_feature(src, src.features[perm])
        src.indices = src.indices[perm]
        counts = torch.bincount(batch_win_inds)
        offsets = F.pad(torch.cumsum(counts, dim=0), (1, 0), mode='constant', value=0)[:-1]

        for i in range(self.num_layers):
            x = self.attn_layers[i](
                self.norm_layers[i](src.features), 
                offsets, counts, batch_win_inds
            )
            src = replace_feature(src, src.features + x)
            src = self.cffn_layers[i](src)

        return src
