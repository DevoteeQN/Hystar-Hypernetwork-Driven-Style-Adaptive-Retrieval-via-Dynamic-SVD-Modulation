# Source: https://github.com/SivanDoveh/TSVLC/blob/main/src/open_clip/loralib/layers.py
#
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
#
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import warnings

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.overrides import handle_torch_function, has_torch_function
from src.model.hyper_net import HyperGenerator


LoRALayer = None
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,latent_dim=792,use_hyper=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.use_hyper = use_hyper
        self.latent_dim = latent_dim
        self.load_svd = False
        self.z = None
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_proj = proj_drop
        self.drop_attn = attn_drop
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map
    def set_z(self,z):
        self.z = z
    def init_svd(self):
        def svd(weight: torch.Tensor):
                with torch.no_grad():
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    r = S.shape[0]
                    U = U[:, :r].contiguous()
                    S = S[:r].contiguous()
                    V = Vh[:r, :].contiguous()
                return nn.Parameter(U,requires_grad=False), nn.Parameter(V,requires_grad=False), nn.Parameter(S,requires_grad=False)

        qkv_U, qkv_V, qkv_S = svd(self.qkv.weight)
        r = qkv_S.shape[0]
        self.U_qkv = qkv_U
        self.V_qkv = qkv_V
        self.S_qkv = qkv_S
        if self.use_hyper:
            self.delta_s_qkv = HyperGenerator(self.latent_dim,[2*r],r)
            # self.delta_s_qkv = HyperGenerator(self.latent_dim, [r], r)
        else:
            self.delta_s_qkv = nn.Parameter(torch.zeros(r))
            self.gate_qkv = nn.Parameter(torch.tensor(0.0))
        self.load_svd = True
    def forward(self, x, register_hook=False):
        assert self.load_svd is True, "svd is not loaded"
        B, N, C = x.shape
        if self.use_hyper:
            z = self.z
            delta_s = self.delta_s_qkv(z)
            out_list = []
            for idx in range(z.shape[0]):
                delta_w = self.U_qkv @ torch.diag(delta_s[idx]) @ self.V_qkv
                w_qkv = self.qkv.weight + delta_w
                qkv = F.linear(x[idx:idx+1,:,:], w_qkv, self.qkv.bias)
                qkv = qkv.reshape(1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                if register_hook:
                    self.save_attention_map(attn)
                    attn.register_hook(self.save_attn_gradients)

                out = (attn @ v).transpose(1, 2).reshape(1, N, C)
                out = self.proj(out)
                out = self.proj_drop(out)
                out_list.append(out)
            x = torch.cat(out_list, dim=0)
        else:
            delta_s = self.delta_s_qkv  # [r]
            delta_s = delta_s * torch.sigmoid(self.gate_qkv)
            delta_w = self.U_qkv @ torch.diag(delta_s) @ self.V_qkv
            w_qkv = self.qkv.weight +delta_w
            qkv = F.linear(x, w_qkv, self.qkv.bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if register_hook:
                self.save_attention_map(attn)
                attn.register_hook(self.save_attn_gradients)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x
class Linear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = None,
        lora_alpha: int = 1,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        use_hyper: bool = False,
        latent_dim: int = 256,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.use_hyper = use_hyper
        self.merged = False
        self.load_svd = False
        self.latent_dim = latent_dim
        self.z = None
    def init_svd(self):
        W = self.weight.data
        if self.fan_in_fan_out:
            W = W.T
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        if self.r is None:
            r = S.shape[0]
            # print(r)
        else:
            r = self.r
        U = U[:, :r]
        S = S[:r]
        V = Vh[:r, :]  # Vh → V
        self.U = nn.Parameter(U,requires_grad=False)
        self.S = nn.Parameter(S,requires_grad=False)
        self.V = nn.Parameter(V,requires_grad=False)
        self.register_buffer('svd_U', U)  # [out, r]
        self.register_buffer('svd_S_pre', S)  # [r]
        self.register_buffer('svd_V', V)  # [r, in]
        self.weight.requires_grad = False  # freeze pretrained

        if self.use_hyper:
            self.delta_s = HyperGenerator(self.latent_dim,[2*r],r)
            # self.delta_s = HyperGenerator(self.latent_dim, [r], r)
        else:
            self.delta_s = nn.Parameter(torch.zeros(r))
            self.gate = nn.Parameter(torch.tensor(0.0))
        self.load_svd = True
    def set_z(self,z):
        self.z = z
    def forward(self, x: torch.Tensor):
        z = self.z
        assert self.load_svd is True, "svd is not loaded"

        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.use_hyper:
            delta_s = self.delta_s(z)  # [r]
        else:
            delta_s = self.delta_s  # [r]
            delta_s = delta_s * torch.sigmoid(self.gate)

        s_total = delta_s
        out = F.linear(x, T(self.weight), self.bias)
        if not self.use_hyper:
            delta_weight = self.U @ torch.diag(s_total) @ self.V
            delta_out = F.linear(x, delta_weight, bias=None)
        else:
            S = torch.diag_embed(s_total)  # [24, 1024, 1024]

            U_exp = self.U.unsqueeze(0)  # [1, 4096, 1024]

            V_exp = self.V.unsqueeze(0)  # [1, 1024, 1024]

            M = torch.matmul(S, V_exp)  # [24, 1024, 1024]

            result = torch.matmul(U_exp, M)  # [24, 4096, 1024]
            result_T = result.transpose(1, 2).contiguous()  # [24,1024,4096]
            x_T = x.transpose(0,1).contiguous()  # [24,257,1024]
            delta_out = torch.bmm(x_T, result_T)  # [24,257,4096]
            delta_out = delta_out.transpose(0,1).contiguous() # [257,24,4096]

        return out + delta_out


class MultiheadAttention(nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,latent_dim=1792,r=None,use_hyper=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.diag_r = r
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim
        self.use_hyper = use_hyper
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.z = None
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()
        self.load_svd = False
    def set_z(self,z):
        self.z = z
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)
    def init_svd(self):
        def svd(weight: torch.Tensor):
                with torch.no_grad():
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    if self.diag_r is None:
                        r = S.shape[0]
                        # print(r)
                    else:
                        r = self.diag_r
                    U = U[:, :r].contiguous()
                    S = S[:r].contiguous()
                    V = Vh[:r, :].contiguous()
                return nn.Parameter(U,requires_grad=False), nn.Parameter(V,requires_grad=False), nn.Parameter(S,requires_grad=False)

        if self._qkv_same_embed_dim:
            assert self.in_proj_weight is not None
            # PyTorch stores the self-attention Q/K/V projections as one packed
            # matrix W_qkv = [W_q; W_k; W_v] with shape [3d, d]. Hystar
            # treats this packed projection as a single spectral object: one
            # hypernetwork predicts delta_s, and U_qkv diag(delta_s) V_qkv
            # reconstructs a joint [3d, d] update for Q, K, and V together.
            qkv_U, qkv_V, qkv_S = svd(self.in_proj_weight)

            if self.diag_r is None:
                r = qkv_S.shape[0]
            else:
                r = self.diag_r
            self.U_qkv = qkv_U
            self.V_qkv = qkv_V
            self.S_qkv = qkv_S
            if self.use_hyper:
                self.delta_s_qkv = HyperGenerator(self.latent_dim,[2*r],r)
                # self.delta_s_qkv = HyperGenerator(self.latent_dim, [r], r)
            else:
                self.delta_s_qkv = nn.Parameter(torch.zeros(r))
                self.gate_qkv = nn.Parameter(torch.tensor(0.0))

        else:
            assert self.q_proj_weight is not None and self.k_proj_weight is not None and self.v_proj_weight is not None
            self.U_q, self.V_q, self.S_q = svd(self.q_proj_weight)
            self.U_k, self.V_k, self.S_k = svd(self.k_proj_weight)
            self.U_v, self.V_v, self.S_v = svd(self.v_proj_weight)
            if self.diag_r is None:
                r = self.S_q.shape[0]
            else:
                r = self.diag_r
            if self.use_hyper:
                self.delta_s_q = HyperGenerator(self.latent_dim,[2*r],r)
                # self.delta_s_q = HyperGenerator(self.latent_dim, [r], r)
            else:
                self.delta_s_q = nn.Parameter(torch.zeros(r))
                self.gate_q = nn.Parameter(torch.tensor(0.0))
            if self.diag_r is None:
                r = self.S_k.shape[0]
            else:
                r = self.diag_r
            if self.use_hyper:
                self.delta_s_k = HyperGenerator(self.latent_dim,[2*r],r)
                # self.delta_s_k = HyperGenerator(self.latent_dim, [r], r)
            else:
                self.delta_s_k = nn.Parameter(torch.zeros(r))
                self.gate_k = nn.Parameter(torch.tensor(0.0))
            if self.diag_r is None:
                r = self.S_v.shape[0]
            else:
                r = self.diag_r
            if self.use_hyper:
                self.delta_s_v = HyperGenerator(self.latent_dim,[2*r],r)
                # self.delta_s_v = HyperGenerator(self.latent_dim, [r], r)
            else:
                self.delta_s_v = nn.Parameter(torch.zeros(r))
                self.gate_v = nn.Parameter(torch.tensor(0.0))
        self.load_svd = True
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            why_not_fast_path = True
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            if self.use_hyper:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, average_attn_weights=average_attn_weights,z=self.z,delta_s_net=self.delta_s_qkv,U=self.U_qkv,V=self.V_qkv,use_hyper=True)
            else:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                    delta_s_net=(self.delta_s_qkv,self.gate_qkv), U=self.U_qkv, V=self.V_qkv,use_hyper=False)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

import math
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    enable_style: bool = True,
    delta_w = None
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            if not enable_style:
                return F.linear(q, w, b).chunk(3, dim=-1)
            if enable_style:
                w = w.repeat(delta_w.shape[0], 1, 1)
                w = w + delta_w
                out = torch.bmm(q.transpose(0,1),w.transpose(1, 2)).transpose(0, 1)
                out = out + b.view(1,1,-1)
                return out.chunk(3,dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched
def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    delta_s_net = None,
    z = None,
    U = None,
    V = None,
    use_hyper=True
) -> Tuple[Tensor, Optional[Tensor]]:
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported")
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection

    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        if use_hyper:
            delta_s = delta_s_net(z)
            S = torch.diag_embed(delta_s)  # [B, r, r]

            U_exp = U.unsqueeze(0)  # [1, 3d, r]

            V_exp = V.unsqueeze(0)  # [1, r, d]

            M = torch.matmul(S, V_exp)  # [B, r, d]

            # U comes from the packed [3d, d] QKV projection, so this is
            # already a full per-sample QKV update with shape [B, 3d, d].
            # No Q-only update or repeat across Q/K/V is needed.
            delta_in_proj_weight = torch.matmul(U_exp, M)
            q, k, v = _in_projection_packed(q=query, k=key, v=value, w=in_proj_weight, b=in_proj_bias,delta_w=delta_in_proj_weight,enable_style=True)
        else:
            delta_s , gate = delta_s_net
            delta_s = delta_s * torch.sigmoid(gate)
            # Joint packed-QKV spectral update: [3d, r] @ [r, r] @ [r, d].
            delta_weight = U @ torch.diag(delta_s) @ V
            in_proj_weight = in_proj_weight + delta_weight
            q, k, v = _in_projection_packed(q=query, k=key, v=value, w=in_proj_weight, b=in_proj_bias,enable_style=False)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


