from typing import Sequence
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath
from torch.utils.cpp_extension import load
from mmcv.cnn.bricks.transformer import to_2tuple
from mmengine.model import BaseModule

class StripConv(nn.Module):
    def __init__(self, channels, strip_size=5, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 4)

        self.conv_h = nn.Conv2d(mid_channels, mid_channels,
                                kernel_size=(1, strip_size),
                                padding=(0, strip_size // 2),
                                groups=mid_channels, bias=False)
        self.conv_v = nn.Conv2d(mid_channels, mid_channels,
                                kernel_size=(strip_size, 1),
                                padding=(strip_size // 2, 0),
                                groups=mid_channels, bias=False)

        self.proj = nn.Conv2d(channels, mid_channels, kernel_size=1, groups=mid_channels//2, bias=False)
        self.expand = nn.Conv2d(mid_channels, channels, kernel_size=1, groups=mid_channels//2, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        feat = self.proj(x)
        out = self.conv_h(feat) + self.conv_v(feat)
        out = self.bn(self.expand(out)) + x
        return out

class MultiScaleStripConv(nn.Module):
    def __init__(self, channels, strip_sizes=[3, 7], reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 4)

        self.proj = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)

        self.conv_h1 = nn.Conv2d(mid_channels, mid_channels,
                                 kernel_size=(1, strip_sizes[0]), padding=(0, strip_sizes[0] // 2),
                                 groups=mid_channels, bias=False)
        self.conv_v1 = nn.Conv2d(mid_channels, mid_channels,
                                 kernel_size=(strip_sizes[0], 1), padding=(strip_sizes[0] // 2, 0),
                                 groups=mid_channels, bias=False)
        self.conv_h2 = nn.Conv2d(mid_channels, mid_channels,
                                 kernel_size=(1, strip_sizes[1]), padding=(0, strip_sizes[1] // 2),
                                 groups=mid_channels, bias=False)
        self.conv_v2 = nn.Conv2d(mid_channels, mid_channels,
                                 kernel_size=(strip_sizes[1], 1), padding=(strip_sizes[1] // 2, 0),
                                 groups=mid_channels, bias=False)

        self.expand = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        feat = self.proj(x)
        out = self.conv_h1(feat) + self.conv_v1(feat) + self.conv_h2(feat) + self.conv_v2(feat)
        out = self.bn(self.expand(out)) + x
        return out

class PatchEmbed(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=256,
                 stride=4,
                 input_size=(512, 512),
                 norm_cfg=None,
                 use_strip_conv=True,
                 strip_size=5,
                 multi_scale=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.stride = stride
        self.use_strip_conv = use_strip_conv

        num_layers = int(math.log2(stride))
        layers = []
        current_channels = in_channels

        for i in range(num_layers):
            if i == 0:
                out_channels = embed_dims // stride
            elif i == num_layers - 1:
                out_channels = embed_dims
            else:
                out_channels = embed_dims // (stride // (2 ** (i + 1)))

            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

            if use_strip_conv and i == num_layers - 1:
                if multi_scale:
                    layers.append(MultiScaleStripConv(out_channels, strip_sizes=[3, 7]))
                else:
                    layers.append(StripConv(out_channels, strip_size=strip_size))

            current_channels = out_channels

        self.proj = nn.Sequential(*layers)
        output_h = input_size[0] // stride
        output_w = input_size[1] // stride
        self.pos_h = nn.Parameter(torch.randn(1, embed_dims, output_h, 1))
        self.pos_w = nn.Parameter(torch.randn(1, embed_dims, 1, output_w))

        if current_channels != embed_dims:
            self.channel_adapter = nn.Conv2d(current_channels, embed_dims, kernel_size=1, bias=False)
        else:
            self.channel_adapter = None

        if norm_cfg is not None:
            from mmcv.cnn import build_norm_layer
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        self.init_input_size = to_2tuple(input_size)
        self.init_out_size = (self.init_input_size[0] // stride, self.init_input_size[1] // stride)

    def forward(self, x):
        x = self.proj(x)

        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        B, C, H, W = x.shape

        pos_h = self.pos_h[:, :, :H, :] if H <= self.pos_h.size(2) else F.interpolate(self.pos_h, size=(H, 1), mode='bilinear', align_corners=False)
        pos_w = self.pos_w[:, :, :, :W] if W <= self.pos_w.size(3) else F.interpolate(self.pos_w, size=(1, W), mode='bilinear', align_corners=False)

        pos_emb = pos_h + pos_w
        x = x + pos_emb

        out_size = (H, W)
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x, out_size

wkv_cuda = load(
    name="wkv",
    sources=["models/wkv.cpp", "models/wkv_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86']
)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)

        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)

def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())

class CohGate(nn.Module):

    def __init__(self, n_embd, n_layer, layer_id, spatial=True):
        super().__init__()
        self.n_embd = n_embd
        self.spatial = spatial

        self.self_proj = nn.Linear(n_embd, n_embd, bias=True)

        if spatial:

            self.neighbor_proj = nn.Linear(n_embd, n_embd, bias=False)
            self.neighbor_conv = nn.Conv2d(
                n_embd, n_embd, 3, 1, 1, groups=n_embd, bias=False
            )

            self.connectivity_weight = nn.Parameter(torch.zeros(1))

        layer_ratio = layer_id / max(1, n_layer - 1)
        init_strength = 0.3 + 0.4 * layer_ratio
        self.gate_strength = nn.Parameter(
            torch.ones(n_embd) * math.log(init_strength / (1 - init_strength))
        )

        self.gate_floor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, C = x.shape

        self_gate = torch.sigmoid(self.self_proj(x))

        if self.spatial:
            H = W = int(math.sqrt(T))

            nb = torch.sigmoid(self.neighbor_proj(x))
            nb_2d = nb.reshape(B, H, W, C).permute(0, 3, 1, 2)
            nb_agg = self.neighbor_conv(nb_2d)
            nb_agg = torch.sigmoid(nb_agg)
            nb_agg = nb_agg.reshape(B, C, T).permute(0, 2, 1)

            conn_w = torch.sigmoid(self.connectivity_weight)
            gate = self_gate * ((1.0 - conn_w) + conn_w * nb_agg)
        else:
            gate = self_gate

        strength = torch.sigmoid(self.gate_strength)
        gate = (1.0 - strength) + strength * gate

        floor = torch.sigmoid(self.gate_floor) * 0.3
        gate = floor + (1.0 - floor) * gate

        return gate

class AGP(nn.Module):

    def __init__(self, n_embd, context_dim=None):
        super().__init__()
        self.n_embd = n_embd

        self.probe_near = nn.Conv2d(n_embd, n_embd, 3, 1, 1,
                                    groups=n_embd, bias=False)
        self.probe_mid = nn.Conv2d(n_embd, n_embd, 3, 1, 2, dilation=2,
                                   groups=n_embd, bias=False)
        self.probe_far = nn.Conv2d(n_embd, n_embd, 3, 1, 4, dilation=4,
                                   groups=n_embd, bias=False)

        self.ani_h = nn.Conv2d(n_embd, n_embd, (1, 3), padding=(0, 1),
                               groups=n_embd, bias=False)
        self.ani_v = nn.Conv2d(n_embd, n_embd, (3, 1), padding=(1, 0),
                               groups=n_embd, bias=False)

        self.scale_selector = nn.Sequential(
            nn.Conv2d(2, 8, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 3, 1, bias=False),
        )

        self.out_proj = nn.Conv2d(n_embd, n_embd, 1, bias=False)

        self.has_context = context_dim is not None and context_dim > 0
        if self.has_context:
            self.context_proj = nn.Conv2d(context_dim, n_embd, 1, bias=False)
            self.context_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(n_embd, max(4, n_embd // 4), 1, bias=False),
                nn.GELU(),
                nn.Conv2d(max(4, n_embd // 4), n_embd, 1, bias=False),
                nn.Sigmoid(),
            )

        self.residual_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, context=None):
        B, T, C = x.shape
        H = W = int(math.sqrt(T))
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)

        p_near = self.probe_near(x_2d)
        p_mid = self.probe_mid(x_2d)
        p_far = self.probe_far(x_2d)

        resp_h = self.ani_h(x_2d).abs().mean(dim=1, keepdim=True)
        resp_v = self.ani_v(x_2d).abs().mean(dim=1, keepdim=True)
        ani = torch.cat([resp_h, resp_v], dim=1)

        w = F.softmax(self.scale_selector(ani), dim=1)
        fused = (w[:, 0:1] * p_near +
                 w[:, 1:2] * p_mid +
                 w[:, 2:3] * p_far)
        fused = self.out_proj(fused)

        if self.has_context and context is not None:
            ctx_2d = context.permute(0, 2, 1).reshape(B, C, H, W)
            ctx_proj = self.context_proj(ctx_2d)
            gate = self.context_gate(ctx_proj)
            fused = fused * gate

        scale = torch.sigmoid(self.residual_scale)
        out_2d = x_2d + scale * fused

        return out_2d.reshape(B, C, T).permute(0, 2, 1)

class Needle_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)

        self.key = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.Linear(n_embd//4, attn_sz, bias=False),
        )

        self.value = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.Linear(n_embd//4, attn_sz, bias=False),
        )

        self.receptance = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.Linear(n_embd//4, attn_sz, bias=False),
        )

        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.k_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(attn_sz, attn_sz // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attn_sz // 8, attn_sz, 1),
            nn.Sigmoid()
        )

        self.coh_gate = CohGate(n_embd, n_layer, layer_id, spatial=True)

        self.conv_63_to_32 = nn.Conv2d(64, 32, 1, groups=32)

        self.geo_dir_proj = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4, bias=False),
            nn.GELU(),
            nn.Linear(n_embd // 4, 2, bias=True),
        )

        self.geo_scale = nn.Parameter(torch.zeros(1))

        self.cross_modal_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.cross_modal_alpha = nn.Parameter(torch.zeros(1))

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad():
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))

                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd

        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))

        else:
            raise NotImplementedError

    def jit_func(self, x):
        xk = x
        xv = x
        xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution, context_feature):
        B, T, C = x.size()
        H = W = int(math.sqrt(T))
        context_B, context_T, context_C = context_feature.size()
        context_H = context_W = int(math.sqrt(context_T))

        context_aligned = self.conv_63_to_32(
            nn.Upsample(size=(H, W), mode="bilinear")(
                context_feature.permute(0, 2, 1).reshape(B, context_C, context_H, context_W)
            )
        )
        context_aligned = context_aligned.reshape(B, C, H * W).permute(0, 2, 1)

        sr, k, v = self.jit_func(x)
        k = k * self.k_gate(k.transpose(1, 2)).transpose(1, 2)
        v = v * self.k_gate(v.transpose(1, 2)).transpose(1, 2)

        w = 8.0 * torch.tanh(self.spatial_decay / 8.0)
        u = 5.0 * torch.tanh(self.spatial_first  / 5.0)
        k_safe = k.clamp(-8.0, 8.0)
        v_safe = v.clamp(-8.0, 8.0)

        dir_raw = self.geo_dir_proj(context_aligned)

        dir_vec = F.normalize(dir_raw, dim=-1, p=2, eps=1e-6)
        direction_align = dir_vec[..., 0:1] ** 2
        geo_strength = torch.sigmoid(self.geo_scale)
        k_safe = k_safe * (1.0 + geo_strength * direction_align)

        rwkv = RUN_CUDA(w / T, u / T, k_safe, v_safe)

        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv

        cross_signal = self.cross_modal_proj(context_aligned)
        alpha = torch.sigmoid(self.cross_modal_alpha)
        rwkv = rwkv + alpha * torch.tanh(cross_signal) * sr

        output_gate = self.coh_gate(x)
        rwkv = rwkv * output_gate

        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                gate_sparsity = (output_gate < 0.2).float().mean().item()
                gate_mean = output_gate.mean().item()
                geo_s = geo_strength.item()
                cm_a = alpha.item()
                conn_w = torch.sigmoid(self.coh_gate.connectivity_weight).item()
                print(f"[CohGate-Spatial] Layer {self.layer_id} | "
                      f"Sparsity: {gate_sparsity:.3f} | Mean: {gate_mean:.3f} | "
                      f"ConnW: {conn_w:.3f} | GeoScale: {geo_s:.3f} | CMAlpha: {cm_a:.3f}")

        rwkv = self.output(rwkv + context_aligned)

        return rwkv

class Needle_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)

        self.key = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.Linear(n_embd//4, attn_sz, bias=False),
        )

        self.value = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.Linear(n_embd//4, attn_sz, bias=False),
        )

        self.receptance = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.Linear(n_embd//4, attn_sz, bias=False),
        )

        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.k_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(attn_sz, attn_sz // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attn_sz // 8, attn_sz, 1),
            nn.Sigmoid()
        )

        self.coh_gate = CohGate(n_embd, n_layer, layer_id, spatial=True)

        self.conv_63_to_32 = nn.Conv2d(64, 32, 1, groups=32)

        self.geo_dir_proj = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4, bias=False),
            nn.GELU(),
            nn.Linear(n_embd // 4, 2, bias=True),
        )

        self.geo_scale = nn.Parameter(torch.zeros(1))

        self.cross_modal_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.cross_modal_alpha = nn.Parameter(torch.zeros(1))

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad():
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))

                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd

        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))

        else:
            raise NotImplementedError

    def _to_vertical_order(self, x, H, W):
        B, T, C = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C)
        return x

    def _from_vertical_order(self, x, H, W):
        B, T, C = x.shape
        x = x.view(B, W, H, C)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C)
        return x

    def jit_func(self, x):
        xk = x
        xv = x
        xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution, context_feature):
        B, T, C = x.size()
        H = W = int(math.sqrt(T))
        context_B, context_T, context_C = context_feature.size()
        context_H = context_W = int(math.sqrt(context_T))

        context_aligned = self.conv_63_to_32(
            nn.Upsample(size=(H, W), mode="bilinear")(
                context_feature.permute(0, 2, 1).reshape(B, context_C, context_H, context_W)
            )
        )
        context_aligned = context_aligned.reshape(B, C, H * W).permute(0, 2, 1)

        sr, k, v = self.jit_func(x)
        k = k * self.k_gate(k.transpose(1, 2)).transpose(1, 2)
        v = v * self.k_gate(v.transpose(1, 2)).transpose(1, 2)

        w = 8.0 * torch.tanh(self.spatial_decay / 8.0)
        u = 5.0 * torch.tanh(self.spatial_first  / 5.0)
        k_safe = k.clamp(-8.0, 8.0)
        v_safe = v.clamp(-8.0, 8.0)

        dir_raw = self.geo_dir_proj(context_aligned)
        dir_vec = F.normalize(dir_raw, dim=-1, p=2, eps=1e-6)

        geo_strength = torch.sigmoid(self.geo_scale)

        align_h = dir_vec[..., 0:1] ** 2
        k_h = k_safe * (1.0 + geo_strength * align_h)

        align_v = dir_vec[..., 1:2] ** 2
        k_v = k_safe * (1.0 + geo_strength * align_v)

        rwkv_h = RUN_CUDA(w / T, u / T, k_h, v_safe)

        k_v_reorder = self._to_vertical_order(k_v, H, W)
        v_v_reorder = self._to_vertical_order(v_safe, H, W)

        rwkv_v = RUN_CUDA(w / T, u / T, k_v_reorder, v_v_reorder)
        rwkv_v = self._from_vertical_order(rwkv_v, H, W)

        rwkv = 0.5 * (rwkv_h + rwkv_v)

        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv

        cross_signal = self.cross_modal_proj(context_aligned)
        alpha = torch.sigmoid(self.cross_modal_alpha)
        rwkv = rwkv + alpha * torch.tanh(cross_signal) * sr

        output_gate = self.coh_gate(x)
        rwkv = rwkv * output_gate

        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                gate_sparsity = (output_gate < 0.2).float().mean().item()
                gate_mean = output_gate.mean().item()
                geo_s = geo_strength.item()
                cm_a = alpha.item()
                conn_w = torch.sigmoid(self.coh_gate.connectivity_weight).item()
                print(f"[CohGate-Spatial] Layer {self.layer_id} | "
                      f"Sparsity: {gate_sparsity:.3f} | Mean: {gate_mean:.3f} | "
                      f"ConnW: {conn_w:.3f} | GeoScale: {geo_s:.3f} | CMAlpha: {cm_a:.3f}")

        rwkv = self.output(rwkv + context_aligned)

        return rwkv

class CM_process(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // r, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pro = self.fc(x.transpose(1, 2)).transpose(1, 2)
        return x * pro

class Needle_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False, context_dim=64):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.hidden_sz = hidden_rate * n_embd

        self.key = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_embd//4, self.hidden_sz, bias=False),
        )

        self.receptance = nn.Sequential(
            nn.Linear(n_embd, n_embd//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_embd//4, n_embd, bias=False),
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_sz, n_embd//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_embd//4, n_embd, bias=False),
        )

        self.value.scale_init = 0
        self.receptance.scale_init = 0

        self.key_norm = nn.LayerNorm(self.hidden_sz) if key_norm else nn.Identity()
        self.cm_process = CM_process(self.hidden_sz, r=8)
        self._init_weights(init_mode)

        self.coh_gate = CohGate(n_embd, n_layer, layer_id, spatial=True)

    def _init_weights(self, init_mode):
        ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))
        if init_mode in ['fancy', 'local', 'global']:
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
        else:
            raise NotImplementedError

    def forward(self, x):
        xk = x
        xr = x

        k = self.key(xk)
        k = torch.square(F.relu(k))
        k = self.cm_process(k)
        k = self.key_norm(k)

        kv = self.value(k)
        rkv = torch.sigmoid(self.receptance(xr)) * kv

        ffn_gate = self.coh_gate(x)
        rkv = rkv * ffn_gate

        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                gate_sparsity = (ffn_gate < 0.2).float().mean().item()
                gate_mean = ffn_gate.mean().item()
                conn_w = torch.sigmoid(self.coh_gate.connectivity_weight).item()
                print(f"[CohGate-FFN] Layer {self.layer_id} | "
                      f"Sparsity: {gate_sparsity:.3f} | "
                      f"Mean: {gate_mean:.3f} | "
                      f"ConnW: {conn_w:.3f}")

        return rkv

class Block(nn.Module):
    def __init__(self, n_layer, layer_id,
                 drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(32)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(32)

        self.att = Needle_SpatialMix(32, n_layer, layer_id, init_mode,
                                   key_norm=key_norm)

        self.agp_pre_att = AGP(n_embd=32, context_dim=32)
        self.agp_pre_ffn = AGP(n_embd=32, context_dim=32)

        self.ffn = Needle_ChannelMix(32, n_layer, layer_id, hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((32)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((32)), requires_grad=True)
        self.with_cp = with_cp

        self.Conv256_to_32 = nn.Sequential(nn.Conv2d(256, 16, 1, groups=8, bias=True),
                             nn.Conv2d(16, 16, 1, 1, 0, groups=16, bias=False),
                             nn.Conv2d(16, 32, 1, groups=8, bias=False))

        self.Conv32_to_256 = nn.Sequential(nn.Conv2d(32, 16, 1, groups=8, bias=True),
                             nn.Conv2d(16, 16, 1, 1, 0, groups=16, bias=False),
                             nn.Conv2d(16, 256, 1, groups=8, bias=False))
        self.conv_ctx_align = nn.Conv2d(64, 32, 1, groups=32)

    def forward(self, x, patch_resolution, context_feature):
        def _inner_forward(x, ctx_feat):
            B, L, C = x.shape
            H = W = int(math.sqrt(L))

            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = self.Conv256_to_32(x)
            C_inner = 32
            x = x.reshape(B, C_inner, H*W).permute(0, 2, 1)

            ctx_B, ctx_T, ctx_C = ctx_feat.shape
            ctx_H = ctx_W = int(math.sqrt(ctx_T))
            ctx_aligned = self.conv_ctx_align(
                nn.Upsample(size=(H, W), mode="bilinear")(
                    ctx_feat.permute(0, 2, 1).reshape(B, ctx_C, ctx_H, ctx_W)
                )
            )
            ctx_aligned = ctx_aligned.reshape(B, C_inner, H * W).permute(0, 2, 1)

            if self.layer_id == 0:
                x = self.ln0(x)

            if self.post_norm:
                if self.layer_scale:
                    x = self.agp_pre_att(x, ctx_aligned)
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution, ctx_feat)))
                    x = self.agp_pre_ffn(x, ctx_aligned)
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x)))
                    x = self.agp_pre_ffn(x, ctx_aligned)
                else:
                    x = self.agp_pre_att(x, ctx_aligned)
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution, ctx_feat)))
                    x = self.agp_pre_ffn(x, ctx_aligned)
                    x = x + self.drop_path(self.ln2(self.ffn(x)))
                    x = self.agp_pre_ffn(x, ctx_aligned)
            else:
                if self.layer_scale:
                    x = self.agp_pre_att(x, ctx_aligned)
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution, ctx_feat))
                    x = self.agp_pre_ffn(x, ctx_aligned)
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x)))
                    x = self.agp_pre_ffn(x, ctx_aligned)
                else:
                    x = self.agp_pre_att(x, ctx_aligned)
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution, ctx_feat))
                    x = self.agp_pre_ffn(x, ctx_aligned)
                    x = x + self.drop_path(self.ffn(self.ln2(x)))
                    x = self.agp_pre_ffn(x, ctx_aligned)

            x = self.Conv32_to_256(x.reshape(B, H, W, C_inner).permute(0, 3, 1, 2)).reshape(B, C, H*W).permute(0, 2, 1)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, context_feature)
        else:
            x = _inner_forward(x, context_feature)
        return x

class Needle(nn.Module):
    def __init__(self,
                 img_size=512,
                 patch_size=4,
                 in_channels=3,
                 out_indices=[0, 1, 2, 3],
                 drop_rate=0.,
                 embed_dims=256,
                 depth=4,
                 drop_path_rate=0.,
                 init_values=0.1,
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 **kwargs):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        if isinstance(patch_size, (list, tuple)):
            assert len(patch_size) == depth, f"patch_size list length {len(patch_size)} must equal depth {depth}"
            self.patch_sizes = patch_size
        else:
            self.patch_sizes = [patch_size] * depth

        self.patch_embeds = nn.ModuleList()
        self.patch_resolutions = []

        for i, ps in enumerate(self.patch_sizes):
            input_channels = in_channels if i == 0 else self.embed_dims
            patch_embed = PatchEmbed(
                in_channels=input_channels,
                input_size=(512, 512),
                embed_dims=self.embed_dims,
                stride=ps
            )
            self.patch_embeds.append(patch_embed)
            self.patch_resolutions.append(patch_embed.init_out_size)

        self.interpolate_mode = interpolate_mode
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_layer=depth,
                layer_id=i,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                init_values=init_values,
                post_norm=post_norm,
                key_norm=key_norm,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

        self.conv256to256 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=1, groups=64, bias=False),
            nn.Conv2d(64, 256, kernel_size=1, bias=False)
        )

        self.conv256to128 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=1, groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False)
        )

        self.conv256to32_1 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=1, groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False)
        )

        self.conv256to32_2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=1, groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False)
        )

        self.gn256 = nn.GroupNorm(num_channels=256, num_groups=16)
        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=8)

    def forward(self, x, context_feature):
        outs = []
        current_input = x

        for i, (patch_embed, layer) in enumerate(zip(self.patch_embeds, self.layers)):
            feat, patch_resolution = patch_embed(current_input)
            feat = self.drop_after_pos(feat)

            processed_feat = layer(feat, patch_resolution, context_feature)

            if i == len(self.layers) - 1 and self.final_norm:
                processed_feat = self.ln1(processed_feat)

            if i in self.out_indices:
                B, _, C = processed_feat.shape
                patch_token = processed_feat.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                if i == 0:
                    patch_token = nn.Upsample(size=(256, 256), mode="bilinear")(self.gn256(self.conv256to256(patch_token)))
                    outs.append(patch_token)
                elif i == 1:
                    outs.append(self.gn128(self.conv256to128(patch_token)))
                elif i == 2:
                    patch_token = nn.Upsample(size=(64, 64), mode="bilinear")(self.gn32(self.conv256to32_1(patch_token)))
                    outs.append(patch_token)
                elif i == 3:
                    patch_token = nn.Upsample(size=(64, 64), mode="bilinear")(self.gn32(self.conv256to32_2(patch_token)))
                    outs.append(patch_token)

            if i < len(self.layers) - 1:
                B, _, C = processed_feat.shape
                H, W = patch_resolution
                current_input = processed_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
                target_size = (512//2, 512//2)
                if current_input.shape[2:] != target_size:
                    current_input = nn.Upsample(size=target_size, mode="bilinear")(current_input)

        return outs
