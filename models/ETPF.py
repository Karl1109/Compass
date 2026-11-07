import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class FrequencyDecomposer(nn.Module):

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
        self.cutoff_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        fft_x = torch.fft.rfft2(x, norm='ortho')

        freq_h = torch.arange(H, device=x.device, dtype=torch.float32) / H
        freq_h = torch.min(freq_h, 1.0 - freq_h)
        freq_w = torch.arange(W // 2 + 1, device=x.device, dtype=torch.float32) / W

        dist = (freq_h.unsqueeze(1) ** 2 + freq_w.unsqueeze(0) ** 2).sqrt()
        dist = dist.unsqueeze(0).unsqueeze(0)

        cutoff = self.cutoff_logit.sigmoid() * 0.4 + 0.05
        low_mask = torch.sigmoid((cutoff - dist) * 20.0)

        low_fft = fft_x * low_mask
        low_freq = torch.fft.irfft2(low_fft, s=(H, W), norm='ortho')
        high_freq = x - low_freq

        return low_freq, high_freq

class FDMR(nn.Module):

    def __init__(self, dim: int = 3, expand_ratio: int = 2, **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_decomposer = FrequencyDecomposer(dim)
        hidden = dim * expand_ratio

        self.low_path = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.high_path = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = self.freq_decomposer(x)

        low_enhanced = low + self.low_path(low)
        high_enhanced = high + self.high_path(high)

        combined = torch.cat([low_enhanced, high_enhanced], dim=1)
        freq_weights = self.freq_attn(combined)
        w_low = freq_weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        w_high = freq_weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)

        merged = w_low * low_enhanced + w_high * high_enhanced

        return x + self.gamma * merged

class DynamicManifoldAggregation(nn.Module):

    def __init__(self, channels: int, max_modalities: int = 8, reduction: int = 2):
        super().__init__()
        self.channels = channels
        self.max_modalities = max_modalities

        mid_dim = max(4, channels // reduction)

        self.confidence_estimator = nn.Sequential(
            nn.Linear(channels, mid_dim),
            nn.SiLU(inplace=True),
            nn.Linear(mid_dim, 1)
        )

        self.recalibration_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1),
            nn.Sigmoid()
        )

        self.lambda_res = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, modal_feats: List[torch.Tensor]) -> torch.Tensor:
        if len(modal_feats) == 0:
            raise ValueError("At least one modality feature is required")

        if len(modal_feats) == 1:
            feat = modal_feats[0]
            scale = self.recalibration_gate(feat)
            return feat * scale

        B, C, H, W = modal_feats[0].shape
        num_modals = len(modal_feats)

        modal_scores = []
        for feat in modal_feats:
            descriptor = F.adaptive_avg_pool2d(feat, 1).view(B, C)
            score = self.confidence_estimator(descriptor)
            modal_scores.append(score)

        modal_scores = torch.cat(modal_scores, dim=1)
        modal_weights = F.softmax(modal_scores, dim=1)

        stacked_feats = torch.stack(modal_feats, dim=1)
        modal_weights = modal_weights.view(B, num_modals, 1, 1, 1)
        fused = (stacked_feats * modal_weights).sum(dim=1)

        channel_scale = self.recalibration_gate(fused)
        fused_enhanced = fused * channel_scale

        simple_mean = torch.stack(modal_feats, dim=0).mean(dim=0)
        residual_scale = self.lambda_res.clamp(0, 1)
        output = residual_scale * fused_enhanced + (1 - residual_scale) * simple_mean

        return output

class Enhancement(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels // 8, 1,
                      groups=in_channels // 8, bias=True),
            nn.Conv2d(in_channels // 8, in_channels // 8, 1, 1, 0,
                      groups=in_channels // 8, bias=False),
            nn.Conv2d(in_channels // 8, in_channels, 1,
                      groups=in_channels // 8, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + x

class EvidenceHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.K = num_classes
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1, bias=True),
        )

    def forward(self, x: torch.Tensor):
        e = F.softplus(self.proj(x))
        alpha = e + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        belief = e / (S + 1e-8)
        uncert = self.K / (S + 1e-8)
        return {'evidence': e, 'belief': belief, 'uncertainty': uncert}

class DSCombination(nn.Module):

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.K = num_classes
        self.ds_blend = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _ds_combine_two(b1, u1, b2, u2):
        b1_sum = b1.sum(dim=1, keepdim=True)
        b2_sum = b2.sum(dim=1, keepdim=True)
        cross_agree = (b1 * b2).sum(dim=1, keepdim=True)
        conflict = (b1_sum * b2_sum - cross_agree).clamp(min=0.0)
        Z = (1.0 - conflict).clamp(min=1e-8)
        b_c = (b1 * b2 + b1 * u2 + u1 * b2) / Z
        u_c = (u1 * u2) / Z
        return b_c, u_c

    def forward(self, evidence_list: List[dict]):
        if len(evidence_list) == 1:
            return evidence_list[0]

        b_ds = evidence_list[0]['belief']
        u_ds = evidence_list[0]['uncertainty']
        for i in range(1, len(evidence_list)):
            b_ds, u_ds = self._ds_combine_two(
                b_ds, u_ds,
                evidence_list[i]['belief'],
                evidence_list[i]['uncertainty']
            )

        e_sum = sum(ev['evidence'] for ev in evidence_list)
        alpha_sum = e_sum + 1.0
        S_sum = alpha_sum.sum(dim=1, keepdim=True)
        b_sum = e_sum / (S_sum + 1e-8)
        u_sum = self.K / (S_sum + 1e-8)

        blend = torch.sigmoid(self.ds_blend)
        b_fused = blend * b_ds + (1.0 - blend) * b_sum
        u_fused = blend * u_ds + (1.0 - blend) * u_sum

        return {
            'evidence': e_sum,
            'belief': b_fused,
            'uncertainty': u_fused,
        }

class AnisotropicEvidencePropagation(nn.Module):

    def __init__(self, feat_channels: int, num_classes: int = 2):
        super().__init__()
        self.K = num_classes

        self.dir_h = nn.Conv2d(feat_channels, feat_channels, (1, 7),
                               padding=(0, 3), groups=feat_channels, bias=False)
        self.dir_v = nn.Conv2d(feat_channels, feat_channels, (7, 1),
                               padding=(3, 0), groups=feat_channels, bias=False)

        self.prop_h = nn.Conv2d(num_classes, num_classes, (1, 7),
                                padding=(0, 3), groups=num_classes, bias=False)
        self.prop_v = nn.Conv2d(num_classes, num_classes, (7, 1),
                                padding=(3, 0), groups=num_classes, bias=False)
        self.prop_local = nn.Conv2d(num_classes, num_classes, 3, 1, 1,
                                    groups=num_classes, bias=False)

        self.prop_strength = nn.Parameter(torch.zeros(1))

    def forward(self, belief: torch.Tensor, uncert: torch.Tensor,
                feat: torch.Tensor):
        resp_h = self.dir_h(feat).abs().mean(dim=1, keepdim=True)
        resp_v = self.dir_v(feat).abs().mean(dim=1, keepdim=True)
        total = resp_h + resp_v + 1e-8
        w_h = resp_h / total
        w_v = resp_v / total

        prop = (w_h * self.prop_h(belief) +
                w_v * self.prop_v(belief) +
                0.3 * self.prop_local(belief))

        strength = torch.sigmoid(self.prop_strength)
        belief_prop = belief + strength * (prop - belief)
        uncert_prop = uncert * (1.0 - strength * 0.3)

        return belief_prop, uncert_prop

class MultiScaleEvidenceConsistency(nn.Module):

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.K = num_classes
        self.consistency_weight = nn.Parameter(torch.zeros(1))

    def forward(self, belief_fine: torch.Tensor, belief_coarse: torch.Tensor):
        prior = F.interpolate(
            belief_coarse, size=belief_fine.shape[2:],
            mode='bilinear', align_corners=False
        )
        w = torch.sigmoid(self.consistency_weight)
        return belief_fine * (1.0 - w) + belief_fine * prior * w + w * prior * 0.1

class EvidenceWeightedDecoder(nn.Module):

    def __init__(self, feat_channels: tuple):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        channels = feat_channels[::-1]

        for i in range(len(channels) - 1):
            self.up_blocks.append(self._make_up_block(channels[i], channels[i + 1]))

        self.skip_uncertainty_projs = nn.ModuleList()
        for i in range(len(channels) - 1):
            skip_ch = channels[i + 1]
            self.skip_uncertainty_projs.append(
                nn.Conv2d(1, skip_ch, 1, bias=True)
            )

        self.final_conv = nn.Sequential(
            nn.Conv2d(feat_channels[0], feat_channels[0] // 4, 3, 1, 1,
                      groups=feat_channels[0] // 4),
            nn.Conv2d(feat_channels[0] // 4, 32, 1,
                      groups=min(feat_channels[0] // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1),
            nn.Conv2d(16, 8, 1),
            nn.Conv2d(8, 1, 1),
            nn.Conv2d(1, 1, 1)
        )

    def _make_up_block(self, in_ch: int, skip_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, in_ch + skip_ch, 3, 1, 1,
                      groups=in_ch + skip_ch, bias=False),
            nn.Conv2d(in_ch + skip_ch, skip_ch // 8, 1,
                      groups=min(in_ch + skip_ch, skip_ch // 8), bias=True),
            nn.Conv2d(skip_ch // 8, skip_ch // 8, 1, 1, 0,
                      groups=skip_ch // 8, bias=False),
            nn.Conv2d(skip_ch // 8, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats: List[torch.Tensor],
                uncertainties: List[torch.Tensor]) -> torch.Tensor:
        feats_rev = feats[::-1]
        uncerts_rev = uncertainties[::-1]

        current = feats_rev[0]

        for i, up_block in enumerate(self.up_blocks):
            skip = feats_rev[i + 1]
            skip_uncert = uncerts_rev[i + 1]
            target_size = skip.shape[2:]

            certainty_gate = torch.sigmoid(
                self.skip_uncertainty_projs[i](1.0 - skip_uncert)
            )
            skip = skip * certainty_gate

            current = F.interpolate(current, size=target_size,
                                    mode='bilinear', align_corners=False)
            current = up_block(torch.cat([current, skip], dim=1))

        current = F.interpolate(current, size=(512, 512),
                                mode='bilinear', align_corners=False)
        out = self.final_conv(current)
        return out

class ETPF(nn.Module):

    def __init__(self, feat_channels: tuple = (256, 128, 32, 32),
                 num_classes: int = 2):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.scales = [(128, 128), (64, 64), (32, 32), (16, 16)]

        self.enhancements = nn.ModuleDict()
        for c in feat_channels:
            self.enhancements[str(c)] = Enhancement(c)

        self.evidence_heads = nn.ModuleList([
            EvidenceHead(c, num_classes) for c in feat_channels
        ])

        self.ds_combiners = nn.ModuleList([
            DSCombination(num_classes) for _ in feat_channels
        ])

        self.feat_fuse_convs = nn.ModuleList()
        for c in feat_channels:
            self.feat_fuse_convs.append(nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, groups=max(1, min(c // 4, 8)),
                          bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ))

        self.evidence_props = nn.ModuleList([
            AnisotropicEvidencePropagation(c, num_classes) for c in feat_channels
        ])

        self.evidence_consistency = nn.ModuleList([
            MultiScaleEvidenceConsistency(num_classes)
            for _ in range(len(feat_channels) - 1)
        ])

        self.decoder = EvidenceWeightedDecoder(feat_channels)

    def forward(self, backbone_outs: List[List[torch.Tensor]]) -> torch.Tensor:
        num_modalities = len(backbone_outs)
        num_scales = len(backbone_outs[0])

        fused_beliefs = []
        fused_uncerts = []
        fused_feats = []

        for s in range(num_scales):
            target_size = self.scales[s]
            ev_list = []
            feat_sum = None

            for m in range(num_modalities):
                feat = backbone_outs[m][s]
                _, C, _, _ = feat.shape
                feat_enh = self.enhancements[str(C)](feat)
                feat_enh = F.interpolate(feat_enh, size=target_size,
                                         mode='bilinear', align_corners=False)
                ev = self.evidence_heads[s](feat_enh)
                ev_list.append(ev)

                if feat_sum is None:
                    feat_sum = feat_enh
                else:
                    feat_sum = feat_sum + feat_enh

            feat_avg = feat_sum / max(num_modalities, 1)
            feat_fused = self.feat_fuse_convs[s](feat_avg)
            ds_out = self.ds_combiners[s](ev_list)

            fused_beliefs.append(ds_out['belief'])
            fused_uncerts.append(ds_out['uncertainty'])
            fused_feats.append(feat_fused)

        for s in range(num_scales):
            fused_beliefs[s], fused_uncerts[s] = self.evidence_props[s](
                fused_beliefs[s], fused_uncerts[s], fused_feats[s]
            )

        for s in range(num_scales - 1, 0, -1):
            fused_beliefs[s - 1] = self.evidence_consistency[s - 1](
                fused_beliefs[s - 1], fused_beliefs[s]
            )
            fused_uncerts[s - 1] = 1.0 - fused_beliefs[s - 1].sum(dim=1, keepdim=True)

        out = self.decoder(fused_feats, fused_uncerts)

        return out
