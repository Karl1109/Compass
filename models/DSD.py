import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

class ScaleFAPT(nn.Module):

    def __init__(self, channels: int, num_prototypes: int = 8):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes

        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels) * 0.02
        )

        self.spatial_assign = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=max(1, channels // 4), bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, num_prototypes, 1, bias=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1,
                      groups=max(1, channels // 4), bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, available_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = available_feat.shape

        assign_logits = self.spatial_assign(available_feat)
        assign_weights = F.softmax(assign_logits, dim=1)

        protos = self.prototypes
        reconstructed = torch.einsum('bkhw,kc->bchw', assign_weights, protos)

        combined = torch.cat([reconstructed, available_feat], dim=1)
        refined = self.refine(combined)

        g = self.gate(refined)
        g = g.unsqueeze(-1).unsqueeze(-1)
        output = reconstructed + g * refined

        return output

class FAPT(nn.Module):

    def __init__(
        self,
        feat_channels: tuple = (256, 128, 32, 32),
        num_prototypes: int = 8
    ):
        super().__init__()
        self.num_scales = len(feat_channels)
        self.scale_modules = nn.ModuleList([
            ScaleFAPT(c, num_prototypes) for c in feat_channels
        ])

    def forward(
        self,
        all_backbone_feats: List[List[torch.Tensor]],
        modal_available: List[bool]
    ) -> List[List[torch.Tensor]]:
        num_modalities = len(all_backbone_feats)
        completed_feats = [list(feats) for feats in all_backbone_feats]

        available_idxs = [i for i, a in enumerate(modal_available) if a]
        missing_idxs = [i for i, a in enumerate(modal_available) if not a]

        if not missing_idxs or not available_idxs:
            return completed_feats

        for scale_idx in range(self.num_scales):
            avail_feats = [all_backbone_feats[i][scale_idx] for i in available_idxs]
            avg_available = torch.stack(avail_feats, dim=0).mean(dim=0)
            completed = self.scale_modules[scale_idx](avg_available)
            for miss_idx in missing_idxs:
                completed_feats[miss_idx][scale_idx] = completed

        return completed_feats

    def compute_fcl(
        self,
        all_backbone_feats: List[List[torch.Tensor]],
        completed_feats: List[List[torch.Tensor]],
        partial_mask: List[bool],
        original_available: Optional[List[bool]] = None
    ) -> torch.Tensor:
        if original_available is None:
            original_available = [True] * len(partial_mask)

        newly_masked = [
            i for i in range(len(partial_mask))
            if original_available[i] and not partial_mask[i]
        ]
        if not newly_masked:
            return torch.tensor(0.0, device=all_backbone_feats[0][0].device)

        fcl_loss = torch.tensor(0.0, device=all_backbone_feats[0][0].device)
        count = 0

        for miss_idx in newly_masked:
            for scale_idx in range(self.num_scales):
                real = all_backbone_feats[miss_idx][scale_idx].detach()
                pred = completed_feats[miss_idx][scale_idx]
                fcl_loss = fcl_loss + F.mse_loss(pred, real)
                count += 1

        if count > 0:
            fcl_loss = fcl_loss / count

        return fcl_loss

class IMDLoss(nn.Module):

    def __init__(self, tau: float = 2.0, alpha: float = 0.3):
        super().__init__()
        self.tau = tau
        self.alpha = alpha

    def forward(self, P_f: torch.Tensor, P_p: torch.Tensor) -> torch.Tensor:
        soft_f = torch.sigmoid(P_f.detach() / self.tau)
        soft_p = torch.sigmoid(P_p.detach() / self.tau)

        forward_kd = F.binary_cross_entropy_with_logits(
            P_p / self.tau, soft_f, reduction='mean'
        ) * (self.tau ** 2)

        backward_kd = F.binary_cross_entropy_with_logits(
            P_f / self.tau, soft_p, reduction='mean'
        ) * (self.tau ** 2)

        return forward_kd + self.alpha * backward_kd

class MGCLoss(nn.Module):

    def __init__(self, erosion_iters: int = 3):
        super().__init__()
        self.erosion_iters = erosion_iters

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _soft_skeleton(self, logits: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(logits)
        skeleton = soft.clone()
        for _ in range(self.erosion_iters):
            eroded = -F.max_pool2d(-skeleton, 3, 1, 1)
            skeleton = skeleton - eroded
        return skeleton

    def _gradient_directions(self, prob: torch.Tensor):
        gx = F.conv2d(prob, self.sobel_x, padding=1)
        gy = F.conv2d(prob, self.sobel_y, padding=1)
        mag_sq = gx ** 2 + gy ** 2
        magnitude = (mag_sq + 1e-8).sqrt()
        nx = gx / magnitude
        ny = gy / magnitude
        return magnitude, nx, ny

    def forward(self, P_f: torch.Tensor, P_p: torch.Tensor) -> torch.Tensor:
        skel_f = self._soft_skeleton(P_f)
        skel_p = self._soft_skeleton(P_p)
        skel_loss = F.l1_loss(skel_p, skel_f.detach())

        prob_f = torch.sigmoid(P_f)
        prob_p = torch.sigmoid(P_p)

        mag_f, nx_f, ny_f = self._gradient_directions(prob_f)
        _, nx_p, ny_p = self._gradient_directions(prob_p)

        cos_sim = nx_f.detach() * nx_p + ny_f.detach() * ny_p
        direction_inconsistency = (1.0 - cos_sim).clamp(min=0.0)
        grad_loss = (direction_inconsistency * mag_f.detach()).mean()

        return skel_loss + grad_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super().__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

class BCE_Dice(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, pred, true):
        BCE = self.bce_fn(pred, true)
        Dice = self.dice_fn(pred.sigmoid(), true)
        return self.args.BCELoss_ratio * BCE + self.args.DiceLoss_ratio * Dice

class MILoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, P_f: torch.Tensor, P_p: torch.Tensor) -> torch.Tensor:

        prob_f = torch.sigmoid(P_f)
        prob_p = torch.sigmoid(P_p)
        m = 0.5 * (prob_f + prob_p)

        eps = 1e-6
        prob_f = prob_f.clamp(eps, 1 - eps)
        prob_p = prob_p.clamp(eps, 1 - eps)
        m = m.clamp(eps, 1 - eps)
        kl_f = (prob_f * (prob_f / m).log()).mean()
        kl_p = (prob_p * (prob_p / m).log()).mean()
        js = 0.5 * (kl_f + kl_p)
        return js

class DualFlowCriterion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bce_dice = BCE_Dice(args)

        self.imd = IMDLoss(
            tau=getattr(args, 'fpmd_tau', 2.0),
            alpha=getattr(args, 'fpmd_alpha', 0.3)
        )
        self.mgc = MGCLoss(
            erosion_iters=getattr(args, 'dgsc_erosion_iters', 3)
        )
        self.mi = MILoss()

        self.imd_weight = getattr(args, 'fpmd_weight', 1.0)
        self.mgc_weight = getattr(args, 'dgsc_weight', 0.5)
        self.recon_weight = getattr(args, 'recon_weight', 1.0)
        self.seg_p_weight = getattr(args, 'seg_p_weight', 1.0)

    def forward(
        self,
        P_f: torch.Tensor,
        target: torch.Tensor,
        distill_outputs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        losses['seg_f'] = self.bce_dice(P_f, target)

        if distill_outputs is not None:
            P_p = distill_outputs['P_p']
            losses['seg_p'] = self.seg_p_weight * self.bce_dice(P_p, target)
            losses['imd'] = self.imd_weight * self.imd(P_f, P_p)
            losses['mgc'] = self.mgc_weight * self.mgc(P_f, P_p)

            fcl_loss = distill_outputs.get('fcl_loss', torch.tensor(0.0))
            if isinstance(fcl_loss, torch.Tensor):
                losses['fcl'] = self.recon_weight * fcl_loss

        losses['total'] = sum(losses.values())
        return losses
