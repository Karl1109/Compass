import random
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, List, Dict

from models.ETPF import ETPF, FDMR, DynamicManifoldAggregation
from models.Needle import Needle
from models.DSD import FAPT, DualFlowCriterion

class DualFlowDecoder(nn.Module):

    def __init__(self, backbones, args=None):
        super().__init__()
        self.args = args
        self.num_modalities = len(args.modals)
        self.training_mode = True

        self.FDMR = FDMR(dim=3, expand_ratio=2)

        self.backbones = nn.ModuleList(backbones)

        self.modal_fusion = DynamicManifoldAggregation(
            channels=3, max_modalities=self.num_modalities, reduction=2
        )
        self.conv_3_to_64 = nn.Conv2d(3, 64, 1)

        self.ETPF = ETPF(feat_channels=(256, 128, 32, 32))

        self.FAPT = FAPT(
            feat_channels=(256, 128, 32, 32),
            num_prototypes=getattr(args, 'fapt_num_prototypes', 8)
        )

    def set_train_mode(self, mode: bool = True):
        self.training_mode = mode

    def _random_partial_mask(self, modal_available: List[bool]) -> List[bool]:
        available_indices = [i for i, a in enumerate(modal_available) if a]
        if len(available_indices) < 2:
            return list(modal_available)
        mask = list(modal_available)
        mask_idx = random.choice(available_indices)
        mask[mask_idx] = False
        return mask

    def _make_zero_feats(self, reference_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.zeros_like(f) for f in reference_feats]

    def forward(
        self,
        modal_imgs: List[torch.Tensor],
        gt_mask: Optional[torch.Tensor] = None,
        return_distill_outputs: bool = False,
        modal_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        B = modal_imgs[0].shape[0]

        if modal_mask is not None:
            modal_available = [bool(modal_mask[0, i].item() > 0.5) for i in range(modal_mask.shape[1])]
        else:
            modal_available = [True] * len(modal_imgs)

        enhanced_imgs = []
        for i, (img, avail) in enumerate(zip(modal_imgs, modal_available)):
            if avail:
                enhanced_imgs.append(self.FDMR(img))
            else:
                enhanced_imgs.append(img)

        available_enhanced = [e for e, a in zip(enhanced_imgs, modal_available) if a]
        if available_enhanced:
            fused = self.modal_fusion(available_enhanced)
        else:
            fused = torch.zeros_like(modal_imgs[0])
        context = self.conv_3_to_64(fused)
        _, C_ctx, H_ctx, W_ctx = context.shape
        context = context.reshape(B, C_ctx, H_ctx * W_ctx).permute(0, 2, 1)

        all_backbone_feats = [None] * len(modal_imgs)
        for i, (img, avail) in enumerate(zip(enhanced_imgs, modal_available)):
            if avail:
                all_backbone_feats[i] = self.backbones[i](img, context)

        ref_feats = next((f for f in all_backbone_feats if f is not None), None)
        if ref_feats is None:

            ref_feats = self.backbones[0](enhanced_imgs[0], context)
        for i in range(len(modal_imgs)):
            if all_backbone_feats[i] is None:
                all_backbone_feats[i] = self._make_zero_feats(ref_feats)

        if all(modal_available):
            full_feats = all_backbone_feats
        else:
            full_feats = self.FAPT(all_backbone_feats, modal_available)
        P_f = self.ETPF(full_feats)

        num_available = sum(modal_available)
        if self.training_mode and return_distill_outputs and num_available >= 2:

            partial_mask = self._random_partial_mask(modal_available)

            partial_feats = []
            for i in range(self.num_modalities):
                if partial_mask[i]:
                    partial_feats.append(list(all_backbone_feats[i]))
                else:
                    partial_feats.append(self._make_zero_feats(all_backbone_feats[i]))

            completed_feats = self.FAPT(partial_feats, partial_mask)

            P_p = self.ETPF(completed_feats)

            fcl_loss = self.FAPT.compute_fcl(
                all_backbone_feats, completed_feats, partial_mask,
                original_available=modal_available
            )

            distill_outputs = {
                'P_f': P_f,
                'P_p': P_p,
                'partial_mask': partial_mask,
                'fcl_loss': fcl_loss,
            }
            return P_f, distill_outputs

        if return_distill_outputs:
            return P_f, {}

        if gt_mask is not None:
            return P_f, {}

        return P_f

def build(args):
    device = torch.device(args.device)
    args.device = device
    modals = args.modals

    backbones = []

    for i in range(len(modals)):
        backbone = Needle(
            img_size=512,
            patch_size=[4, 4, 4, 4],
            in_channels=3,
            out_indices=[0, 1, 2, 3],
            drop_rate=0.2,
            embed_dims=256,
            depth=4,
        )
        backbones.append(backbone.cuda())

    model = DualFlowDecoder(backbones, args)

    criterion = DualFlowCriterion(args)
    criterion.to(device)

    return model, criterion
