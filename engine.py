from typing import Iterable
import math
import torch
from tqdm import tqdm

def _reset_bn_stats(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if torch.isnan(module.running_mean).any() or torch.isnan(module.running_var).any():
                module.running_mean.zero_()
                module.running_var.fill_(1.0)

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args=None,
    logger=None,
    log_interval: int = 50,
    save_vis_interval: int = 9999
):
    model.train()
    criterion.train()
    model.to(args.device)
    criterion.to(args.device)

    if hasattr(model, 'set_train_mode'):
        model.set_train_mode(True)

    modal_num = len(args.modals)
    warmup_epochs = getattr(args, 'dualflow_warmup_epochs', 0)
    use_dual_flow = (epoch >= warmup_epochs)

    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Epoch {epoch}")
    epoch_losses = {'total': 0.0, 'seg_f': 0.0}
    if use_dual_flow:
        epoch_losses.update({'seg_p': 0.0, 'imd': 0.0, 'mgc': 0.0, 'fcl': 0.0})
    num_batches = 0

    for j, data in enumerate(data_loader):
        modal_imgs = []
        for i in range(modal_num):
            items = list(data.items())
            key, value = items[i]
            modal_imgs.append(value.to(args.device))

        targets = data['label'].to(args.device)
        modal_mask = data['modal_mask'].to(args.device) if 'modal_mask' in data else None

        if use_dual_flow:

            P_f, distill_outputs = model(
                modal_imgs, gt_mask=targets, return_distill_outputs=True,
                modal_mask=modal_mask
            )

            if torch.isnan(P_f).any():
                if logger:
                    logger.warning(f"Epoch {epoch} Batch {j}: NaN in model output, resetting BN stats and skipping")
                _reset_bn_stats(model)
                pbar.update(1)
                continue

            effective_distill = distill_outputs if (distill_outputs and 'P_p' in distill_outputs) else None
            losses = criterion(P_f, targets.float(), effective_distill)
            loss_final = losses['total']

            for key in ['seg_f', 'seg_p', 'imd', 'mgc', 'fcl']:
                if key in losses:
                    val = losses[key]
                    epoch_losses[key] = epoch_losses.get(key, 0.0) + (
                        val.item() if isinstance(val, torch.Tensor) else val
                    )

        else:

            output_fused, extras = model(modal_imgs, gt_mask=targets, modal_mask=modal_mask)

            if torch.isnan(output_fused).any():
                if logger:
                    logger.warning(f"Epoch {epoch} Batch {j}: NaN in model output, resetting BN stats and skipping")
                _reset_bn_stats(model)
                pbar.update(1)
                continue

            losses = criterion(output_fused, targets.float())
            loss_final = losses['total']
            epoch_losses['seg_f'] += losses.get('seg_f', torch.tensor(0.0)).item()

        epoch_losses['total'] += loss_final.item()
        num_batches += 1

        if j % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            if use_dual_flow:
                logger.info(
                    f"Epoch {epoch} | Batch {j}/{len(data_loader.dataloader)} | "
                    f"Loss: {loss_final.item():.4f} | Seg_f: {losses.get('seg_f', 0):.4f} | "
                    f"Seg_p: {losses.get('seg_p', 0):.4f} | "
                    f"IMD: {losses.get('imd', 0):.4f} | MGC: {losses.get('mgc', 0):.4f} | "
                    f"FCL: {losses.get('fcl', 0):.4f} | LR: {lr:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch} [Warmup] | Batch {j}/{len(data_loader.dataloader)} | "
                    f"Loss: {loss_final.item():.4f} | LR: {lr:.6f}"
                )

        optimizer.zero_grad()

        if torch.isnan(loss_final) or torch.isinf(loss_final):
            logger.warning(f"Epoch {epoch} Batch {j}: NaN/Inf loss detected, skipping batch")
            continue
        loss_final.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        pbar.set_postfix({
        })
        pbar.update(1)

    pbar.close()

    if num_batches > 0:
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        logger.info("=" * 60)
        phase_str = "Dual-Flow" if use_dual_flow else "Warmup"
        logger.info(f"Epoch {epoch} Summary ({phase_str}):")
        logger.info(f"  Avg Total Loss: {avg_losses['total']:.4f}")
        logger.info(f"  Avg Seg_f Loss: {avg_losses['seg_f']:.4f}")
        if use_dual_flow:
            logger.info(f"  Avg Seg_p Loss: {avg_losses.get('seg_p', 0):.4f}")
            logger.info(f"  Avg IMD Loss: {avg_losses.get('imd', 0):.4f}")
            logger.info(f"  Avg MGC Loss: {avg_losses.get('mgc', 0):.4f}")
            logger.info(f"  Avg FCL Loss: {avg_losses.get('fcl', 0):.4f}")
        logger.info("=" * 60)
