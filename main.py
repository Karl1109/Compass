import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval
from util.logger import get_logger

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Compass — Multi-Modal Crack Segmentation',
        add_help=False
    )

    parser.add_argument('--BCELoss_ratio', default=1.0, type=float)
    parser.add_argument('--DiceLoss_ratio', default=1.0, type=float)
    parser.add_argument('--dataset_path', default="your_dataset_path", type=str)
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--modals', nargs='+', default=['RGB', 'aop', 'dop'])
    parser.add_argument('--miss_rates', nargs='+', default=[0.0, 0.7, 0.7], type=float)
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR', choices=['StepLR', 'CosLR', 'PolyLR'])
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='checkpoints/weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load_width', type=int, default=512)
    parser.add_argument('--load_height', type=int, default=512)
    parser.add_argument('--dualflow_warmup_epochs', default=5, type=int,
                        help='Epochs of warmup before enabling dual-flow')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm for clipping')
    parser.add_argument('--fpmd_weight', default=1.0, type=float,
                        help='Weight for FPMD mutual distillation loss')
    parser.add_argument('--fpmd_tau', default=2.0, type=float,
                        help='Temperature for FPMD soft targets')
    parser.add_argument('--fpmd_alpha', default=0.3, type=float,
                        help='Backward KD weight in FPMD')
    parser.add_argument('--dgsc_weight', default=0.5, type=float,
                        help='Weight for DGSC structural consistency loss')
    parser.add_argument('--dgsc_erosion_iters', default=3, type=int,
                        help='Soft erosion iterations for skeleton approximation')
    parser.add_argument('--recon_weight', default=1.0, type=float,
                        help='Weight for FAPT feature reconstruction loss')
    parser.add_argument('--seg_p_weight', default=1.0, type=float,
                        help='Weight for partial stream segmentation loss')
    parser.add_argument('--fapt_num_prototypes', default=8, type=int,
                        help='Number of semantic prototypes in FAPT')
    parser.add_argument('--log_interval', type=int, default=50)

    return parser

def main(args):
    checkpoints_path = "checkpoints"
    curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = args.dataset_path.split('/')[-1]
    modals_name = ''.join(f'_{modal}' for modal in args.modals)
    modals_miss_ratio = '+'.join(str(rate) for rate in args.miss_rates)
    process_folder_path = os.path.join(
        checkpoints_path,
        f'{curTime}_Dataset_{dataset_name}_modals_{modals_name}_missratio_{modals_miss_ratio}_Compass'
    )

    if not os.path.exists(process_folder_path):
        os.makedirs(process_folder_path)

    log_train = get_logger(process_folder_path, 'train')
    log_val = get_logger(process_folder_path, 'val')
    log_eval = get_logger(process_folder_path, 'eval')

    log_train.info(f"args _ {args}")
    log_train.info(f"dataset _ {args.dataset_path}")
    log_train.info(f"processing modals _ {args.modals}")
    log_train.info("=" * 60)
    log_train.info("Compass Dual-Flow Distillation Configuration:")
    log_train.info(f"  Warmup epochs: {args.dualflow_warmup_epochs}")
    log_train.info(f"  Max grad norm: {args.max_grad_norm}")
    log_train.info(f"  FPMD weight: {args.fpmd_weight}, tau: {args.fpmd_tau}, alpha: {args.fpmd_alpha}")
    log_train.info(f"  DGSC weight: {args.dgsc_weight}")
    log_train.info(f"  Recon weight: {args.recon_weight}")
    log_train.info(f"  FAPT prototypes: {args.fapt_num_prototypes}")
    log_train.info("=" * 60)

    modal_num = len(args.modals)
    print(f'Processing modals: {args.modals}')
    print(f'Compass Dual-Flow Distillation enabled')

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print(f'Number of training images: {dataset_size}')
    log_train.info(f'Number of training images: {dataset_size}')

    param_dicts = [{
        "params": [p for n, p in model.named_parameters() if p.requires_grad],
        "lr": args.lr,
    }]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-5
        )
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr,
                              begin=args.start_epoch, end=args.epochs)

    output_dir = (
        f'{args.output_dir}/{curTime}_Dataset_{dataset_name}'
        f'_modals_{modals_name}_missratio_{modals_miss_ratio}_Compass'
    )
    if args.output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Starting training!")
    log_train.info("Starting training!")
    start_time = time.time()
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'F1': 0}

    for epoch in range(args.start_epoch, args.epochs):
        args.phase = 'train'
        print("-" * 87)
        print(f"Training epoch start _ {epoch}")

        train_one_epoch(
            model, criterion, train_dataLoader, optimizer, epoch, args, log_train,
            log_interval=args.log_interval
        )

        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        print(f"Training epoch finish _ {epoch}")
        print("-" * 87)

        print(f"Validation epoch start _ {epoch}")
        results_path = (
            f'{curTime}_Dataset_{dataset_name}_modal_{modals_name}'
            f'_missratio_{modals_miss_ratio}_Compass'
        )
        save_root = f'results/{results_path}/results_{epoch}'
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        val_dataloader = create_dataset(args)
        pbar = tqdm(total=len(val_dataloader), desc="Validation")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)

        if hasattr(model, 'set_train_mode'):
            model.set_train_mode(False)

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                modal_imgs = []
                for i in range(modal_num):
                    items = list(data.items())
                    key, value = items[i]
                    modal_imgs.append(value.to(torch.device(args.device)))

                target = data["label"]
                target = target.to(dtype=torch.int64).cuda()

                modal_mask = data['modal_mask'].to(torch.device(args.device)) if 'modal_mask' in data else None
                root_name = data["image_path"][0].split("/")[-1][0:-4]
                out = model(modal_imgs, modal_mask=modal_mask)

                loss = criterion.bce_dice(out, target.float())

                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                target = 255 * (target / np.max(target)) if np.max(target) > 0 else target
                out = 255 * (out / np.max(out)) if np.max(out) > 0 else out

                log_val.info(f"loss _ {loss}")
                cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), target)
                cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

        pbar.close()
        log_val.info(f"Model epoch {epoch} validation finish!")

        print(f"Evaluation epoch start _ {epoch}")
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(f'{key} _ {value}')

        if metrics['F1'] > max_Metrics['F1']:
            max_Metrics['epoch'] = epoch
            max_Metrics['mIoU'] = metrics['mIoU']
            max_Metrics['F1'] = metrics['F1']

            if args.output_dir:
                best_checkpoint_path = output_dir / 'checkpoint_best.pth'
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_metrics': max_Metrics,
                }, best_checkpoint_path)
                print(f"*** New best model at epoch {epoch} with F1={metrics['F1']:.4f} ***")

        print(f'Max F1 _ {max_Metrics["F1"]} | Max Epoch _ {max_Metrics["epoch"]}')
        print("-" * 87)

        log_eval.info(f'Max F1 _ {max_Metrics["F1"]} | Max Epoch _ {max_Metrics["epoch"]}')

    for key, value in max_Metrics.items():
        log_eval.info(f'{key} _ {value}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    log_train.info(f'Training time: {total_time_str}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Compass Multi-Modal Crack Segmentation',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
