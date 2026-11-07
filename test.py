import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from datasets import create_dataset
from eval.evaluate import eval
from util.logger import get_logger

def get_args_parser():
    parser = argparse.ArgumentParser('Inference for Multi-Modal Crack Segmentation', add_help=False)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', default="your_path_to_data", help='path to images')
    parser.add_argument('--batch_size', type=int, default=1, help='test input batch size')
    parser.add_argument('--modals', nargs='+', default=['RGB', 'aop', 'dop'], help='modal name for process')
    parser.add_argument('--miss_rates', nargs='+', type=float, default=[0.0, 0.5, 0.5], help='Missing rates for each modal data during testing')
    parser.add_argument('--output_dir', default='your_path_to_output', help='save path for results')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--save_predictions', action='store_true', default=True, help='Save prediction images')
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--load_width', type=int, default=512)
    parser.add_argument('--load_height', type=int, default=512)
    parser.add_argument('--visualize_completion', action='store_true', default=False, help='Visualize modality completion results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')

    return parser

def load_model(checkpoint_path: str, args):
    from models.build import DualFlowDecoder
    from models.Needle import Needle

    device = torch.device(args.device)

    backbones = []
    for i in range(len(args.modals)):
        backbone = Needle(
            img_size=512,
            patch_size=[4, 4, 4, 4],
            in_channels=3,
            out_indices=[0, 1, 2, 3],
            drop_rate=0.2,
            embed_dims=256,
            depth=4
        )
        backbones.append(backbone.to(device))

    model = DualFlowDecoder(backbones, args)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model

def inference_single(model, modal_imgs, args, modal_mask=None):
    with torch.no_grad():
        output = model(modal_imgs, modal_mask=modal_mask)
        prob = torch.sigmoid(output)
        pred = (prob > args.threshold).float()

    return output, prob, pred

def visualize_results(
    modal_imgs: list,
    pred: torch.Tensor,
    target: torch.Tensor,
    save_path: str,
    modal_names: list
):
    import matplotlib.pyplot as plt

    num_modals = len(modal_imgs)
    fig, axes = plt.subplots(1, num_modals + 2, figsize=(4 * (num_modals + 2), 4))
    for i, (img, name) in enumerate(zip(modal_imgs, modal_names)):
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        if img.abs().sum() < 1e-6:
            axes[i].text(0.5, 0.5, 'MISSING', ha='center', va='center',
                        fontsize=20, color='red', transform=axes[i].transAxes)
            axes[i].set_facecolor('lightgray')
        else:
            axes[i].imshow(img_np)
        axes[i].set_title(name)
        axes[i].axis('off')

    pred_np = pred[0, 0].cpu().numpy()
    axes[num_modals].imshow(pred_np, cmap='gray')
    axes[num_modals].set_title('Prediction')
    axes[num_modals].axis('off')
    target_np = target[0, 0].cpu().numpy()
    axes[num_modals + 1].imshow(target_np, cmap='gray')
    axes[num_modals + 1].set_title('Ground Truth')
    axes[num_modals + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_test = get_logger(str(output_dir), 'test')

    log_test.info("=" * 60)
    log_test.info("Inference Mode")
    log_test.info("=" * 60)
    log_test.info(f"Checkpoint: {args.checkpoint}")
    log_test.info(f"Dataset: {args.dataset_path}")
    log_test.info(f"Modalities: {args.modals}")
    log_test.info(f"Missing rates: {args.miss_rates}")
    log_test.info("=" * 60)

    model = load_model(args.checkpoint, args)
    args.batch_size_test = args.batch_size
    test_dataloader = create_dataset(args)

    print(f"Number of test images: {len(test_dataloader)}")
    log_test.info(f"Number of test images: {len(test_dataloader)}")
    results_dir = output_dir / 'predictions'
    results_dir.mkdir(exist_ok=True)

    if args.visualize_completion:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

    pbar = tqdm(total=len(test_dataloader), desc="Inference")
    modal_num = len(args.modals)

    all_preds = []
    all_targets = []

    for batch_idx, data in enumerate(test_dataloader):
        modal_imgs = []
        for i in range(modal_num):
            items = list(data.items())
            key, value = items[i]
            modal_imgs.append(value.to(args.device))

        target = data['label'].to(args.device)
        modal_mask = data['modal_mask'].to(args.device) if 'modal_mask' in data else None
        image_name = Path(data['image_path'][0]).stem
        output, prob, pred = inference_single(model, modal_imgs, args, modal_mask=modal_mask)

        if args.save_predictions:
            pred_np = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(results_dir / f"{image_name}_pred.png"), pred_np)
            prob_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(results_dir / f"{image_name}_prob.png"), prob_np)
            target_np = (target[0, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(results_dir / f"{image_name}_gt.png"), target_np)

        if args.visualize_completion:
            visualize_results(
                modal_imgs, prob, target.float(),
                str(vis_dir / f"{image_name}_vis.png"),
                args.modals
            )

        all_preds.append(pred_np)
        all_targets.append(target_np)

        pbar.update(1)

    pbar.close()

    log_test.info("Computing metrics...")
    metrics = eval(log_test, str(results_dir), 0)

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        log_test.info(f"{key}: {value}")
    print("=" * 60)

    log_test.info("Inference completed!")

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test Multi-Modal Crack Segmentation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
