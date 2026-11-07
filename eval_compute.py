from thop import profile
import torch
import argparse
from main import get_args_parser
from models.build import build, DualFlowDecoder
from models.ETPF import FDMR, FrequencyDecomposer
from models.DSD import FAPT, ScaleFAPT

def MoRe_count(module, input, output):
    x = input[0]
    if x.dim() == 4:
        B, C, H, W = x.shape
        total_pixels = B * H * W
    elif x.dim() == 3:
        B, N, C = x.shape
        total_pixels = B * N
    else:
        total_pixels = x.numel() // x.shape[-1]
        C = x.shape[-1]

    out_c = output.shape[1] if output.dim() >= 4 else output.shape[-1]
    flops = C * out_c * total_pixels
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += flops

def freq_decomposer_count(module, input, output):
    x = input[0]
    B, C, H, W = x.shape

    N = H * W
    import math
    fft_flops = B * C * N * math.log2(max(N, 1)) * 2
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += int(fft_flops)

def scale_fapt_count(module, input, output):
    x = input[0]
    B, C, H, W = x.shape
    K = module.num_prototypes

    einsum_flops = B * K * H * W * C
    if not hasattr(module, '__flops__'):
        module.__flops__ = 0
    module.__flops__ += einsum_flops

custom_ops = {
    FDMR: MoRe_count,
    FrequencyDecomposer: freq_decomposer_count,
    ScaleFAPT: scale_fapt_count,
}

def compute_flops_params(args):
    model, _ = build(args)
    model.to(args.device)
    model.eval()
    model.set_train_mode(False)

    num_modals = len(args.modals)
    inputs = [torch.randn(1, 3, 512, 512).to(args.device) for _ in range(num_modals)]

    with torch.no_grad():
        flops, params = profile(model, (inputs,), custom_ops=custom_ops, verbose=False)

    return flops, params

def analyze_module_flops(model, args):
    device = args.device
    num_modals = len(args.modals)

    print("\n" + "=" * 60)
    print("Module-wise FLOPs Analysis")
    print("=" * 60)

    fdmr_input = torch.randn(1, 3, 512, 512).to(device)
    fdmr_flops, fdmr_params = profile(
        model.FDMR, (fdmr_input,), custom_ops=custom_ops, verbose=False
    )
    print(f"  FDMR: {fdmr_flops / 1e9:.4f} G FLOPs, {fdmr_params / 1e6:.4f} M params")

    backbone_input = torch.randn(1, 3, 512, 512).to(device)
    context = torch.randn(1, 128 * 128, 64).to(device)
    backbone_flops, backbone_params = profile(
        model.backbones[0], (backbone_input, context), verbose=False
    )
    print(f"  Single Backbone (Needle): {backbone_flops / 1e9:.4f} G FLOPs, {backbone_params / 1e6:.4f} M params")
    print(f"  All Backbones ({num_modals}x): {backbone_flops * num_modals / 1e9:.4f} G FLOPs")

    dummy_scales = [
        (1, 256, 256, 256),
        (1, 128, 64, 64),
        (1, 32, 64, 64),
        (1, 32, 64, 64),
    ]

    dummy_feats = []
    for _ in range(num_modals):
        modal_f = [torch.randn(*s).to(device) for s in dummy_scales]
        dummy_feats.append(modal_f)

    modal_avail = [True] * num_modals
    if num_modals > 1:
        modal_avail[0] = False
        dummy_feats[0] = [torch.zeros(*s).to(device) for s in dummy_scales]

    try:

        with torch.no_grad():
            fapt_params = sum(p.numel() for p in model.FAPT.parameters())

            print(f"  FAPT: ~{fapt_params / 1e6:.4f} M params (prototype completion)")
    except Exception:
        print(f"  FAPT: (profiling skipped)")

    try:
        etpf_params = sum(p.numel() for p in model.ETPF.parameters())
        print(f"  ETPF: {etpf_params / 1e6:.4f} M params")
    except Exception:
        print(f"  ETPF: (profiling skipped)")

    try:
        dma_params = sum(p.numel() for p in model.modal_fusion.parameters())
        print(f"  DynamicManifoldAggregation: {dma_params / 1e6:.6f} M params")
    except Exception:
        pass

    total_flops = fdmr_flops * num_modals + backbone_flops * num_modals
    print(f"\n  Estimated main FLOPs: {total_flops / 1e9:.2f} G "
          f"(FDMR×{num_modals} + Backbone×{num_modals})")

def main():
    parser = argparse.ArgumentParser(
        'Compass Model Complexity Analysis',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if not hasattr(args, 'modals') or args.modals is None:
        args.modals = ['RGB', 'dep']

    print("=" * 60)
    print("Compass Model Complexity Analysis")
    print("=" * 60)
    print(f"Modalities: {args.modals}")
    print(f"Input size: 512 x 512")
    print(f"Note: FPMD/DGSC losses are NOT measured (training only)")
    print(f"Note: FAPT is included (used at inference for missing modality)")
    print("=" * 60)

    flops, params = compute_flops_params(args)

    print(f"\nResults:")
    print(f"  FLOPs: {flops / 1e9:.4f} G")
    print(f"  Params: {params / 1e6:.4f} M")
    print("=" * 60)

    model, _ = build(args)
    model.to(args.device)
    model.eval()
    model.set_train_mode(False)
    analyze_module_flops(model, args)

    print("\nComplexity with different modality configurations:")
    for num_modals in [1, 2, 3]:
        args.modals = ['RGB', 'dep', 'aop'][:num_modals]
        args.miss_rates = [0.0] * num_modals
        flops, params = compute_flops_params(args)
        print(f"  {num_modals} modalities: {flops / 1e9:.4f} GFLOPs, {params / 1e6:.4f} M params")

if __name__ == '__main__':
    main()
