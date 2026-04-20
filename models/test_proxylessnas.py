"""
Test script for ProxylessNAS-Mobile model adaptation.
Tests the full pipeline: FP32 load → PTQ quantize → Extract layers → Simulate INT8

This file lives in models/ as a standalone test, NOT modifying any existing code.

Usage:
    cd /home/kyonqi/Project/RustProjects/Python_Sim_Infer
    python -m models.test_proxylessnas
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.protocol import LayerConfig, LayerType, QuantParams


# ============================================================
# Step 1: Load ProxylessNAS-Mobile FP32 model via torch.hub
# ============================================================

def load_proxylessnas_fp32() -> nn.Module:
    """Load ProxylessNASNets-Mobile from MIT-HAN-Lab via torch.hub."""
    print("=" * 60)
    print("Step 1: Loading ProxylessNAS-Mobile (FP32) via torch.hub")
    print("=" * 60)
    model = torch.hub.load(
        'mit-han-lab/proxylessnas', 'proxyless_mobile',
        pretrained=True, trust_repo=True, verbose=False
    )
    model.eval()
    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Total blocks: {len(model.blocks)}")
    return model


# ============================================================
# Step 2: FP32 Layer Extractor  (Conv+BN fuse → FP32 weights)
# ============================================================

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d (returns numpy w, b)."""
    with torch.no_grad():
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        scale = gamma / torch.sqrt(var + bn.eps)
        view_shape = [-1] + [1] * (w.ndim - 1)
        w_fused = w * scale.view(view_shape)
        b_fused = (b - mean) * scale + beta
    return w_fused.cpu().numpy(), b_fused.cpu().numpy()


def extract_proxylessnas_fp32_layers(model) -> list:
    """
    Extract FP32 layers from ProxylessNAS-Mobile.
    
    ProxylessNAS structure:
        first_conv: ConvLayer (Conv2d + BN + ReLU6)
        blocks[0..21]: MobileInvertedResidualBlock
            - mobile_inverted_conv: MBInvertedConvLayer or ZeroLayer
                - inverted_bottleneck (optional): Sequential(conv, bn, relu)
                - depth_conv: Sequential(conv, bn, relu)
                - point_linear: Sequential(conv, bn)   ← no activation!
            - shortcut: IdentityLayer or None
        feature_mix_layer: ConvLayer (Conv2d + BN + ReLU6)
        classifier: LinearLayer (Linear)
    
    Returns: list of (LayerConfig, weights_np, bias_np)
    """
    sim_layers = []

    def add(conv: nn.Conv2d, bn: nn.BatchNorm2d, name: str,
            res_add=None, res_conn=None):
        w, b = fuse_conv_bn(conv, bn)
        if conv.groups == conv.in_channels and conv.in_channels > 1:
            l_type = LayerType.DEPTHWISE
        elif conv.kernel_size[0] == 1:
            l_type = LayerType.POINTWISE
        else:
            l_type = LayerType.CONV2D

        cfg = LayerConfig(
            name=name, type=l_type,
            in_channels=conv.in_channels, out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0], stride=conv.stride[0],
            padding=conv.padding[0], groups=conv.groups,
            residual_add_to=res_add, residual_connect_from=res_conn,
        )
        sim_layers.append((cfg, w, b))

    # 1. first_conv
    fc = model.first_conv
    add(fc.conv, fc.bn, "init_conv")

    # 2. blocks
    for i, blk in enumerate(model.blocks):
        mic = blk.mobile_inverted_conv
        mic_type = type(mic).__name__

        # ZeroLayer → pure identity shortcut, no computation needed, skip
        if mic_type == 'ZeroLayer':
            # The shortcut is an IdentityLayer, feature map passes through unchanged
            continue

        has_shortcut = (blk.shortcut is not None 
                        and type(blk.shortcut).__name__ == 'IdentityLayer')
        res_key = f"blk{i}_cache" if has_shortcut else None

        has_expand = (hasattr(mic, 'inverted_bottleneck') 
                      and mic.inverted_bottleneck is not None)

        if has_expand:
            # Expand → DW → Project
            exp = mic.inverted_bottleneck
            add(exp.conv, exp.bn, f"blk{i}_exp", res_add=res_key)

            dw = mic.depth_conv
            add(dw.conv, dw.bn, f"blk{i}_dw")

            proj = mic.point_linear
            add(proj.conv, proj.bn, f"blk{i}_proj", res_conn=res_key)
        else:
            # DW → Project (no expand, block 0 style)
            dw = mic.depth_conv
            add(dw.conv, dw.bn, f"blk{i}_dw", res_add=res_key)

            proj = mic.point_linear
            add(proj.conv, proj.bn, f"blk{i}_proj", res_conn=res_key)

    # 3. feature_mix_layer (1x1 Conv + BN + ReLU6)
    fml = model.feature_mix_layer
    add(fml.conv, fml.bn, "final_conv")

    # 4. classifier (Linear)
    fc_linear = model.classifier.linear
    fc_cfg = LayerConfig(
        name="fc_final", type=LayerType.LINEAR,
        in_channels=fc_linear.in_features, out_channels=fc_linear.out_features,
    )
    sim_layers.append((fc_cfg, 
                       fc_linear.weight.detach().cpu().numpy(),
                       fc_linear.bias.detach().cpu().numpy()))

    return sim_layers


# ============================================================
# Step 3: PTQ Quantization for ProxylessNAS
# ============================================================

def make_quantizable_proxylessnas(model):
    """
    Wrap a ProxylessNAS-Mobile model to be PTQ-friendly.
    
    PyTorch eager-mode PTQ needs:
    1. QuantStub / DeQuantStub at input/output
    2. Conv+BN+ReLU fused
    3. torch.quantization.prepare() + calibration + convert()
    
    Since ProxylessNAS is NOT a torchvision model, we cannot call fuse_model().
    We manually build a quantizable wrapper.
    """

    class QuantizableProxylessNAS(nn.Module):
        """Wrapper that adds quant/dequant stubs and uses FloatFunctional for residual add."""

        def __init__(self, base_model):
            super().__init__()
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.base = base_model

            # Create a FloatFunctional for each block that has a shortcut (residual add)
            self.skip_adds = nn.ModuleDict()
            for i, blk in enumerate(self.base.blocks):
                if (blk.shortcut is not None 
                    and type(blk.shortcut).__name__ == 'IdentityLayer'):
                    self.skip_adds[str(i)] = torch.ao.nn.quantized.FloatFunctional()

        def forward(self, x):
            x = self.quant(x)

            # first_conv
            x = self.base.first_conv(x)

            # blocks
            for i, blk in enumerate(self.base.blocks):
                mic = blk.mobile_inverted_conv
                mic_type = type(mic).__name__

                if mic_type == 'ZeroLayer':
                    # pure identity
                    continue

                out = mic(x)

                has_shortcut = (blk.shortcut is not None
                                and type(blk.shortcut).__name__ == 'IdentityLayer')
                if has_shortcut:
                    # Use FloatFunctional for quantized add
                    out = self.skip_adds[str(i)].add(out, x)

                x = out

            # feature_mix_layer
            x = self.base.feature_mix_layer(x)

            # GAP
            x = self.base.global_avg_pooling(x)
            x = x.view(x.size(0), -1)

            # classifier
            x = self.base.classifier(x)

            x = self.dequant(x)
            return x

    return QuantizableProxylessNAS(model)


def _replace_relu6_with_relu(module: nn.Module):
    """
    Replace all ReLU6 with ReLU in-place throughout the module tree.
    
    This is necessary because PyTorch's fuse_modules only knows
    Conv+BN+ReLU, not Conv+BN+ReLU6. In quantized INT8 mode, the output
    is clamped to [0, 255] anyway, so ReLU6's upper bound at 6 is 
    irrelevant — the requantization step handles the clamp.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            new_relu = nn.ReLU(inplace=child.inplace)
            new_relu.training = child.training  # preserve train/eval mode
            setattr(module, name, new_relu)
        else:
            _replace_relu6_with_relu(child)


def fuse_proxylessnas(q_model):
    """Fuse Conv+BN+ReLU in ProxylessNAS base model."""
    base = q_model.base

    # Step 0: Replace all ReLU6 → ReLU so fuse_modules can handle them
    _replace_relu6_with_relu(base)

    # Fuse first_conv: conv + bn + activation (now ReLU)
    torch.quantization.fuse_modules(
        base.first_conv, ['conv', 'bn', 'activation'], inplace=True
    )

    # Fuse each block's MBInvertedConvLayer
    for blk in base.blocks:
        mic = blk.mobile_inverted_conv
        if type(mic).__name__ == 'ZeroLayer':
            continue

        # inverted_bottleneck (expand): conv + bn + relu
        if hasattr(mic, 'inverted_bottleneck') and mic.inverted_bottleneck is not None:
            torch.quantization.fuse_modules(
                mic.inverted_bottleneck, ['conv', 'bn', 'relu'], inplace=True
            )

        # depth_conv: conv + bn + relu
        torch.quantization.fuse_modules(
            mic.depth_conv, ['conv', 'bn', 'relu'], inplace=True
        )

        # point_linear: conv + bn (NO activation)
        torch.quantization.fuse_modules(
            mic.point_linear, ['conv', 'bn'], inplace=True
        )

    # feature_mix_layer: conv + bn + activation (now ReLU)
    torch.quantization.fuse_modules(
        base.feature_mix_layer, ['conv', 'bn', 'activation'], inplace=True
    )


def quantize_proxylessnas(model, save_path="./models/proxylessnas_mobile_quantized.pth"):
    """Full PTQ pipeline: wrap → fuse → prepare → calibrate → convert."""
    print("=" * 60)
    print("Step 3: PTQ Quantization for ProxylessNAS-Mobile")
    print("=" * 60)

    q_model = make_quantizable_proxylessnas(model)
    q_model.eval()

    # Fuse
    fuse_proxylessnas(q_model)
    print("  Conv+BN+ReLU fused")

    # QConfig
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    q_model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Prepare
    torch.quantization.prepare(q_model, inplace=True)
    print("  Prepared for calibration")

    # Check saved model
    if os.path.exists(save_path):
        print(f"  [Fast Load] Found saved model at {save_path}")
        q_model(torch.randn(1, 3, 224, 224))
        torch.quantization.convert(q_model, inplace=True)
        q_model.load_state_dict(torch.load(save_path, weights_only=True))
        print("  Loaded quantized model from cache")
        return q_model

    # Calibrate with random data (for pipeline test only)
    print("  Calibrating with random data (10 batches)...")
    q_model.eval()
    with torch.no_grad():
        for _ in range(10):
            q_model(torch.randn(1, 3, 224, 224))

    # Convert
    torch.quantization.convert(q_model, inplace=True)
    torch.save(q_model.state_dict(), save_path)
    print(f"  Quantized model saved to {save_path}")

    return q_model


# ============================================================
# Step 4: INT8 Layer Extractor (from quantized model)
# ============================================================

def extract_quantized_conv_params(q_layer, s_in, z_in):
    """
    Extract int8 weights, int32 bias, and quant params from a quantized Conv2d/Linear.
    Reuses the same logic as quant_model_utils.extract_conv_params.
    """
    w_int8 = q_layer.weight().int_repr().numpy().astype(np.int8)

    if q_layer.weight().qscheme() in [torch.per_channel_affine, torch.per_channel_symmetric]:
        s_w = q_layer.weight().q_per_channel_scales().numpy()
        z_w = q_layer.weight().q_per_channel_zero_points().numpy()
        if z_w.size == 1:
            z_w = np.zeros(s_w.shape, dtype=np.int32)
    else:
        s_w = float(q_layer.weight().q_scale())
        z_w = int(q_layer.weight().q_zero_point())

    if q_layer.bias() is not None:
        b_float = q_layer.bias().detach().numpy()
        bias_scale = s_in * s_w
        b_int32 = np.round(b_float / bias_scale).astype(np.int32)
    else:
        out_channels = w_int8.shape[0]
        b_int32 = np.zeros(out_channels, dtype=np.int32)

    s_out = float(q_layer.scale)
    z_out = int(q_layer.zero_point)

    return w_int8, b_int32, s_w, z_w, s_out, z_out


def extract_proxylessnas_quantized_layers(q_model):
    """
    Extract INT8 layers from quantized QuantizableProxylessNAS.
    
    After quantization, the structure is:
        q_model.quant  → QuantStub (provides input scale/zp)
        q_model.base.first_conv.conv  → QuantizedConvReLU2d (fused conv+bn+relu)
        q_model.base.blocks[i].mobile_inverted_conv:
            .inverted_bottleneck.conv → QuantizedConvReLU2d
            .depth_conv.conv → QuantizedConvReLU2d
            .point_linear.conv → QuantizedConv2d (no relu!)
        q_model.skip_adds[str(i)] → FloatFunctional with .scale/.zero_point
        q_model.base.feature_mix_layer.conv → QuantizedConvReLU2d
        q_model.base.classifier.linear → QuantizedLinear
    
    Returns: list of (LayerConfig, weights_int8, bias_int32, qp_dict)
    """
    sim_layers = []
    base = q_model.base

    # Input quant params from QuantStub
    current_scale = float(q_model.quant.scale.item())
    current_zp = int(q_model.quant.zero_point.item())
    print(f"\n  [Extractor] Model Input: Scale={current_scale:.6f}, ZP={current_zp}")

    # 1. first_conv (after fuse: conv is QuantizedConvReLU2d)
    init_layer = base.first_conv.conv
    w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
        init_layer, current_scale, current_zp
    )
    cfg = LayerConfig(
        name="init_conv", type=LayerType.CONV2D,
        in_channels=init_layer.in_channels, out_channels=init_layer.out_channels,
        kernel_size=init_layer.kernel_size[0], stride=init_layer.stride[0],
        padding=init_layer.padding[0],
    )
    qp = {"s_in": current_scale, "z_in": current_zp,
          "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
    sim_layers.append((cfg, w, b, qp))
    current_scale, current_zp = s_out, z_out

    # 2. blocks
    for i, blk in enumerate(base.blocks):
        mic = blk.mobile_inverted_conv
        mic_type = type(mic).__name__

        # ZeroLayer → identity skip, no computation
        if mic_type == 'ZeroLayer':
            continue

        has_shortcut = str(i) in q_model.skip_adds
        res_key = f"blk{i}_cache" if has_shortcut else None

        # Get residual quant params
        if has_shortcut:
            skip_fn = q_model.skip_adds[str(i)]
            res_out_scale = float(skip_fn.scale)
            res_out_zp = int(skip_fn.zero_point)
        else:
            res_out_scale, res_out_zp = None, None

        has_expand = (hasattr(mic, 'inverted_bottleneck')
                      and mic.inverted_bottleneck is not None)

        if has_expand:
            # A. Expand → DW → Project
            # A.1 Expand
            exp_layer = mic.inverted_bottleneck.conv
            w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
                exp_layer, current_scale, current_zp
            )
            cfg = LayerConfig(
                name=f"blk{i}_exp", type=LayerType.CONV2D,
                in_channels=exp_layer.in_channels, out_channels=exp_layer.out_channels,
                kernel_size=1, stride=1, padding=0,
                residual_add_to=res_key,
            )
            qp = {"s_in": current_scale, "z_in": current_zp,
                  "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            sim_layers.append((cfg, w, b, qp))
            current_scale, current_zp = s_out, z_out

            # A.2 Depthwise
            dw_layer = mic.depth_conv.conv
            w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
                dw_layer, current_scale, current_zp
            )
            cfg = LayerConfig(
                name=f"blk{i}_dw", type=LayerType.DEPTHWISE,
                in_channels=dw_layer.in_channels, out_channels=dw_layer.out_channels,
                kernel_size=dw_layer.kernel_size[0], stride=dw_layer.stride[0],
                padding=dw_layer.padding[0], groups=dw_layer.groups,
            )
            qp = {"s_in": current_scale, "z_in": current_zp,
                  "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            sim_layers.append((cfg, w, b, qp))
            current_scale, current_zp = s_out, z_out

            # A.3 Project
            proj_layer = mic.point_linear.conv
            w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
                proj_layer, current_scale, current_zp
            )
            cfg = LayerConfig(
                name=f"blk{i}_proj", type=LayerType.CONV2D,
                in_channels=proj_layer.in_channels, out_channels=proj_layer.out_channels,
                kernel_size=1, stride=1, padding=0,
                residual_connect_from=res_key,
            )
            qp = {"s_in": current_scale, "z_in": current_zp,
                  "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            if has_shortcut:
                qp['residual_out_scale'] = res_out_scale
                qp['residual_out_zp'] = res_out_zp
                current_scale, current_zp = res_out_scale, res_out_zp
            else:
                current_scale, current_zp = s_out, z_out
            sim_layers.append((cfg, w, b, qp))

        else:
            # B. DW → Project (no expand, e.g., block 0)
            dw_layer = mic.depth_conv.conv
            w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
                dw_layer, current_scale, current_zp
            )
            cfg = LayerConfig(
                name=f"blk{i}_dw", type=LayerType.DEPTHWISE,
                in_channels=dw_layer.in_channels, out_channels=dw_layer.out_channels,
                kernel_size=dw_layer.kernel_size[0], stride=dw_layer.stride[0],
                padding=dw_layer.padding[0], groups=dw_layer.groups,
                residual_add_to=res_key,
            )
            qp = {"s_in": current_scale, "z_in": current_zp,
                  "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            sim_layers.append((cfg, w, b, qp))
            current_scale, current_zp = s_out, z_out

            proj_layer = mic.point_linear.conv
            w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
                proj_layer, current_scale, current_zp
            )
            cfg = LayerConfig(
                name=f"blk{i}_proj", type=LayerType.CONV2D,
                in_channels=proj_layer.in_channels, out_channels=proj_layer.out_channels,
                kernel_size=1, stride=1, padding=0,
                residual_connect_from=res_key,
            )
            qp = {"s_in": current_scale, "z_in": current_zp,
                  "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            if has_shortcut:
                qp['residual_out_scale'] = res_out_scale
                qp['residual_out_zp'] = res_out_zp
                current_scale, current_zp = res_out_scale, res_out_zp
            else:
                current_scale, current_zp = s_out, z_out
            sim_layers.append((cfg, w, b, qp))

    # 3. feature_mix_layer (1x1 Conv + BN + ReLU6, fused)
    final_layer = base.feature_mix_layer.conv
    w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
        final_layer, current_scale, current_zp
    )
    cfg = LayerConfig(
        name="final_conv", type=LayerType.CONV2D,
        in_channels=final_layer.in_channels, out_channels=final_layer.out_channels,
        kernel_size=1, stride=1, padding=0,
    )
    qp = {"s_in": current_scale, "z_in": current_zp,
          "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
    sim_layers.append((cfg, w, b, qp))
    current_scale, current_zp = s_out, z_out

    # 4. classifier (Linear)
    fc_layer = base.classifier.linear
    w, b, s_w, z_w, s_out, z_out = extract_quantized_conv_params(
        fc_layer, current_scale, current_zp
    )
    cfg = LayerConfig(
        name="fc_final", type=LayerType.LINEAR,
        in_channels=fc_layer.in_features, out_channels=fc_layer.out_features,
    )
    qp = {"s_in": current_scale, "z_in": current_zp,
          "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
    sim_layers.append((cfg, w, b, qp))

    print(f"  [Extractor] Successfully extracted {len(sim_layers)} layers")
    return sim_layers


# ============================================================
# Step 5: Run full pipeline test
# ============================================================

def test_fp32_extraction():
    """Test FP32 layer extraction (same format as model_utils.extract_layers)."""
    print("\n" + "=" * 60)
    print("Step 2: FP32 Layer Extraction")
    print("=" * 60)
    model = load_proxylessnas_fp32()
    layers = extract_proxylessnas_fp32_layers(model)

    print(f"\n  Extracted {len(layers)} FP32 layers:")
    for i, (cfg, w, b) in enumerate(layers):
        res_info = ""
        if cfg.residual_add_to:
            res_info += f" [save→{cfg.residual_add_to}]"
        if cfg.residual_connect_from:
            res_info += f" [add←{cfg.residual_connect_from}]"
        print(f"    {i:2d}. {cfg.name:20s} {cfg.type.name:10s} "
              f"w={str(w.shape):20s} b={str(b.shape):10s} "
              f"k={cfg.kernel_size} s={cfg.stride} g={cfg.groups}"
              f"{res_info}")
    return model, layers


def test_int8_extraction(model):
    """Test PTQ + INT8 layer extraction."""
    print("\n" + "=" * 60)
    print("Step 4: INT8 Layer Extraction")
    print("=" * 60)

    q_model = quantize_proxylessnas(model)
    layers = extract_proxylessnas_quantized_layers(q_model)

    print(f"\n  Extracted {len(layers)} INT8 layers:")
    for i, (cfg, w, b, qp) in enumerate(layers):
        res_info = ""
        if cfg.residual_add_to:
            res_info += f" [save→{cfg.residual_add_to}]"
        if cfg.residual_connect_from:
            res_info += f" [add←{cfg.residual_connect_from}]"
        sw_shape = f"({len(qp['s_w'])})" if isinstance(qp['s_w'], np.ndarray) else "scalar"
        print(f"    {i:2d}. {cfg.name:20s} {cfg.type.name:10s} "
              f"w={str(w.shape):20s} b={str(b.shape):10s} "
              f"s_w={sw_shape:8s} k={cfg.kernel_size} s={cfg.stride} g={cfg.groups}"
              f"{res_info}")

    return q_model, layers


def test_simulation(q_model, sim_layers):
    """Test running INT8 inference through the existing QuantCoordinator."""
    print("\n" + "=" * 60)
    print("Step 5: Simulation Test (QuantCoordinator)")
    print("=" * 60)

    from coordinator import QuantCoordinator

    # Prepare input
    input_tensor = torch.randn(1, 3, 224, 224)
    img_np = input_tensor.numpy().squeeze(0)  # (3, 224, 224)

    input_scale = sim_layers[0][3]['s_in']
    input_zp = sim_layers[0][3]['z_in']

    # PyTorch INT8 reference
    q_model.eval()
    with torch.no_grad():
        pt_out = q_model(input_tensor)
    pt_pred = int(torch.argmax(pt_out))

    # Simulator
    coord = QuantCoordinator(num_workers=4)
    coord.quantize_input(img_np, input_scale, input_zp)
    print(f"  Input quantized: shape={coord.feature_map.shape}, dtype={coord.feature_map.dtype}")

    start = time.time()
    sim_out_uint8, last_name = coord.execute_inference(sim_layers)
    sim_time = time.time() - start

    sim_pred = int(np.argmax(sim_out_uint8))
    print(f"  Simulation time: {sim_time:.4f}s")
    print(f"  PT INT8 prediction:  class {pt_pred}")
    print(f"  SIM INT8 prediction: class {sim_pred}")
    print(f"  Match: {'YES ✓' if pt_pred == sim_pred else 'NO ✗'}")

    # Stats
    print(f"\n  Performance Stats:")
    print(f"    Total inference time: {coord.stats['total_inference_time']:.4f}s")
    print(f"    Total compute time:   {coord.stats['total_compute_time']:.4f}s")
    print(f"    Total codec time:     {coord.stats['total_codec_time']:.4f}s")
    print(f"    Total comm volume:    {coord.stats['total_comm_volume'] / 1024:.2f} KB")


def main():
    print("=" * 60)
    print("ProxylessNAS-Mobile Full Pipeline Test")
    print("=" * 60)
    print()

    # Step 1+2: Load FP32 and extract layers
    model, fp32_layers = test_fp32_extraction()

    # Step 3+4: Quantize and extract INT8 layers
    q_model, int8_layers = test_int8_extraction(model)

    # Step 5: Run simulation
    test_simulation(q_model, int8_layers)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
