import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from typing import Optional

from models.base import ModelAdapter, SimLayerINT8
from models.registry import register
from protocol.protocol import LayerConfig, LayerType

class QuantizableMNASNet(nn.Module):
    def __init__(self, base_model: models.MNASNet):
        super().__init__()
        self.base = base_model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.skip_adds = nn.ModuleDict()
        flat_idx = 0
        for stage_idx in range(len(self.base.layers)):
            layer = self.base.layers[stage_idx]
            if isinstance(layer, nn.Sequential):
                for blk in layer:
                    if hasattr(blk, "apply_residual"):
                        if blk.apply_residual:
                            self.skip_adds[str(flat_idx)] = (
                                torch.ao.nn.quantized.FloatFunctional()
                            )
                        flat_idx += 1

    def forward(self, x):
        x = self.quant(x)

        # stem: layers[0..7]
        for i in range(8):
            x = self.base.layers[i](x)

        flat_idx = 0
        for stage_idx in range(8, len(self.base.layers)): # layers[8..13]
            layer = self.base.layers[stage_idx]
            if isinstance(layer, nn.Sequential):
                for blk in layer:
                    if hasattr(blk, "apply_residual"):
                        if blk.apply_residual:
                            out = blk.layers(x)
                            x = self.skip_adds[str(flat_idx)].add(out, x)
                        else:
                            x = blk(x)
                        flat_idx += 1
                    else:
                        x = blk(x)
            else:
                # final conv layer (14)
                x = layer(x)

        # GAP + classifier
        x = x.mean([2, 3]) # global average pooling
        x = self.base.classifier(x)

        x = self.dequant(x) 
        return x           

    def fuse_model(self):
        """ Fuse Conv+BN+ReLU / Conv+BN layers for quantization. """
        base = self.base

        # stem layers[0] + [1] + [2] (conv + bn + relu)
        torch.quantization.fuse_modules(base, ['layers.0', 'layers.1', 'layers.2'], inplace=True)
        # stem layers[3] + [4] + [5] (conv + bn + relu)
        torch.quantization.fuse_modules(base, ['layers.3', 'layers.4', 'layers.5'], inplace=True)
        # stem layers[6] + [7] (conv + bn)
        torch.quantization.fuse_modules(base, ['layers.6', 'layers.7'], inplace=True)

        # blocks in layers[8..13]
        for stage_idx in range(8, len(base.layers)):
            stage = base.layers[stage_idx]
            if not isinstance(stage, nn.Sequential):
                continue

            if all(not hasattr(blk, "apply_residual") for blk in stage):
                continue

            for blk_idx, blk in enumerate(stage):
                if not hasattr(blk, "apply_residual"):
                    continue

                # _InvertedResidual has child "layers" (Sequential of 8 ops)
                # [0] + [1] + [2] (expand conv + bn + relu)
                torch.quantization.fuse_modules(blk.layers, ['0', '1', '2'], inplace=True)
                # [3] + [4] + [5] (dw conv + bn + relu)
                torch.quantization.fuse_modules(blk.layers, ['3', '4', '5'], inplace=True)
                # [6] + [7] (project conv + bn)
                torch.quantization.fuse_modules(blk.layers, ['6', '7'], inplace=True)
        
        # final conv layers[14] + [15] + [16]
        torch.quantization.fuse_modules(base, ['layers.14', 'layers.15', 'layers.16'], inplace=True)

@register("mnasnet0_5")
class MNASNetAdapter(ModelAdapter):
    name = "mnasnet0_5"

    def load_fp32(self):
        return models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1).eval()
    
    def make_quantizable(self):
        base = self.load_fp32()
        q_model = QuantizableMNASNet(base)
        q_model.eval()
        q_model.fuse_model()
        return q_model

    def extract_fp32_layers(self, model: models.MNASNet):
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
        
        layers = model.layers

        add(layers[0], layers[1], "init_conv")
        add(layers[3], layers[4], "init_dw")
        add(layers[6], layers[7], "init_proj")

        flat_idx = 0
        for stage_idx in range(8, 14): # layers[8..13]
            stage = layers[stage_idx]
            for blk in stage:
                use_res = blk.apply_residual
                res_key = f"blk{flat_idx}_cache" if use_res else None
                ops = blk.layers

                # expand
                add(ops[0], ops[1], f"blk{flat_idx}_exp", res_add=res_key)
                # dw
                add(ops[3], ops[4], f"blk{flat_idx}_dw") 
                # project
                add(ops[6], ops[7], f"blk{flat_idx}_proj", res_conn=res_key)

                flat_idx += 1
        
        add(layers[14], layers[15], "final_conv")

        # classifier
        fc = model.classifier[1]
        cfg = LayerConfig(
            name="fc_final", type=LayerType.LINEAR,
            in_channels=fc.in_features, out_channels=fc.out_features,
        )
        sim_layers.append((cfg, fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()))

        return sim_layers

    def extract_quantized_layers(self, q_model: QuantizableMNASNet) -> list[SimLayerINT8]:
        sim_layers = []
        base = q_model.base

        cur_s = float(q_model.quant.scale.item())
        cur_z = int(q_model.quant.zero_point.item())
        print(f"\n  [Extractor] MNASNet Input: Scale={cur_s:.6f}, ZP={cur_z}")

        def add_q(q_layer, name, ltype, cfg_extra=None):
            nonlocal cur_s, cur_z
            w, b, s_w, z_w, s_out, z_out = _extract_q_params(q_layer, cur_s, cur_z)

            in_ch = q_layer.in_channels if hasattr(q_layer, 'in_channels') else q_layer.in_features
            out_ch = q_layer.out_channels if hasattr(q_layer, 'out_channels') else q_layer.out_features
            k = q_layer.kernel_size[0] if hasattr(q_layer, 'kernel_size') else 1
            s = q_layer.stride[0] if hasattr(q_layer, 'stride') else 1
            p = q_layer.padding[0] if hasattr(q_layer, 'padding') else 0
            g = q_layer.groups if hasattr(q_layer, 'groups') else 1

            kwargs = dict(
                name=name, type=ltype,
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=k, stride=s, padding=p, groups=g,
            )
            if cfg_extra:
                kwargs.update(cfg_extra)
            cfg = LayerConfig(**kwargs)

            qp = {"s_in": cur_s, "z_in": cur_z,
                   "s_w": s_w, "z_w": z_w,
                   "s_out": s_out, "z_out": z_out}

            sim_layers.append((cfg, w, b, qp))
            cur_s, cur_z = s_out, z_out
            return qp  # caller may annotate with residual params

        # 1. Stem conv (fused: Conv+BN+ReLU → QuantizedConvReLU2d)
        add_q(base.layers[0], "init_conv", LayerType.CONV2D)

        # 2. Stem DW (fused: DWConv+BN+ReLU → QuantizedConvReLU2d)
        add_q(base.layers[3], "init_dw", LayerType.DEPTHWISE)

        # 3. Stem project (fused: Conv+BN → QuantizedConv2d)
        add_q(base.layers[6], "init_proj", LayerType.CONV2D)

        # 4. IR blocks
        flat_idx = 0
        for stage_idx in range(8, 14):
            stage = base.layers[stage_idx]
            if not isinstance(stage, nn.Sequential):
                continue
            for blk in stage:
                if not hasattr(blk, 'apply_residual'):
                    continue

                use_res = blk.apply_residual
                res_key = f"blk{flat_idx}_cache" if use_res else None

                if use_res:
                    skip_fn = q_model.skip_adds[str(flat_idx)]
                    res_out_s = float(skip_fn.scale)
                    res_out_z = int(skip_fn.zero_point)
                else:
                    res_out_s, res_out_z = None, None

                ops = blk.layers  # Sequential, after fuse: [QConvReLU, Identity, Identity, QConvReLU, Identity, Identity, QConv, Identity]

                # Expand (1×1 + ReLU)
                add_q(ops[0], f"blk{flat_idx}_exp", LayerType.CONV2D,
                      {"residual_add_to": res_key})

                # DW (3×3 or 5×5 + ReLU)
                dw_layer = ops[3]
                add_q(dw_layer, f"blk{flat_idx}_dw", LayerType.DEPTHWISE)

                # Project (1×1, no activation)
                proj_qp = add_q(ops[6], f"blk{flat_idx}_proj", LayerType.CONV2D,
                                {"residual_connect_from": res_key})
                if use_res:
                    proj_qp['residual_out_scale'] = res_out_s
                    proj_qp['residual_out_zp'] = res_out_z
                    cur_s, cur_z = res_out_s, res_out_z

                flat_idx += 1

        # 5. Final 1×1 conv (fused: Conv+BN+ReLU → QuantizedConvReLU2d)
        add_q(base.layers[14], "final_conv", LayerType.CONV2D)

        # 6. Classifier
        fc = base.classifier[1]
        add_q(fc, "fc_final", LayerType.LINEAR)

        print(f"  [Extractor] Extracted {len(sim_layers)} layers from MNASNet")
        return sim_layers

def _extract_q_params(q_layer, s_in, z_in):
    """Extract int8 weights, int32 bias, and scale/zp from a quantized Conv/Linear."""
    w_int8 = q_layer.weight().int_repr().numpy().astype(np.int8)

    if q_layer.weight().qscheme() in [
        torch.per_channel_affine, torch.per_channel_symmetric
    ]:
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
        b_int32 = np.zeros(w_int8.shape[0], dtype=np.int32)

    s_out = float(q_layer.scale)
    z_out = int(q_layer.zero_point)
    return w_int8, b_int32, s_w, z_w, s_out, z_out

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