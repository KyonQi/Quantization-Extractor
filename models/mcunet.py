import sys
import os
import torch
import torch.nn as nn
import numpy as np

from models.base import ModelAdapter, SimLayerINT8
from models.registry import register
from protocol.protocol import LayerConfig, LayerType

# mcunet hub path — needed for torch.load to unpickle ProxylessNASNets
_MCUNET_HUB = os.path.expanduser("~/.cache/torch/hub/mit-han-lab_mcunet_master")
if _MCUNET_HUB not in sys.path:
    sys.path.insert(0, _MCUNET_HUB)


@register("mcunet_in4")
class MCUNetAdapter(ModelAdapter):
    name = "mcunet_in4"
    input_size = 160

    def load_fp32(self) -> nn.Module:
        model = torch.load("./models/mcu_model.pth", weights_only=False, map_location="cpu")
        model.eval()
        return model

    def make_quantizable(self) -> nn.Module:
        q_model = QuantizableMCUNet(self.load_fp32())
        q_model.eval()
        q_model.fuse_model()
        return q_model

    def extract_fp32_layers(self, model):
        """
        Extract FP32 layers from MCUNet-in4.

        MCUNet-in4 structure (ProxylessNASNets):
            first_conv: ConvLayer (Conv2d + BN + ReLU6)
            blocks[0..16]: MobileInvertedResidualBlock
                - mobile_inverted_conv: MBInvertedConvLayer
                    - inverted_bottleneck (optional, absent in block 0)
                    - depth_conv: Sequential(conv, bn, act)
                    - point_linear: Sequential(conv, bn)
                - shortcut: IdentityLayer or None
            classifier: LinearLayer (Linear)

        No feature_mix_layer.
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

            has_shortcut = (blk.shortcut is not None
                            and type(blk.shortcut).__name__ == 'IdentityLayer')
            res_key = f"blk{i}_cache" if has_shortcut else None

            has_expand = (hasattr(mic, 'inverted_bottleneck')
                          and mic.inverted_bottleneck is not None)

            if has_expand:
                exp = mic.inverted_bottleneck
                add(exp.conv, exp.bn, f"blk{i}_exp", res_add=res_key)

                dw = mic.depth_conv
                add(dw.conv, dw.bn, f"blk{i}_dw")

                proj = mic.point_linear
                add(proj.conv, proj.bn, f"blk{i}_proj", res_conn=res_key)
            else:
                dw = mic.depth_conv
                add(dw.conv, dw.bn, f"blk{i}_dw", res_add=res_key)

                proj = mic.point_linear
                add(proj.conv, proj.bn, f"blk{i}_proj", res_conn=res_key)

        # 3. classifier (Linear) — no feature_mix_layer in MCUNet-in4
        fc_linear = model.classifier.linear
        fc_cfg = LayerConfig(
            name="fc_final", type=LayerType.LINEAR,
            in_channels=fc_linear.in_features, out_channels=fc_linear.out_features,
        )
        sim_layers.append((fc_cfg,
                           fc_linear.weight.detach().cpu().numpy(),
                           fc_linear.bias.detach().cpu().numpy()))

        return sim_layers

    def extract_quantized_layers(self, q_model):
        """
        Extract INT8 layers from quantized QuantizableMCUNet.

        After quantization, the structure is:
            q_model.quant  → QuantStub
            q_model.base.first_conv.conv  → QuantizedConvReLU2d
            q_model.base.blocks[i].mobile_inverted_conv:
                .inverted_bottleneck.conv → QuantizedConvReLU2d
                .depth_conv.conv → QuantizedConvReLU2d
                .point_linear.conv → QuantizedConv2d (no relu)
            q_model.skip_adds[str(i)] → FloatFunctional
            q_model.base.classifier.linear → QuantizedLinear
        """
        sim_layers = []
        base = q_model.base

        current_scale = float(q_model.quant.scale.item())
        current_zp = int(q_model.quant.zero_point.item())
        print(f"\n  [Extractor] MCUNet-in4 Input: Scale={current_scale:.6f}, ZP={current_zp}")

        # 1. first_conv
        init_layer = base.first_conv.conv
        w, b, s_w, z_w, s_out, z_out = _extract_q_params(init_layer, current_scale, current_zp)
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

            has_shortcut = str(i) in q_model.skip_adds
            res_key = f"blk{i}_cache" if has_shortcut else None

            if has_shortcut:
                skip_fn = q_model.skip_adds[str(i)]
                res_out_scale = float(skip_fn.scale)
                res_out_zp = int(skip_fn.zero_point)
            else:
                res_out_scale, res_out_zp = None, None

            has_expand = (hasattr(mic, 'inverted_bottleneck')
                          and mic.inverted_bottleneck is not None)

            if has_expand:
                # Expand
                exp_layer = mic.inverted_bottleneck.conv
                w, b, s_w, z_w, s_out, z_out = _extract_q_params(exp_layer, current_scale, current_zp)
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

                # Depthwise
                dw_layer = mic.depth_conv.conv
                w, b, s_w, z_w, s_out, z_out = _extract_q_params(dw_layer, current_scale, current_zp)
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

                # Project
                proj_layer = mic.point_linear.conv
                w, b, s_w, z_w, s_out, z_out = _extract_q_params(proj_layer, current_scale, current_zp)
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
                # DW → Project (block 0, no expand)
                dw_layer = mic.depth_conv.conv
                w, b, s_w, z_w, s_out, z_out = _extract_q_params(dw_layer, current_scale, current_zp)
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
                w, b, s_w, z_w, s_out, z_out = _extract_q_params(proj_layer, current_scale, current_zp)
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

        # 3. classifier (no feature_mix_layer)
        fc_layer = base.classifier.linear
        w, b, s_w, z_w, s_out, z_out = _extract_q_params(fc_layer, current_scale, current_zp)
        cfg = LayerConfig(
            name="fc_final", type=LayerType.LINEAR,
            in_channels=fc_layer.in_features, out_channels=fc_layer.out_features,
        )
        qp = {"s_in": current_scale, "z_in": current_zp,
              "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
        sim_layers.append((cfg, w, b, qp))

        print(f"  [Extractor] Successfully extracted {len(sim_layers)} layers from MCUNet-in4")
        return sim_layers


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


class QuantizableMCUNet(nn.Module):
    """Wrapper that adds quant/dequant stubs and FloatFunctional for residual adds."""

    def __init__(self, base_model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.base = base_model

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
            out = mic(x)

            has_shortcut = (blk.shortcut is not None
                            and type(blk.shortcut).__name__ == 'IdentityLayer')
            if has_shortcut:
                out = self.skip_adds[str(i)].add(out, x)

            x = out

        # GAP + classifier
        x = x.mean([2, 3])
        x = x.view(x.size(0), -1)
        x = self.base.classifier(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU in MCUNet base model.

        MCUNet uses 'act' as the activation attribute name (not 'relu' or 'activation').
        """
        base = self.base

        # Replace all ReLU6 → ReLU for fusion compatibility
        self._replace_relu6_with_relu(base)

        # first_conv: conv + bn + act (now ReLU)
        torch.quantization.fuse_modules(
            base.first_conv, ['conv', 'bn', 'act'], inplace=True
        )

        # blocks
        for blk in base.blocks:
            mic = blk.mobile_inverted_conv

            # inverted_bottleneck (expand): conv + bn + act
            if hasattr(mic, 'inverted_bottleneck') and mic.inverted_bottleneck is not None:
                torch.quantization.fuse_modules(
                    mic.inverted_bottleneck, ['conv', 'bn', 'act'], inplace=True
                )

            # depth_conv: conv + bn + act
            torch.quantization.fuse_modules(
                mic.depth_conv, ['conv', 'bn', 'act'], inplace=True
            )

            # point_linear: conv + bn (NO activation)
            torch.quantization.fuse_modules(
                mic.point_linear, ['conv', 'bn'], inplace=True
            )

    def _replace_relu6_with_relu(self, module: nn.Module):
        """Replace all ReLU6 with ReLU in-place."""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU6):
                new_relu = nn.ReLU(inplace=child.inplace)
                new_relu.training = child.training
                setattr(module, name, new_relu)
            else:
                self._replace_relu6_with_relu(child)
