import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from models.base import ModelAdapter, SimLayerFP32, SimLayerINT8
from models.registry import register
from protocol.protocol import LayerConfig, LayerType


# ============================================================
# FP32 Model
# ============================================================

class DSConvBlock(nn.Module):
    def __init__(self, ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, stride=stride, padding=1, groups=ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        x = F.relu(self.dw_bn(self.dw(x)))
        x = F.relu(self.pw_bn(self.pw(x)))
        return x


class DSCNNLarge(nn.Module):
    """
    DS-CNN Large for Keyword Spotting

    Input: [B, 1, 49, 10]  (MFCC)
    Output: [B, 12]
    """
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 276, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(276)
        self.blocks = nn.ModuleList([
            DSConvBlock(276, 2),   # 49x10 -> 25x5
            DSConvBlock(276, 2),   # 25x5  -> 13x3
            DSConvBlock(276, 1),   # 13x3
            DSConvBlock(276, 1),   # 13x3
            DSConvBlock(276, 1),   # 13x3
            DSConvBlock(276, 1),   # 13x3
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(276, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Quantization-friendly Model
# ============================================================

class DSConvBlockQuant(nn.Module):
    """Quantization-friendly: nn.ReLU modules instead of F.relu"""
    def __init__(self, ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, stride=stride, padding=1, groups=ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(ch)
        self.dw_relu = nn.ReLU()
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(ch)
        self.pw_relu = nn.ReLU()

    def forward(self, x):
        x = self.dw_relu(self.dw_bn(self.dw(x)))
        x = self.pw_relu(self.pw_bn(self.pw(x)))
        return x


class DSCNNLargeQuant(nn.Module):
    """Quantization-friendly DS-CNN Large with QuantStub/DeQuantStub"""
    def __init__(self, trained_model=None, num_classes=12):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 276, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(276)
        self.relu1 = nn.ReLU()
        self.blocks = nn.ModuleList([
            DSConvBlockQuant(276, 2),
            DSConvBlockQuant(276, 2),
            DSConvBlockQuant(276, 1),
            DSConvBlockQuant(276, 1),
            DSConvBlockQuant(276, 1),
            DSConvBlockQuant(276, 1),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(276, num_classes)
        self.dequant = torch.ao.quantization.DeQuantStub()

        if trained_model is not None:
            self._copy_weights(trained_model)

    def _copy_weights(self, src):
        self.conv1.load_state_dict(src.conv1.state_dict())
        self.bn1.load_state_dict(src.bn1.state_dict())
        for i in range(len(self.blocks)):
            self.blocks[i].dw.load_state_dict(src.blocks[i].dw.state_dict())
            self.blocks[i].dw_bn.load_state_dict(src.blocks[i].dw_bn.state_dict())
            self.blocks[i].pw.load_state_dict(src.blocks[i].pw.state_dict())
            self.blocks[i].pw_bn.load_state_dict(src.blocks[i].pw_bn.state_dict())
        self.fc.load_state_dict(src.fc.state_dict())

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        for i in range(len(self.blocks)):
            torch.ao.quantization.fuse_modules(
                self, [f'blocks.{i}.dw', f'blocks.{i}.dw_bn', f'blocks.{i}.dw_relu'],
                inplace=True)
            torch.ao.quantization.fuse_modules(
                self, [f'blocks.{i}.pw', f'blocks.{i}.pw_bn', f'blocks.{i}.pw_relu'],
                inplace=True)


# ============================================================
# Helpers
# ============================================================

def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fuse Conv2d + BatchNorm2d into numpy arrays (w, b)."""
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


# ============================================================
# Adapter
# ============================================================

@register("dscnn_large")
class DSCNNAdapter(ModelAdapter):
    name = "dscnn_large"
    input_size = 49       # H dimension (for base class compatibility)
    input_channels = 1
    input_height = 49
    input_width = 10
    num_classes = 12

    def load_fp32(self) -> nn.Module:
        model = DSCNNLarge(num_classes=self.num_classes)
        state_dict = torch.load("./models/dscnn.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def load_datasets(self):
        train_ds = MFCCDataset("./data/test_samples.npz")
        val_ds = MFCCDataset("./data/test_samples.npz")
        print(f"Train: {len(train_ds)} samples")
        print(f"Test: {len(val_ds)} samples")
        return train_ds, val_ds

    def make_quantizable(self) -> nn.Module:
        fp32_model = self.load_fp32()
        q_model = DSCNNLargeQuant(trained_model=fp32_model, num_classes=self.num_classes)
        q_model.eval()
        q_model.fuse_model()
        return q_model

    def quantize(self, calibration_loader, num_calibration_batches, save_path) -> nn.Module:
        """Override base quantize() for 1-channel 49x10 MFCC input."""
        q_model = self.make_quantizable()

        backend = "fbgemm"
        torch.backends.quantized.engine = backend
        q_model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(q_model, inplace=True)

        # fast load if quantized model already exists
        if save_path and os.path.exists(save_path):
            print(f"\n[Fast Load] Found saved model at {save_path}, loading...")
            q_model(torch.randn(1, self.input_channels, self.input_height, self.input_width))
            torch.quantization.convert(q_model, inplace=True)
            q_model.load_state_dict(torch.load(save_path, weights_only=True))
            return q_model

        # calibrate
        with torch.no_grad():
            for i, (mfcc, _) in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                q_model(mfcc)
                if (i + 1) % 50 == 0:
                    print(f"  Calibrated {i + 1}/{num_calibration_batches} batches")

        torch.ao.quantization.convert(q_model, inplace=True)
        if save_path:
            torch.save(q_model.state_dict(), save_path)

        return q_model

    def extract_fp32_layers(self, model) -> list[SimLayerFP32]:
        sim_layers = []

        def add(conv: nn.Conv2d, bn: nn.BatchNorm2d, name: str):
            w, b = _fuse_conv_bn(conv, bn)
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
            )
            sim_layers.append((cfg, w, b))

        # 1. conv1
        add(model.conv1, model.bn1, "init_conv")

        # 2. blocks (dw + pw each)
        for i, blk in enumerate(model.blocks):
            add(blk.dw, blk.dw_bn, f"blk{i}_dw")
            add(blk.pw, blk.pw_bn, f"blk{i}_pw")

        # 3. classifier
        fc = model.fc
        cfg = LayerConfig(
            name="fc_final", type=LayerType.LINEAR,
            in_channels=fc.in_features, out_channels=fc.out_features,
        )
        sim_layers.append((cfg,
                           fc.weight.detach().cpu().numpy(),
                           fc.bias.detach().cpu().numpy()))

        return sim_layers

    def extract_quantized_layers(self, q_model) -> list[SimLayerINT8]:
        """
        Extract INT8 layers from quantized DSCNNLargeQuant.

        After fusion + convert:
            q_model.quant       -> QuantStub (input scale/zp)
            q_model.conv1       -> QuantizedConvReLU2d (fused conv1+bn1+relu1)
            q_model.blocks[i]:
                .dw             -> QuantizedConvReLU2d (fused dw+dw_bn+dw_relu)
                .pw             -> QuantizedConvReLU2d (fused pw+pw_bn+pw_relu)
            q_model.fc          -> QuantizedLinear
            q_model.dequant     -> DeQuantStub
        """
        sim_layers = []

        current_scale = float(q_model.quant.scale.item())
        current_zp = int(q_model.quant.zero_point.item())
        print(f"\n  [Extractor] DSCNN-Large Input: Scale={current_scale:.6f}, ZP={current_zp}")

        # 1. conv1 (QuantizedConvReLU2d)
        conv1_layer = q_model.conv1
        w, b, s_w, z_w, s_out, z_out = _extract_q_params(conv1_layer, current_scale, current_zp)
        cfg = LayerConfig(
            name="init_conv", type=LayerType.CONV2D,
            in_channels=conv1_layer.in_channels, out_channels=conv1_layer.out_channels,
            kernel_size=conv1_layer.kernel_size[0], stride=conv1_layer.stride[0],
            padding=conv1_layer.padding[0],
        )
        qp = {"s_in": current_scale, "z_in": current_zp,
              "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
        sim_layers.append((cfg, w, b, qp))
        current_scale, current_zp = s_out, z_out

        # 2. blocks (no residual connections in DSCNN)
        for i, blk in enumerate(q_model.blocks):
            # depthwise
            dw_layer = blk.dw
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

            # pointwise
            pw_layer = blk.pw
            w, b, s_w, z_w, s_out, z_out = _extract_q_params(pw_layer, current_scale, current_zp)
            cfg = LayerConfig(
                name=f"blk{i}_pw", type=LayerType.POINTWISE,
                in_channels=pw_layer.in_channels, out_channels=pw_layer.out_channels,
                kernel_size=1, stride=1, padding=0,
            )
            qp = {"s_in": current_scale, "z_in": current_zp,
                  "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            sim_layers.append((cfg, w, b, qp))
            current_scale, current_zp = s_out, z_out

        # 3. classifier (QuantizedLinear)
        fc_layer = q_model.fc
        w, b, s_w, z_w, s_out, z_out = _extract_q_params(fc_layer, current_scale, current_zp)
        cfg = LayerConfig(
            name="fc_final", type=LayerType.LINEAR,
            in_channels=fc_layer.in_features, out_channels=fc_layer.out_features,
        )
        qp = {"s_in": current_scale, "z_in": current_zp,
              "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
        sim_layers.append((cfg, w, b, qp))

        print(f"  [Extractor] Successfully extracted {len(sim_layers)} layers from DSCNN-Large")
        return sim_layers

# ==============================================================
# Dataset（from .npz return (tensor, label)）
# ==============================================================
class MFCCDataset(Dataset):
    """
    Dataset for MFCC features of Speech Commands.
        mfccs:       [N, 1, 49, 10] float32
        labels:      [N] int64
        label_names: [12] str
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.mfccs = data["mfccs"]          # [N, 1, 49, 10]
        self.labels = data["labels"]         # [N]
        self.label_names = data["label_names"].tolist()
        self.classes = self.label_names
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        x = torch.from_numpy(self.mfccs[idx].copy())   # [1, 49, 10]
        y = int(self.labels[idx])
        return x, y
 