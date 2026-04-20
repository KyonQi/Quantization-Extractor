import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from typing import Optional

from models.base import ModelAdapter
from models.registry import register
from protocol.protocol import LayerConfig, LayerType

@register("mbv2_1.0")
class MobileNetV2Adapter(ModelAdapter):
    name = "mbv2_1.0"
    input_size = 224

    def load_fp32(self):
        return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    
    def make_quantizable(self):
        q_model = models.quantization.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
        q_model.eval()
        q_model.fuse_model()
        return q_model
    
    def extract_fp32_layers(self, model: models.MobileNetV2):
        """ Parse the PyTorch model to extract configs for simulation """
        sim_layers = []

        def add(conv: nn.Conv2d, bn: nn.BatchNorm2d, name: str, res_add: Optional[str] = None, res_conn: Optional[str] = None) -> None:
            w, b = self.fuse_conv_bn(conv=conv, bn=bn)
            # judge layer type
            if conv.groups == conv.in_channels and conv.in_channels > 1:
                l_type = LayerType.DEPTHWISE
            elif conv.kernel_size[0] == 1:
                l_type = LayerType.POINTWISE
            else:
                l_type = LayerType.CONV2D
            
            cfg = LayerConfig(
                name=name, type=l_type,
                in_channels=conv.in_channels, out_channels=conv.out_channels,
                kernel_size=conv.kernel_size[0], stride=conv.stride[0], padding=conv.padding[0],
                groups=conv.groups, residual_add_to=res_add, residual_connect_from=res_conn
            )
            sim_layers.append((cfg, w, b))
            
        # 1. Process the Initial Convolution Layer ---
        # The first layer in MobileNetV2 features is Conv2dNormActivation (Conv+BN+ReLU6)
        # We name it 'init_conv' (ReLU6 will be applied by Worker)
        add(model.features[0][0], model.features[0][1], "init_conv")

        # 2. Iterate through Inverted Residual Blocks ---
        for i, block in enumerate(model.features[1:]):
            if hasattr(block, 'conv'):
                ops = list(block.conv)
                use_res = block.use_res_connect
                # If residual connection is active, generate key for the buffer
                res_key = f"block_{i}" if use_res else None
                
                # Case A: Expansion -> Depthwise -> Projection (Expansion Factor t > 1)
                # Structure: [ExpandSeq(Conv+BN+ReLU), DWSeq(Conv+BN+ReLU), ProjConv, ProjBN]
                if len(ops) == 4:
                    # 1. Expand Layer (Start of block: cache input if residual needed)
                    add(ops[0][0], ops[0][1], f"blk{i}_exp", res_add=res_key)
                    
                    # 2. Depthwise Layer
                    add(ops[1][0], ops[1][1], f"blk{i}_dw")
                    
                    # 3. Projection Layer (End of block: add residual if needed)
                    add(ops[2], ops[3], f"blk{i}_proj", res_conn=res_key)
                    
                # Case B: Depthwise -> Projection (Expansion Factor t = 1, No Expansion)
                # Structure: [DWSeq(Conv+BN+ReLU), ProjConv, ProjBN]
                elif len(ops) == 3:
                    # 1. Depthwise Layer (Start of block: cache input if residual needed)
                    add(ops[0][0], ops[0][1], f"blk{i}_dw", res_add=res_key)
                    
                    # 2. Projection Layer (End of block: add residual if needed)
                    add(ops[1], ops[2], f"blk{i}_proj", res_conn=res_key)
                
                else:
                    print(f"Warning: Skipping unknown block structure at index {i} with len {len(ops)}")
            
            # 3. Process the Final Feature Layer ---
            # The last layer in .features is usually a 1x1 Conv (Conv2dNormActivation)
            # Structure: [Conv2d, BatchNorm2d, ReLU6]
            elif isinstance(block, (nn.Sequential, torchvision.ops.misc.Conv2dNormActivation)):
                add(block[0], block[1], f"final_conv_{i}")
            else:
                print(f"Warning: Unknown layer type at index {i}: {type(block)}")
        
        return sim_layers

    def fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
        """ Fuse Conv2d and BatchNorm2d layers """
        with torch.no_grad():
            w, b = conv.weight, conv.bias
            if b is None:
                b: torch.Tensor = torch.zeros_like(bn.running_mean)

            mean, var = bn.running_mean, bn.running_var
            gamma, beta = bn.weight, bn.bias
            scale = gamma / torch.sqrt(var + bn.eps)

            if isinstance(conv, nn.Conv2d):
                # Reshape scale for broadcasting
                view_shape = [-1] + [1] * (w.ndim - 1)
                w_fused = w * scale.view(view_shape)
                b_fused = (b - mean) * scale + beta
            return w_fused.cpu().numpy(), b_fused.cpu().numpy()
    
    def extract_quantized_layers(self, q_model):
        """
        Parse QuantizableMobileNetV2
        Return: list of (LayerConfig, weights_int8, bias_int32, qp_dict)
        """
        sim_layers = []
        
        # 获取全局输入的量化参数 (Input QuantStub)
        current_scale = float(q_model.quant.scale.item())
        current_zp = int(q_model.quant.zero_point.item())
        
        print(f"[Extractor] Model Input: Scale={current_scale}, ZP={current_zp}")

        # ==========================================
        # 1. Initial Conv Layer
        # ==========================================
        # features[0] 是 ConvNormActivation，里面 [0] 是 QuantizedConvReLU2d
        init_layer = q_model.features[0][0]
        w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(init_layer, current_scale, current_zp)
        
        cfg = LayerConfig(
            name="init_conv", type=LayerType.CONV2D,
            in_channels=init_layer.in_channels, out_channels=init_layer.out_channels,
            kernel_size=init_layer.kernel_size[0], stride=init_layer.stride[0], 
            padding=init_layer.padding[0]
        )
        
        qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
        sim_layers.append((cfg, w, b, qp))
        
        # 更新下一层的输入参数
        current_scale, current_zp = s_out, z_out

        # ==========================================
        # 2. Inverted Residual Blocks
        # ==========================================
        for i, block in enumerate(q_model.features[1:-1]):
            # block.conv 是一个 Sequential，包含了 Conv/BN/ReLU 的组合
            ops = list(block.conv)
            use_res = block.use_res_connect
            res_key = f"blk{i}_cache" if use_res else None
            
            # 获取 Add 操作后的量化参数 (用于 Residual 对齐)
            if use_res:
                res_out_scale = float(block.skip_add.scale)
                res_out_zp = int(block.skip_add.zero_point)
                # res_out_scale = float(block.skip_add.scale.item())
                # res_out_zp = int(block.skip_add.zero_point.item())
            else:
                res_out_scale, res_out_zp = None, None

            # --- Case A: Expand -> Depthwise -> Project (len=3) ---
            if len(ops) == 4:
                # A.1 Expand (1x1 Conv + ReLU)
                exp_layer = ops[0][0]
                w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(exp_layer, current_scale, current_zp)
                
                # 如果有残差，在这里缓存输入 (residual_add_to)
                cfg = LayerConfig(
                    name=f"blk{i}_exp", type=LayerType.CONV2D,
                    in_channels=exp_layer.in_channels, out_channels=exp_layer.out_channels,
                    kernel_size=1, stride=1, padding=0,
                    residual_add_to=res_key # Start of block
                )
                qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
                sim_layers.append((cfg, w, b, qp))
                current_scale, current_zp = s_out, z_out

                # A.2 Depthwise (3x3 Conv + ReLU)
                dw_layer = ops[1][0]
                w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(dw_layer, current_scale, current_zp)
                
                cfg = LayerConfig(
                    name=f"blk{i}_dw", type=LayerType.DEPTHWISE,
                    in_channels=dw_layer.in_channels, out_channels=dw_layer.out_channels,
                    kernel_size=dw_layer.kernel_size[0], stride=dw_layer.stride[0],
                    padding=dw_layer.padding[0], groups=dw_layer.groups
                )
                qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
                sim_layers.append((cfg, w, b, qp))
                current_scale, current_zp = s_out, z_out

                # A.3 Project (1x1 Conv, No ReLU)
                proj_layer = ops[2] # 通常是 QuantizedConv2d (无 ReLU)
                w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(proj_layer, current_scale, current_zp)
                
                cfg = LayerConfig(
                    name=f"blk{i}_proj", type=LayerType.CONV2D,
                    in_channels=proj_layer.in_channels, out_channels=proj_layer.out_channels,
                    kernel_size=1, stride=1, padding=0,
                    residual_connect_from=res_key # End of block
                )
                qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
                
                # 如果有残差，把 Block 最终的 Scale 带上
                if use_res:
                    qp['residual_out_scale'] = res_out_scale
                    qp['residual_out_zp'] = res_out_zp
                    current_scale, current_zp = res_out_scale, res_out_zp # Block 输出作为下一层输入
                else:
                    current_scale, current_zp = s_out, z_out
                    
                sim_layers.append((cfg, w, b, qp))

            # --- Case B: Depthwise -> Project (len=2, No Expand) ---
            elif len(ops) == 3:
                # B.1 Depthwise
                dw_layer = ops[0][0]
                w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(dw_layer, current_scale, current_zp)
                
                cfg = LayerConfig(
                    name=f"blk{i}_dw", type=LayerType.DEPTHWISE,
                    in_channels=dw_layer.in_channels, out_channels=dw_layer.out_channels,
                    kernel_size=dw_layer.kernel_size[0], stride=dw_layer.stride[0],
                    padding=dw_layer.padding[0], groups=dw_layer.groups,
                    residual_add_to=res_key # Start of block
                )
                qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
                sim_layers.append((cfg, w, b, qp))
                current_scale, current_zp = s_out, z_out
                
                # B.2 Project
                proj_layer = ops[1]
                w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(proj_layer, current_scale, current_zp)
                
                cfg = LayerConfig(
                    name=f"blk{i}_proj", type=LayerType.CONV2D,
                    in_channels=proj_layer.in_channels, out_channels=proj_layer.out_channels,
                    kernel_size=1, stride=1, padding=0,
                    residual_connect_from=res_key # End of block
                )
                qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}

                if use_res:
                    qp['residual_out_scale'] = res_out_scale
                    qp['residual_out_zp'] = res_out_zp
                    current_scale, current_zp = res_out_scale, res_out_zp
                else:
                    current_scale, current_zp = s_out, z_out
                
                sim_layers.append((cfg, w, b, qp))

        # ==========================================
        # 3. Final Features Layer (1x1 Conv)
        # ==========================================
        final_conv = q_model.features[-1][0]
        w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(final_conv, current_scale, current_zp)
        
        cfg = LayerConfig(
            name="final_conv", type=LayerType.CONV2D,
            in_channels=final_conv.in_channels, out_channels=final_conv.out_channels,
            kernel_size=1, stride=1, padding=0
        )
        qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
        sim_layers.append((cfg, w, b, qp))
        current_scale, current_zp = s_out, z_out

        # ==========================================
        # 4. Classifier (Linear)
        # ==========================================
        # PyTorch QuantizedLinear 在 classifier[1]
        fc_layer = q_model.classifier[1]
        w, b, s_w, z_w, s_out, z_out = self.extract_conv_params(fc_layer, current_scale, current_zp)
        
        cfg = LayerConfig(
            name="fc_final", type=LayerType.LINEAR,
            in_channels=fc_layer.in_features, out_channels=fc_layer.out_features
        )
        qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
        sim_layers.append((cfg, w, b, qp))

        print(f"[Extractor] Successfully extracted {len(sim_layers)} layers.")
        return sim_layers
    
    def extract_conv_params(self, q_layer, s_in, z_in):
        """
        通用函数：从 PyTorch 量化层提取 Int8 权重、Int32 偏置和量化参数
        """
        # 1. 提取权重 (Int8)
        # int_repr() 返回底层存储的 int8 Tensor
        w_int8 = q_layer.weight().int_repr().numpy().astype(np.int8)
        
        # 2. 提取权重 Scale 和 ZeroPoint
        # 检查是否是 Per-Channel 量化
        if q_layer.weight().qscheme() in [torch.per_channel_affine, torch.per_channel_symmetric]:
            s_w = q_layer.weight().q_per_channel_scales().numpy() # Vector: (Out_Channels,)
            z_w = q_layer.weight().q_per_channel_zero_points().numpy()
            # 通常 Per-Channel ZP 都是 0，但我们还是提取出来
            if z_w.size == 1: z_w = np.zeros(s_w.shape, dtype=np.int32)
        else:
            s_w = float(q_layer.weight().q_scale()) # Scalar
            z_w = int(q_layer.weight().q_zero_point())

        # 3. 提取并计算 Bias (Int32)
        # PyTorch 存储的是 float bias，我们需要手动量化
        # Formula: b_int = Round( b_float / (s_in * s_w) )
        if q_layer.bias() is not None:
            b_float = q_layer.bias().detach().numpy()
            
            # 处理 Broadcasting: s_in (scalar) * s_w (vector) -> bias_scale (vector)
            bias_scale = s_in * s_w
            b_int32 = np.round(b_float / bias_scale).astype(np.int32)
        else:
            # 如果没有 bias，创建一个全 0 的 int32 数组
            out_channels = w_int8.shape[0]
            b_int32 = np.zeros(out_channels, dtype=np.int32)

        # 4. 提取输出参数
        s_out = float(q_layer.scale)
        z_out = int(q_layer.zero_point)

        return w_int8, b_int32, s_w, z_w, s_out, z_out
    

@register("mbv2_0.35")
class MobileNetV2035Adapter(MobileNetV2Adapter):
    name = "mbv2_0.35"
    input_size = 224

    def load_fp32(self):
        return models.mobilenet_v2(weights=None, width_mult=0.35).eval()
    
    def make_quantizable(self):
        q_model = models.quantization.mobilenet_v2(weights=None, quantize=False, width_mult=0.35)
        q_model.eval()
        q_model.fuse_model()
        return q_model