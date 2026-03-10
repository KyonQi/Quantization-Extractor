import torch
import torch.nn as nn
import numpy as np
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from torchvision.models.quantization import QuantizableMobileNetV2
from protocol.protocol import LayerConfig, LayerType, QuantParams

from typing import List, Dict, Tuple, Any

def extract_conv_params(q_layer, s_in, z_in):
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

def extract_quantized_layers(q_model):
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
    w, b, s_w, z_w, s_out, z_out = extract_conv_params(init_layer, current_scale, current_zp)
    
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
            w, b, s_w, z_w, s_out, z_out = extract_conv_params(exp_layer, current_scale, current_zp)
            
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
            w, b, s_w, z_w, s_out, z_out = extract_conv_params(dw_layer, current_scale, current_zp)
            
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
            w, b, s_w, z_w, s_out, z_out = extract_conv_params(proj_layer, current_scale, current_zp)
            
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
            w, b, s_w, z_w, s_out, z_out = extract_conv_params(dw_layer, current_scale, current_zp)
            
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
            w, b, s_w, z_w, s_out, z_out = extract_conv_params(proj_layer, current_scale, current_zp)
            
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
    w, b, s_w, z_w, s_out, z_out = extract_conv_params(final_conv, current_scale, current_zp)
    
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
    w, b, s_w, z_w, s_out, z_out = extract_conv_params(fc_layer, current_scale, current_zp)
    
    cfg = LayerConfig(
        name="fc_final", type=LayerType.LINEAR,
        in_channels=fc_layer.in_features, out_channels=fc_layer.out_features
    )
    qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
    sim_layers.append((cfg, w, b, qp))

    print(f"[Extractor] Successfully extracted {len(sim_layers)} layers.")
    return sim_layers


# ==========================================
# Test Block: 验证脚本是否可用
# ==========================================
if __name__ == "__main__":
    print("Testing Extractor...")
    # 1. Prepare Model
    q_model = q_mobilenet_v2(pretrained=True, quantize=False)
    q_model.fuse_model()
    q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(q_model, inplace=True)
    # Simple Calibration
    q_model(torch.randn(1, 3, 224, 224))
    torch.quantization.convert(q_model, inplace=True)
    
    # 2. Extract
    layers = extract_quantized_layers(q_model)
    
    # 3. Inspect First Layer
    l0 = layers[0]
    cfg, w, b, qp = l0
    print(f"\nLayer 0: {cfg.name}")
    print(f"Weight Shape: {w.shape}, Dtype: {w.dtype}")
    print(f"Bias Shape: {b.shape}, Dtype: {b.dtype}")
    print(f"Scale_W Shape: {np.array(qp['s_w']).shape}") # Check if vector or scalar
    print(f"Sample Weight (Int8): {w.flatten()[:5]}")


# def extract_quantized_layers(q_model):
#     """ 
#     Parse the quantizable model
#     """
#     sim_layers = []

#     # 1. QuantStub records the input scale/zp
#     current_scale = q_model.quant.scale.item()
#     current_zp = int(q_model.quant.zero_point.item())

#     def extract_conv_params(q_layer,
#                             name: str, 
#                             s_in: float, z_in: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
#         # extract the weights, qint8 as default for weights in Pytorch
#         w_int8 = q_layer[0].weight.detach().numpy().astype(np.int8)
#         print(w_int8, type(w_int8))
#         # extract the weights' scale/zp
#         if q_layer[0].weight().qscheme() in [torch.per_channel_symmetric, torch.per_channel_affine]:
#             s_w = q_layer.weight.q_per_channel_scales().numpy()
#             z_w = np.zeros(s_w.shape, dtype=np.int32)
#         else:
#             print(f"WARNING!")
        
#         # extract bias
#         b_float = q_layer.bias().detach().numpy()
#         bias_scale = s_in * s_w
#         b_int32 = np.round(b_float / bias_scale).astype(np.int32) #??????

#         # output
#         s_out = float(q_layer.scale)
#         z_out = int(q_layer.zero_point)

#         return w_int8, b_int32, s_w, z_w, s_out, z_out

#     # 1. init layer Features[0]
#     first_layer: torch.ao.nn.quantized.Conv2d = q_model.features[0][0]
#     w, b, s_w, z_w, s_out, z_out = extract_conv_params(first_layer, "init_conv", current_scale, current_zp)
#     cfg = LayerConfig("init_conv", LayerType.CONV2D, 
#                       first_layer.in_channels, first_layer.out_channels, 
#                       stride=first_layer.stride[0], padding=first_layer.padding[0])
#     qp_dict = {
#         "s_in": current_scale,
#         "z_in": current_zp,
#         "s_out": s_out,
#         "z_out": z_out,
#     }
#     sim_layers.append((cfg, w, b, qp_dict))

#     current_scale = s_out
#     current_zp = z_out
#     # 2. Inverted Residual Blocks
#     for i, block in enumerate(q_model.features[1:]):
#         ops = list(block.conv)
#         use_res = block.use_res_connect
#         res_key = f"blk{i}_out" if use_res else None

#         # if using residual, the final output scale should be determined by QFunctional
#         # otherwise by Conv
#         block_out_scale = float(block.skip_add.scale) if use_res else None
#         block_out_zp = int(block.skip_add.zero_point) if use_res else None

#         if len(ops) == 3:
#             # 2.1 (Expand -> DW -> Project)
#             # expand
#             exp_layer = ops[0][0] #QuantizedConvReLU2d
#             w, b, s_w, z_w, s_out, z_out = extract_conv_params(exp_layer, f"blk{i}_exp", current_scale, current_zp)
#             cfg = LayerConfig(f"blk{i}_exp", LayerType.CONV2D, 
#                               exp_layer.in_channels, exp_layer.out_channels, kernel_size=1,
#                               residual_add_to=res_key if use_res else None) # if having residual, cache it
#             qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
#             sim_layers.append((cfg, w, b, qp))
#             current_scale, current_zp = s_out, z_out # update

#             # dw
#             dw_layer = ops[1][0]
#             w, b, s_w, z_w, s_out, z_out = extract_conv_params(dw_layer, f"blk{i}_dw", current_scale, current_zp)
#             cfg = LayerConfig(f"blk{i}_dw", LayerType.DEPTHWISE, 
#                               dw_layer.in_channels, dw_layer.out_channels, stride=dw_layer.stride[0], 
#                               padding=dw_layer.padding[0], groups=dw_layer.groups)
#             qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
#             sim_layers.append((cfg, w, b, qp))
#             current_scale, current_zp = s_out, z_out

#             # project
#             proj_layer = ops[2] # QuantizedConv2d (注意: 没有 ReLU)
#             w, b, s_w, z_w, s_out, z_out = extract_conv_params(proj_layer, f"blk{i}_proj", current_scale, current_zp)   
#             cfg = LayerConfig(f"blk{i}_proj", LayerType.CONV2D, 
#                               proj_layer.in_channels, proj_layer.out_channels, kernel_size=1,
#                               residual_connect_from=res_key) # add_residual
#             qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
            
#             # since we have skip_add layer,
#             # next layer's Input Scale should be Scale of QFunctional
#             if use_res:
#                 # 把 QFunctional 的参数附带在 Layer 上，供 Coordinator 做 Add 后 Requantize 使用
#                 qp["residual_out_scale"] = block_out_scale
#                 qp["residual_out_zp"] = block_out_zp
#                 sim_layers.append((cfg, w, b, qp))
#                 current_scale, current_zp = block_out_scale, block_out_zp
#             else:
#                 sim_layers.append((cfg, w, b, qp))
#                 current_scale, current_zp = s_out, z_out # No residual, flow continues
#         elif len(ops) == 2:
#             # 2.2 (DW -> Project, Skip Expand)
#             dw_layer = ops[0][0]
#             w, b, s_w, z_w, s_out, z_out = extract_conv_params(dw_layer, f"blk{i}_dw", current_scale, current_zp)
#             cfg = LayerConfig(f"blk{i}_dw", LayerType.DEPTHWISE, 
#                               dw_layer.in_channels, dw_layer.out_channels, stride=dw_layer.stride[0], 
#                               padding=dw_layer.padding[0], groups=dw_layer.groups,
#                               residual_add_to=res_key if use_res else None)
#             qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
#             sim_layers.append((cfg, w, b, qp))
#             current_scale, current_zp = s_out, z_out

#             proj_layer = ops[1]
#             w, b, s_w, z_w, s_out, z_out = extract_conv_params(proj_layer, f"blk{i}_proj", current_scale, current_zp)
            
#             cfg = LayerConfig(f"blk{i}_proj", LayerType.CONV2D, 
#                               proj_layer.in_channels, proj_layer.out_channels, kernel_size=1,
#                               residual_connect_from=res_key)
            
#             qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}


#             if use_res:
#                 qp["residual_out_scale"] = block_out_scale
#                 qp["residual_out_zp"] = block_out_zp
#                 sim_layers.append((cfg, w, b, qp))
#                 current_scale, current_zp = block_out_scale, block_out_zp
#             else:
#                 sim_layers.append((cfg, w, b, qp))
#                 current_scale, current_zp = s_out, z_out
        
#     # 3. Last Conv (1280 channels)
#     last_conv = q_model.features[-1][0]
#     w, b, s_w, z_w, s_out, z_out = extract_conv_params(last_conv, "final_conv", current_scale, current_zp)
#     cfg = LayerConfig("final_conv", LayerType.CONV2D, last_conv.in_channels, last_conv.out_channels)
#     qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
#     sim_layers.append((cfg, w, b, qp))
#     current_scale, current_zp = s_out, z_out

#     # 4. Classifier (Linear)
#     fc = q_model.classifier[1]
#     w, b, s_w, z_w, s_out, z_out = extract_conv_params(fc, "fc_final", current_scale, current_zp)
    
#     cfg = LayerConfig("fc_final", LayerType.LINEAR, fc.in_features, fc.out_features)
#     qp = {"s_in": current_scale, "z_in": current_zp, "s_w": s_w, "z_w": z_w, "s_out": s_out, "z_out": z_out}
#     sim_layers.append((cfg, w, b, qp))

#     return sim_layers
