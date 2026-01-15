import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import json
import numpy as np

class NativeQuantExtractor:
    def __init__(self):
        self.params = {}
        self.weights = {}
        self.biases = {}
        self.layer_list = []

    def _get_tensor_min_max(self, weight_tensor, scale, zp, is_per_channel):
        """
        计算反量化后的权重的实际 min/max (float domain)
        """
        # 为了计算准确的 min/max，我们直接把 int8 权重反量化回 float
        w_int = weight_tensor.int_repr().float()
        
        if is_per_channel:
            # Scale 和 ZP 都是 vector，需要 reshape 才能广播
            # 假设权重形状是 [Out, In/Groups, k, k]
            # Per-channel 通常是在 dim 0
            n_channels = len(scale)
            scale = scale.view(n_channels, 1, 1, 1)
            zp = zp.view(n_channels, 1, 1, 1)
            
        w_float = (w_int - zp) * scale
        return w_float.min().item(), w_float.max().item()

    def add_layer_params(self, name, module, param_type="activation"):
        """
        提取层参数
        param_type: "activation" (输出/输入) 或 "weight" (卷积核)
        """
        entry = {}
        
        # === 情况 A: 提取权重参数 ===
        if param_type == "weight":
            if not hasattr(module, 'weight'):
                return
            
            # `module.weight` 在不同的量化/转换后的模块中有时是属性（Tensor/Parameter），
            # 有时是可调用方法（返回 Tensor）。兼容两种情况。
            w_attr = module.weight
            w = w_attr() if callable(w_attr) else w_attr
            qscheme = w.qscheme()
            
            # 判断是否为 Per-Channel
            if qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
                # === Per-Channel 处理 ===
                scales = w.q_per_channel_scales()
                zps = w.q_per_channel_zero_points()
                
                # 存为列表
                entry["scale"] = scales.numpy().tolist()
                entry["zero_point"] = zps.numpy().tolist()
                entry["quant_type"] = "per_channel" # 标记一下方便识别
                
                # 计算实际的 min/max 范围供参考
                f_min, f_max = self._get_tensor_min_max(w, scales, zps, True)
                
            else:
                # === Per-Tensor 处理 ===
                scale = w.q_scale()
                zp = w.q_zero_point()
                
                entry["scale"] = float(scale)
                entry["zero_point"] = int(zp)
                entry["quant_type"] = "per_tensor"
                
                f_min, f_max = self._get_tensor_min_max(w, scale, zp, False)

            entry["min"] = f_min
            entry["max"] = f_max
            
            # 添加到结果
            self.params[f"{name}_weights"] = entry

            # 尝试保存该层的 float 权重 & bias 以供 simulator 使用
            try:
                w_tensor = w.dequantize() if hasattr(w, 'dequantize') else (w.float() if hasattr(w, 'float') else w)
                self.weights[f"{name}_weights"] = w_tensor.cpu().numpy()
            except Exception:
                # 忽略无法导出权重的情况
                pass

            # bias 可能在 module.bias 中（float）
            if hasattr(module, 'bias') and module.bias is not None:
                b_attr = module.bias
                b_tensor = b_attr() if callable(b_attr) else b_attr
                try:
                    self.biases[f"{name}_bias"] = b_tensor.cpu().numpy()
                except Exception:
                    pass

        # === 情况 B: 提取激活参数 (Output Scale) ===
        elif param_type == "activation":
            # 激活通常是 Per-Tensor
            # 有些层（例如 Identity 或未 convert 的容器）没有 `scale`/`zero_point` 属性，
            # 我们先尝试直接读取；如果没有，则在子层中寻找带有这些属性的量化算子；找不到则跳过。
            if not (hasattr(module, 'scale') and hasattr(module, 'zero_point')):
                found = None
                for child in module.children():
                    if hasattr(child, 'scale') and hasattr(child, 'zero_point'):
                        found = child
                        break
                if found is None:
                    return
                module = found

            scale = float(module.scale)
            zp = int(module.zero_point)

            # 对于激活，min/max 由量化范围决定 (0~255)
            # val = (q - zp) * scale
            q_min, q_max = 0, 255  # 通常激活是 quint8
            f_min = (q_min - zp) * scale
            f_max = (q_max - zp) * scale

            entry = {
                "scale": scale,
                "zero_point": zp,
                "min": f_min,
                "max": f_max
            }
            # 添加到结果 (直接用 layer name)
            self.params[name] = entry

    def _unwrap_layer(self, module):
        """
        遍历容器内的所有子层，寻找真正的量化算子（特征是具有 .scale 属性或为 QuantizedConv 类）。
        而不是简单地取第0个元素。
        """
        # 1. 检查当前层是否就是我们要找的 (已经量化且有参数)
        # 注意：QuantizedConv2d 在 convert() 后会有 scale 属性
        if hasattr(module, "scale"):
            return module
            
        # 2. 如果是容器，遍历所有子层寻找
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.modules.container.Sequential)):
            for child in module:
                # 递归查找子层
                found = self._unwrap_layer(child)
                
                # 如果找到了合法的量化层（具备 scale），直接返回
                if hasattr(found, "scale"):
                    return found
                
                # 备用检查：如果是量化卷积但还没有 scale (极少见，除非没 convert)，也先返回
                if isinstance(found, (torch.nn.quantized.Conv2d, torch.nn.intrinsic.quantized.ConvReLU2d)):
                    return found

        # 3. 如果实在找不到，返回 module 本身（此时后续的 add_layer_params 会报错，但至少报错信息会是 'Sequential object has no attribute scale'）
        return module

    def _record_layer_meta(self, name, module):
        """Record basic layer metadata (channels, kernel, stride, padding, groups)."""
        meta = {"name": name}
        # try to read convolution-like attributes
        for attr in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "groups"):
            if hasattr(module, attr):
                val = getattr(module, attr)
                # kernel_size/stride/padding may be tuples
                if isinstance(val, (tuple, list)):
                    meta[attr] = int(val[0])
                else:
                    try:
                        meta[attr] = int(val)
                    except Exception:
                        pass
        # layer type guess
        if hasattr(module, "weight") and hasattr(module, "bias"):
            # Heuristic: groups == in_channels and kernel_size > 1 -> depthwise
            g = meta.get("groups", 1)
            kin = meta.get("in_channels", None)
            ksize = meta.get("kernel_size", 1)
            if g != 1 and kin == g:
                meta["type"] = "DEPTHWISE"
            elif ksize == 1:
                meta["type"] = "POINTWISE"
            else:
                meta["type"] = "CONV2D"
        else:
            meta["type"] = "LINEAR"

        # reference keys for weights/bias
        meta["weight_key"] = f"{name}_weights"
        meta["bias_key"] = f"{name}_bias"
        self.layer_list.append(meta)

    def process_mobilenet(self, model):
        print("Extracting parameters from native PyTorch model (Fixed)...")
        
        # 1. Input
        if hasattr(model, 'quant'):
            self.add_layer_params("input", model.quant, "activation")

        # 2. Features 遍历
        # model.features[0] 是 Conv2dNormActivation，我们需要剥开它
        init_conv = self._unwrap_layer(model.features[0])
        self.add_layer_params("init_conv", init_conv, "weight")
        self._record_layer_meta("init_conv", init_conv)
        self.add_layer_params("init_conv", init_conv, "activation")
        
        # features[1...17] -> InvertedResidual Blocks
        block_idx = 0
        for i in range(1, len(model.features) - 1):
            block = model.features[i]
            ops = block.conv
            
            # ops 是一个 Sequential，里面的子层也是 Sequential (Conv2dNormActivation)
            # 我们需要对 ops[x] 进行 unwrap
            
            if len(ops) == 3: # [Expand, Depthwise, Project]
                # 1. Expand
                exp_layer = self._unwrap_layer(ops[0])
                self.add_layer_params(f"blk{block_idx}_exp", exp_layer, "weight")
                self._record_layer_meta(f"blk{block_idx}_exp", exp_layer)
                self.add_layer_params(f"blk{block_idx}_exp", exp_layer, "activation")
                
                # 2. Depthwise
                dw_layer = self._unwrap_layer(ops[1])
                self.add_layer_params(f"blk{block_idx}_dw", dw_layer, "weight")
                self._record_layer_meta(f"blk{block_idx}_dw", dw_layer)
                self.add_layer_params(f"blk{block_idx}_dw", dw_layer, "activation")
                
                # 3. Project
                proj_layer = self._unwrap_layer(ops[2])
                self.add_layer_params(f"blk{block_idx}_proj", proj_layer, "weight")
                self._record_layer_meta(f"blk{block_idx}_proj", proj_layer)
                self.add_layer_params(f"blk{block_idx}_proj", proj_layer, "activation")
                
            elif len(ops) == 2: # [Depthwise, Project] (Block 0)
                # 1. Depthwise
                dw_layer = self._unwrap_layer(ops[0])
                self.add_layer_params(f"blk{block_idx}_dw", dw_layer, "weight")
                self._record_layer_meta(f"blk{block_idx}_dw", dw_layer)
                self.add_layer_params(f"blk{block_idx}_dw", dw_layer, "activation")
                
                # 2. Project
                proj_layer = self._unwrap_layer(ops[1])
                self.add_layer_params(f"blk{block_idx}_proj", proj_layer, "weight")
                self._record_layer_meta(f"blk{block_idx}_proj", proj_layer)
                self.add_layer_params(f"blk{block_idx}_proj", proj_layer, "activation")
            
            block_idx += 1
            
        # 3. Final Conv (features[18])
        last_conv_idx = len(model.features) - 1
        last_conv = self._unwrap_layer(model.features[last_conv_idx])
        
        name = f"final_conv_{block_idx}"
        self.add_layer_params(name, last_conv, "weight")
        self._record_layer_meta(name, last_conv)
        self.add_layer_params(name, last_conv, "activation")

        # 4. Classifier
        # model.classifier[1] 通常是直接的 Linear，不需要 unwrap，但加上也无妨
        fc_layer = self._unwrap_layer(model.classifier[1])
        self.add_layer_params("fc_final", fc_layer, "weight")
        self._record_layer_meta("fc_final", fc_layer)
        self.add_layer_params("fc_final", fc_layer, "activation")

        return self.params

def main():
    # 1. 正常加载量化模型流程 (保持默认)
    print("1. Setting up PyTorch Default Quantization...")
    model = models.quantization.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
    model.eval()

    with open("./quant/original_model.txt", "w") as file1:
        print(model, file=file1)
    
    # 默认后端 (fbgemm for x86, qnnpack for arm)
    # Quant weights Per-Channel
    torch.backends.quantized.engine = 'fbgemm' 
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 融合与准备
    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)

    # print(model)
    
    # 2. 快速校准 (这里用随机数据，实际请用真实图片)
    print("2. Calibrating (using random data)...")
    # 注意：要得到正确的 min/max，必须输入真实分布的数据
    model(torch.randn(10, 3, 224, 224))
    
    # 3. 转换
    print("3. Converting model...")
    torch.quantization.convert(model, inplace=True)

    with open("./quant/quant_model.txt", "w") as file2:
        print(model, file=file2)
    
    # # 4. 提取参数
    # extractor = NativeQuantExtractor()
    # params = extractor.process_mobilenet(model)
    
    # # 5. 保存
    # # 自定义 JSON Encoder 用于处理 numpy 数组
    # class NumpyEncoder(json.JSONEncoder):
    #     def default(self, obj):
    #         if isinstance(obj, np.integer):
    #             return int(obj)
    #         elif isinstance(obj, np.floating):
    #             return float(obj)
    #         elif isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         return super(NumpyEncoder, self).default(obj)

    # print(f"\nExtracted {len(params)} parameter sets.")
    
    # # 预览其中一个 Per-Channel 的层
    # sample_dw = params.get('blk0_dw_weights')
    # if sample_dw:
    #     print("\n[Example] blk0_dw_weights (Per-Channel check):")
    #     print(f"  Type: {sample_dw.get('quant_type')}")
    #     scales = sample_dw['scale']
    #     if isinstance(scales, list):
    #         print(f"  Scale is a list of length: {len(scales)}")
    #         print(f"  First 5 scales: {scales[:5]}")
    #     else:
    #         print(f"  Scale: {scales}")

    # # Compose simulator-friendly output: ordered layers + ordered quant params
    # quant_ordered = []
    # for meta in extractor.layer_list:
    #     name = meta.get('name')
    #     item = {"name": name, "meta": meta}

    #     # attach activation quant params if present
    #     act = params.get(name)
    #     if act is not None:
    #         item['activation'] = act

    #     # attach weight quant params if present (use weight_key from meta)
    #     wkey = meta.get('weight_key')
    #     if wkey and wkey in params:
    #         wq = dict(params[wkey])
    #         # If per-channel, compress channel arrays into single-line strings to avoid huge JSON expansion
    #         if wq.get('quant_type') == 'per_channel':
    #             scales = wq.get('scale', [])
    #             zps = wq.get('zero_point', [])
    #             try:
    #                 # use repr for floats to preserve precision but keep single-line
    #                 wq['scale_compact'] = ','.join(map(repr, scales))
    #                 wq['zero_point_compact'] = ','.join(map(str, zps))
    #             except Exception:
    #                 wq['scale_compact'] = str(scales)
    #                 wq['zero_point_compact'] = str(zps)
    #             # keep length for reference and remove the verbose lists
    #             wq['scale_len'] = len(scales)
    #             wq.pop('scale', None)
    #             wq.pop('zero_point', None)

    #         item['weight'] = wq

    #     quant_ordered.append(item)

    # out = {
    #     "layers": extractor.layer_list,
    #     "quant_params_ordered": quant_ordered
    # }

    # with open('pytorch_native_params.json', 'w') as f:
    #     json.dump(out, f, cls=NumpyEncoder, indent=4)

    # # Save weights and biases as an .npz for fast loader in simulator
    # if extractor.weights or extractor.biases:
    #     np.savez_compressed('pytorch_native_weights.npz', **{**extractor.weights, **extractor.biases})
    #     print("\n✓ Saved weights/biases to 'pytorch_native_weights.npz'")

    # print("\n✓ Saved to 'pytorch_native_params.json'")

if __name__ == "__main__":
    main()