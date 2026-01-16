import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torchvision.models.quantization as quant_models
from torchvision import transforms
from PIL import Image
import os

# 导入你的模块
from protocol.protocol import MessageType
from quant.quant_model_utils import extract_quantized_layers
from coordinator import QuantCoordinator
from worker import QuantWorker

# ==========================================
# 1. PyTorch Hook 工具: 抓取中间层输出
# ==========================================
class PyTorchProbe:
    def __init__(self):
        self.outputs = {} # layer_name -> numpy_int8_array
        self.hooks = []

    def hook_fn(self, name):
        def fn(module, input, output):
            # output 是 Quantized Tensor
            if hasattr(output, 'int_repr'):
                # 提取底层整数 (Int8/Uint8)
                data = output.int_repr().numpy()
                self.outputs[name] = data
            else:
                # 某些层可能没量化 (如最后的 Linear 输出可能是 Float 如果没 DeQuantStub)
                pass
        return fn

    def register_hooks(self, q_model):
        """
        按照 Extractor 的遍历逻辑，给每一个计算层挂钩子
        """
        layer_idx = 0
        
        # 1. Init Conv
        self._register(q_model.features[0][0], f"Layer_{layer_idx:02d}_init_conv")
        layer_idx += 1
        
        # 2. Blocks
        for i, block in enumerate(q_model.features[1:-1]):
            ops = [op for op in block.conv]
            use_res = block.use_res_connect

            # 按照 Extractor 的逻辑命名，确保顺序一致
            if len(ops) == 4: # Exp, DW, Proj
                self._register(ops[0][0], f"Layer_{layer_idx:02d}_blk{i}_exp")
                layer_idx += 1
                self._register(ops[1][0], f"Layer_{layer_idx:02d}_blk{i}_dw")
                layer_idx += 1
                # Hook 整个 block 以获取 add 后的输出（如果有 residual）
                # Block 的 forward 返回值就是 add 后的结果
                self._register(block, f"Layer_{layer_idx:02d}_blk{i}_proj")
                layer_idx += 1
            elif len(ops) == 3: # DW, Proj
                self._register(ops[0][0], f"Layer_{layer_idx:02d}_blk{i}_dw")
                layer_idx += 1
                # Hook 整个 block 以获取 add 后的输出（如果有 residual）
                self._register(block, f"Layer_{layer_idx:02d}_blk{i}_proj")
                layer_idx += 1
        
        # 3. Final Conv
        self._register(q_model.features[-1][0], f"Layer_{layer_idx:02d}_final_conv")
        layer_idx += 1
        
        # 4. Classifier
        self._register(q_model.classifier[1], f"Layer_{layer_idx:02d}_fc_final")

    def _register(self, real_layer, name):
        # 使用 get_real_layer 逻辑找到真正的层
        # real_layer = get_real_layer(container_or_layer)
        h = real_layer.register_forward_hook(self.hook_fn(name))
        self.hooks.append(h)
    
    def clear(self):
        self.outputs = {}
    
    def remove_hooks(self):
        for h in self.hooks: h.remove()

# # 需要复用 extractor 里的工具
# def get_real_layer(layer):
#     try: return layer[0]
#     except: return layer

# ==========================================
# 2. 主调试逻辑
# ==========================================
def debug_layers():
    print("Loading Quantized Model...")
    # 强制重新校准一次或加载
    # 这里简单起见，直接由 extractor 的流程通过文件加载
    # 假设你已经有了 mobilenet_v2_quantized.pth
    # 如果没有，请先运行之前的 calibrate_and_save_model
    
    q_model = quant_models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
    q_model.eval()
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    q_model.qconfig = torch.quantization.get_default_qconfig(backend)
    q_model.fuse_model()
    torch.quantization.prepare(q_model, inplace=True)
    q_model(torch.randn(1, 3, 224, 224)) # Dummy run to allow convert
    torch.quantization.convert(q_model, inplace=True)
    
    try:
        q_model.load_state_dict(torch.load("mobilenet_v2_quantized.pth"))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # A. 准备数据 (一张随机图，或者真实图)
    # 建议用随机图先测对齐，排除数据加载差异
    input_tensor = torch.randn(1, 3, 224, 224) 
    # 如果想用真实图片，解开下面注释
    # input_tensor = ... (load from image)

    # B. PyTorch 推理 & 抓取
    probe = PyTorchProbe()
    probe.register_hooks(q_model)
    
    print("Running PyTorch Inference...")
    with torch.no_grad():
        q_model(input_tensor)
    
    pt_results = probe.outputs
    # 按名称排序
    sorted_pt_keys = sorted(pt_results.keys())

    # C. Simulator 推理 & 抓取
    print("Running Simulator Step-by-Step...")
    sim_layers = extract_quantized_layers(q_model)
    coord = QuantCoordinator(num_workers=1) # Debug 用 1 个 worker 方便排查
    
    # 1. Input Quantization
    s_in = sim_layers[0][3]['s_in']
    z_in = sim_layers[0][3]['z_in']
    img_np = input_tensor.numpy().squeeze(0)
    
    coord.quantize_input(img_np, s_in, z_in)
    
    # 检查输入的对齐情况
    # PyTorch 输入通常通过 QuantStub，这里我们手动验证一下
    # print(f"Input Check: PyTorch expects input quantized with s={s_in}, z={z_in}")
    
    print("\n" + "="*100)
    print(f"{'Layer Name':<30} | {'Shape (PT vs Sim)':<25} | {'Max Diff':<10} | {'Mean Diff':<10} | {'Status'}")
    print("="*100)

    coord.workers = [QuantWorker(i, coord.task_queue[i], coord.result_queue) for i in range(coord.num_workers)]
    for w in coord.workers:
        w.start()

    # 2. 逐层运行 Simulator 并对比
    for i, (layer_cfg, w, b, qp) in enumerate(sim_layers):
        # 运行一层
        coord._run_layer(layer_cfg, w, b, qp)
        
        # 获取 Simulator 输出
        sim_out = coord.feature_map.copy() # uint8
        
        # 获取对应的 PyTorch 输出
        # 我们假设 extractor 的顺序和 probe hook 的顺序是完全一致的
        pt_key = sorted_pt_keys[i]
        pt_out = pt_results[pt_key].squeeze(0) # 去掉 Batch 维度 (1, C, H, W) -> (C, H, W)
        
        # 特殊处理: Linear 层
        if layer_cfg.type == str(layer_cfg.type) and "LINEAR" in str(layer_cfg.type): 
             # Sim output is (1000,), PT output might be (1000,) or (1, 1000)
             pt_out = pt_out.flatten()
        
        # --- 对比环节 ---
        
        # 1. 形状检查        
        shape_match = (sim_out.shape == pt_out.shape)
        shape_str = f"{str(pt_out.shape)} vs {str(sim_out.shape)}"
        
        # 2. 类型对齐
        # PyTorch 输出通常是 Uint8 (0-255) 或 Int8 (-128-127)
        # 这里的 int_repr() 返回的根据 qscheme 不同而不同
        # FBGEMM 通常使用 Uint8 for activations (quint8)
        # 如果 PyTorch 是 Int8 (qint8)，我们需要转换或者让 Simulator 适配
        
        # 转换 PyTorch 数据为 int32 以便安全计算 diff
        pt_data = pt_out.astype(np.int32)
        sim_data = sim_out.astype(np.int32)
        
        # 处理可能的 signed/unsigned 差异
        # 如果 max diff 接近 128，说明一个是 signed 一个是 unsigned
        
        diff = np.abs(pt_data - sim_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        status = "✅"
        if not shape_match: status = "❌ Shape"
        elif max_diff == 0: status = "✅ Perfect"
        elif max_diff <= 10 or mean_diff <= 2: status = "⚠️ Rounding" # 允许 10 的误差 (舍入方式不同)
        elif max_diff > 120: status = "❌ Sign/Offset" # 可能是 0 vs 128 的问题
        else: status = "❌ Wrong Calc"
        
        print(f"{pt_key:<30} | {shape_str:<25} | {max_diff:<10} | {mean_diff:<10.4f} | {status}")
        
        # 如果遇到错误，打印局部数据详情，并中止
        if max_diff > 10 or mean_diff > 2:
            print("-" * 50)
            print(f"DEBUG FAIL at {pt_key}")
            print(f"PyTorch Sample (First 10): {pt_data.flatten()[:10]}")
            print(f"Simulat Sample (First 10): {sim_data.flatten()[:10]}")
            print(f"PyTorch Min/Max: {pt_data.min()}/{pt_data.max()}")
            print(f"Simulat Min/Max: {sim_data.min()}/{sim_data.max()}")
            
            # 常见错误自检提示
            if abs(pt_data.mean() - sim_data.mean()) > 100:
                print("💡 提示: 均值差异巨大，检查 ZeroPoint 是否被加了两次或忘记减去？")
            
            # 停止后续检查，专注修这层
            # break

    # 3. Clear
    for q in coord.task_queue:
        q.put((MessageType.TERMINATE, None))
    for w in coord.workers:
        w.join()
    probe.remove_hooks()
if __name__ == "__main__":
    debug_layers()