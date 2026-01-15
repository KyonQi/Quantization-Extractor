# import torch
# import numpy as np
# import torchvision.models as models
# from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2

# from .quant_model_utils import extract_quantized_layers
# # import torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d
# def verify_extraction():
#     print("=== 1. 准备 PyTorch 量化模型 ===")
#     # 按照标准流程获取一个量化模型
#     q_model = q_mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
#     q_model.fuse_model()
#     q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#     torch.quantization.prepare(q_model, inplace=True)
#     # 简单校准一下（哪怕是随机数），确保 Scale/ZP 被计算出来
#     q_model(torch.randn(1, 3, 224, 224))
#     torch.quantization.convert(q_model, inplace=True)
    
#     print("=== 2. 运行提取器 ===")
#     # sim_layers 结构: [(LayerConfig, W_int8, B_int32, QP_dict), ...]
#     print(f"Type: {type(q_model.features[0][0])}")
#     sim_layers = extract_quantized_layers(q_model)
    
#     print(f"提取了 {len(sim_layers)} 层。开始逐层比对...")

#     # 我们挑选几个关键层进行“抽检”
#     # 映射关系：
#     # sim_layers[0] -> features.0.0 (Init Conv)
#     # sim_layers[-1] -> classifier.1 (Final Linear)
    
#     # --- Check 1: Init Conv (Standard Conv) ---
#     print("\n--- Checking Layer 0 (Init Conv) ---")
#     sim_layer = sim_layers[0]
#     pt_layer = q_model.features[0][0]
    
#     # 比对权重 (Int8)
#     sim_w = sim_layer[1] # W_int8
#     pt_w = pt_layer.weight().int_repr().numpy().astype(np.int8)
    
#     if np.array_equal(sim_w, pt_w):
#         print("✅ Weights Match (Bit-exact)")
#     else:
#         print(f"❌ Weights Mismatch! Max diff: {np.abs(sim_w - pt_w).max()}")

#     # 比对 Bias (Int32)
#     # 注意：模拟器里的 Bias 是我们算出来的，PyTorch 存的是 Float。
#     # 我们验证计算逻辑是否合理：反推回去看误差
#     sim_b = sim_layer[2] # B_int32
#     pt_b_float = pt_layer.bias().detach().numpy()
    
#     qp = sim_layer[3]
#     # Reconstruct float bias: b_int * (s_in * s_w)
#     # 注意 s_w 可能是 vector
#     scale_factor = qp['s_in'] * qp['s_w']
#     reconstructed_b = sim_b * scale_factor
    
#     # 允许微小的浮点误差
#     if np.allclose(reconstructed_b, pt_b_float, atol=1e-4):
#         print("✅ Bias Re-construction Passed (Logic correct)")
#     else:
#         print("❌ Bias Logic Mismatch")

#     # --- Check 2: Classifier (Linear) ---
#     print("\n--- Checking Last Layer (Classifier) ---")
#     sim_layer = sim_layers[-1]
#     pt_layer = q_model.classifier[1]
    
#     sim_w = sim_layer[1]
#     pt_w = pt_layer.weight().int_repr().numpy().astype(np.int8)
    
#     if np.array_equal(sim_w, pt_w):
#         print("✅ Linear Weights Match (Bit-exact)")
#     else:
#         print("❌ Linear Weights Mismatch")

#     print("\n验证完成！如果全为 ✅，则可以安全修改 Coordinator。")


# def test_conversion():
#     # 1. 加载模型 (FP32)
#     # quantize=False 表示只加载结构，不自动配置，我们需要手动来
#     q_model = q_mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
#     q_model.eval()

#     # 2. Fuse (融合)
#     # 此时变成 ConvBnReLU2d
#     q_model.fuse_model()
#     print(f"[Step 2] After Fuse: {type(q_model.features[0][0])}") 
#     # output: <class 'torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d'>

#     # ==========================================
#     # 3. 配置 QConfig (你可能漏了这步！)
#     # ==========================================
#     # 'fbgemm' 适用于 x86 服务器/桌面
#     # 'qnnpack' 适用于 ARM (手机/M1 Mac)
#     backend = 'fbgemm' 
#     torch.backends.quantized.engine = backend
#     q_model.qconfig = torch.quantization.get_default_qconfig(backend)

#     # ==========================================
#     # 4. Prepare (你可能也漏了这步！)
#     # ==========================================
#     # 这一步会给 ConvBnReLU2d 挂上 Observer
#     torch.quantization.prepare(q_model, inplace=True)
#     print(f"[Step 4] After Prepare: {type(q_model.features[0][0])}")
#     # 注意：此时类型依然是 ConvBnReLU2d，但如果你打印它，会发现多了 activation_post_process (Observer)

#     # 5. Calibrate (校准 - 哪怕喂随机数据也必须做)
#     # 如果不喂数据，Scale 和 ZP 算不出来，Convert 可能会报错或跳过
#     q_model(torch.randn(1, 3, 224, 224))

#     # ==========================================
#     # 6. Convert (转换)
#     # ==========================================
#     torch.quantization.convert(q_model, inplace=True)
    
#     # 5. 提取权重测试
#     # 获取第一层 (已经变成了 QuantizedConvReLU2d)
#     layer = q_model.features[0][0]
    
#     # 提取 int8 权重
#     # 注意：一定要用 .weight() 方法，而不是 .weight 属性
#     w_int8 = layer.weight().int_repr().numpy()
    
#     print(f"\nLayer Type: {type(layer)}")
#     print(f"Weight Shape: {w_int8.shape}")
#     print(f"Weight Type:  {w_int8.dtype}")
#     print(f"Sample Weights (First 10 values):\n{w_int8.flatten()[:10]}")
#     # print(f"[Step 6] After Convert: {type(q_model.features[1][0])}")
#     # 期望 Output: <class 'torch.ao.nn.quantized.modules.conv.ConvReLU2d'>

# if __name__ == "__main__":
#     # verify_extraction()
#     test_conversion()

import torch
import numpy as np
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
import sys

# 引入你的提取器
from .quant_model_utils import extract_quantized_layers

# 颜色辅助，方便看结果
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def assert_allclose(name, actual, expected, atol=1e-6, rtol=1e-5):
    """辅助函数：比较两个数组或数值"""
    if isinstance(expected, torch.Tensor):
        expected = expected.numpy()
    
    # 处理 Scale 为 0 的极端情况（防止除零报错，虽然在量化中很少见）
    if np.any(np.isnan(actual)) or np.any(np.isnan(expected)):
        print(f"{RED}[FAIL] {name}: Found NaN values!{RESET}")
        return False

    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = np.abs(actual - expected)
        print(f"{RED}[FAIL] {name} mismatch! Max diff: {np.max(diff)}{RESET}")
        print(f"   Expected (first 3): {expected.flatten()[:3]}")
        print(f"   Actual   (first 3): {actual.flatten()[:3]}")
        return False
    return True

def assert_equal(name, actual, expected):
    """辅助函数：比较整数完全相等"""
    if isinstance(expected, torch.Tensor):
        expected = expected.numpy()
    
    if not np.array_equal(actual, expected):
        # 找出不一样的个数
        mismatch_count = np.sum(actual != expected)
        print(f"{RED}[FAIL] {name} mismatch! Count: {mismatch_count}/{actual.size}{RESET}")
        print(f"   Expected (sample): {expected.flatten()[:5]}")
        print(f"   Actual   (sample): {actual.flatten()[:5]}")
        return False
    return True

def verify_layer_params(layer_idx, layer_name, pt_layer, sim_data, input_scale, input_zp):
    """
    核心比对逻辑：对比 PyTorch 层 vs 提取出的 sim_data
    sim_data: (LayerConfig, w_int8, b_int32, qp_dict)
    """
    cfg, w_ext, b_ext, qp = sim_data
    print(f"Checking Layer {layer_idx}: {layer_name} ({cfg.name})... ", end="")
    
    all_pass = True
    
    # 1. 验证权重 (Int8 Bit-Exact)
    pt_w_int8 = pt_layer.weight().int_repr().numpy().astype(np.int8)
    if not assert_equal("Weight", w_ext, pt_w_int8): all_pass = False

    # 2. 验证权重 Scale (Float Close)
    if pt_layer.weight().qscheme() in [torch.per_channel_affine, torch.per_channel_symmetric]:
        pt_s_w = pt_layer.weight().q_per_channel_scales().numpy()
    else:
        pt_s_w = float(pt_layer.weight().q_scale())
    
    if not assert_allclose("Weight Scale", qp['s_w'], pt_s_w): all_pass = False

    # 3. 验证 Bias (Logic Check)
    # 提取器是自己算的 int32 bias，PyTorch 存的是 float。
    # 我们这里复现提取器的计算逻辑，看结果是否和提取出的 b_ext 一致。
    # 这证明了提取逻辑是忠实于 PyTorch 参数的。
    if pt_layer.bias() is not None:
        pt_bias_float = pt_layer.bias().detach().numpy()
        bias_scale = input_scale * qp['s_w'] # Broadcasting if per-channel
        expected_b_int32 = np.round(pt_bias_float / bias_scale).astype(np.int32)
        
        # 允许 1 的误差 (因为 numpy.round 和 cpp round 可能在 .5 处行为极微小差异)
        # 通常应该是完全相等的
        if not assert_allclose("Bias (Int32)", b_ext, expected_b_int32, atol=1): 
            all_pass = False
    else:
        # 如果 PyTorch 没有 bias，提取器应该全是 0
        if not assert_equal("Bias (Zero)", b_ext, np.zeros_like(b_ext)): all_pass = False

    # 4. 验证输出 Scale/ZP
    if not assert_allclose("Output Scale", qp['s_out'], pt_layer.scale): all_pass = False
    if not assert_equal("Output ZP", qp['z_out'], int(pt_layer.zero_point)): all_pass = False

    if all_pass:
        print(f"{GREEN}PASS{RESET}")
    else:
        print(f"{RED}FAILED{RESET}")
        sys.exit(1) # 遇到错误直接退出

def verify_extraction_process():
    # --- 1. 准备 PyTorch 模型 ---
    print("Preparing PyTorch Model...")
    q_model = q_mobilenet_v2(pretrained=True, quantize=False)
    q_model.eval()
    q_model.fuse_model()
    q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(q_model, inplace=True)
    q_model.eval()
    # 简单校准以生成 Scale
    q_model(torch.randn(1, 3, 224, 224))
    torch.quantization.convert(q_model, inplace=True)
    q_model.eval()
    
    # --- 2. 运行提取器 ---
    print("Running Extractor...")
    
    sim_layers = extract_quantized_layers(q_model)
    
    # --- 3. 开始同步遍历比对 ---
    print("\n=== Starting Verification ===")
    
    # 追踪全局 Input Scale/ZP (用于 Bias 验证)
    curr_s = float(q_model.quant.scale.item())
    curr_z = int(q_model.quant.zero_point.item())
    
    sim_idx = 0
    
    # 3.1 Init Conv
    verify_layer_params(sim_idx, "features.0.0", q_model.features[0][0], sim_layers[sim_idx], curr_s, curr_z)
    # 更新 scale 为上一层的输出
    curr_s = sim_layers[sim_idx][3]['s_out']
    curr_z = sim_layers[sim_idx][3]['z_out']
    sim_idx += 1
    
    # 3.2 Inverted Residual Blocks
    for i, block in enumerate(q_model.features[1:-1]):
        ops = list(block.conv)
        use_res = block.use_res_connect
        
        # Case A: Expand -> DW -> Proj
        if len(ops) == 4:
            # Expand
            verify_layer_params(sim_idx, f"blk{i}.exp", ops[0][0], sim_layers[sim_idx], curr_s, curr_z)
            curr_s = sim_layers[sim_idx][3]['s_out']
            curr_z = sim_layers[sim_idx][3]['z_out']
            sim_idx += 1
            
            # DW
            verify_layer_params(sim_idx, f"blk{i}.dw", ops[1][0], sim_layers[sim_idx], curr_s, curr_z)
            curr_s = sim_layers[sim_idx][3]['s_out']
            curr_z = sim_layers[sim_idx][3]['z_out']
            sim_idx += 1
            
            # Proj
            verify_layer_params(sim_idx, f"blk{i}.proj", ops[2], sim_layers[sim_idx], curr_s, curr_z)
            # 这里要注意：如果是残差块，Project 输出的 Scale 还是 Project 自己的
            # 但是 Extractor 会把下一层的 curr_s 更新为 block.skip_add 的 Scale
            # 我们需要在验证逻辑里保持一致
            
            qp_proj = sim_layers[sim_idx][3]
            
            if use_res:
                # 验证 Extractor 是否正确捕获了 QFunctional 的 Scale
                expected_skip_s = float(block.skip_add.scale) # .item()
                expected_skip_z = int(block.skip_add.zero_point)
                
                assert_allclose(f"Blk{i} Skip Scale", qp_proj['residual_out_scale'], expected_skip_s)
                assert_equal(f"Blk{i} Skip ZP", qp_proj['residual_out_zp'], expected_skip_z)
                
                curr_s = expected_skip_s
                curr_z = expected_skip_z
            else:
                curr_s = qp_proj['s_out']
                curr_z = qp_proj['z_out']
            
            sim_idx += 1

        # Case B: DW -> Proj
        elif len(ops) == 3:
            # DW
            verify_layer_params(sim_idx, f"blk{i}.dw", ops[0][0], sim_layers[sim_idx], curr_s, curr_z)
            curr_s = sim_layers[sim_idx][3]['s_out']
            curr_z = sim_layers[sim_idx][3]['z_out']
            sim_idx += 1
            
            # Proj
            verify_layer_params(sim_idx, f"blk{i}.proj", ops[1], sim_layers[sim_idx], curr_s, curr_z)
            
            qp_proj = sim_layers[sim_idx][3]
            if use_res:
                expected_skip_s = float(block.skip_add.scale.item())
                expected_skip_z = int(block.skip_add.zero_point.item())
                assert_allclose(f"Blk{i} Skip Scale", qp_proj['residual_out_scale'], expected_skip_s)
                curr_s = expected_skip_s
                curr_z = expected_skip_z
            else:
                curr_s = qp_proj['s_out']
                curr_z = qp_proj['z_out']
            
            sim_idx += 1

    # 3.3 Final Conv
    verify_layer_params(sim_idx, "features.last", q_model.features[-1][0], sim_layers[sim_idx], curr_s, curr_z)
    curr_s = sim_layers[sim_idx][3]['s_out']
    curr_z = sim_layers[sim_idx][3]['z_out']
    sim_idx += 1
    
    # 3.4 Classifier
    verify_layer_params(sim_idx, "classifier", q_model.classifier[1], sim_layers[sim_idx], curr_s, curr_z)
    
    print(f"\n{GREEN}=== Verification Complete: ALL MATCHED ==={RESET}")

if __name__ == "__main__":
    verify_extraction_process()