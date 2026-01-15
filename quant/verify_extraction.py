import torch
import numpy as np
import torchvision.models as models
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
import sys

from .quant_model_utils import extract_quantized_layers

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
    q_model = q_mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
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
    verify_layer_params(sim_idx, "final conv", q_model.features[-1][0], sim_layers[sim_idx], curr_s, curr_z)
    curr_s = sim_layers[sim_idx][3]['s_out']
    curr_z = sim_layers[sim_idx][3]['z_out']
    sim_idx += 1
    
    # 3.4 Classifier
    verify_layer_params(sim_idx, "classifier", q_model.classifier[1], sim_layers[sim_idx], curr_s, curr_z)
    
    print(f"\n{GREEN}=== Verification Complete: ALL MATCHED ==={RESET}")

if __name__ == "__main__":
    verify_extraction_process()