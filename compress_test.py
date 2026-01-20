import torch
import torchvision.models as models
import torchvision.models.quantization as quant
import numpy as np
import lz4.frame
import pandas as pd

from model_utils import preprocess_image
from compression.compress_tools import bitmap_ans_compressor, bitmap_compressor

def analyze_compression_ratio():
    # 1. 准备量化模型
    print("Preparing Model...")
    q_model = quant.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)
    q_model.eval()
    q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    q_model.fuse_model()
    torch.quantization.prepare(q_model, inplace=True)
    q_model.eval()
    q_model(torch.randn(1, 3, 224, 224))
    torch.quantization.convert(q_model, inplace=True)
    
    # 2. 准备钩子
    results = []

    def hook_fn(name, layer_type):
        def fn(module, input, output):
            # 获取 Int8/Uint8 数据
            if hasattr(output, 'int_repr'):
                data = output.int_repr().numpy()
            else:
                # 针对 float 输出 (如未量化的层)，先跳过或模拟量化
                return
            
            # 统计稀疏度 (0 的比例)
            # 注意: 对于 uint8, 0 值取决于 zero_point。
            # 但 LZ4 不在乎是不是 0，只在乎重不重复。
            # 这里我们只统计 bytes 大小

            total_zeros = np.sum(data == 0)
            ratio_zeros = total_zeros / data.size
            original_bytes = data.tobytes()
            raw_size = len(original_bytes)

            # 执行 LZ4 压缩
            # level=0 是默认快速压缩，适合实时通信
            # compressed_bytes = lz4.frame.compress(original_bytes, compression_level=0)
            # comp_size = len(compressed_bytes)
            compressed_bytes = bitmap_ans_compressor.compress(data)
            comp_size = len(compressed_bytes)
            # compressed_bytes = bitmap_compressor.compress(data)
            # comp_size = len(compressed_bytes)
            
            ratio = raw_size / comp_size
            
            # 计算稀疏度 (假设 ZP 附近的数值重复率高)
            # 简单统计一下最频繁出现的字节占比
            vals, counts = np.unique(data, return_counts=True)
            max_freq = counts.max() / data.size
            
            results.append({
                "Layer": name,
                "Type": layer_type,
                "Shape": str(data.shape),
                "Raw (KB)": raw_size / 1024,
                "Comp (KB)": comp_size / 1024,
                "Ratio of Zeros": ratio_zeros,
                "Ratio": ratio,
                "MaxFreq": max_freq # 最常出现的数值占比 (近似稀疏度)
            })
        return fn

    # 3. 注册钩子 (模拟 Extractor 的顺序)
    # Init
    q_model.features[0].register_forward_hook(hook_fn("Init Conv", "Conv"))
    
    for i, block in enumerate(q_model.features[1:-1]):
        ops = [op for op in block.conv if not isinstance(op, torch.nn.Identity)]
        if len(ops) == 4: # Exp -> DW -> Proj
            ops[0].register_forward_hook(hook_fn(f"Blk{i} Expand", "ReLU (High Dim)"))
            ops[1].register_forward_hook(hook_fn(f"Blk{i} Depthwise", "ReLU (High Dim)"))
            # 注意：Project 输出没有 ReLU
            ops[2].register_forward_hook(hook_fn(f"Blk{i} Project", "(Low Dim)"))
        elif len(ops) == 3: # DW -> Proj
            ops[0].register_forward_hook(hook_fn(f"Blk{i} Depthwise", "ReLU (High Dim)"))
            ops[1].register_forward_hook(hook_fn(f"Blk{i} Project", "(Low Dim)"))
    
    q_model.features[-1].register_forward_hook(hook_fn(f"Final Conv", "Conv"))
    
    q_model.classifier[1].register_forward_hook(hook_fn("FC Final", "Linear"))

    # 4. 运行推理
    print("Running Inference...")
    q_model.eval()
    with torch.no_grad():
        input_data = preprocess_image("./img/panda.jpg")
        q_model(input_data)

    # 5. 展示分析
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("Compression Analysis")
    print("="*80)
    # 设置显示格式
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df[['Layer', 'Type', 'Raw (KB)', 'Comp (KB)', 'Ratio of Zeros', 'Ratio', 'MaxFreq']])
    
    print("\nSummary:")
    print(f"Average Ratio (ReLU Layers):   {df[df['Type'].str.contains('ReLU')]['Ratio'].mean():.2f}x")
    print(f"Average Ratio (Linear Layers): {df[df['Type'].str.contains('Linear')]['Ratio'].mean():.2f}x")

    total_raw = sum(result['Raw (KB)'] for result in results)
    total_comp = sum(result['Comp (KB)'] for result in results)
    overall_ratio = total_raw / total_comp
    print("\nOverall original size (KB):      {:.2f} KB".format(total_raw))
    print("Overall compressed size (KB):    {:.2f} KB".format(total_comp))
    print(f"Overall Compression Ratio:      {overall_ratio:.2f}x")


if __name__ == "__main__":
    analyze_compression_ratio()