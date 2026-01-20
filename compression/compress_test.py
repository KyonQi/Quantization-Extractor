import torch
# import torchvision.models as models
# import torchvision.models.quantization as quant
import random
import numpy as np
# import lz4.frame
import pandas as pd

import compression.compressors as compressors
from model_utils import preprocess_image
from models.utils import get_pytorch_quantized_model

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

def analyze_compression_ratio():
    set_seed(42)

    q_model = get_pytorch_quantized_model(train_loader=None, num_calibration_batches=200) # load from saved model
    compressor_list = [compressors.ANSCompressor(), compressors.BitMapCompressor(), compressors.BitMapANSCompressor()]
    for compressor in compressor_list:
        hooks = []
        # 1. prepare the quantized model
        print("Preparing Model...")
        # 2. prepare hooks
        results = []
        def hook_fn(name, layer_type, compressor: compressors.Compressor = compressors.BitMapANSCompressor()):
            def fn(module, input, output: torch.Tensor):
                data = output.int_repr().numpy()

                total_zeros = np.sum(data == 0)
                ratio_zeros = total_zeros / data.size # record ratio of zeros
                
                original_bytes = data.tobytes()
                raw_size = len(original_bytes)

                compressed_bytes = compressor.compress(data=data)
                compress_size = len(compressed_bytes)

                ratio = raw_size / compress_size
                # calculate sparsity approx via max frequency
                vals, counts = np.unique(data, return_counts=True)
                max_freq = counts.max() / data.size

                results.append({
                    "Layer": name,
                    "Type": layer_type,
                    "Shape": str(data.shape),
                    "Raw (KB)": raw_size / 1024,
                    "Comp (KB)": compress_size / 1024,
                    "Ratio of Zeros": ratio_zeros,
                    "Ratio": ratio,
                    "MaxVal": vals[counts.argmax()], # most frequent value
                    "MaxFreq": max_freq # most frequent value ratio
                })

            return fn

        # 3. Register hooks
        print(f"Analyzing with compressor: {compressor.__class__.__name__}")
        h = q_model.features[0].register_forward_hook(hook_fn("Init Conv", "Conv", compressor))
        hooks.append(h)
        for i, block in enumerate(q_model.features[1:-1]):
            ops = [op for op in block.conv if not isinstance(op, torch.nn.Identity)]
            if len(ops) == 4: # Exp -> DW -> Proj
                h = ops[0].register_forward_hook(hook_fn(f"Blk{i} Expand", "ReLU (High Dim)", compressor))
                hooks.append(h)
                h = ops[1].register_forward_hook(hook_fn(f"Blk{i} Depthwise", "ReLU (High Dim)", compressor))
                hooks.append(h)
                # Note: Project output does not have ReLU
                h = ops[2].register_forward_hook(hook_fn(f"Blk{i} Project", "(Low Dim)", compressor))
                hooks.append(h)
            elif len(ops) == 3: # DW -> Proj
                h = ops[0].register_forward_hook(hook_fn(f"Blk{i} Depthwise", "ReLU (High Dim)", compressor))
                hooks.append(h)
                h = ops[1].register_forward_hook(hook_fn(f"Blk{i} Project", "(Low Dim)", compressor))
                hooks.append(h)
        h = q_model.features[-1].register_forward_hook(hook_fn(f"Final Conv", "Conv", compressor))
        hooks.append(h)
        h = q_model.classifier[1].register_forward_hook(hook_fn("FC Final", "Linear", compressor))
        hooks.append(h)

        # 4. Run Inference
        print("Running Inference...")
        q_model.eval()
        input_data = preprocess_image("./img/panda.jpg")
        q_model(input_data)
        
        for hook in hooks:
            hook.remove()

        # with torch.no_grad():
        #     input_data = preprocess_image("./img/panda.jpg")
        #     q_model(input_data)

        # 5. Display Analysis
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("Compression Analysis")
        print("="*80)
        # Set display format
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(df[['Layer', 'Type', 'Raw (KB)', 'Comp (KB)', 'Ratio of Zeros', 'Ratio', 'MaxVal', 'MaxFreq']])
        
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