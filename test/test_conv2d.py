"""
Python reference implementation for MCU conv2d testing
"""

import numpy as np
from quant.quant_model_utils import extract_quantized_layers
from models.utils import get_pytorch_quantized_model

# Same as testing in MCU (3x4x4)
test_input = np.array([
    # Channel 0
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]],
    
    # Channel 1
    [[17, 18, 19, 20],
     [21, 22, 23, 24],
     [25, 26, 27, 28],
     [29, 30, 31, 32]],
    
    # Channel 2
    [[33, 34, 35, 36],
     [37, 38, 39, 40],
     [41, 42, 43, 44],
     [45, 46, 47, 48]]
], dtype=np.uint8)

def python_quantized_conv2d(input_q, weights_q, bias_q, stride, padding, 
                             s_in, z_in, s_w, z_w, s_out, z_out):
    """
    The same convolution implementation as MCU native_conv2d, but in Python for reference. 

    Args:
        input_q: Quantized input tensor (C_in x H_in x W_in)
        weights_q: Quantized weights (C_out x C_in x K x K)
        bias_q: Quantized bias (C_out)
        stride: Convolution stride
        padding: Convolution padding
        s_in: Input scale
        z_in: Input zero point
        s_w: Weight scale (can be per-channel or per-tensor)
        z_w: Weight zero point (can be per-channel or per-tensor)
        s_out: Output scale
        z_out: Output zero point
    Returns:
        Quantized output tensor (C_out x H_out x W_out)
    """
    C_in, H_in, W_in = input_q.shape
    C_out, _, K, _ = weights_q.shape
    
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1
    
    if padding > 0:
        input_padded = np.pad(input_q, ((0, 0), (padding, padding), (padding, padding)), 
                              mode='constant', constant_values=z_in)
    else:
        input_padded = input_q
    
    output_q = np.zeros((C_out, H_out, W_out), dtype=np.uint8)
    
    # per-channel
    if isinstance(s_w, np.ndarray):
        s_w_array = s_w
        z_w_array = z_w
    else:
        s_w_array = np.full(C_out, s_w)
        z_w_array = np.full(C_out, z_w)
    
    print(f"Input shape: {input_q.shape}")
    print(f"Weight shape: {weights_q.shape}")
    print(f"Output shape: {output_q.shape}")
    print(f"Padding: {padding}, Stride: {stride}, Kernel: {K}")
    print(f"s_in={s_in:.6f}, z_in={z_in}")
    print(f"s_out={s_out:.6f}, z_out={z_out}")
    print(f"s_w[0]={s_w_array[0]:.6f}, z_w[0]={z_w_array[0]}")
    
    for oc in range(C_out):
        multiplier = (s_in * s_w_array[oc]) / s_out
        
        for oh in range(H_out):
            for ow in range(W_out):
                acc = bias_q[oc]
                
                start_y = oh * stride
                start_x = ow * stride
                
                for ic in range(C_in):
                    for kh in range(K):
                        for kw in range(K):
                            in_y = start_y + kh
                            in_x = start_x + kw
                            
                            input_val = int(input_padded[ic, in_y, in_x]) - z_in
                            weight_val = int(weights_q[oc, ic, kh, kw]) - z_w_array[oc]
                            
                            acc += input_val * weight_val
                
                # Requantization
                acc_float = acc * multiplier + z_out
                output_q[oc, oh, ow] = np.clip(np.round(acc_float), 0, 255).astype(np.uint8)
    
    return output_q


def main():
    print("="*60)
    print("Python Reference for MCU Conv2D Test")
    print("="*60)
    
    print("\n1. Loading model...")
    q_model = get_pytorch_quantized_model(train_loader=None)
    q_model.eval()
    
    sim_layers = extract_quantized_layers(q_model)
    
    # Get parameters of the 0th layer (init_conv)
    layer_cfg, weights_q, bias_q, qp_dict = sim_layers[0]
    
    print(f"\n2. Layer 0 info:")
    print(f"   Name: {layer_cfg.name}")
    print(f"   Type: {layer_cfg.type.name}")
    print(f"   Input channels: {layer_cfg.in_channels}")
    print(f"   Output channels: {layer_cfg.out_channels}")
    print(f"   Kernel size: {layer_cfg.kernel_size}")
    print(f"   Stride: {layer_cfg.stride}")
    print(f"   Padding: {layer_cfg.padding}")
    
    # Extract quantization parameters
    s_in = qp_dict['s_in']
    z_in = qp_dict['z_in']
    s_w = qp_dict['s_w']
    z_w = qp_dict['z_w']
    s_out = qp_dict['s_out']
    z_out = qp_dict['z_out']
    
    # Run convolution
    print("\n3. Running convolution...")
    output = python_quantized_conv2d(
        test_input, weights_q, bias_q,
        stride=layer_cfg.stride,
        padding=layer_cfg.padding,
        s_in=s_in, z_in=z_in,
        s_w=s_w, z_w=z_w,
        s_out=s_out, z_out=z_out
    )
    
    print("\n4. Output (C_out x H_out x W_out):")
    print("="*60)
    for c in range(min(5, output.shape[0])): 
        print(f"\nChannel {c}:")
        print(output[c])
    
    if output.shape[0] > 5:
        print(f"\n... (showing first 5 of {output.shape[0]} channels)")
    
    # Save as CSV for MCU comparison
    print("\n5. Saving reference output...")
    output_flat = output.flatten()
    np.savetxt('./test/reference_output.csv', output_flat, delimiter=',', fmt='%d')
    print(f"   Saved {len(output_flat)} values to ./test/reference_output.csv")
    
    print("\n6. Output statistics:")
    print(f"   Min: {output.min()}")
    print(f"   Max: {output.max()}")
    print(f"   Mean: {output.mean():.2f}")
    print(f"   Std: {output.std():.2f}")
    
    print("\n7. First 20 output values (flattened):")
    print("   ", output_flat[:20])
    
    print("\n" + "="*60)
    print("Done! Compare with MCU output.")
    print("="*60)


if __name__ == '__main__':
    main()
