"""
Python reference implementation for MCU depthwise-conv2d testing
"""

import numpy as np
from quant.quant_model_utils import extract_quantized_layers
from models.utils import get_pytorch_quantized_model

# Test input for depthwise conv2d (32x4x4)
# Same data structure as in Worker/include/conv/conv2d_test_data.h
test_input_dw = np.array([
    # Channel 0
    [[120, 130, 125, 135],
     [110, 140, 115, 128],
     [132, 118, 142, 122],
     [138, 126, 130, 145]],
    # Channel 1
    [[95, 105, 100, 110],
     [88, 115, 92, 103],
     [107, 98, 118, 101],
     [113, 104, 108, 120]],
    # Channel 2
    [[150, 160, 155, 165],
     [145, 170, 148, 158],
     [162, 152, 172, 156],
     [168, 154, 160, 175]],
    # Channel 3
    [[75, 85, 80, 90],
     [70, 95, 73, 83],
     [87, 78, 98, 81],
     [93, 84, 88, 100]],
    # Channel 4
    [[200, 210, 205, 215],
     [195, 220, 198, 208],
     [212, 202, 222, 206],
     [218, 204, 210, 225]],
    # Channel 5
    [[50, 60, 55, 65],
     [45, 70, 48, 58],
     [62, 52, 72, 56],
     [68, 54, 60, 75]],
    # Channel 6
    [[128, 138, 133, 143],
     [123, 148, 126, 136],
     [140, 130, 150, 134],
     [146, 132, 138, 153]],
    # Channel 7
    [[180, 190, 185, 195],
     [175, 200, 178, 188],
     [192, 182, 202, 186],
     [198, 184, 190, 205]],
    # Channel 8
    [[112, 122, 117, 127],
     [107, 132, 110, 120],
     [124, 114, 134, 118],
     [130, 116, 122, 137]],
    # Channel 9
    [[85, 95, 90, 100],
     [80, 105, 83, 93],
     [97, 87, 107, 91],
     [103, 89, 95, 110]],
    # Channel 10
    [[155, 165, 160, 170],
     [150, 175, 153, 163],
     [167, 157, 177, 161],
     [173, 159, 165, 180]],
    # Channel 11
    [[65, 75, 70, 80],
     [60, 85, 63, 73],
     [77, 67, 87, 71],
     [83, 69, 75, 90]],
    # Channel 12
    [[190, 200, 195, 205],
     [185, 210, 188, 198],
     [202, 192, 212, 196],
     [208, 194, 200, 215]],
    # Channel 13
    [[40, 50, 45, 55],
     [35, 60, 38, 48],
     [52, 42, 62, 46],
     [58, 44, 50, 65]],
    # Channel 14
    [[135, 145, 140, 150],
     [130, 155, 133, 143],
     [147, 137, 157, 141],
     [153, 139, 145, 160]],
    # Channel 15
    [[170, 180, 175, 185],
     [165, 190, 168, 178],
     [182, 172, 192, 176],
     [188, 174, 180, 195]],
    # Channel 16
    [[105, 115, 110, 120],
     [100, 125, 103, 113],
     [117, 107, 127, 111],
     [123, 109, 115, 130]],
    # Channel 17
    [[80, 90, 85, 95],
     [75, 100, 78, 88],
     [92, 82, 102, 86],
     [98, 84, 90, 105]],
    # Channel 18
    [[145, 155, 150, 160],
     [140, 165, 143, 153],
     [157, 147, 167, 151],
     [163, 149, 155, 170]],
    # Channel 19
    [[60, 70, 65, 75],
     [55, 80, 58, 68],
     [72, 62, 82, 66],
     [78, 64, 70, 85]],
    # Channel 20
    [[185, 195, 190, 200],
     [180, 205, 183, 193],
     [197, 187, 207, 191],
     [203, 189, 195, 210]],
    # Channel 21
    [[35, 45, 40, 50],
     [30, 55, 33, 43],
     [47, 37, 57, 41],
     [53, 39, 45, 60]],
    # Channel 22
    [[125, 135, 130, 140],
     [120, 145, 123, 133],
     [137, 127, 147, 131],
     [143, 129, 135, 150]],
    # Channel 23
    [[160, 170, 165, 175],
     [155, 180, 158, 168],
     [172, 162, 182, 166],
     [178, 164, 170, 185]],
    # Channel 24
    [[100, 110, 105, 115],
     [95, 120, 98, 108],
     [112, 102, 122, 106],
     [118, 104, 110, 125]],
    # Channel 25
    [[75, 85, 80, 90],
     [70, 95, 73, 83],
     [87, 77, 97, 81],
     [93, 79, 85, 100]],
    # Channel 26
    [[140, 150, 145, 155],
     [135, 160, 138, 148],
     [152, 142, 162, 146],
     [158, 144, 150, 165]],
    # Channel 27
    [[55, 65, 60, 70],
     [50, 75, 53, 63],
     [67, 57, 77, 61],
     [73, 59, 65, 80]],
    # Channel 28
    [[180, 190, 185, 195],
     [175, 200, 178, 188],
     [192, 182, 202, 186],
     [198, 184, 190, 205]],
    # Channel 29
    [[30, 40, 35, 45],
     [25, 50, 28, 38],
     [42, 32, 52, 36],
     [48, 34, 40, 55]],
    # Channel 30
    [[120, 130, 125, 135],
     [115, 140, 118, 128],
     [132, 122, 142, 126],
     [138, 124, 130, 145]],
    # Channel 31
    [[155, 165, 160, 170],
     [150, 175, 153, 163],
     [167, 157, 177, 161],
     [173, 159, 165, 180]]
], dtype=np.uint8)


def python_quantized_depthwise_conv2d(input_q, weights_q, bias_q, stride, padding, 
                                       s_in, z_in, s_w, z_w, s_out, z_out):
    """
    Python reference implementation of depthwise convolution, matching MCU implementation.
    
    In depthwise convolution:
    - Each input channel is convolved with its own set of filters independently
    - input_channels == output_channels
    - weights shape: (C_out, 1, K, K) or (C, K, K) for depthwise
    
    Args:
        input_q: Quantized input tensor (C x H_in x W_in)
        weights_q: Quantized weights (C x K x K) for depthwise
        bias_q: Quantized bias (C)
        stride: Convolution stride
        padding: Convolution padding
        s_in: Input scale
        z_in: Input zero point
        s_w: Weight scale (per-channel array)
        z_w: Weight zero point (per-channel array)
        s_out: Output scale
        z_out: Output zero point
    Returns:
        Quantized output tensor (C x H_out x W_out)
    """
    C, H_in, W_in = input_q.shape
    if len(weights_q.shape) == 4:
        # weights shape: (C, 1, K, K)
        _, _, K, _ = weights_q.shape
        weights_q = weights_q.squeeze(1)  # (C, K, K)
    else:
        # weights shape: (C, K, K)
        _, K, _ = weights_q.shape
    
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1
    
    output_q = np.zeros((C, H_out, W_out), dtype=np.uint8)
    
    # Handle per-channel scales
    if isinstance(s_w, np.ndarray):
        s_w_array = s_w
        z_w_array = z_w
    else:
        s_w_array = np.full(C, s_w)
        z_w_array = np.full(C, z_w)
    
    print(f"Input shape: {input_q.shape}")
    print(f"Weight shape: {weights_q.shape}")
    print(f"Output shape: {output_q.shape}")
    print(f"Padding: {padding}, Stride: {stride}, Kernel: {K}")
    print(f"s_in={s_in:.6f}, z_in={z_in}")
    print(f"s_out={s_out:.6f}, z_out={z_out}")
    print(f"s_w[0]={s_w_array[0]:.6f}, z_w[0]={z_w_array[0]}")
    
    # Depthwise convolution - each channel processes independently
    for oc in range(C):
        multiplier = (s_in * s_w_array[oc]) / s_out
        
        for oh in range(H_out):
            for ow in range(W_out):
                acc = bias_q[oc]
                
                start_y = oh * stride - padding
                start_x = ow * stride - padding
                
                # For depthwise, only convolve with the same channel
                for kh in range(K):
                    for kw in range(K):
                        in_y = start_y + kh
                        in_x = start_x + kw
                        
                        # Check valid coordinates (handle padding)
                        if 0 <= in_y < H_in and 0 <= in_x < W_in:
                            input_val = int(input_q[oc, in_y, in_x]) - z_in
                            weight_val = int(weights_q[oc, kh, kw]) - z_w_array[oc]
                            acc += input_val * weight_val
                
                # Requantization
                acc_float = acc * multiplier + z_out
                output_q[oc, oh, ow] = np.clip(np.round(acc_float), 0, 255).astype(np.uint8)
    
    return output_q


def main():
    print("="*60)
    print("Python Reference for MCU Depthwise Conv2D Test")
    print("="*60)
    
    print("\n1. Loading model...")
    q_model = get_pytorch_quantized_model(train_loader=None)
    q_model.eval()
    
    sim_layers = extract_quantized_layers(q_model)
    
    # Find a depthwise conv layer in the model
    # MobileNetV2 typically has depthwise conv layers after the initial conv
    depthwise_layer_idx = None
    for idx, (layer_cfg, _, _, _) in enumerate(sim_layers):
        if 'depthwise' in layer_cfg.name.lower() or \
           (layer_cfg.in_channels == layer_cfg.out_channels and layer_cfg.groups == layer_cfg.in_channels):
            depthwise_layer_idx = idx
            print(f"   Found depthwise layer at index {idx}: {layer_cfg.name}")
            break
    
    if depthwise_layer_idx is None:
        print("   Warning: No depthwise layer found, using layer 1 as fallback")
        depthwise_layer_idx = 1
    
    # Get parameters of the depthwise layer
    layer_cfg, weights_q, bias_q, qp_dict = sim_layers[depthwise_layer_idx]
    
    print(f"\n2. Layer {depthwise_layer_idx} info:")
    print(f"   Name: {layer_cfg.name}")
    print(f"   Type: {layer_cfg.type.name}")
    print(f"   Input channels: {layer_cfg.in_channels}")
    print(f"   Output channels: {layer_cfg.out_channels}")
    print(f"   Groups: {layer_cfg.groups}")
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
    
    # Use test input if channels match, otherwise use first N channels
    if test_input_dw.shape[0] >= layer_cfg.in_channels:
        test_input = test_input_dw[:layer_cfg.in_channels]
        print(f"\n3. Using test input (first {layer_cfg.in_channels} channels of 32)")
    else:
        # Generate random test input if needed
        print(f"\n3. Generating random test input ({layer_cfg.in_channels} channels)")
        test_input = np.random.randint(0, 256, 
                                       size=(layer_cfg.in_channels, 4, 4), 
                                       dtype=np.uint8)
    
    # Run depthwise convolution
    print("\n4. Running depthwise convolution...")
    output = python_quantized_depthwise_conv2d(
        test_input, weights_q, bias_q,
        stride=layer_cfg.stride,
        padding=layer_cfg.padding,
        s_in=s_in, z_in=z_in,
        s_w=s_w, z_w=z_w,
        s_out=s_out, z_out=z_out
    )
    
    print("\n5. Output (C x H_out x W_out):")
    print("="*60)
    for c in range(min(5, output.shape[0])): 
        print(f"\nChannel {c}:")
        print(output[c])
    
    if output.shape[0] > 5:
        print(f"\n... (showing first 5 of {output.shape[0]} channels)")
    
    # Save as CSV for MCU comparison
    print("\n6. Saving reference output...")
    output_flat = output.flatten()
    np.savetxt('./test/reference_output_depthwise.csv', output_flat, delimiter=',', fmt='%d')
    print(f"   Saved {len(output_flat)} values to ./test/reference_output_depthwise.csv")
    
    print("\n7. Output statistics:")
    print(f"   Min: {output.min()}")
    print(f"   Max: {output.max()}")
    print(f"   Mean: {output.mean():.2f}")
    print(f"   Std: {output.std():.2f}")
    
    print("\n8. First 20 output values (flattened):")
    print("   ", output_flat[:20])
    
    # Save test input for MCU
    print("\n9. Saving test input...")
    test_input_flat = test_input.flatten()
    np.savetxt('./test/test_input_depthwise.csv', test_input_flat, delimiter=',', fmt='%d')
    print(f"   Saved {len(test_input_flat)} values to ./test/test_input_depthwise.csv")
    
    print("\n" + "="*60)
    print("Done! Compare with MCU output.")
    print("MCU Implementation:")
    print("  - Uses depthwise_conv2d() function")
    print("  - Input: uint8_t array [C][H][W]")
    print("  - Weights: int8_t array [C][K][K]")
    print("  - Each channel processes independently")
    print("="*60)


if __name__ == '__main__':
    main()