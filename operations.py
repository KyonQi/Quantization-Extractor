""" Calculating operations for simulation """
import numpy as np

from protocol import QuantParams

def pad_input(x: np.ndarray, padding: int) -> np.ndarray:
    """ Pad the input feature map with zeros """
    if padding == 0:
        return x
    # x: (C, H, W) -> Pad H and W dimension with 0
    return np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

def relu6(x: np.ndarray) -> np.ndarray:
    """ ReLU6 activation function """
    return np.minimum(np.maximum(0, x), 6)

def numpy_conv2d(input_patch: np.ndarray, weights: np.ndarray, bias: np.ndarray, stride: int, groups: int) -> np.ndarray:
    """
    Conv 2d operation using numpy
    Input: (C_in, H_in, W_in)
    Weights: (C_out, C_in/groups, K, K)
    """
    C_in, H_in, W_in = input_patch.shape
    C_out, _, K, _ = weights.shape
    
    H_out = (H_in - K) // stride + 1
    W_out = (W_in - K) // stride + 1
    output = np.zeros((C_out, H_out, W_out), dtype=np.float32)
    
    # === Depthwise Convolution ===
    if groups == C_in and C_in == C_out:
        for c in range(C_in):
            w_k = weights[c, 0] # (K, K)
            b_k = bias[c]
            # slide over output feature map
            for i in range(H_out):
                for j in range(W_out):
                    h_s, w_s = i * stride, j * stride
                    patch = input_patch[c, h_s:h_s+K, w_s:w_s+K]
                    output[c, i, j] = np.sum(patch * w_k) + b_k
    # === Pointwise or Standard Convolution ===
    else:
        # simplify version of im2col
        weights_flat = weights.reshape(C_out, -1) # (C_out, C_in/groups * K * K)
        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * stride, j * stride
                patch = input_patch[:, h_s:h_s+K, w_s:w_s+K].reshape(-1) # (C_in * K * K, 1)
                patch_flat = patch.flatten()
                output[:, i, j] = weights_flat @ patch_flat + bias # matrix multiplication (C_out, C_in * K * K) @ (C_in * K * K, 1) + (C_out, 1)
    return output

def numpy_linear(input_vec: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Fully connected layer using numpy
    Input: (in_features,)
    Weights: (out_features, in_features)
    Output: (out_features,)
    """
    return weights @ input_vec + bias  # (out_features, in_features) @ (in_features,) + (out_features,)


def requantize(acc_int32: np.ndarray, m: float, z_out: int, is_activation: bool = True) -> np.ndarray:
    """
    Requantize the int32 accumulator to uint8 using the formula:
    output = clip(round(acc * m) + z_out, 0, 255)
    If is_activation is True, apply ReLU6 after requantization.
    """
    # Apply multiplier and add zero point
    output = np.round(acc_int32.astype(np.float32) * m) + z_out
    # Clip to uint8 range
    output = np.clip(output, 0, 255).astype(np.uint8)
    # if is_activation:
        ## Apply ReLU6
        # output = relu6(output.astype(np.float32)).astype(np.uint8)
    return output

def quantized_conv2d(input_patch: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                     stride: int, groups: int, quant_params: QuantParams) -> np.ndarray:
    """ Quantized convolution operation """
    # 1. cast to int32 for accumulation
    x_shifted = input_patch.astype(np.int32) - quant_params.z_in
    w_shifted = weights.astype(np.int32) - quant_params.z_w
    
    # 2. perform convolution in int32
    C_in, H_in, W_in = input_patch.shape
    C_out, _, K, _ = weights.shape
    
    H_out = (H_in - K) // stride + 1
    W_out = (W_in - K) // stride + 1
    
    # Accumulator
    acc = np.zeros((C_out, H_out, W_out), dtype=np.int32)
    # === Depthwise Convolution ===
    if groups == C_in and C_in == C_out:
        for c in range(C_in):
            w_k = w_shifted[c, 0] # (K, K)
            b_k = bias[c].astype(np.int32)
            for i in range(H_out):
                for j in range(W_out):
                    h_s, w_s = i * stride, j * stride
                    patch = x_shifted[c, h_s:h_s+K, w_s:w_s+K]
                    acc[c, i, j] = np.sum(patch * w_k) + b_k
    # === Pointwise or Standard Convolution ===
    else:
        weights_flat = w_shifted.reshape(C_out, -1) # (C_out, C_in/groups * K * K)
        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * stride, j * stride
                patch = x_shifted[:, h_s:h_s+K, w_s:w_s+K].reshape(-1) # (C_in * K * K, 1)
                patch_flat = patch.flatten()
                acc[:, i, j] = weights_flat @ patch_flat + bias.astype(np.int32)
    # 3. requantize
    output = requantize(acc, quant_params.m, quant_params.z_out, True)
    return output

def quantized_linear(input_vec: np.ndarray, weights: np.ndarray, bias: np.ndarray, quant_params: QuantParams) -> np.ndarray:
    """ quantized fully connected layer """
    # 1. cast to int32 for accumulation
    x_shifted = input_vec.astype(np.int32) - quant_params.z_in
    w_shifted = weights.astype(np.int32) - quant_params.z_w

    # 2. perform linear in int32
    acc = w_shifted @ x_shifted + bias.astype(np.int32)

    # 3. requantize
    output = requantize(acc, quant_params.m, quant_params.z_out, True)
    return output


            
