""" Calculating operations for simulation """
import numpy as np
from numba import njit

from protocol.protocol import QuantParams

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


# def requantize(acc_int32: np.ndarray, m: float, z_out: int, is_activation: bool = True) -> np.ndarray:
#     """
#     Requantize the int32 accumulator to uint8 using the formula:
#     output = clip(round(acc * m) + z_out, 0, 255)
#     If is_activation is True, apply ReLU6 after requantization.
#     """
#     # Apply multiplier and add zero point
#     output = np.round(acc_int32.astype(np.float32) * m) + z_out
#     # Clip to uint8 range
#     output = np.clip(output, 0, 255).astype(np.uint8)
#     # if is_activation:
#         ## Apply ReLU6
#         # output = relu6(output.astype(np.float32)).astype(np.uint8)
#     return output

def quantized_pad_input(x: np.ndarray, padding: int, z_in: int) -> np.ndarray:
    """ Pad the input feature map with zeros """
    if padding == 0:
        return x
    # x: (C, H, W) -> Pad H and W dimension with 0
    return np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=z_in)

@njit(fastmath=True, cache=True)
def requantize(acc_int32: np.ndarray, m, z_out: int) -> np.ndarray:
    acc_float = acc_int32.astype(np.float32) * m
    q_out = np.round(acc_float) + z_out
    return np.minimum(np.maximum(q_out, 0), 255).astype(np.uint8)

@njit(fastmath=True, cache=True) # 去掉 parallel=True，使用 cache=True
def _quantized_conv2d_jit(input_patch: np.ndarray, weights: np.ndarray, bias_int32: np.ndarray,
                          stride: int, groups: int, z_in: int, z_w, m, z_out: int) -> np.ndarray:
    
    # 1. 预处理：只做一次大的类型转换，绝对不要在循环里做
    x_shifted = input_patch.astype(np.int32) - z_in
    w_shifted = weights.astype(np.int32) - z_w
    
    C_in, H_in, W_in = input_patch.shape
    C_out, _, K, _ = weights.shape
    
    H_out = (H_in - K) // stride + 1
    W_out = (W_in - K) // stride + 1
    
    acc = np.zeros((C_out, H_out, W_out), dtype=np.int32)

    # === Depthwise Convolution ===
    if groups == C_in and C_in == C_out:
        for c in range(C_in):
            # 取出标量 bias
            b_val = bias_int32[c]
            # 这里的 w_k 是 (K, K)
            # 优化：不提取 w_k 子矩阵，直接用索引访问 w_shifted
            
            for i in range(H_out):
                for j in range(W_out):
                    h_s = i * stride
                    w_s = j * stride
                    
                    # 纯标量累加，寄存器级速度
                    val = 0
                    for ky in range(K):
                        for kx in range(K):
                            # 直接索引，无切片开销
                            val += x_shifted[c, h_s + ky, w_s + kx] * w_shifted[c, 0, ky, kx]
                    
                    acc[c, i, j] = val + b_val

    # === Standard Convolution (Groups=1) ===
    else:
        # 即使是标准卷积，也展开为纯循环
        # 这种写法 Numba 会自动进行 SIMD 向量化优化
        for i in range(H_out):
            for j in range(W_out):
                h_s = i * stride
                w_s = j * stride
                
                for co in range(C_out):
                    # 初始化累加器为 bias
                    val = bias_int32[co]
                    
                    # 核心卷积循环：遍历输入通道和卷积核
                    for ci in range(C_in):
                        for ky in range(K):
                            for kx in range(K):
                                # 这里的访问模式对 CPU 缓存非常友好 (连续内存访问)
                                # x: (C, H, W) -> W 维度连续
                                # w: (Out, In, K, K) -> Kx 维度连续
                                val += x_shifted[ci, h_s + ky, w_s + kx] * w_shifted[co, ci, ky, kx]
                    
                    acc[co, i, j] = val

    # 3. Requantize
    return requantize(acc, m, z_out)


def quantized_conv2d(input_patch: np.ndarray, weights: np.ndarray, bias: np.ndarray,
                     stride: int, groups: int, quant_params) -> np.ndarray:
    z_in = int(quant_params.z_in)
    z_out = int(quant_params.z_out)
    bias_int32 = bias.astype(np.int32)
    
    z_w = quant_params.z_w
    m = quant_params.m
    
    # 维度处理保持不变
    if isinstance(z_w, np.ndarray):
        z_w = z_w.astype(np.int32).reshape(-1, 1, 1, 1)
    else:
        z_w = int(z_w)
        
    if isinstance(m, np.ndarray):
        m = m.astype(np.float32).reshape(-1, 1, 1)
    else:
        m = float(m)
        
    return _quantized_conv2d_jit(input_patch, weights, bias_int32, stride, groups, z_in, z_w, m, z_out)


@njit(fastmath=True, cache=True) # 同样加上 cache=True
def _quantized_linear_jit(input_vec: np.ndarray, weights: np.ndarray, bias_int32: np.ndarray,
                          z_in: int, z_w, m, z_out: int) -> np.ndarray:
    
    x_shifted = input_vec.astype(np.int32) - z_in
    w_shifted = weights.astype(np.int32) - z_w 

    C_out, C_in = w_shifted.shape
    acc = np.zeros(C_out, dtype=np.int32)
    
    for i in range(C_out):
        val = 0
        for j in range(C_in):
            val += w_shifted[i, j] * x_shifted[j]
        acc[i] = val + bias_int32[i]

    return requantize(acc, m, z_out)

def quantized_linear(input_vec: np.ndarray, weights: np.ndarray, bias: np.ndarray, quant_params) -> np.ndarray:
    z_in = int(quant_params.z_in)
    z_out = int(quant_params.z_out)
    bias_int32 = bias.astype(np.int32)
    z_w = quant_params.z_w
    m = quant_params.m
    
    if isinstance(z_w, np.ndarray):
        z_w = z_w.astype(np.int32).reshape(-1, 1)
    else:
        z_w = int(z_w)
    
    if isinstance(m, np.ndarray):
        m = m.astype(np.float32).reshape(-1)
    else:
        m = float(m)
    
    return _quantized_linear_jit(input_vec, weights, bias_int32, z_in, z_w, m, z_out)

