import numpy as np
import math
from abc import ABC, abstractmethod
import struct
import zstandard as zstd

import numpy as np
import struct
import zstandard as zstd
from abc import ABC, abstractmethod

# ==========================================
# 1. 基础协议与工具
# ==========================================
class Compressor(ABC):
    @abstractmethod
    def compress(self, data: np.ndarray) -> bytes:
        """输入 Numpy 数组，返回包含 Shape 信息的自包含压缩字节流"""
        pass

    @abstractmethod
    def decompress(self, data_bytes: bytes, dtype=np.uint8) -> np.ndarray:
        """输入字节流，自动解析 Shape 并还原数组"""
        pass

def serialize_shape(shape: tuple) -> bytes:
    """
    协议头格式: [Dim Count (1 byte)] + [Dim 1 (2 bytes)] + [Dim 2 (2 bytes)] ...
    支持任意维度的数组。
    """
    dim_count = len(shape)
    # 'B' is unsigned char (1 byte), 'H' is unsigned short (2 bytes)
    # Format: >B (Big-endian dim_count) + H*dim_count
    fmt = f'>B{dim_count}H' 
    return struct.pack(fmt, dim_count, *shape)

def deserialize_shape(data_bytes: bytes) -> tuple[tuple, int]:
    """
    解析 Header，返回 (shape_tuple, header_length_in_bytes)
    """
    # 1. 读取维度数量 (第1个字节)
    dim_count = data_bytes[0]
    # 2. 计算 Header 总长度: 1 byte (count) + 2 bytes * dim_count
    header_len = 1 + dim_count * 2
    
    fmt = f'>B{dim_count}H'
    unpacked = struct.unpack(fmt, data_bytes[:header_len])
    shape = tuple(unpacked[1:]) # Skip the dim_count itself
    
    return shape, header_len

# ==========================================
# 2. 标准 ANS (Zstd) 压缩器
# ==========================================
class ANSCompressor(Compressor):
    def __init__(self, level: int = 1):
        # 复用 Context 避免频繁内存分配
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()
    
    def compress(self, data: np.ndarray) -> bytes:
        # 1. 生成 Shape Header
        header = serialize_shape(data.shape)
        
        # 2. 压缩数据体
        # 注意: 即使是 uint8，tobytes() 也是最高效的序列化方式
        body_bytes = data.tobytes()
        compressed_body = self.cctx.compress(body_bytes)

        # 3. 拼接
        return header + compressed_body
    
    def decompress(self, data_bytes: bytes, dtype=np.uint8) -> np.ndarray:
        # 1. 解析 Shape
        shape, header_len = deserialize_shape(data_bytes)
        
        # 2. 解压 Body
        compressed_body = data_bytes[header_len:]
        decompressed_bytes = self.dctx.decompress(compressed_body)

        # 3. 重构数组
        return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)

# ==========================================
# 3. Bitmap + ANS 稀疏压缩器
# ==========================================
class BitMapANSCompressor(Compressor):
    def __init__(self, level: int = 1, zero_point: int = 0):
        # 既然是基于 Zstd，直接持有 Zstd Context 即可，不要继承 ANSCompressor 导致逻辑耦合
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()
        self.zp = zero_point # 支持非0的稀疏基准值
    
    def compress(self, data: np.ndarray) -> bytes:
        # 1. 生成 Shape Header (这对所有 Compressor 都是通用的)
        shape_header = serialize_shape(data.shape)
        
        # 2. 处理稀疏逻辑
        flat_data = data.flatten()

        mask = (flat_data != 0)
        
        # # 生成 Mask (非ZP元素为 True)
        # if self.zp == 0:
        #     mask = (flat_data != 0)
        # else:
        #     mask = (flat_data != self.zp)
            
        values = flat_data[mask]
        
        # 3. 打包 Bitmap
        # np.packbits 将 8个bool 压成 1个byte，极大减小体积
        bitmap_bytes = np.packbits(mask).tobytes()
        values_bytes = values.tobytes()
        
        # 4. 构建中间 Payload
        # 格式: [Bitmap Length (4 bytes)] + [Bitmap Body] + [Values Body]
        # 我们需要知道 Bitmap 在哪里结束，Values 在哪里开始
        bitmap_len = len(bitmap_bytes)
        # 'I' is unsigned int (4 bytes)
        intermediate_data = struct.pack('>I', bitmap_len) + bitmap_bytes + values_bytes
        
        # 5. Zstd 压缩整个中间 Payload
        # 这样做的好处是 Bitmap 本身也会被 Zstd 进一步压缩 (RLE效果)
        compressed_body = self.cctx.compress(intermediate_data)
        
        # 6. 返回: Shape Header + Compressed(Bitmap + Values)
        return shape_header + compressed_body
        
    
    def decompress(self, data_bytes: bytes, dtype=np.uint8) -> np.ndarray:
        # 1. 解析 Shape (为了最终 reshape)
        shape, header_len = deserialize_shape(data_bytes)
        total_pixels = np.prod(shape)
        
        # 2. Zstd 解压 Body
        compressed_body = data_bytes[header_len:]
        raw_payload = self.dctx.decompress(compressed_body)
        
        # 3. 解析中间 Payload
        # 读取 Bitmap 长度
        bitmap_len_size = 4
        bitmap_len = struct.unpack('>I', raw_payload[:bitmap_len_size])[0]
        
        # 切分
        bitmap_bytes = raw_payload[bitmap_len_size : bitmap_len_size + bitmap_len]
        values_bytes = raw_payload[bitmap_len_size + bitmap_len :]
        
        # 4. 还原 Mask
        # unpackbits 会还原成 uint8 (0 或 1)
        # 注意: packbits 会补全到 8 的倍数，所以 unpack 后可能比 total_pixels 长，需要切片
        mask_uint8 = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
        mask = mask_uint8[:total_pixels].astype(bool)
        
        # 5. 还原 Values
        values = np.frombuffer(values_bytes, dtype=dtype)
        
        # 6. 稀疏重构
        reconstructed = np.full(total_pixels, self.zp, dtype=dtype) # 先填满 ZP
        reconstructed[mask] = values # 填入非稀疏值

        return reconstructed.reshape(shape)
    
class BitMapCompressor(Compressor):
    def __init__(self, zero_point: int = 0):
        """
        :param zero_point: 定义哪个值被视为"零" (稀疏背景值)
        """
        self.zp = zero_point

    def compress(self, data: np.ndarray) -> bytes:
        # 1. 生成 Shape Header
        shape_header = serialize_shape(data.shape)

        # 2. 扁平化处理
        flat_data = data.flatten()

        # 3. 生成掩码 (Mask)
        if self.zp == 0:
            mask = (flat_data != 0)
        else:
            mask = (flat_data != self.zp)
            
        # 4. 提取非零值 (Values)
        # 这些是"干货"，必须原样传输
        values = flat_data[mask]
        
        # 5. 压缩位图 (Bitmap)
        # np.packbits: 8个布尔值 -> 1个字节
        # 比如 [1, 0, 0, 1, 1, 1, 1, 1] -> 0x9F
        bitmap_bytes = np.packbits(mask).tobytes()
        
        # 6. 拼装 Payload
        # 结构: [Bitmap Length (4 bytes)] + [Bitmap Bytes] + [Values Bytes]
        # 我们需要 Bitmap Length 来在解压时切分 Bitmap 和 Values
        bitmap_len = len(bitmap_bytes)
        
        # values.tobytes() 极其高效，就是内存拷贝
        payload = struct.pack('>I', bitmap_len) + bitmap_bytes + values.tobytes()

        # 7. 返回完整数据包
        return shape_header + payload

    def decompress(self, data_bytes: bytes, dtype=np.uint8) -> np.ndarray:
        # 1. 解析 Shape Header
        shape, header_len = deserialize_shape(data_bytes)
        total_pixels = np.prod(shape)
        
        # 2. 读取 Payload
        body = data_bytes[header_len:]
        
        # 3. 解析 Bitmap 长度
        bitmap_len_size = 4
        bitmap_len = struct.unpack('>I', body[:bitmap_len_size])[0]
        
        # 4. 切分 Bitmap 和 Values
        bitmap_end = bitmap_len_size + bitmap_len
        bitmap_bytes = body[bitmap_len_size : bitmap_end]
        values_bytes = body[bitmap_end :]
        
        # 5. 还原 Mask
        # np.unpackbits: 1个字节 -> 8个 uint8 (0或1)
        # 注意: packbits 会在末尾补0以凑齐8位，所以 unpack 后长度可能 >= total_pixels
        mask_uint8 = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
        # 截取有效部分并转为 bool
        mask = mask_uint8[:total_pixels].astype(bool)
        
        # 6. 还原 Values
        values = np.frombuffer(values_bytes, dtype=dtype)
        
        # 7. 稀疏重构 (Scatter)
        # 先创建一个全为 ZP 的底板
        reconstructed = np.full(total_pixels, self.zp, dtype=dtype)
        # 将 Values 填入 Mask 为 True 的位置
        reconstructed[mask] = values
        
        return reconstructed.reshape(shape)    
    
bitmap_ans_compressor = BitMapANSCompressor()
bitmap_compressor = BitMapCompressor()

# # ==========================================
# # 4. 单元测试 (Verify)
# # ==========================================
# if __name__ == "__main__":
#     # 模拟一个高稀疏度的 FeatureMap (32, 112, 112)
#     shape = (32, 112, 112)
#     # 90% 稀疏度
#     data = np.random.choice([0, 1, 2, 255], size=shape, p=[0.9, 0.03, 0.03, 0.04]).astype(np.uint8)
    
#     print(f"Original Data Size: {data.nbytes / 1024:.2f} KB")

#     # 测试 Standard ANS
#     ans = ANSCompressor()
#     compressed_ans = ans.compress(data)
#     recon_ans = ans.decompress(compressed_ans)
#     print(f"ANS Compressed: {len(compressed_ans) / 1024:.2f} KB (Ratio: {data.nbytes/len(compressed_ans):.2f}x)")
#     assert np.array_equal(data, recon_ans), "ANS Decompression failed!"

#     # 测试 BitMap ANS
#     # 注意：如果稀疏度不够高，BitMap 反而会变大。但 Zstd 通常能兜底。
#     bmp = BitMapANSCompressor(zero_point=0)
#     compressed_bmp = bmp.compress(data)
#     recon_bmp = bmp.decompress(compressed_bmp)
#     print(f"BitMap Compressed: {len(compressed_bmp) / 1024:.2f} KB (Ratio: {data.nbytes/len(compressed_bmp):.2f}x)")
#     assert np.array_equal(data, recon_bmp), "BitMap Decompression failed!"
    
#     print("All tests passed!")



# class Compressor:
#     @abstractmethod
#     def compress(self, data: np.ndarray) -> bytes:
#         pass

#     @abstractmethod
#     def decompress(self, data_bytes: bytes, dtype=np.uint8) -> np.ndarray:
#         pass

# class ANSCompressor(Compressor):
#     def __init__(self, level: int = 1):
#         super().__init__()
#         self.compress_ctx = zstd.ZstdCompressor(level=level) # level 1 is fast
#         self.decompress_ctx = zstd.ZstdDecompressor()
    
#     def compress(self, data: np.ndarray) -> bytes:
#         shape_bytes = np.array(data.shape, dtype=np.uint16).tobytes()
#         body_bytes = data.tobytes()
#         compressed_body = self.compress_ctx.compress(body_bytes)

#         return shape_bytes + compressed_body
    
#     def decompress(self, data_bytes: bytes, dtype=np.uint8) -> np.ndarray:
#         header_len = 6
#         shape_arr = np.frombuffer(data_bytes[:header_len], dtype=np.uint16)

#         shape = tuple(s for s in shape_arr if s > 0)
#         decompressed_body = self.decompress_ctx.decompress(data=data_bytes[header_len:])

#         return np.frombuffer(buffer=decompressed_body, dtype=dtype).reshape(shape)
        

# class BitMapANSCompressor(Compressor):
#     def __init__(self):
#         super().__init__()
#         self.ans = ANSCompressor(level=1)
#         self.dtype = np.uint8
    
#     def compress(self, data: np.ndarray) -> bytes:
#         flat_data = data.flatten()
#         # using bitmap to record the non-zero positions
#         non_zero_mask = (flat_data != 0)
#         values = flat_data[non_zero_mask]
#         bitmap_bytes = np.packbits(non_zero_mask).tobytes()
#         values_bytes = values.tobytes()
        
#         raw_payload = bitmap_bytes + values_bytes
#         payload = struct.pack('I', len(bitmap_bytes)) + bitmap_bytes + values_bytes
#         bitmap_len = len(bitmap_bytes)
#         header = np.array([bitmap_len], dtype=np.uint32).tobytes()

#         return header + self.ans.compress(raw_payload)
        
    
#     def decompress(self, data_bytes: bytes, shape) -> np.ndarray:
#         header_len = 4
#         bitmap_len = np.frombuffer(data_bytes[:header_len], dtype=np.uint32)[0] # length of bitmap in bytes
#         raw_payload = self.ans.decompress(data_bytes=data_bytes[header_len:])
        
#         bitmap_bytes = raw_payload[:bitmap_len]
#         values_bytes = raw_payload[bitmap_len:]

#         total_pixels = np.prod(shape)
#         mask = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))[:total_pixels].astype(bool)
#         values = np.frombuffer(values_bytes, dtype=self.dtype)

#         reconstructed = np.zeros(total_pixels, dtype=self.dtype)
#         reconstructed[mask] = values

#         return reconstructed.reshape(shape)

# bitmap_ans_compressor = BitMapANSCompressor()