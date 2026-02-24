import struct
import zstandard as zstd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, __all__

__all__ = ['Compressor', 'ANSCompressor', 'BitMapCompressor', 'BitMapANSCompressor']

class Compressor(ABC):
    @abstractmethod
    def compress(self, data: np.ndarray) -> bytes:
        """
        Compress the input numpy array and return compressed bytes.
        """
        pass

    @abstractmethod
    def decompress(self, compressed_data: bytes, dtype=np.uint8) -> np.ndarray:
        """
        Decompress the input bytes and return the original numpy array.
        """
        pass

def serialize_shape(shape: tuple) -> bytes:
    """ 
    Serialize the shape of a numpy array into bytes.
    Header fommat:
      [Dim Count (1 bytes)] + [Dim 1 (2 bytes)] + ... + [Dim N (2 bytes)]
    """
    dim_count = len(shape)
    # B: unsigned char for dimension count, H: unsigned short for each dimension size
    fmt = f'>B{dim_count}H'  
    return struct.pack(f'>B{dim_count}H', dim_count, *shape)

def deserialzie_shape(data_bytes: bytes) -> Tuple[tuple, int]:
    """
    Deserialize the shape from the header bytes.
    """
    dim_count = data_bytes[0]
    header_len = 1 + dim_count * 2
    fmt = f'>B{dim_count}H'
    unpacked = struct.unpack(fmt, data_bytes[:header_len])
    shape = tuple(unpacked[1:])
    return shape, header_len

class BitMapCompressor(Compressor):
    """
    Compressor that uses bitmap representation for compression.    
    """
    def __init__(self):
        super().__init__()

    def compress(self, data: np.ndarray) -> bytes:
        shape_header = serialize_shape(data.shape)
        flat_data = data.flatten()
        mask = (flat_data != 0)
        values = flat_data[mask]
        
        bitmap_bytes = np.packbits(mask).tobytes()
        values_bytes = values.tobytes()
        # payload := [bitmap length (4bytes)] + [bitmap bytes] + [values bytes]
        bitmap_len = len(bitmap_bytes)
        compressed_payload = struct.pack('>I', bitmap_len) + bitmap_bytes + values_bytes
        
        return shape_header + compressed_payload
    
    def decompress(self, compressed_data: bytes, dtype=np.uint8) -> np.ndarray:
        shape, header_len = deserialzie_shape(compressed_data)
        total_pixels = np.prod(shape)
        
        body = compressed_data[header_len:]
        bitmap_len_size = 4 # hardcoded size
        bitmap_len = struct.unpack('>I', body[:bitmap_len_size])[0]
        
        bitmap_bytes = body[bitmap_len_size : bitmap_len_size + bitmap_len]
        values_bytes = body[bitmap_len_size + bitmap_len :]

        mask_uint8 = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
        mask = mask_uint8[:total_pixels].astype(bool)

        values = np.frombuffer(values_bytes, dtype=dtype)
        
        # reconstruct the original array
        reconstructed = np.zeros(total_pixels, dtype=dtype)
        reconstructed[mask] = values

        return reconstructed.reshape(shape)
    
class ANSCompressor(Compressor):
    """
    Compressor that uses ANS compression.  
    """
    def __init__(self, level: int = 1):
        super().__init__()
        self.compress_ctx = zstd.ZstdCompressor(level=level)
        self.decompress_ctx = zstd.ZstdDecompressor()
        
    def compress(self, data: np.ndarray) -> bytes:
        shape_header = serialize_shape(data.shape)
        flat_data = data.flatten()
        compressed_payload = self.compress_ctx.compress(flat_data.tobytes())
        return shape_header + compressed_payload
    
    def decompress(self, compressed_data: bytes, dtype=np.uint8) -> np.ndarray:
        shape, header_len = deserialzie_shape(compressed_data)
        compressed_payload = compressed_data[header_len:]
        decompressed_bytes = self.decompress_ctx.decompress(compressed_payload)
        decompressed_array = np.frombuffer(decompressed_bytes, dtype=dtype)
        return decompressed_array.reshape(shape)

class BitMapANSCompressor(Compressor):
    """
    Compressor that uses bitmap representation combined with ANS compression.  
    """
    def __init__(self, level: int = 1):
        super().__init__()
        self.compress_ctx = zstd.ZstdCompressor(level=level)
        self.decompress_ctx = zstd.ZstdDecompressor()
        
    def compress(self, data: np.ndarray) -> bytes:
        shape_header = serialize_shape(data.shape)
        flat_data = data.flatten()
        mask = (flat_data != 0)
        values = flat_data[mask]
        
        # pack the bitmap
        bitmap_bytes = np.packbits(mask).tobytes()
        values_bytes = values.tobytes()
        # payload := [bitmap length (4bytes)] + [bitmap bytes] + [values bytes]
        bitmap_len = len(bitmap_bytes)
        intermediate = struct.pack('>I', bitmap_len) + bitmap_bytes + values_bytes # >I: big-endian unsigned int: 4 bytes
        compressed_payload = self.compress_ctx.compress(intermediate)
        
        return shape_header + compressed_payload
    
    def decompress(self, compressed_data: bytes, dtype=np.uint8) -> np.ndarray:
        shape, header_len = deserialzie_shape(compressed_data)
        total_pixels = np.prod(shape)
        
        compressed_payload = compressed_data[header_len:]
        decompressed_payload = self.decompress_ctx.decompress(compressed_payload)
        
        bitmap_len_size = 4 # hardcoded size
        bitmap_len = struct.unpack('>I', decompressed_payload[:bitmap_len_size])[0]
        
        bitmap_bytes = decompressed_payload[bitmap_len_size : bitmap_len_size + bitmap_len]
        values_bytes = decompressed_payload[bitmap_len_size + bitmap_len :]

        mask_uint8 = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
        mask = mask_uint8[:total_pixels].astype(bool)

        values = np.frombuffer(values_bytes, dtype=dtype)

        # reconstruct the original array
        reconstructed = np.zeros(total_pixels, dtype=dtype)
        reconstructed[mask] = values

        return reconstructed.reshape(shape)
    
# bitmap_compressor = BitMapCompressor()
# bitmap_ans_compressor = BitMapANSCompressor(level=1)