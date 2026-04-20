"""
Protocol definitions for simulation
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional, Union
import numpy as np

class MessageType(Enum):
    TASK = auto()
    RESULT = auto()
    TERMINATE = auto()

class LayerType(Enum):
    CONV2D = auto() # normal conv
    DEPTHWISE = auto() # depthwise conv
    POINTWISE = auto() # pointwise conv
    LINEAR = auto() # fully connected

@dataclass
class LayerConfig:
    name: str
    type: LayerType
    in_channels: int
    out_channels: int
    kernel_size: int = 1
    stride: int = 1
    padding: int = 0
    groups: int = 1
    residual_add_to: Optional[str] = None # if not None, coordinator will save input to this key
    residual_connect_from: Optional[str] = None # if not None, coordinator will add saved tensor from this key to output

@dataclass
class QuantParams:
    """ quantization parameters needs to be shared between coordinator and workers """
    s_in: float
    z_in: int
    s_w: Union[float, np.ndarray]
    z_w: Union[float, np.ndarray]
    s_out: float
    z_out: int
    m: Union[float, np.ndarray] #float # precomputing multiplier for requantization m = (s_in * s_w) / s_out

@dataclass
class TaskPayload:
    """ send to the worker """
    layer_config: LayerConfig
    slice_idx: Tuple[int, int] # output row for conv, output feature for linear
    weights: np.ndarray # slice of weights int8
    bias: np.ndarray # slice of bias int32
    quant_params: QuantParams
    # Full-patch mode
    input_patch: Optional[np.ndarray] = None # slice of input feature map uint8
    input_patch_compressed: Optional[bytes] = None # compressed input patch
    # Halo mode
    prev_layer_name: Optional[str] = None # if not None, worker needs to fetch halo data from coordinator using this key
    halo_top: Optional[np.ndarray] = None # halo data on the top, only for conv
    halo_bottom: Optional[np.ndarray] = None # halo data on the bottom, only for
    cache_use_range: Optional[Tuple[int, int]] = None # if not None, worker can only use this range of the input feature map for computation, only for conv



@dataclass
class ResultPayload:
    """ send back to the coordinator """
    worker_id: int
    slice_idx: Tuple[int, int]
    output_patch: np.ndarray
    output_patch_compressed: bytes# compressed output patch
    compute_time: float
    codec_time: float = 0.0
    compressed_size: int = 0