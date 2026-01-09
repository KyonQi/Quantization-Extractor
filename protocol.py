"""
Protocol definitions for simulation
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional
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
    s_w: float
    z_w: int
    s_out: float
    z_out: int
    m: float # precomputing multiplier for requantization m = (s_in * s_w) / s_out

@dataclass
class TaskPayload:
    """ send to the worker """
    layer_config: LayerConfig
    slice_idx: Tuple[int, int] # output row for conv, output feature for linear
    input_patch: np.ndarray # slice of input feature map
    weights: np.ndarray # slice of weights
    bias: np.ndarray # slice of bias
    quant_params: QuantParams

@dataclass
class ResultPayload:
    """ send back to the coordinator """
    worker_id: int
    slice_idx: Tuple[int, int]
    output_patch: np.ndarray
    compute_time: float