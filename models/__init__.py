from .registry import get_adapter, list_models
from .mobilenet_v2 import MobileNetV2Adapter, MobileNetV2035Adapter  # 强制执行该模块，触发 @register
from .proxylessnas import ProxylessNASAdapter
from .mnasnet import MNASNetAdapter
from .mcunet import MCUNetAdapter