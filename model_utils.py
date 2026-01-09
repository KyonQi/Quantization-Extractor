import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import json
import urllib.request
import numpy as np
from PIL import Image
from typing import Optional

from protocol import LayerConfig, LayerType

def get_imagenet_labels(path: str) -> list[str]:
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    # filename = "imagenet_labels.json"
    if not os.path.exists(path):
        print(f"Downloading ImageNet labels...")
        with urllib.request.urlopen(url=url) as response, open(path, 'wb') as out_file:
            out_file.write(response.read()) # save to local file

    try:
        with open(path, 'r') as f:
            return json.load(f)
        # with urllib.request.urlopen(url=url) as response:
        #     return json.loads(response.read())
    except:
        print(f"Failed to fetch ImageNet labels from {url}, using fallback labels.")
        return [str(i) for i in range(1000)]  # Fallback to dummy labels

def preprocess_image(path: str) -> torch.Tensor:
    prepocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(path):
        print(f"Image path {path} does not exist. Downloading sample image...")
        url = "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG"
        with urllib.request.urlopen(url=url) as response, open(path, 'wb') as out_file:
            out_file.write(response.read()) # save to local file
    img = Image.open(path).convert('RGB')
    return prepocess(img).unsqueeze(0)  # add batch dimension

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """ Fuse Conv2d and BatchNorm2d layers """
    with torch.no_grad():
        w, b = conv.weight, conv.bias
        if b is None:
            b: torch.Tensor = torch.zeros_like(bn.running_mean)

        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        scale = gamma / torch.sqrt(var + bn.eps)

        if isinstance(conv, nn.Conv2d):
            # Reshape scale for broadcasting
            view_shape = [-1] + [1] * (w.ndim - 1)
            w_fused = w * scale.view(view_shape)
            b_fused = (b - mean) * scale + beta
        return w_fused.cpu().numpy(), b_fused.cpu().numpy()

def extract_layers(model: models.MobileNetV2) -> list[tuple[LayerConfig, np.ndarray, np.ndarray]]:
    """ Parse the PyTorch model to extract configs for simulation """
    sim_layers = []

    def add(conv: nn.Conv2d, bn: nn.BatchNorm2d, name: str, res_add: Optional[str] = None, res_conn: Optional[str] = None) -> None:
        w, b = fuse_conv_bn(conv=conv, bn=bn)
        # judge layer type
        if conv.groups == conv.in_channels and conv.in_channels > 1:
            l_type = LayerType.DEPTHWISE
        elif conv.kernel_size[0] == 1:
            l_type = LayerType.POINTWISE
        else:
            l_type = LayerType.CONV2D
        
        cfg = LayerConfig(
            name=name, type=l_type,
            in_channels=conv.in_channels, out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0], stride=conv.stride[0], padding=conv.padding[0],
            groups=conv.groups, residual_add_to=res_add, residual_connect_from=res_conn
        )
        sim_layers.append((cfg, w, b))
        
    # 1. Process the Initial Convolution Layer ---
    # The first layer in MobileNetV2 features is Conv2dNormActivation (Conv+BN+ReLU6)
    # We name it 'init_conv' (ReLU6 will be applied by Worker)
    add(model.features[0][0], model.features[0][1], "init_conv")

    # 2. Iterate through Inverted Residual Blocks ---
    for i, block in enumerate(model.features[1:]):
        if hasattr(block, 'conv'):
            ops = list(block.conv)
            use_res = block.use_res_connect
            # If residual connection is active, generate key for the buffer
            res_key = f"block_{i}" if use_res else None
            
            # Case A: Expansion -> Depthwise -> Projection (Expansion Factor t > 1)
            # Structure: [ExpandSeq(Conv+BN+ReLU), DWSeq(Conv+BN+ReLU), ProjConv, ProjBN]
            if len(ops) == 4:
                # 1. Expand Layer (Start of block: cache input if residual needed)
                add(ops[0][0], ops[0][1], f"blk{i}_exp", res_add=res_key)
                
                # 2. Depthwise Layer
                add(ops[1][0], ops[1][1], f"blk{i}_dw")
                
                # 3. Projection Layer (End of block: add residual if needed)
                add(ops[2], ops[3], f"blk{i}_proj", res_conn=res_key)
                
            # Case B: Depthwise -> Projection (Expansion Factor t = 1, No Expansion)
            # Structure: [DWSeq(Conv+BN+ReLU), ProjConv, ProjBN]
            elif len(ops) == 3:
                # 1. Depthwise Layer (Start of block: cache input if residual needed)
                add(ops[0][0], ops[0][1], f"blk{i}_dw", res_add=res_key)
                
                # 2. Projection Layer (End of block: add residual if needed)
                add(ops[1], ops[2], f"blk{i}_proj", res_conn=res_key)
            
            else:
                print(f"Warning: Skipping unknown block structure at index {i} with len {len(ops)}")
        
        # 3. Process the Final Feature Layer ---
        # The last layer in .features is usually a 1x1 Conv (Conv2dNormActivation)
        # Structure: [Conv2d, BatchNorm2d, ReLU6]
        elif isinstance(block, (nn.Sequential, torchvision.ops.misc.Conv2dNormActivation)):
            add(block[0], block[1], f"final_conv_{i}")
        else:
            print(f"Warning: Unknown layer type at index {i}: {type(block)}")
    
    # 3. Return
    return sim_layers
        


