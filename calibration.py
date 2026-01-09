import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import os
import json
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

from model_utils import extract_layers, fuse_conv_bn
from protocol import LayerConfig, LayerType

class QuantCalibrator:
    def __init__(self, model: nn.Module, data_path: str):
        self.model: models.MobileNetV2 = model
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Store activation statistics (Min, Max)
        # Key: layer name, Value: [Global Min, Global Max]
        self.activation_stats: Dict[str, List[np.ndarray]] = {}
        # Store quantization parameters (scale, zero_point)
        self.quant_params: Dict[str, Dict[str, np.ndarray]] = {}
    
    def _get_dataloader(self, num_images=1000) -> DataLoader:
        """ Create a dataloader for calibration images """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            dataset = datasets.ImageFolder(self.data_path, transform=transform)
        except:
            print(f"Failed to load dataset from {self.data_path}. Using fake data for calibration.")
            dataset = torch.utils.data.TensorDataset(
                torch.randn(num_images, 3, 224, 224),
                torch.zeros(num_images)
            )
        indices = torch.arange(min(len(dataset), num_images))
        subset = torch.utils.data.Subset(dataset=dataset, indices=indices)
        return DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    def _update_stats(self, layer_name: str, activations: torch.Tensor):
        """ Update min/max statistics for a given layer """
        # store the statistic for activations
        data = activations.detach().cpu().numpy()
        current_min = data.min()
        current_max = data.max()
        if layer_name not in self.activation_stats:
            self.activation_stats[layer_name] = [float(current_min), float(current_max)]
        else:
            global_min, global_max = self.activation_stats[layer_name]
            self.activation_stats[layer_name] = [
                min(global_min, current_min),
                max(global_max, current_max)
            ]

    def _calculate_quant_params(self, min_val: float, max_val: float, num_bits=8) -> Tuple[float, int]:
        """ 
        Calculate scale and zero_point for asymmetric quantization
        scale = (max - min) / (qmax - qmin)
        zero_point = qmin - min / scale
        where qmin=0, qmax=2^num_bits - 1
        """
        qmin = 0
        qmax = (1 << num_bits) - 1
        # ensure zero point is in range
        min_val = min(min_val, 0.0)
        max_val = max(max_val, 0.0)
        if min_val == max_val:
            max_val = min_val + 1e-5  # prevent division by zero
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point)) # clamp to [qmin, qmax]
        
        return scale, zero_point

    def calibrate(self, num_images=1000) -> Dict[str, Dict[str, np.ndarray]]:
        """ Run calibration to collect activation statistics """
        # 1. parse layers and fuse Conv+BN
        print("=== Parsing layers and fusing Conv+BN ===")
        sim_layers = extract_layers(self.model)

        hooks: List[torch.utils.hooks.RemovableHandle] = []
        layer_map: List[str] = [] # record layer_name
        idx_cnt = 0

        def hook_fn(module, input, output, layer_name):
            self._update_stats(layer_name, output)
        
        # 2. register hooks to collect activation stats
        print("=== Registering hooks ===")
        def register(module: nn.Module, name: str):
            h = module.register_forward_hook(
                lambda m, inp, outp, layer_name=name: hook_fn(m, inp, outp, layer_name)
            )
            hooks.append(h)
            layer_map.append(name)
        
        # start registering hooks
        # 1. Init conv
        register(self.model.features[0][0], "init_conv")
        
        # 2. Blocks
        for i, block in enumerate(self.model.features[1:]):
            if hasattr(block, "conv"):
                ops = list(block.conv)
                if len(ops) == 4: # Inverted Residual - Exp, DW, Proj
                    register(ops[0][0], f"blk{i}_exp")
                    register(ops[1][0], f"blk{i}_dw")
                    register(ops[3], f"blk{i}_proj") # Proj conv only
                elif len(ops) == 3: # Inverted Residual - DW, Proj
                    register(ops[0][0], f"blk{i}_dw")
                    register(ops[2], f"blk{i}_proj") # Proj conv only
            elif isinstance(block, (nn.Sequential, torchvision.ops.misc.Conv2dNormActivation)):
                register(block, f"final_conv_{i}")
        
        # 3. classifier
        register(self.model.classifier[1], "fc_final")

        print(f"Total hooks registered: {len(hooks)}")
        
        print("=== Running calibration dataset through the model ===")
        dataloader = self._get_dataloader(num_images=num_images)
        self.activation_stats['input'] = [100.0, -100.0] # Initialize input stats
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader):
                images: torch.Tensor = images.to(self.device)
                # Update input stats
                self._update_stats('input', images)
                # forward with hooks
                self.model(images)
        
        # remove hooks
        for h in hooks:
            h.remove()

        # 4. calculate quantization parameters
        print("=== Calculating quantization parameters ===")
        # compute the scale and zero_point for each layer's activations
        print("Calculating activation quantization parameters")
        for layer_name, (min_val, max_val) in self.activation_stats.items():
            s, z = self._calculate_quant_params(min_val, max_val)
            self.quant_params[layer_name] = {
                "scale": float(s),
                "zero_point": int(z),
                "min": float(min_val),
                "max": float(max_val)
            }
        
        print("Calculating conv weight parameters quantization")
        for layer_cfg, weights, bias in sim_layers:
            w_min = weights.min()
            w_max = weights.max()
            s, z = self._calculate_quant_params(w_min, w_max)
            self.quant_params[f"{layer_cfg.name}_weights"] = {
                "scale": float(s),
                "zero_point": int(z),
                "min": float(w_min),
                "max": float(w_max)
            }
            # bias quantization (if exists)
            # b_quant = b_float / (s_input * s_weight), so we only need to store s_input and s_weight
        
        print("Calculating fc weight parameters quantization")
        fc_layer: nn.Linear = self.model.classifier[1]
        fc_weights = fc_layer.weight.detach().cpu().numpy()
        w_min = fc_weights.min()
        w_max = fc_weights.max()
        s, z = self._calculate_quant_params(w_min, w_max)
        self.quant_params["fc_final_weights"] = {
            "scale": float(s),
            "zero_point": int(z),
            "min": float(w_min),
            "max": float(w_max)
        }

        return self.quant_params
        
    def save_json(self, filename: str = "quant_params.json"):
        """ Save quantization parameters to a JSON file """
        with open(filename, 'w') as f:
            json.dump(self.quant_params, f, indent=4)
        print(f"Quantization parameters saved to {filename}")

if __name__ == "__main__":
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    data_dir = "./data/imagenette2-320/val"
    calib = QuantCalibrator(model=model, data_path=data_dir)
    params = calib.calibrate(100)
    calib.save_json("mobilenetv2_quant_params.json")
    
    print("\n Sample Quantization Parameters:")
    keys = list(params.keys())
    for k in keys[:5]:
        print(f"{k}: {params[k]}")

    


        
                
                    

        

        
