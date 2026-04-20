import os
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from protocol.protocol import LayerConfig

SimLayerFP32 = tuple[LayerConfig, np.ndarray, np.ndarray]
SimLayerINT8 = tuple[LayerConfig, np.ndarray, np.ndarray, dict]

class ModelAdapter(ABC):
    """ Abstract base class for model adapters. Each model architecture should have its own adapter that inherits from this class. """
    
    name: str
    input_size: int = 224
    num_classes: int = 1000

    @abstractmethod
    def load_fp32(self) -> nn.Module:
        """ Load the FP32 version of the model. """
        pass

    @abstractmethod
    def make_quantizable(self) -> nn.Module:
        """ Modify the model architecture to make it quantizable for models that don't support it natively. """
        pass

    def quantize(self, calibration_loader: torch.utils.data.DataLoader, num_calibration_batches: int, save_path: str) -> nn.Module:
        """ Quantize the model using the provided calibration data. """
        q_model = self.make_quantizable()
        # q_model.eval()

        # configure the quant backend
        backend= "fbgemm"
        torch.backends.quantized.engine = backend
        q_model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(q_model, inplace=True)

        # load the pth if exists
        if save_path and os.path.exists(save_path):
            print(f"\n[Fast Load] Found saved model at {save_path}, loading...")
            q_model(torch.randn(1, 3, self.input_size, self.input_size))
            torch.quantization.convert(q_model, inplace=True)
            q_model.load_state_dict(torch.load(save_path))
            return q_model
        
        # calibrate
        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                q_model(images)
                if (i + 1) % 50 == 0:
                    print(f"  Calibrated {i + 1}/{num_calibration_batches} batches")

        torch.ao.quantization.convert(q_model, inplace=True)
        if save_path:
            torch.save(q_model.state_dict(), save_path) # save the quantized model

        return q_model
    
    @abstractmethod
    def extract_fp32_layers(self, model: nn.Module) -> list[SimLayerFP32]:
        """ Extract the layers from the FP32 model for simulation. """
        pass

    @abstractmethod
    def extract_quantized_layers(self, q_model: nn.Module) -> list[SimLayerINT8]:
        """ Extract the layers from the quantized model for simulation. """
        pass




