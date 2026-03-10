import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from tqdm import tqdm
import os
import numpy as np

def get_pytorch_quantized_model(train_loader: DataLoader, num_calibration_batches=200, 
                                    save_path: str = "./models/mobilenet_v2_quantized.pth", width_mult: float = 1.0):
    """
    PyTorch quantization.
    It will load the predefined model if exists
    """
    if width_mult == 1.0:
        q_model = q_mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, 
                                 quantize=False)
    else:
        q_model = q_mobilenet_v2(weights=None, quantize=False, width_mult=width_mult)

    q_model.eval()
    
    # configure the quant backend
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    q_model.qconfig = torch.quantization.get_default_qconfig(backend=backend)
    
    # fuse model
    q_model.fuse_model()
    
    # prepare to quant
    torch.quantization.prepare(q_model, inplace=True)

    # load the pth if exists
    if os.path.exists(save_path):
        print(f"\n[Fast Load] Found saved model at {save_path}, loading...")
        q_model(torch.randn(1, 3, 224, 224)) 
        torch.quantization.convert(q_model, inplace=True)
        q_model.load_state_dict(torch.load(save_path))
        return q_model
    
    # calibrate
    print(f"Calibrating with {num_calibration_batches} batches...")
    q_model.eval() # eval is important: make sure dropout doesn't work, and batchnorm doesn't update the params
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= num_calibration_batches:
                break
            q_model(images)
            if (i + 1) % 50 == 0:
                print(f"  Calibrated {i + 1}/{num_calibration_batches} batches")
    
    # convert to the quant model
    torch.quantization.convert(q_model, inplace=True)
    torch.save(q_model.state_dict(), save_path) # save the quantized model
    print("Quantization complete!\n")
    
    return q_model