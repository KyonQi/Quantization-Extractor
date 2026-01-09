import torch
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models.quantization.mobilenetv2 import QuantizableMobileNetV2
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2

import numpy as np
import os
from tqdm import tqdm

from model_utils import extract_layers, get_imagenet_labels
from protocol import LayerConfig, LayerType, QuantParams
from coordinator import QuantCoordinator

def get_pytorch_quantized_model(data_laoder: DataLoader):
    q_model: QuantizableMobileNetV2 = q_mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, 
                             quantize=False)
    q_model.eval()

    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    q_model.qconfig = torch.quantization.get_default_qconfig(backend=backend)

    q_model.fuse_model()

    torch.quantization.prepare(q_model, inplace=True)
    # Calibrate with the training set
    with torch.no_grad():
        for i, (image, _) in enumerate(data_laoder):
            if i >= 100:
                break
            q_model(image)

    torch.quantization.convert(q_model, inplace=True)

    return q_model

def evaluate_distributed():
    # 1. prepare model and dataset
    data_dir = "./data/imagenette2-320/val"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        return

    ## Use 10 images for quick evaluation
    subset_size = 100
    subset = torch.utils.data.Subset(dataset=dataset, indices=torch.arange(subset_size))
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)

    ## pytorch fp32 model
    pt_fp32_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    ## pytorch quantized model
    pt_int8_model = get_pytorch_quantized_model(data_laoder=dataloader)
    ## simulated quantized model
    sim_layers = extract_layers(pt_fp32_model)
    fc: nn.Linear = pt_fp32_model.classifier[1]
    fc_cfg = LayerConfig(
        "fc_final", LayerType.LINEAR, fc.in_features, fc.out_features
    )
    sim_layers.append((fc_cfg, fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()))
    ## init coordinator
    coor = QuantCoordinator(num_workers=3, quant_params_path="./mobilenetv2_quant_params.json")
    labels_map = get_imagenet_labels("./img/imagenet_labels.json")
    
    # 2. evaluate
    results = {'FP32': 0, 'PT_INT8': 0, 'SIM_INT8': 0}
    total = 0

    print(f"\n{'='*30} FULL SPECTRUM COMPARISON {'='*30}")
    print(f"{'GT':<15} | {'FP32':<15} | {'PT INT8':<15} | {'SIM INT8':<15}")
    print("-" * 85)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            label_idx = int(labels.item())

            # 2.1 Groundtruth with FP32 model
            fp32_out = pt_fp32_model(inputs)
            fp32_pred = int(torch.argmax(fp32_out))

            # 2.2 Pytorch quantized model
            pt_int8_out = pt_int8_model(inputs)
            pt_int8_pred = int(torch.argmax(pt_int8_out))

            # 2.3 Simulated quantized model
            img_numpy = inputs.numpy().squeeze(0)
            coor.quantize_input(img_numpy)
            sim_uint8_out, last_name = coor.execute_inference(sim_layers)
            sim_uint8_pred = int(np.argmax(sim_uint8_out))

            if fp32_pred == label_idx:
                results['FP32'] += 1
            if pt_int8_pred == label_idx:
                results['PT_INT8'] += 1
            if sim_uint8_pred == label_idx:
                results['SIM_INT8'] += 1
            total += 1

            if total % 10 == 0:
                gt_name = labels_map[label_idx][:13]
                n1 = labels_map[fp32_pred][:13]
                n2 = labels_map[pt_int8_pred][:13]
                n3 = labels_map[sim_uint8_pred][:13]

                print(f"{gt_name:<15} | {n1:<15} | {n2:<15} | {n3:<15}")
    
    print(f"\n{'='*20} Final Scoreboard {'='*20}")
    print(f"Total Samples:      {total}")
    print(f"1. PyTorch FP32:    {results['FP32']/total:.2%}")
    print(f"2. PyTorch INT8:    {results['PT_INT8']/total:.2%}")
    print(f"3. Your Simulator:  {results['SIM_INT8']/total:.2%}")
    print("-" * 50)
            
if __name__ == "__main__":
    evaluate_distributed()

    