import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from tqdm import tqdm
import os
import numpy as np

from quant.quant_model_utils import extract_quantized_layers
from coordinator import QuantCoordinator

# Imagenette 到 ImageNet 的映射
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # tench
    1: 217,  # English springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # French horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701   # parachute
}

def get_pytorch_quantized_model(train_loader: DataLoader, num_calibration_batches=200, save_path: str = "mobilenet_v2_quantized.pth"):
    """
    PyTorch quantization.
    It will load the predefined model if exists
    """
    q_model = q_mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, 
                             quantize=False)
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

def evaluate_distributed():
    # 1. prepare the dataset
    data_dir = "./data/imagenette2-320"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # calibrate data
    calibration_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # small sample
    eval_subset = torch.utils.data.Subset(
        val_dataset, 
        indices=torch.arange(1000)
    )
    eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False)
    
    # 2. load the model
    print("="*60)
    print("Loading FP32 model...")
    pt_fp32_model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    pt_fp32_model.eval()
    
    print("\nLoading Pytorch INT8 model...")
    pt_int8_model = get_pytorch_quantized_model(
        train_loader=calibration_loader,
        num_calibration_batches=200
    )

    print("\nLoading My INT8 model...")
    sim_layers = extract_quantized_layers(pt_int8_model)
    coord = QuantCoordinator(num_workers=4)
    input_scale = sim_layers[0][3]['s_in']
    input_zp = sim_layers[0][3]['z_in']
    output_scale = sim_layers[-1][3]['s_out']
    output_zp = sim_layers[-1][3]['z_out']

    
    # 3. evaluating
    results = {'FP32': 0, 'PT_INT8': 0, 'SIM_INT8': 0}
    total = 0
    
    print("="*60)
    print("EVALUATION (100 samples)")
    print("="*60)
    
    with torch.no_grad():
        for inputs, labels in tqdm(eval_loader, desc="Evaluating"):
            label_idx = int(labels.item())
            true_global_label = IMAGENETTE_TO_IMAGENET[label_idx]
            
            # FP32 prediction
            fp32_out = pt_fp32_model(inputs)
            fp32_pred = int(torch.argmax(fp32_out))
            
            # INT8 prediction
            pt_int8_out = pt_int8_model(inputs)
            pt_int8_pred = int(torch.argmax(pt_int8_out))

            # My INT8
            img_np = inputs.numpy().squeeze(0)
            coord.quantize_input(img_np, input_scale, input_zp)
            sim_out_uint8, _ = coord.execute_inference(sim_layers)
            sim_pred = int(np.argmax(sim_out_uint8))
            
            if fp32_pred == true_global_label:
                results['FP32'] += 1
            if pt_int8_pred == true_global_label:
                results['PT_INT8'] += 1
            if sim_pred == true_global_label:
                results['SIM_INT8'] += 1
            
            total += 1
    
    print(f"\n{'='*25} Results (100 samples) {'='*25}")
    print(f"Total Samples:      {total}")
    print(f"FP32 Accuracy:      {results['FP32']/total:.2%} ({results['FP32']}/{total})")
    print(f"INT8 Accuracy:      {results['PT_INT8']/total:.2%} ({results['PT_INT8']}/{total})")
    print(f"SIM INT8 Accuracy:  {results['SIM_INT8']/total:.2%} ({results['SIM_INT8']}/{total})")
    print(f"Accuracy Drop:      {(results['FP32']-results['PT_INT8'])/total:.2%}")
    print(f"SIM INT8 Accuracy Drop: {(results['FP32']-results['SIM_INT8'])/total:.2%}")
    print("-" * 60)
    
    # if small test passes, run the full test set
    if results['PT_INT8']/total > 0.85:
        pass
        print("\n✓ Small test passed! Running full evaluation...")
        
        full_eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        results_full = {'FP32': 0, 'PT_INT8': 0, 'SIM_INT8': 0}
        total_full = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(full_eval_loader, desc="Full Evaluation"):
                label_idx = int(labels.item())
                true_global_label = IMAGENETTE_TO_IMAGENET[label_idx]
                
                fp32_out = pt_fp32_model(inputs)
                fp32_pred = int(torch.argmax(fp32_out))
                
                pt_int8_out = pt_int8_model(inputs)
                pt_int8_pred = int(torch.argmax(pt_int8_out))

                # My INT8
                img_np = inputs.numpy().squeeze(0)
                coord.quantize_input(img_np, input_scale, input_zp)
                sim_out_uint8, _ = coord.execute_inference(sim_layers)
                sim_pred = int(np.argmax(sim_out_uint8))
                
                if fp32_pred == true_global_label:
                    results_full['FP32'] += 1
                if pt_int8_pred == true_global_label:
                    results_full['PT_INT8'] += 1
                if sim_pred == true_global_label:
                    results_full['SIM_INT8'] += 1
                
                total_full += 1
        
        print(f"\n{'='*20} Final Results (Full Dataset) {'='*20}")
        print(f"Total Samples:      {total_full}")
        print(f"FP32 Accuracy:      {results_full['FP32']/total_full:.2%} ({results_full['FP32']}/{total_full})")
        print(f"INT8 Accuracy:      {results_full['PT_INT8']/total_full:.2%} ({results_full['PT_INT8']}/{total_full})")
        print(f"SIM INT8 Accuracy:  {results_full['SIM_INT8']/total_full:.2%} ({results_full['SIM_INT8']}/{total_full})")
        print(f"Accuracy Drop:      {(results_full['FP32']-results_full['PT_INT8'])/total_full:.2%}")
        print(f"SIM INT8 Accuracy Drop: {(results_full['FP32']-results_full['SIM_INT8'])/total_full:.2%}")
        print("-" * 60)
    else:
        print("\n✗ Small test failed! Check the label mapping above.")

if __name__ == "__main__":
    evaluate_distributed()