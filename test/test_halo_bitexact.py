import os, sys
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.registry import get_adapter
from coordinator import QuantCoordinator

def _build_eval_loader(adapter):
    input_size = adapter.input_size
    resize = int(input_size * 256 / 224)
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val = datasets.ImageFolder("./data/imagenette2-320/val", transform=tf)
    subset = torch.utils.data.Subset(val, indices=torch.arange(min(32, len(val))))
    return DataLoader(subset, batch_size=1, shuffle=False)

def _run(coord: QuantCoordinator, sim_layers, img_np, s_in, z_in):
    coord.quantize_input(img_np, s_in, z_in)
    out, _ = coord.execute_inference(sim_layers)
    return out

def main():
    adapter = get_adapter("mbv2_0.35")
    input_size = adapter.input_size

    train_tf = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder("./data/imagenette2-320/train", transform=train_tf)
    calib_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)

    pt_int8 = adapter.quantize(calibration_loader=calib_loader,
                                                num_calibration_batches=200,
                                                save_path=f"./models/{adapter.name}_quantized.pth")
    sim_layers = adapter.extract_quantized_layers(pt_int8)
    s_in = sim_layers[0][3]['s_in']
    z_in = sim_layers[0][3]['z_in']

    eval_loader = _build_eval_loader(adapter)

    mismatch = 0
    total = 0
    with torch.no_grad():
        for inputs, _ in eval_loader:
            img_np = inputs.numpy().squeeze(0) # (3, H, W)

            coord_off = QuantCoordinator(num_workers=4, use_halo=False)
            coord_on = QuantCoordinator(num_workers=4, use_halo=True)

            out_off = _run(coord_off, sim_layers, img_np, s_in, z_in)
            out_on = _run(coord_on, sim_layers, img_np, s_in, z_in)

            if not np.array_equal(out_off, out_on):
                mismatch += 1
                diff = np.abs(out_off.astype(np.int32) - out_on.astype(np.int32))
                print(f"[MISMATCH] sample {total} max|diff|={diff.max()}")

            total += 1

    print(f"Total samples: {total}, Mismatched samples: {mismatch}")

if __name__ == "__main__":
    main()