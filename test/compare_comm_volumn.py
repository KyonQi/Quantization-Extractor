import os, sys
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.registry import get_adapter
from coordinator import QuantCoordinator

def run_one(coord: QuantCoordinator, sim_layers, img_np, s_in, z_in):
    coord.quantize_input(img_np, s_in, z_in)
    out, _ = coord.execute_inference(sim_layers)
    return coord.stats["per_layer_comm"], out


def main():
    adapter = get_adapter("mbv2_0.35")
    input_size = adapter.input_size

    tf = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder("./data/imagenette2-320/train", transform=tf)
    calib_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)

    pt_int8 = adapter.quantize(calibration_loader=calib_loader,
                                                num_calibration_batches=200,
                                                save_path=f"./models/{adapter.name}_quantized.pth")
    sim_layers = adapter.extract_quantized_layers(pt_int8)
    s_in = sim_layers[0][3]['s_in']
    z_in = sim_layers[0][3]['z_in']

    val = datasets.ImageFolder("./data/imagenette2-320/val", transform=tf)
    img_np = val[0][0].numpy().squeeze() # (3, H, W)
    
    off, __ = run_one(QuantCoordinator(num_workers=4, use_halo=False), sim_layers, img_np, s_in, z_in)
    on, ___ = run_one(QuantCoordinator(num_workers=4, use_halo=True), sim_layers, img_np, s_in, z_in)

    print(f"{'layer':30s} {'halo':>5s} {'off.down':>10s} {'off.up':>10s} "
          f"{'on.down':>10s} {'on.up':>10s} {'save%':>7s}")
    tot_off = tot_on = 0
    for name in off:
        o = off[name]; n = on[name]
        tot_off += o['total']; tot_on += n['total']
        save = 100.0 * (o['total'] - n['total']) / max(o['total'], 1)
        print(f"{name:30s} {str(n['halo']):>5s} "
              f"{o['down']:>10d} {o['up']:>10d} "
              f"{n['down']:>10d} {n['up']:>10d} {save:>6.1f}%")
    print(f"{'TOTAL':30s} {'':>5s} "
          f"{'':>10s} {tot_off:>10d} {'':>10s} {tot_on:>10d} "
          f"{100.0*(tot_off-tot_on)/tot_off:>6.1f}%") 

if __name__ == "__main__":
    main()