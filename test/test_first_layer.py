import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from quant.quant_model_utils import extract_quantized_layers
from coordinator import QuantCoordinator
from models.utils import get_pytorch_quantized_model
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def prepocess_image(image_path: str) -> np.ndarray:
    prepocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor: torch.Tensor = prepocess(img)
    return input_tensor.numpy()


def save_hex_dump(arr: np.ndarray, path: str):
    """Save a 2D uint8 array to a text file in a hex dump format"""
    flat = arr.flatten()
    with open(path, 'w') as f:
        f.write(f"# shape: {arr.shape}  dtype: {arr.dtype}  total: {flat.size} bytes\n")
        for i in range(0, len(flat), arr.shape[1]):
            chunk = flat[i:i+arr.shape[1]]
            hex_part = ' '.join(f'{b:02X}' for b in chunk)
            f.write(f"{i:08X}  {hex_part:<47}\n")


def test_first_layer():
    img_path = "./img/panda.jpg"
    input_image = prepocess_image(img_path)
    # 1. Load quantized model and extract layers
    q_model = get_pytorch_quantized_model(train_loader=None)
    sim_layers = extract_quantized_layers(q_model)

    layer0_cfg, layer0_w, layer0_b, layer0_qp = sim_layers[0]
    print(f"Layer 0: {layer0_cfg.name}  "
      f"in_ch={layer0_cfg.in_channels}, out_ch={layer0_cfg.out_channels}, "
      f"k={layer0_cfg.kernel_size}, s={layer0_cfg.stride}, p={layer0_cfg.padding}")
    print(f"  s_in={layer0_qp['s_in']:.6f}, z_in={layer0_qp['z_in']}")
    print(f"  s_out={layer0_qp['s_out']:.6f}, z_out={layer0_qp['z_out']}")

    coord = QuantCoordinator(num_workers=4)
    coord.quantize_input(input_image, layer0_qp['s_in'], layer0_qp['z_in'])

    output_uint8_layer0, _ = coord.execute_inference([sim_layers[0]])
    print(f"\n Layer 0 output shape: {output_uint8_layer0.shape}, dtype: {output_uint8_layer0.dtype}")
    # save it in text file hex for debug in beautiful way
    save_hex_dump(output_uint8_layer0[1, :, :], "./test/layer0_output_hex.txt")

    # np.savetxt("./test/layer0_output.txt", output_uint8_layer0.flatten(), fmt="%02x")

if __name__ == "__main__":
    test_first_layer()