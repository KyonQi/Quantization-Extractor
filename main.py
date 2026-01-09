import torch
import torch.nn as nn
import torchvision.models as models

import time
import numpy as np
import multiprocessing

from model_utils import get_imagenet_labels, preprocess_image, extract_layers
from protocol import LayerConfig, LayerType
from coordinator import Coordinator

def main() -> None:
    # 1. Prepare data
    labels = get_imagenet_labels("./img/imagenet_labels.json")
    input_tensor = preprocess_image("./img/panda.jpg")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Labels: {labels[:5]}\n")
    
    torch_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()
    
    # 2. Inference with PyTorch (groundtruth)
    print("=== Running inference with PyTorch ===")
    start = time.time()
    with torch.no_grad():
        pt_logits: torch.Tensor = torch_model(input_tensor)
        pt_prob: torch.Tensor = torch.nn.functional.softmax(pt_logits[0], dim=0)
    print(f"Pytorch Time: {time.time() - start:.4f} s")
    print(f"Prediction: {labels[torch.argmax(pt_prob)]}, Probability: {pt_prob.max().item():.4f}")
    
    # 3. Simulation
    print("=== Running inference with simulated model ===")
    sim_conv_layers = extract_layers(torch_model)
    fc: nn.Linear  = torch_model.classifier[1] # [0]dropout, [1]linear
    fc_cfg = LayerConfig(name="fc_final", type=LayerType.LINEAR, in_channels=fc.in_features, out_channels=fc.out_features)
    sim_fc_layer = (fc_cfg, fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy())

    # 4. Run simulation
    print(f"=== Running simulation ===")
    coord = Coordinator(num_workers=3, input_shape=input_tensor.squeeze(0).shape)
    coord.set_input(input_tensor.squeeze(0).numpy())

    start_sim = time.time()
    
    # 4.1 Convolutional layers
    features = coord.execute_inference(sim_conv_layers)
    
    # 42. Global Average Pooling
    ## Note: feature shape: (1280, 7, 7) -> (1280,)
    gap_output = np.mean(features, axis=(1, 2))
    
    # 4.3 Fully connected layer
    coord.feature_map = gap_output
    coord.execute_inference([sim_fc_layer])

    sim_logits = coord.feature_map
    sim_time = time.time() - start_sim
    print(f"Simulation Time: {sim_time:.4f} s")
    
    # 5. Compare results
    ## compare logits
    diff = np.abs(pt_logits.numpy().flatten() - sim_logits)
    print(f"Logits Mean Error: {np.mean(diff):.6f}")
    
    # Compare Top-5 predictions
    def get_top5(logits: np.ndarray) -> list[tuple[str, float]]:
        top5_idx = np.argsort(logits)[::-1][:5]
        return [(labels[idx], logits[idx]) for idx in top5_idx]

    print(f"{'PyTorch':<30} {'Simulation':<30}")
    print("-" * 50)

    pt_top5 = get_top5(pt_logits.numpy().flatten())
    sim_top5 = get_top5(sim_logits)
    for (p_l, p_s), (s_l, s_s) in zip(pt_top5, sim_top5):
        print(f"{p_l[:25]:<25} {p_s:.2f} | {s_l[:25]:<25} {s_s:.2f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()