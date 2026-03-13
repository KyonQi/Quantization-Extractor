"""
MNASNet-0.5 Full Pipeline Test

Steps:
  1. Load FP32 model and verify forward pass
  2. Extract FP32 layers and print summary
  3. PTQ quantization (calibration or load saved)
  4. Extract INT8 layers and print summary
  5. Run QuantCoordinator simulation and compare with PT INT8 prediction
"""

import sys
import time
import torch
import numpy as np

sys.path.insert(0, ".")

from models import get_adapter
from coordinator import QuantCoordinator


def test_fp32_extraction():
    """Step 1+2: Load FP32 model and extract layers."""
    print("\n" + "=" * 60)
    print("Step 1-2: FP32 Model Load & Layer Extraction")
    print("=" * 60)

    adapter = get_adapter("mnasnet0_5")
    model = adapter.load_fp32()

    # Quick forward pass
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"  FP32 forward OK — output shape: {out.shape}")

    layers = adapter.extract_fp32_layers(model)
    print(f"\n  Extracted {len(layers)} FP32 layers:")
    for i, (cfg, w, b) in enumerate(layers):
        res_info = ""
        if cfg.residual_add_to:
            res_info += f" [save→{cfg.residual_add_to}]"
        if cfg.residual_connect_from:
            res_info += f" [add←{cfg.residual_connect_from}]"
        print(f"    {i:2d}. {cfg.name:20s} {cfg.type.name:10s} "
              f"w={str(w.shape):20s} b={str(b.shape):10s} "
              f"k={cfg.kernel_size} s={cfg.stride} g={cfg.groups}"
              f"{res_info}")

    return model, layers


def test_quantization_and_extraction():
    """Step 3+4: PTQ + INT8 layer extraction."""
    print("\n" + "=" * 60)
    print("Step 3-4: PTQ & INT8 Layer Extraction")
    print("=" * 60)

    adapter = get_adapter("mnasnet0_5")

    # Try to load from saved; if not found, quantize with dummy calibration
    save_path = "./models/mnasnet0_5_quantized.pth"
    import os
    if os.path.exists(save_path):
        print(f"  Loading saved quantized model from {save_path}")
        q_model = adapter.quantize(calibration_loader=None,
                                   num_calibration_batches=0,
                                   save_path=save_path)
    else:
        print("  No saved model found — running PTQ with dummy calibration")
        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = TensorDataset(torch.randn(64, 3, 224, 224),
                                   torch.zeros(64, dtype=torch.long))
        dummy_loader = DataLoader(dummy_data, batch_size=32)
        q_model = adapter.quantize(calibration_loader=dummy_loader,
                                   num_calibration_batches=2,
                                   save_path=save_path)

    # Verify forward pass
    q_model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = q_model(dummy)
    print(f"  INT8 forward OK — output shape: {out.shape}")

    # Extract layers
    layers = adapter.extract_quantized_layers(q_model)
    print(f"\n  Extracted {len(layers)} INT8 layers:")
    for i, (cfg, w, b, qp) in enumerate(layers):
        res_info = ""
        if cfg.residual_add_to:
            res_info += f" [save→{cfg.residual_add_to}]"
        if cfg.residual_connect_from:
            res_info += f" [add←{cfg.residual_connect_from}]"
        sw_shape = (f"({len(qp['s_w'])})" if isinstance(qp['s_w'], np.ndarray)
                    else "scalar")
        print(f"    {i:2d}. {cfg.name:20s} {cfg.type.name:10s} "
              f"w={str(w.shape):20s} b={str(b.shape):10s} "
              f"s_w={sw_shape:8s} k={cfg.kernel_size} s={cfg.stride} g={cfg.groups}"
              f"{res_info}")

    return q_model, layers


def test_simulation(q_model, sim_layers):
    """Step 5: QuantCoordinator simulation vs PyTorch INT8."""
    print("\n" + "=" * 60)
    print("Step 5: Simulation Test (QuantCoordinator)")
    print("=" * 60)

    input_tensor = torch.randn(1, 3, 224, 224)
    img_np = input_tensor.numpy().squeeze(0)  # (3, 224, 224)

    input_scale = sim_layers[0][3]['s_in']
    input_zp = sim_layers[0][3]['z_in']

    # PyTorch INT8 reference
    q_model.eval()
    with torch.no_grad():
        pt_out = q_model(input_tensor)
    pt_pred = int(torch.argmax(pt_out))

    # Simulator
    coord = QuantCoordinator(num_workers=4)
    coord.quantize_input(img_np, input_scale, input_zp)
    print(f"  Input quantized: shape={coord.feature_map.shape}, "
          f"dtype={coord.feature_map.dtype}")

    start = time.time()
    sim_out_uint8, last_name = coord.execute_inference(sim_layers)
    sim_time = time.time() - start

    sim_pred = int(np.argmax(sim_out_uint8))
    print(f"\n  Simulation time: {sim_time:.4f}s")
    print(f"  PT  INT8 prediction:  class {pt_pred}")
    print(f"  SIM INT8 prediction:  class {sim_pred}")
    print(f"  Match: {'YES ✓' if pt_pred == sim_pred else 'NO ✗'}")

    print(f"\n  Performance Stats:")
    print(f"    Total inference time: {coord.stats['total_inference_time']:.4f}s")
    print(f"    Total compute time:   {coord.stats['total_compute_time']:.4f}s")
    print(f"    Total codec time:     {coord.stats['total_codec_time']:.4f}s")
    print(f"    Total comm volume:    {coord.stats['total_comm_volume'] / 1024:.2f} KB")

    return pt_pred == sim_pred


def main():
    print("=" * 60)
    print("MNASNet-0.5 Full Pipeline Test")
    print("=" * 60)

    # Step 1+2
    model, fp32_layers = test_fp32_extraction()

    # Step 3+4
    q_model, int8_layers = test_quantization_and_extraction()

    # Step 5
    match = test_simulation(q_model, int8_layers)

    print("\n" + "=" * 60)
    if match:
        print("ALL TESTS PASSED ✓")
    else:
        print("PREDICTION MISMATCH — check quantization quality")
    print("=" * 60)


if __name__ == "__main__":
    main()