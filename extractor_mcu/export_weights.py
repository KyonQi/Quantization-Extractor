import argparse
from pathlib import Path

import torch
import numpy as np
import torchvision.models as models
from quant.quant_model_utils import extract_quantized_layers
from models.utils import get_pytorch_quantized_model
from protocol.protocol import LayerConfig


class Exporter:
    def __init__(self):
        pass
    
    def export_weights(self, sim_layers: list[tuple[LayerConfig, np.ndarray, np.ndarray, dict]], output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        weights_file = output_path / "weights.h"

        with open(weights_file, 'w') as f:
            f.write("// Auto-generated weights header file\n\n")
            f.write("#ifndef WEIGHTS_H\n")
            f.write("#define WEIGHTS_H\n")
            f.write("#include <stdint.h>\n")
            f.write("// reading pgm_read_byte(&array[i])\n")
            
            total_weights_size = 0
            total_bias_size = 0
            
            for idx, (layer_cfg, w_int8, b_int32, qp_dict) in enumerate(sim_layers):
                layer_name = layer_cfg.name
                # weight_scale = qp_dict['s_w']
                # weight_zp = qp_dict['z_w']

                weights_flattened = w_int8.flatten()
                weights_size = len(weights_flattened)
                total_weights_size += weights_size
                f.write(f"// Layer {idx}: {layer_name}\n")
                # f.write(f"// Weights: scale={weight_scale}, zero_point={weight_zp}\n")
                f.write(f"const int8_t {layer_name}_weights[] = {{\n")
                for i in range(0, len(weights_flattened), 16):
                    line = ', '.join(str(x) for x in weights_flattened[i:i+16])
                    f.write(f"    {line},\n")
                f.write("};\n\n")

                bias_size = len(b_int32)
                total_bias_size += bias_size
                f.write(f"const int32_t {layer_name}_bias[] = {{\n")
                for i in range(0, len(b_int32), 16):
                    line = ', '.join(str(x) for x in b_int32[i:i+16])
                    f.write(f"    {line},\n")
                f.write("};\n\n")
            
            # add the pointer for all weights and biases
            f.write("// Pointers to layer weights and biases\n")
            f.write("struct LayerWeights {\n")
            f.write("    const int8_t* weights;\n")
            f.write("    const int32_t* bias;\n")
            f.write("    uint32_t weights_size;\n")
            f.write("    uint32_t bias_size;\n")
            f.write("};\n\n")
            f.write("const struct LayerWeights model_weights[] = {\n")
            for idx, (layer_cfg, _, _, _) in enumerate(sim_layers):
                layer_name = layer_cfg.name
                f.write(f"    {{{layer_name}_weights, {layer_name}_bias, sizeof({layer_name}_weights), sizeof({layer_name}_bias) / sizeof(int32_t)}},\n")
            f.write("};\n\n")

            f.write(f"#define NUM_LAYERS {len(sim_layers)}\n\n")
            
            f.write("#endif // WEIGHTS_H\n")
        
        print(f"Exported weights and biases to {weights_file}")
        print(f"Total weights size: {total_weights_size / 1024:.2f} KB")
        print(f"Total bias size: {total_bias_size / 1024:.2f} KB")

        return weights_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export quantized weights from PyTorch INT8 model")
    parser.add_argument('--output-dir', type=str, default='../../PlatformIO_MCU/Download/include', help='Output directory for header files')
    parser.add_argument('--model-path', type=str, default='../models/mobilenet_v2_quantized.pth', help='Path to quantized model')

    args = parser.parse_args()
    # load the quantized model
    q_model = get_pytorch_quantized_model(train_loader=None)
    q_model.eval()

    sim_layers = extract_quantized_layers(q_model)

    exporter = Exporter()
    exporter.export_weights(sim_layers=sim_layers, output_path=args.output_dir)