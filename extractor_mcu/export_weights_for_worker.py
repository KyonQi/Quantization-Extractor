import argparse
from pathlib import Path

import torch
import numpy as np
import torchvision.models as models
from quant.quant_model_utils import extract_quantized_layers
from models.utils import get_pytorch_quantized_model
from protocol.protocol import LayerConfig, LayerType


class Exporter:
    def __init__(self, num_mcus=1):
        self.num_mcus = num_mcus
        
    def export_weights(self, sim_layers: list[tuple[LayerConfig, np.ndarray, np.ndarray, dict]], output_path, mcu_id: int):
        """ This function export the weights and bias into a C header file for MCU usage. """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        weights_file = output_path / "weights.h"
        # weights_file = output_path / "weights_test.h"

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

                if layer_cfg.type == LayerType.LINEAR:
                    # for linear layers, weights and bias are partitioned by colomn-wise
                    # which means each MCU contains (output_channels / num_mcus) columns of weights and the corresponding bias
                    out_ch_per_mcu = layer_cfg.out_channels // self.num_mcus
                    start_ch = mcu_id * out_ch_per_mcu
                    end_ch = min(start_ch + out_ch_per_mcu, layer_cfg.out_channels)
                    w_local = w_int8[start_ch:end_ch, :]
                    b_local = b_int32[start_ch:end_ch]
                else:
                    w_local = w_int8
                    b_local = b_int32
                    
                weights_flattened = w_local.flatten()
                weights_size = len(weights_flattened)
                total_weights_size += weights_size
                f.write(f"// Layer {idx}: {layer_name}\n")
                f.write(f"const int8_t {layer_name}_weights[] PROGMEM = {{\n")
                for i in range(0, len(weights_flattened), 16):
                    line = ', '.join(str(x) for x in weights_flattened[i:i+16])
                    f.write(f"    {line},\n")
                f.write("};\n\n")

                bias_size = len(b_local)
                total_bias_size += bias_size
                f.write(f"const int32_t {layer_name}_bias[] PROGMEM = {{\n")
                for i in range(0, len(b_local), 16):
                    line = ', '.join(str(x) for x in b_local[i:i+16])
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
    
    def export_layer_config(self, sim_layers: list[tuple[LayerConfig, np.ndarray, np.ndarray, dict]], output_path):
        """ This function export the layer configuration into a C header file for MCU usage. """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        config_file = output_path / "layer_config.h"
        
        with open(config_file, 'w') as f:
            f.write("// Auto-generated layer configuration header file\n\n")
            f.write("#ifndef LAYER_CONFIG_H\n")
            f.write("#define LAYER_CONFIG_H\n\n")
            f.write("#include <stdint.h>\n\n")
            
            f.write("struct LayerConfig {\n")
            f.write("    const char* name;\n")
            f.write("    uint32_t input_channels;\n")
            f.write("    uint32_t output_channels;\n")
            f.write("    uint32_t kernel_size;\n")
            f.write("    uint32_t stride;\n")
            f.write("    uint32_t padding;\n")
            f.write("};\n\n")

            f.write("const struct LayerConfig model_layer_config[] = {\n")
            for idx, (layer_cfg, _, _, _) in enumerate(sim_layers):
                f.write(f"    {{\"{layer_cfg.name}\", {layer_cfg.in_channels}, {layer_cfg.out_channels}, {layer_cfg.kernel_size}, {layer_cfg.stride}, {layer_cfg.padding}}},\n")
            f.write("};\n\n")
            # f.write(f"#define NUM_LAYERS {len(sim_layers)}\n\n")
            f.write("#endif // LAYER_CONFIG_H\n")

        print(f"Exported layer configuration to {config_file}")
        return config_file
    
    def export_quant_params(self, sim_layers: list[tuple[LayerConfig, np.ndarray, np.ndarray, dict]], output_path, mcu_id: int):
        """ This function export the quantization parameters into a C header file for MCU usage. """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        quant_file = output_path / "quant_params.h"
        
        with open(quant_file, 'w') as f:
            f.write("// Auto-generated quantization parameters header file\n\n")
            f.write("#ifndef QUANT_PARAMS_H\n")
            f.write("#define QUANT_PARAMS_H\n\n")
            f.write("#include <stdint.h>\n\n")
            
            # Note: weights quant params are per-channel for conv layers, but input/output quant params are per-tensor
                # if backend == "fbgemm":
                #     qconfig = QConfig(
                #         activation=HistogramObserver.with_args(reduce_range=True),
                #         weight=default_per_channel_weight_observer,
                #     )
            for idx, (layer_cfg, _, _, qp_dict) in enumerate(sim_layers):
                layer_name = layer_cfg.name
                if layer_cfg.type == LayerType.LINEAR:
                    # for linear layers, weights and bias are partitioned by colomn-wise
                    # which means each MCU contains (output_channels / num_mcus) columns of weights and the corresponding bias
                    out_ch_per_mcu = layer_cfg.out_channels // self.num_mcus
                    start_ch = mcu_id * out_ch_per_mcu
                    end_ch = min(start_ch + out_ch_per_mcu, layer_cfg.out_channels)
                    weight_scale = qp_dict['s_w'][start_ch:end_ch]
                    weight_zp = qp_dict['z_w'][start_ch:end_ch]
                else:
                    weight_scale = qp_dict['s_w']
                    weight_zp = qp_dict['z_w']

                # weight_scale = qp_dict['s_w']
                # weight_zp = qp_dict['z_w']
                
                # check if it is per-channel
                if isinstance(weight_scale, np.ndarray):
                    # Per-channel: export arrays
                    f.write(f"// Layer {idx}: {layer_name} - Per-channel weight scales\n")
                    f.write(f"const float {layer_name}_weight_scales[] PROGMEM = {{\n")
                    for i in range(0, len(weight_scale), 8):
                        line = ', '.join(f"{s:.10f}f" for s in weight_scale[i:i+8])
                        f.write(f"    {line},\n")
                    f.write("};\n\n")
                    
                    f.write(f"const int32_t {layer_name}_weight_zps[] PROGMEM = {{\n")
                    for i in range(0, len(weight_zp), 16):
                        line = ', '.join(str(int(z)) for z in weight_zp[i:i+16])
                        f.write(f"    {line},\n")
                    f.write("};\n\n")
            
            # Define quantization parameters struct
            f.write("struct QuantParams {\n")
            f.write("    const float* weight_scales;  // Pointer to per-channel scales array\n")
            f.write("    const int32_t* weight_zps;   // Pointer to per-channel zero_points array\n")
            f.write("    uint32_t num_channels;        // Number of channels (used for per-channel)\n")
            f.write("    float input_scale;\n")
            f.write("    int32_t input_zero_point;\n")
            f.write("    float output_scale;\n")
            f.write("    int32_t output_zero_point;\n")
            f.write("};\n\n")

            f.write("const struct QuantParams model_quant_params[] = {\n")
            for idx, (layer_cfg, _, _, qp_dict) in enumerate(sim_layers):
                layer_name = layer_cfg.name
                if layer_cfg.type == LayerType.LINEAR:
                    out_ch_per_mcu = layer_cfg.out_channels // self.num_mcus
                    start_ch = mcu_id * out_ch_per_mcu
                    end_ch = min(start_ch + out_ch_per_mcu, layer_cfg.out_channels)
                    weight_scale = qp_dict['s_w'][start_ch:end_ch]
                    weight_zp = qp_dict['z_w'][start_ch:end_ch]
                else:
                    weight_scale = qp_dict['s_w']
                    weight_zp = qp_dict['z_w']

                # weight_scale = qp_dict['s_w']
                # weight_zp = qp_dict['z_w']
                input_scale = qp_dict['s_in']
                input_zp = qp_dict['z_in']
                output_scale = qp_dict['s_out']
                output_zp = qp_dict['z_out']
                
                # check if it is per-channel
                if isinstance(weight_scale, np.ndarray):
                    num_channels = len(weight_scale)
                    f.write(f"    {{{layer_name}_weight_scales, {layer_name}_weight_zps, {num_channels}, "
                           f"{input_scale:.10f}f, {input_zp}, {output_scale:.10f}f, {output_zp}}},  // {layer_name}\n")
                else:
                    # Per-tensor
                    f.write(f"    {{(const float[]){{{weight_scale:.10f}f}}, (const int32_t[]){{{weight_zp}}}, 1, "
                           f"{input_scale:.10f}f, {input_zp}, {output_scale:.10f}f, {output_zp}}},  // {layer_name}\n")
            
            f.write("};\n\n")
            f.write("#endif // QUANT_PARAMS_H\n")
        
        print(f"Exported quantization parameters to {quant_file}")
        return quant_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export quantized weights from PyTorch INT8 model")
    parser.add_argument('--num-mcus', type=int, default=1, help='Number of MCUs for partitioning the weights (default: 1)')
    parser.add_argument('--output-dir', type=str, default='../PlatformIO_MCU/Download/include', help='Output directory for header files')
    parser.add_argument('--model-path', type=str, default='./models/mobilenet_v2_quantized.pth', help='Path to quantized model')

    args = parser.parse_args()
    num_mcus = args.num_mcus
    # load the quantized model
    q_model = get_pytorch_quantized_model(train_loader=None)
    q_model.eval()

    sim_layers = extract_quantized_layers(q_model)

    exporter = Exporter(num_mcus=num_mcus)
    for worker_id in range(num_mcus):
        output_path = args.output_dir + f"/mcu_{worker_id}"
        exporter.export_weights(sim_layers=sim_layers, output_path=output_path, mcu_id=worker_id)
        exporter.export_layer_config(sim_layers=sim_layers, output_path=output_path)
        exporter.export_quant_params(sim_layers=sim_layers, output_path=output_path, mcu_id=worker_id)