import argparse
import json
import torch
import numpy as np
import torchvision.models as models
from pathlib import Path
from models import list_models, get_adapter
from protocol.protocol import LayerType, LayerConfig, QuantParams

class MCUExporter:
    """ Exporter for MCU deployment. Converts quantized model layers and parameters into a JSON format suitable for MCU inference. """
    
    def __init__(self, model_name: str, num_mcus: int = 1):
        self.model_name = model_name
        self.num_mcus = num_mcus

    def export(self, sim_layers, output_path):
        """ Main export function that generates all necessary files for MCU deployment (coord + workers). """
        self.export_model_config(sim_layers, output_path)

        for mcu_id in range(self.num_mcus):
            mcu_dir = f"{output_path}/mcu_{mcu_id}"
            self.export_weights_h(sim_layers, mcu_dir, mcu_id)
            self.export_layer_config_h(sim_layers, mcu_dir, mcu_id)
            self.export_quant_params_h(sim_layers, mcu_dir, mcu_id)

    def export_model_config(self, sim_layers, output_path):
        data = {
            "model": self.model_name,
            "num_layers": len(sim_layers),
            "layers": []
        }

        for cfg, _, _, qp_dict in sim_layers:
            layer_data = {
                'layer_config': layer_config_to_dict(cfg),
                'quant_params': quant_params_to_dict(qp_dict)
            }
            data["layers"].append(layer_data)
        
        output_path: Path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        config_file = output_path / f"model_config_{self.model_name}.json"
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Exporter]: Exported model configuration to {config_file}")

    def export_weights_h(self, sim_layers, output_path, mcu_id):
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
        
        print(f"[Exporter]: Exported weights and biases to {weights_file}")
        print(f"[Exporter]: Total weights size: {total_weights_size / 1024:.2f} KB")
        print(f"[Exporter]: Total bias size: {total_bias_size / 1024:.2f} KB")

        return weights_file

    def export_layer_config_h(self, sim_layers, output_path, mcu_id):
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

        print(f"[Exporter]:Exported layer configuration to {config_file}")
        return config_file

    def export_quant_params_h(self, sim_layers, output_path, mcu_id):
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
        
        print(f"[Exporter]: Exported quantization parameters to {quant_file}")
        return quant_file


def layer_config_to_dict(layer_config: LayerConfig) -> dict:
    return {
        "name": layer_config.name,
        "type": layer_config.type.value,
        "in_channels": layer_config.in_channels,
        "out_channels": layer_config.out_channels,
        "kernel_size": layer_config.kernel_size,
        "stride": layer_config.stride,
        "padding": layer_config.padding,
        "groups": layer_config.groups,
        "residual_add_to": layer_config.residual_add_to,
        "residual_connect_from": layer_config.residual_connect_from

    }

def quant_params_to_dict(quant_params: dict) -> dict:
    m = quant_params.get("m", quant_params["s_in"] * quant_params["s_w"] / quant_params["s_out"])
    return {
        "s_in": quant_params["s_in"],
        "z_in": quant_params["z_in"],
        "s_w": quant_params["s_w"].tolist() if isinstance(quant_params["s_w"], np.ndarray) else quant_params["s_w"],
        "z_w": quant_params["z_w"].tolist() if isinstance(quant_params["z_w"], np.ndarray) else quant_params["z_w"],
        "s_out": quant_params["s_out"],
        "z_out": quant_params["z_out"],
        "m": m.tolist() if isinstance(m, np.ndarray) else m,
        "s_residual_out": quant_params.get("residual_out_scale", None),
        "z_residual_out": quant_params.get("residual_out_zp", None)
    }

def main():
    available_models = list_models()
    parser = argparse.ArgumentParser(description="Export quantized model to MCU C headers + JSON config from PyTorch INT8 model")
    parser.add_argument('--model', type=str, required=True, choices=available_models, help=f'Model name to export. Available models: {available_models}')
    parser.add_argument('--num-mcus', type=int, default=4, help='Number of MCUs for partitioning the weights (default: 1)')
    parser.add_argument('--output-dir', type=str, default='./extractor_mcu/', help='Output directory for header files')
    parser.add_argument('--model-path', type=str, required=True, help='Path to quantized model')

    args = parser.parse_args()
    model = args.model
    num_mcus = args.num_mcus
    model_path = args.model_path
    output_path = args.output_dir
    # load the quantized model
    adapter = get_adapter(model)
    print(f"[Exporter] Model: {model}, Num MCUs: {num_mcus}, Model Path: {model_path}")
    q_model = adapter.quantize(calibration_loader=None, num_calibration_batches=0, save_path=model_path)
    
    # q_model = get_pytorch_quantized_model(train_loader=None, save_path=model_path, width_mult=0.35)
    q_model.eval()

    sim_layers = adapter.extract_quantized_layers(q_model)

    exporter = MCUExporter(model, 4)
    exporter.export(sim_layers, output_path)

if __name__ == "__main__":
    main()