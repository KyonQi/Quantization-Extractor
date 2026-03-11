import json
import numpy as np
from pathlib import Path
from models.utils import get_pytorch_quantized_model
from protocol.protocol import LayerConfig, QuantParams
from quant.quant_model_utils import extract_quantized_layers

WIDTH_MULT = 0.35
QUANTIZED_SAVE_PATH = f'./models/mobilenet_v2_quantized_{WIDTH_MULT}.pth'

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

def save_model_config(layers: list[LayerConfig], quant_params: list[dict], output_path: str):
    if len(layers) != len(quant_params):
        raise ValueError("Layers and quant_params must have the same length")
    
    data = {
        "num_layers": len(layers),
        "layers": []
    }
    for cfg, qp_dict in zip(layers, quant_params):
        layer_data = {
            'layer_config': layer_config_to_dict(cfg),
            'quant_params': quant_params_to_dict(qp_dict)
        }
        data["layers"].append(layer_data)

        output_path: Path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        config_file = output_path / f"model_config_{WIDTH_MULT}.json"
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    q_model = get_pytorch_quantized_model(train_loader=None, save_path=QUANTIZED_SAVE_PATH, width_mult=WIDTH_MULT)
    q_model.eval()

    sim_layers = extract_quantized_layers(q_model)

    layer_configs = []
    quant_params = []
    
    for cfg, weights, bias, qp_dict in sim_layers:
        layer_configs.append(cfg)
        quant_params.append(qp_dict)
        # qp = QuantParams(
        #     s_in=qp_dict["s_in"],
        #     z_in=qp_dict["z_in"],
        #     s_w=qp_dict["s_w"],
        #     z_w=qp_dict["z_w"],
        #     s_out=qp_dict["s_out"],
        #     z_out=qp_dict["z_out"],
        #     m=qp_dict.get("m", qp_dict["s_in"] * qp_dict["s_w"] / qp_dict["s_out"])
        # )
        # quant_params.append(qp)
    save_model_config(layers=layer_configs, quant_params=quant_params, output_path="./extractor_mcu/")

if __name__ == "__main__":
    main()