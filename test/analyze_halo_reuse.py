"""
Halo-reuse analysis: compare per-layer coord->worker input-patch volume
under (a) current layer-wise redistribute vs (b) each worker caches its
own output slice and coord only ships the halo rows from neighbors.

Assumes uint8 activations (quantized path). Weights and return-trip
output are excluded - they are identical between the two schemes.
"""
import os, sys
import math
import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.registry import get_adapter
from protocol.protocol import LayerType

DTYPE_BYTES = 1  # uint8 quantized activations

def analyze(model_name: str, num_workers: int, input_shape=(3, 224, 224)):
    adapter = get_adapter(model_name)
    model = adapter.load_fp32()
    sim_layers = adapter.extract_fp32_layers(model)

    C, H, W = input_shape
    prev_worker_rows = None  # list[(start,end)] in output-of-prev-layer coords, None = no cache

    total_current = 0
    total_halo = 0

    print(f"\nModel: {model_name}  |  workers: {num_workers}  |  input: {input_shape}")
    print("-" * 108)
    print(f"{'#':>3}  {'layer':<18} {'type':<10} {'kspd':<8} {'in(C,H,W)':<15} "
          f"{'cur avg/w':>10} {'halo avg/w':>10} {'cur total':>11} {'halo total':>11} {'save%':>7}")
    print("-" * 108)

    for idx, (cfg, _, _) in enumerate(sim_layers):
        if cfg.type == LayerType.LINEAR:
            # GAP -> linear: coord sends full (C,) vector to each worker in current scheme.
            # In halo scheme, each worker still needs all channels -> no saving.
            cur = C * DTYPE_BYTES * num_workers
            halo = cur
            total_current += cur
            total_halo += halo
            per = C * DTYPE_BYTES
            print(f"{idx:>3}  {cfg.name:<18} {'LINEAR':<10} {'-':<8} {f'({C},)':<15} "
                  f"{per:>10} {per:>10} {cur:>11} {halo:>11} {'0.0':>7}")
            prev_worker_rows = None
            continue

        k, s, p = cfg.kernel_size, cfg.stride, cfg.padding
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1
        C_in = C
        C_out = cfg.out_channels

        rows_per_worker = int(math.ceil(H_out / num_workers))

        layer_current = 0
        layer_halo = 0
        new_rows = []
        per_w_cur = []
        per_w_halo = []

        for i in range(num_workers):
            start_row = i * rows_per_worker
            end_row = min(start_row + rows_per_worker, H_out)
            if start_row >= H_out:
                continue
            in_start_y = start_row * s              # padded-input coords
            in_end_y = (end_row - 1) * s + k
            patch_rows = in_end_y - in_start_y
            patch_w = W + 2 * p                     # current scheme ships padded width

            cur_b = C_in * patch_rows * patch_w * DTYPE_BYTES

            if prev_worker_rows is None:
                halo_b = cur_b  # no local cache (input image or post-linear)
            else:
                # unpadded rows this worker needs (in prev-layer output coords)
                need_u_start = max(0, in_start_y - p)
                need_u_end = min(H, in_end_y - p)
                need_u_len = max(0, need_u_end - need_u_start)

                cached_s, cached_e = prev_worker_rows[i]
                ov_s = max(need_u_start, cached_s)
                ov_e = min(need_u_end, cached_e)
                overlap = max(0, ov_e - ov_s)

                new_from_peers = need_u_len - overlap
                # padding (top/bottom/left/right) is recreated locally, not shipped
                halo_b = C_in * new_from_peers * W * DTYPE_BYTES

            layer_current += cur_b
            layer_halo += halo_b
            new_rows.append((start_row, end_row))
            per_w_cur.append(cur_b)
            per_w_halo.append(halo_b)

        total_current += layer_current
        total_halo += layer_halo

        save = 0.0 if layer_current == 0 else (1 - layer_halo / layer_current) * 100
        tname = cfg.type.name[:8]
        kspd = f"k{k}s{s}p{p}"
        shape = f"({C_in},{H},{W})"
        nw = len(per_w_cur)
        avg_cur = layer_current // nw
        avg_halo = layer_halo // nw
        print(f"{idx:>3}  {cfg.name:<18} {tname:<10} {kspd:<8} {shape:<15} "
              f"{avg_cur:>10} {avg_halo:>10} {layer_current:>11} {layer_halo:>11} {save:>6.1f}")

        prev_worker_rows = new_rows
        C, H, W = C_out, H_out, W_out

    print("-" * 108)
    save_total = (1 - total_halo / total_current) * 100
    print(f"{'TOTAL input-patch volume (all layers, all workers)':<72} "
          f"{total_current:>11} {total_halo:>11} {save_total:>6.1f}")
    print(f"  current:  {total_current/1024:.2f} KB")
    print(f"  halo:     {total_halo/1024:.2f} KB")
    print(f"  saved:    {(total_current-total_halo)/1024:.2f} KB  ({save_total:.1f}%)")
    return total_current, total_halo


if __name__ == "__main__":
    analyze("mbv2_0.35", num_workers=4, input_shape=(3, 224, 224))
