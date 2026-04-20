import numpy as np
import multiprocessing
import json
import time
from typing import List, Dict, Tuple, Optional

from worker import Worker, QuantWorker
from protocol.protocol import MessageType, TaskPayload, ResultPayload, LayerConfig, LayerType, QuantParams
from operations import pad_input, relu6, numpy_conv2d, numpy_linear, quantized_pad_input
from compression.compressors import Compressor, BitMapANSCompressor

class Coordinator:
    def __init__(self, num_workers: int, input_shape: Tuple[int, int, int] = (3, 244, 244)) -> None:
        self.num_workers = num_workers
        self.feature_map = np.zeros(input_shape, dtype=np.float32) # C, H, W
        
        self.task_queue = [multiprocessing.Queue() for _ in range(num_workers)]
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        
        self.residual_buffers: Dict[str, np.ndarray] = {} # to store residual inputs
    
    def set_input(self, img: np.ndarray) -> None:
        """ Set the input image as the initial feature map """
        self.feature_map = img.astype(np.float32)

    def execute_inference(self, layers: List[Tuple[LayerConfig, np.ndarray, np.ndarray]]) -> np.ndarray:
        """ Execute inference on the model layers """
        self.workers = [Worker(worker_id=i, task_queue=self.task_queue[i], result_queue=self.result_queue) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()
        
        try:
            for layer_cfg, weights, bias in layers:
                self._run_layer(layer=layer_cfg, weights=weights, bias=bias)
        finally:
            # Terminate workers
            for q in self.task_queue:
                q.put((MessageType.TERMINATE, None))
            for w in self.workers:
                w.join()

        return self.feature_map        

    def _run_layer(self, layer: LayerConfig, weights: np.ndarray, bias: np.ndarray) -> None:
        print(f"[Coordinator] Running layer: {layer.name} ({layer.type.name})")
        
        # 1. store residual if needed
        if layer.residual_add_to:
            self.residual_buffers[layer.residual_add_to] = self.feature_map.copy()
        
        # 2. distribute tasks based on layer type
        if layer.type == LayerType.LINEAR:
            self._distribute_linear(layer=layer, weights=weights, bias=bias)
        else:
            self._distribute_conv(layer=layer, weights=weights, bias=bias)

        # 3. add residual if needed
        if layer.residual_connect_from:
            key = layer.residual_connect_from
            if key in self.residual_buffers:
                cached = self.residual_buffers[key]
                if cached.shape == self.feature_map.shape:
                    self.feature_map += cached
                else:
                    raise ValueError(f"Residual shape mismatch for layer {layer.name}: cached {cached.shape}, current {self.feature_map.shape}")

    def _distribute_conv(self, layer: LayerConfig, weights: np.ndarray, bias: np.ndarray) -> None:
        """ Convolution layer distribution """
        C_in, H_in, W_in = self.feature_map.shape
        H_out = (H_in + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        W_out = (W_in + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

        padded_input = pad_input(self.feature_map, layer.padding)

        rows_per_worker = int(np.ceil(H_out / self.num_workers))
        active_workers = 0
        
        for i in range(self.num_workers):
            start_row = i * rows_per_worker
            end_row = min(start_row + rows_per_worker, H_out)
            if start_row >= H_out:
                continue
            
            # Calculate input patch for this slice
            in_start_y = start_row * layer.stride
            in_end_y = (end_row - 1) * layer.stride + layer.kernel_size
            input_patch = padded_input[:, in_start_y:in_end_y, :]
            
            task = TaskPayload(layer_config=layer, slice_idx=(start_row, end_row), input_patch=input_patch, weights=weights, bias=bias)
            self.task_queue[i].put((MessageType.TASK, task))
            active_workers += 1
        
        # Collect results
        new_map = np.zeros((layer.out_channels, H_out, W_out), dtype=np.float32)
        collected = 0
        while collected < active_workers:
            type, payload = self.result_queue.get()
            if type == MessageType.RESULT:
                res: ResultPayload = payload
                start_row, end_row = res.slice_idx
                new_map[:, start_row:end_row, :] = res.output_patch
                collected += 1
            else:
                raise ValueError("Unexpected message type from worker")
        self.feature_map = new_map

    def _distribute_linear(self, layer: LayerConfig, weights: np.ndarray, bias: np.ndarray) -> None:
        """ Fully connected layer distribution """
        input_vec = self.feature_map.flatten() # flatten input
        total_classes = layer.out_channels
        classes_per_worker = int(np.ceil(total_classes / self.num_workers))
        active_workers = 0
        
        for i in range(self.num_workers):
            start_cls = i * classes_per_worker
            end_cls = min(start_cls + classes_per_worker, total_classes)
            if start_cls >= end_cls:
                continue
            
            # Slice weights and bias, weights: (out_features, in_features), bias: (out_features,)
            # every worker gets all input features but only a subset of output classes
            w_chunk = weights[start_cls:end_cls, :]
            b_chunk = bias[start_cls:end_cls]
            
            task = TaskPayload(layer_config=layer, slice_idx=(start_cls, end_cls), input_patch=input_vec, weights=w_chunk, bias=b_chunk)
            self.task_queue[i].put((MessageType.TASK, task))
            active_workers += 1
        
        # Collect results
        final_logits = np.zeros((total_classes,), dtype=np.float32)
        collected = 0
        while collected < active_workers:
            type, payload = self.result_queue.get()
            if type == MessageType.RESULT:
                res: ResultPayload = payload
                start_cls, end_cls = res.slice_idx
                final_logits[start_cls:end_cls] = res.output_patch
                collected += 1
            else:
                raise ValueError("Unexpected message type from worker")
        self.feature_map = final_logits

        
class QuantCoordinator:
    def __init__(self, num_workers: int, quant_params_path: str = "NoUse", use_halo: bool = False) -> None:
        self.num_workers = num_workers
        self.task_queue = [multiprocessing.Queue() for _ in range(num_workers)]
        self.result_queue = multiprocessing.Queue()
        self.workers = []

        # with open(quant_params_path, 'r') as f:
        #     self.q_params_dict = json.load(f)
        
        self.feature_map: np.ndarray = None
        self.residual_buffer: Dict[str, Tuple[np.ndarray, float, int]] = {} # name -> (tensor, s, z)

        self.compressor = BitMapANSCompressor()
        self.stats = {
            "total_inference_time": 0.0, # end to end time
            "total_comm_volume": 0, # total communication volume in bytes
            "total_codec_time": 0.0, # total time spent in compression/decompression
            "total_compute_time": 0.0 # total time spent in computation
        }

        # halo state
        self.use_halo = use_halo
        self._prev_conv_name: Optional[str] = None
        self._prev_worker_rows: Optional[list] = None # List[(start_row, end_row)] per worker
        self._prev_H_out: Optional[int] = None

    def get_quant_params(self, layer_name: str) -> Tuple[float, int]:
        """ Get scale and zero point for a given layer """
        if layer_name not in self.q_params_dict:
            raise ValueError(f"Quantization parameters for layer {layer_name} not found")
        
        p = self.q_params_dict[layer_name]
        return float(p['scale']), int(p['zero_point'])
    
    def quantize_input(self, img_float: np.ndarray, s_in: float, z_in: int) -> None:
        """ Quantize input image to uint8 """
        # s, z = self.get_quant_params('input')
        self.feature_map = np.clip(np.round(img_float / s_in + z_in), 0, 255).astype(np.uint8)

    def execute_inference(self, layers: List[Tuple[LayerConfig, np.ndarray, np.ndarray, QuantParams]]) -> Tuple[np.ndarray, str]:
        # self.stats = {k: 0 for k in self.stats}
        
        start_time = time.perf_counter()

        self._prev_conv_name = None
        self._prev_worker_rows = None
        self._prev_H_out = None

        self.workers = [QuantWorker(i, self.task_queue[i], self.result_queue) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

        last_layer_name = ""
        try:
            for layer_cfg, w_int8, b_int32, qp_dict in layers:
                self._run_layer(layer=layer_cfg, weights=w_int8, bias=b_int32, qp_dict=qp_dict)
                last_layer_name = layer_cfg.name
        finally:
            for q in self.task_queue:
                q.put((MessageType.TERMINATE, None))
            for w in self.workers:
                w.join()
        self.stats["total_inference_time"] += time.perf_counter() - start_time
        return self.feature_map, last_layer_name
        
    def _run_layer(self, layer: LayerConfig, weights: np.ndarray, bias: np.ndarray, qp_dict: dict) -> None:
        s_in, z_in = qp_dict['s_in'], qp_dict['z_in']
        s_out, z_out = qp_dict['s_out'], qp_dict['z_out']
        s_w, z_w = qp_dict['s_w'], qp_dict['z_w']

        # calculate multiplier m
        m = (s_in * s_w) / s_out
        qp = QuantParams(s_in=s_in, z_in=z_in, s_out=s_out, z_out=z_out, s_w=s_w, z_w=z_w, m=m)
        
        if layer.residual_add_to:
            self.residual_buffer[layer.residual_add_to] = (self.feature_map.copy(), s_in, z_in)
        
        if layer.type == LayerType.LINEAR and self.feature_map.ndim == 3:
            gap_output = np.mean(self.feature_map, axis=(1, 2)) # global average pooling, shape: (C, H, W ) -> (C, ) 
            self.feature_map = np.round(gap_output).astype(np.uint8)

        
        if layer.type == LayerType.LINEAR:
            self._distribute_linear(layer=layer, weights_q=weights, bias_q=bias, quant_params=qp)
        else:
            self._distribute_conv(layer=layer, weights_q=weights, bias_q=bias, quant_params=qp)
        
        if layer.residual_connect_from:
            target_s = qp_dict.get('residual_out_scale', s_out)
            target_z = qp_dict.get('residual_out_zp', z_out)
            self._apply_residual(res_key=layer.residual_connect_from, curr_s=s_out, curr_z=z_out, target_s=target_s, target_z=target_z)
            
    def _distribute_linear(self, layer: LayerConfig, weights_q: np.ndarray, bias_q: np.ndarray, quant_params: QuantParams) -> None:
        input_vec = self.feature_map.flatten() # (C_in, )
        total_classes = layer.out_channels
        classes_per_worker = int(np.ceil(total_classes / self.num_workers))
        active_workers = 0

        t0 = time.perf_counter()
        input_vec_compressed = self.compressor.compress(input_vec)
        self.stats["total_codec_time"] += (time.perf_counter() - t0)

        for i in range(self.num_workers):
            start_cls = i * classes_per_worker
            end_cls = min(start_cls + classes_per_worker, total_classes)
            if start_cls >= end_cls:
                continue

            # split weights and bias by output classes
            w_chunk = weights_q[start_cls:end_cls, :]
            b_chunk = bias_q[start_cls:end_cls]
            # split the quantparam
            ## Slice Multiplier (m)
            m_chunk = quant_params.m[start_cls:end_cls] if isinstance(quant_params.m, np.ndarray) else quant_params.m
            ## Slice Weight Zero Point (z_w)
            zw_chunk = quant_params.z_w[start_cls:end_cls] if isinstance(quant_params.z_w, np.ndarray) else quant_params.z_w
            ## Slice Weight Scale (s_w)
            sw_chunk = quant_params.s_w[start_cls:end_cls] if isinstance(quant_params.s_w, np.ndarray) else quant_params.s_w
            task_qp = QuantParams(
                s_in=quant_params.s_in,
                z_in=quant_params.z_in,
                s_out=quant_params.s_out,
                z_out=quant_params.z_out,
                s_w=sw_chunk,
                z_w=zw_chunk,
                m=m_chunk
            )

            self.stats["total_comm_volume"] += len(input_vec_compressed)

            task = TaskPayload(
                layer_config=layer,
                slice_idx=(start_cls, end_cls),
                input_patch=input_vec,
                input_patch_compressed=input_vec_compressed,
                weights=w_chunk,
                bias=b_chunk,
                quant_params=task_qp
            )
            self.task_queue[i].put((MessageType.TASK, task))
            active_workers += 1
        
        final_logits = np.zeros((total_classes, ), dtype=np.uint8)
        collected = 0
        while collected < active_workers:
            type_, res = self.result_queue.get()
            if type_ == MessageType.RESULT:
                res: ResultPayload = res

                self.stats["total_comm_volume"] += len(res.output_patch_compressed)
                self.stats["total_codec_time"] += res.codec_time
                self.stats["total_compute_time"] += res.compute_time

                start_cls, end_cls = res.slice_idx
                # final_logits[start_cls:end_cls] = res.output_patch
                t1 = time.perf_counter()
                output_decompressed = self.compressor.decompress(res.output_patch_compressed)
                self.stats["total_codec_time"] += (time.perf_counter() - t1)
                final_logits[start_cls:end_cls] = output_decompressed                
                collected += 1
            else:
                raise ValueError("Unexpected message type from worker")
        self.feature_map = final_logits

        # clear the cache since linear layer won't be used for halo
        self._prev_conv_name = None
        self._prev_worker_rows = None
        self._prev_H_out = None

    def _distribute_conv(self, layer: LayerConfig, weights_q: np.ndarray, bias_q: np.ndarray, quant_params: QuantParams) -> None:
        C, H, W = self.feature_map.shape
        if layer.padding > 0:
            padded_input = quantized_pad_input(self.feature_map, layer.padding, quant_params.z_in)
        else:
            padded_input = self.feature_map
        
        H_out = (H + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        W_out = (W + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

        # halo mode
        halo_mode = (
            self.use_halo 
            and self._prev_conv_name is not None 
            and layer.residual_add_to is None
            and layer.residual_connect_from is None
        )
        new_worker_rows: List[Tuple[int, int]] = []

        # distribute rows to workers
        rows_per_worker = int(np.ceil(H_out / self.num_workers))
        active_workers = 0
        
        for i in range(self.num_workers):
            start_row = i * rows_per_worker
            end_row = min(start_row + rows_per_worker, H_out)
            if start_row >= H_out:
                continue

            in_start_y = start_row * layer.stride
            in_end_y = (end_row - 1) * layer.stride + layer.kernel_size
            input_patch = padded_input[:, in_start_y:in_end_y, :]
            t0 = time.perf_counter()
            input_patch_compressed = self.compressor.compress(input_patch)
            self.stats["total_codec_time"] += (time.perf_counter() - t0)
            self.stats["total_comm_volume"] += len(input_patch_compressed)

            if not halo_mode:
                task = TaskPayload(
                    layer_config=layer,
                    slice_idx=(start_row, end_row),
                    input_patch=input_patch,
                    input_patch_compressed=input_patch_compressed,
                    weights=weights_q,
                    bias=bias_q,
                    quant_params=quant_params
                )
            else:
                prev_start, prev_end = self._prev_worker_rows[i]
                cache_padded_start = layer.padding + prev_start
                cache_padded_end = layer.padding + prev_end
                ov_start = max(in_start_y, cache_padded_start)
                ov_end = min(in_end_y, cache_padded_end)
                if ov_end <= ov_start:
                    raise ValueError(f"No overlap between worker {i} input patch and cached halo data for layer {layer.name}")
                cache_use_start = ov_start - cache_padded_start
                cache_use_end = cache_use_start + (ov_end - ov_start)
                halo_top = padded_input[:, in_start_y:ov_start, :]
                halo_bottom = padded_input[:, ov_end:in_end_y, :]
                task = TaskPayload(
                    layer_config=layer,
                    slice_idx=(start_row, end_row),
                    input_patch=input_patch,
                    input_patch_compressed=input_patch_compressed,
                    weights=weights_q,
                    bias=bias_q,
                    quant_params=quant_params,
                    prev_layer_name=self._prev_conv_name,
                    halo_top=halo_top,
                    halo_bottom=halo_bottom,
                    cache_use_range=(cache_use_start, cache_use_end)
                )

            self.task_queue[i].put((MessageType.TASK, task))

            new_worker_rows.append((start_row, end_row))

            active_workers += 1
        
        # collect results
        new_map = np.zeros((layer.out_channels, H_out, W_out), dtype=np.uint8)
        collected = 0
        while collected < active_workers:
            type_, res = self.result_queue.get()
            if type_ == MessageType.RESULT:
                res: ResultPayload = res
                
                self.stats["total_comm_volume"] += len(res.output_patch_compressed)
                self.stats["total_codec_time"] += res.codec_time
                self.stats["total_compute_time"] += res.compute_time

                start_row, end_row = res.slice_idx
                # new_map[:, start_row:end_row, :] = res.output_patch
                
                t1 = time.perf_counter()
                decompressed_patch= self.compressor.decompress(res.output_patch_compressed)
                self.stats["total_codec_time"] += (time.perf_counter() - t1)

                new_map[:, start_row:end_row, :] = decompressed_patch
                collected += 1
            else:
                raise ValueError("Unexpected message type from worker")
        self.feature_map = new_map

        # update halo state
        self._prev_conv_name = layer.name
        self._prev_worker_rows = new_worker_rows
        self._prev_H_out = H_out

    def _apply_residual(self, res_key: str, curr_s, curr_z, target_s, target_z) -> None:
        res_data, res_s, res_z = self.residual_buffer[res_key]
        curr_f = (self.feature_map.astype(np.float32)- curr_z) * curr_s
        res_f = (res_data.astype(np.float32) - res_z) * res_s
        sum_f = curr_f + res_f
        self.feature_map = np.clip(np.round(sum_f / target_s + target_z), 0, 255).astype(np.uint8)
        
    