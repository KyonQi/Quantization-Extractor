import numpy as np
import multiprocessing
import json
from typing import List, Dict, Tuple, Any

from worker import Worker, QuantWorker
from protocol import MessageType, TaskPayload, ResultPayload, LayerConfig, LayerType, QuantParams
from operations import pad_input, relu6, numpy_conv2d, numpy_linear

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
    def __init__(self, num_workers: int, quant_params_path: str) -> None:
        self.num_workers = num_workers
        self.task_queue = [multiprocessing.Queue() for _ in range(num_workers)]
        self.result_queue = multiprocessing.Queue()
        self.workers = []

        with open(quant_params_path, 'r') as f:
            self.q_params_dict = json.load(f)
        
        self.feature_map: np.ndarray = None
        self.residual_buffer: Dict[str, Tuple[np.ndarray, float, int]] = {} # name -> (tensor, s, z)

    def get_quant_params(self, layer_name: str) -> Tuple[float, int]:
        """ Get scale and zero point for a given layer """
        if layer_name not in self.q_params_dict:
            raise ValueError(f"Quantization parameters for layer {layer_name} not found")
        
        p = self.q_params_dict[layer_name]
        return float(p['scale']), int(p['zero_point'])
    
    def quantize_input(self, img_float: np.ndarray) -> None:
        """ Quantize input image to uint8 """
        s, z = self.get_quant_params('input')
        self.feature_map = np.clip(np.round(img_float / s + z), 0, 255).astype(np.uint8)

    def execute_inference(self, layers: List[Tuple[LayerConfig, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, str]:
        self.workers = [QuantWorker(i, self.task_queue[i], self.result_queue) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()
        curr_layer_name = 'input'
        try:
            for layer_cfg, weights, bias in layers:
                curr_layer_name = self._run_layer(layer=layer_cfg, weights_float=weights,
                                                  bias_float=bias, 
                                                  prev_layer_name=curr_layer_name)
        finally:
            # Terminate workers
            for q in self.task_queue:
                q.put((MessageType.TERMINATE, None))
            for w in self.workers:
                w.join()
        return self.feature_map, curr_layer_name
        
    def _run_layer(self, layer: LayerConfig, weights_float: np.ndarray, bias_float: np.ndarray, prev_layer_name: str) -> str:
        """ Run a quantized layer """
        # 1. get quantization parameters
        s_in, z_in = self.get_quant_params(prev_layer_name)
        s_out, z_out = self.get_quant_params(layer_name=layer.name)

        w_name = f"{layer.name}_weights"
        s_w, z_w = self.get_quant_params(w_name)
        
        # 2. quantize weights and bias
        weights_q = np.clip(np.round(weights_float / s_w + z_w), 0, 255).astype(np.uint8)
        bias_q = np.round(bias_float / (s_in * s_w)).astype(np.int32)

        # 3. prepare quantization parameters object
        qb = QuantParams(
            s_in=s_in,
            z_in=z_in,
            s_w=s_w,
            z_w=z_w,
            s_out=s_out,
            z_out=z_out,
            m=(s_in * s_w) / s_out
        )

        # 4. cache residual if needed
        if layer.residual_add_to:
            self.residual_buffer[layer.residual_add_to] = (self.feature_map.copy(), s_in, z_in)

        # 5. global average pooling handling
        if layer.type == LayerType.LINEAR and self.feature_map.ndim == 3:
            C, H, W = self.feature_map.shape
            gap_output = np.mean(self.feature_map.astype(np.float32), axis=(1, 2)) # float32
            # quantize gap output
            self.feature_map = np.clip(np.round(gap_output / s_in + z_in), 0, 255).astype(np.uint8)

        # 6. distribute tasks
        if layer.type == LayerType.LINEAR:
            self._distribute_linear(layer=layer, weights_q=weights_q, bias_q=bias_q, quant_params=qb)
        else:
            self._distribute_conv(layer=layer, weights_q=weights_q, bias_q=bias_q, quant_params=qb)
        
        # 7. add residual if needed
        if layer.residual_connect_from:
            self._apply_residual(layer.residual_connect_from, s_out, z_out)
        
        return layer.name
    
    def _distribute_linear(self, layer: LayerConfig, weights_q: np.ndarray, bias_q: np.ndarray, quant_params: QuantParams) -> None:
        input_vec = self.feature_map.flatten() # (C_in, )
        total_classes = layer.out_channels
        classes_per_worker = int(np.ceil(total_classes / self.num_workers))
        active_workers = 0
        
        for i in range(self.num_workers):
            start_cls = i * classes_per_worker
            end_cls = min(start_cls + classes_per_worker, total_classes)
            if start_cls >= end_cls:
                continue

            # split weights and bias by output classes
            w_chunk = weights_q[start_cls:end_cls, :]
            b_chunk = bias_q[start_cls:end_cls]
            task = TaskPayload(
                layer_config=layer,
                slice_idx=(start_cls, end_cls),
                input_patch=input_vec,
                weights=w_chunk,
                bias=b_chunk,
                quant_params=quant_params
            )
            self.task_queue[i].put((MessageType.TASK, task))
            active_workers += 1
        
        final_logits = np.zeros((total_classes, ), dtype=np.uint8)
        collected = 0
        while collected < active_workers:
            type_, res = self.result_queue.get()
            if type_ == MessageType.RESULT:
                res: ResultPayload = res
                start_cls, end_cls = res.slice_idx
                final_logits[start_cls:end_cls] = res.output_patch
                collected += 1
            else:
                raise ValueError("Unexpected message type from worker")
        self.feature_map = final_logits

    def _distribute_conv(self, layer: LayerConfig, weights_q: np.ndarray, bias_q: np.ndarray, quant_params: QuantParams) -> None:
        C, H, W = self.feature_map.shape
        if layer.padding > 0:
            padded_input = pad_input(self.feature_map, layer.padding)
        else:
            padded_input = self.feature_map
        
        H_out = (H + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        W_out = (W + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

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

            task = TaskPayload(
                layer_config=layer,
                slice_idx=(start_row, end_row),
                input_patch=input_patch,
                weights=weights_q,
                bias=bias_q,
                quant_params=quant_params
            )
            self.task_queue[i].put((MessageType.TASK, task))
            active_workers += 1
        
        # collect results
        new_map = np.zeros((layer.out_channels, H_out, W_out), dtype=np.uint8)
        collected = 0
        while collected < active_workers:
            type_, res = self.result_queue.get()
            if type_ == MessageType.RESULT:
                res: ResultPayload = res
                start_row, end_row = res.slice_idx
                new_map[:, start_row:end_row, :] = res.output_patch
                collected += 1
            else:
                raise ValueError("Unexpected message type from worker")
        self.feature_map = new_map

    def _apply_residual(self, res_key: str, s_current: float, z_current: int) -> None:
        """ apply residual connection """
        if res_key not in self.residual_buffer:
            raise ValueError(f"Residual key {res_key} not found in buffer")
        
        res_data, s_res, z_res = self.residual_buffer[res_key]
        if s_res != s_current or z_res != z_current:
            # rescale: res_data -> float32 -> current quantization
            # (q_res - z_res) * s_res -> float32
            # float32 -> ( / s_current + z_current) -> q_current
            res_float = (res_data.astype(np.float32) - z_res) * s_res
            res_quant = np.clip(np.round(res_float / s_current + z_current), 0, 255).astype(np.uint8)
        else:
            res_quant = res_data
        
        # q_out = q_current + q_res - z_current ????
        sum_result = self.feature_map.astype(np.int32) + res_quant.astype(np.int32) - z_current
        self.feature_map = np.clip(sum_result, 0, 255).astype(np.uint8)
    