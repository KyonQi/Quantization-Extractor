import numpy as np
import multiprocessing
from typing import List, Dict, Tuple, Any

from worker import Worker
from protocol import MessageType, TaskPayload, ResultPayload, LayerConfig, LayerType
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

        

                  