import multiprocessing
import numpy as np
import time
import queue
from typing import List, Dict, Tuple, Any

from protocol.protocol import MessageType, TaskPayload, ResultPayload, LayerConfig, LayerType
from operations import pad_input, relu6, numpy_conv2d, numpy_linear, quantized_conv2d, quantized_linear
from compression.compressors import Compressor, BitMapANSCompressor

class Worker(multiprocessing.Process):
    def __init__(self, worker_id: int, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
    
    def run(self) -> None:
        print(f"[Worker {self.worker_id}] Starting worker process.")
        while True:
            try:
                msg = self.task_queue.get(timeout=1) # wait for a task
            except queue.Empty:
                continue
            
            type_, payload = msg
            if type_ == MessageType.TASK:
                task: TaskPayload = payload
                start_t = time.time()
                
                # 1. perform operation based on layer type
                if task.layer_config.type == LayerType.LINEAR:
                    # Fully connected layer
                    # input is already flattened
                    out = numpy_linear(task.input_patch, task.weights, task.bias)
                else:
                    # Convolutional layer
                    out = numpy_conv2d(task.input_patch, task.weights, task.bias, task.layer_config.stride, task.layer_config.groups)
                    # Pointwise does not need activation while expanded conv and depthwise need ReLU6
                    name = task.layer_config.name
                    if "proj" not in name and "fc" not in name:
                        out = relu6(out)
                
                duration = time.time() - start_t
                
                res = ResultPayload(
                    worker_id=self.worker_id,
                    slice_idx=task.slice_idx,
                    output_patch=out,
                    compute_time=duration
                )
                self.result_queue.put((MessageType.RESULT, res))
            elif type_ == MessageType.TERMINATE:
                print(f"[Worker {self.worker_id}] Terminating worker process.")
                break
    

class QuantWorker(multiprocessing.Process):
    def __init__(self, worker_id: int, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.compressor = BitMapANSCompressor()

        # cache output for halo usage
        self.local_cache: Dict[str, np.ndarray] = {}
    
    def run(self) -> None:
        # print(f"[Worker {self.worker_id}] Starting quant worker process.")
        while True:
            try:
                msg = self.task_queue.get(timeout=1) # wait for a task
            except queue.Empty:
                continue

            type_, payload = msg
            if type_ == MessageType.TASK:
                task: TaskPayload = payload
                start_t = time.time()
                # record the time
                t0 = time.perf_counter()
                # input_patch = self.compressor.decompress(task.input_patch_compressed)
                input_patch = self._fetch_input(task)
                t_decomp = time.perf_counter() - t0

                # 1. perform quantized operation based on layer type
                t_compute_start = time.perf_counter()
                if task.layer_config.type == LayerType.LINEAR:
                    # Fully connected layer
                    out = quantized_linear(input_patch, task.weights, task.bias, task.quant_params)
                else:
                    # Convolutional layer
                    out = quantized_conv2d(input_patch, task.weights, 
                                           task.bias, task.layer_config.stride,
                                           task.layer_config.groups, task.quant_params)
                t_compute = time.perf_counter() - t_compute_start

                # cache the conv output for next halo layer
                if task.layer_config.type != LayerType.LINEAR:
                    self.local_cache[task.layer_config.name] = out


                t1 = time.perf_counter()
                output_patch_compressed = self.compressor.compress(out)
                t_comp = time.perf_counter() - t1
                upstream_size = len(output_patch_compressed)

                duration = time.time() - start_t
                res = ResultPayload(
                    worker_id=self.worker_id,
                    slice_idx=task.slice_idx,
                    output_patch=out,
                    output_patch_compressed=output_patch_compressed,
                    # compute_time=duration,
                    compute_time=t_compute,
                    codec_time=t_decomp + t_comp,
                    compressed_size=upstream_size
                )
                self.result_queue.put((MessageType.RESULT, res))

            elif type_ == MessageType.TERMINATE:
                # print(f"[Worker {self.worker_id}] Terminating quant worker process.")
                break

    def _fetch_input(self, task: TaskPayload) -> np.ndarray:
        """ fetch input for convolutional layer, either from task payload or from local cache for halo data """
        if task.prev_layer_name is None:
            return self.compressor.decompress(task.input_patch_compressed)
        
        # halo reconstruction
        cached = self.local_cache[task.prev_layer_name] # (C, rows_cached, cols)
        use_start, use_end = task.cache_use_range
        cache_slice = cached[:, use_start:use_end, :]
        padding = task.layer_config.padding
        z_in = int(task.quant_params.z_in)

        if padding > 0:
            cache_wpad = np.pad(
                cache_slice, ((0, 0), (0, 0), (padding, padding)),
                mode="constant", constant_values=z_in
            )
        else:
            cache_wpad = cache_slice

        parts = []
        top = task.halo_top
        if top.size > 0:
            parts.append(top)
        parts.append(cache_wpad)
        bottom = task.halo_bottom
        if bottom.size > 0:
            parts.append(bottom)
        return np.concatenate(parts, axis=1) # axis = 1?
                
        
        