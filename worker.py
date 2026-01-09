import multiprocessing
import numpy as np
import time
import queue
from typing import List, Dict, Tuple, Any

from protocol import MessageType, TaskPayload, ResultPayload, LayerConfig, LayerType
from operations import pad_input, relu6, numpy_conv2d, numpy_linear, quantized_conv2d, quantized_linear

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
    
    def run(self) -> None:
        print(f"[Worker {self.worker_id}] Starting quant worker process.")
        while True:
            try:
                msg = self.task_queue.get(timeout=1) # wait for a task
            except queue.Empty:
                continue

            type_, payload = msg
            if type_ == MessageType.TASK:
                task: TaskPayload = payload
                start_t = time.time()

                # 1. perform quantized operation based on layer type
                if task.layer_config.type == LayerType.LINEAR:
                    # Fully connected layer
                    out = quantized_linear(task.input_patch, task.weights, task.bias, task.quant_params)
                else:
                    # Convolutional layer
                    out = quantized_conv2d(task.input_patch, task.weights, 
                                           task.bias, task.layer_config.stride,
                                           task.layer_config.groups, task.quant_params)
                    name = task.layer_config.name
                    if "proj" not in name and "fc" not in name:
                        s_out = task.quant_params.s_out
                        z_out = task.quant_params.z_out
                        q_6 = np.clip(round(6.0 / s_out + z_out), 0, 255)
                        out = np.minimum(out, q_6).astype(np.uint8) ## ????
            
                duration = time.time() - start_t
                res = ResultPayload(
                    worker_id=self.worker_id,
                    slice_idx=task.slice_idx,
                    output_patch=out,
                    compute_time=duration
                )
                self.result_queue.put((MessageType.RESULT, res))

            elif type_ == MessageType.TERMINATE:
                print(f"[Worker {self.worker_id}] Terminating quant worker process.")
                break

        
                
        
        