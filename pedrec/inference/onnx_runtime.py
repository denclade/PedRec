"""
onnxruntime backend for the exported networks (see ``pedrec/tools/networks/export_onnx.py``).

With ``onnxruntime-gpu`` the TensorRT execution provider (if TensorRT is installed) or the CUDA execution provider is
used. Inputs and outputs stay on the GPU (IO binding), so the pipeline code is the same as for the PyTorch backend.
"""
import logging
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from pedrec.configs.default_paths import data_path

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")


def default_onnx_dir(data_root: Optional[str] = None) -> str:
    return data_path("models", "onnx", data_root=data_root)


class OnnxModule:
    """Callable like the PyTorch module: takes a torch tensor, returns a tuple of torch tensors on the same device."""

    def __init__(self, path: str, device: torch.device, providers: Optional[Sequence[str]] = None,
                 trt_cache_dir: Optional[str] = None):
        import onnxruntime as ort
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ONNX model {path} not found. Export it with "
                                    f"'python pedrec/tools/networks/export_onnx.py' first.")
        available = ort.get_available_providers()
        wanted = list(providers or DEFAULT_PROVIDERS)
        if device.type != "cuda":
            wanted = ["CPUExecutionProvider"]
        selected = []
        for provider in wanted:
            if provider not in available:
                continue
            if provider == "TensorrtExecutionProvider":
                cache = trt_cache_dir or os.path.join(os.path.dirname(path), "trt_cache")
                os.makedirs(cache, exist_ok=True)
                selected.append((provider, {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                            "trt_engine_cache_path": cache}))
            elif provider == "CUDAExecutionProvider":
                selected.append((provider, {"device_id": device.index or 0}))
            else:
                selected.append(provider)
        self.session = ort.InferenceSession(path, providers=selected)
        self.device = device
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"Loaded {path} with onnxruntime providers {self.session.get_providers()}")

    def __call__(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        inputs = inputs.float().contiguous()
        if self.device.type == "cuda":
            binding = self.session.io_binding()
            binding.bind_input(self.input_name, "cuda", self.device.index or 0, np.float32, tuple(inputs.shape),
                               inputs.data_ptr())
            for name in self.output_names:
                binding.bind_output(name, "cuda", self.device.index or 0)
            self.session.run_with_iobinding(binding)
            outputs = [torch.from_dlpack(o.to_dlpack()) if hasattr(o, "to_dlpack")
                       else torch.from_numpy(o.numpy()).to(self.device) for o in binding.get_outputs()]
        else:
            outputs = [torch.from_numpy(o) for o in
                       self.session.run(self.output_names, {self.input_name: inputs.cpu().numpy()})]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
