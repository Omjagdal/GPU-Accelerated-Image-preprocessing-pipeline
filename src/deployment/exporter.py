"""
ONNX & TensorRT Export Pipeline.

Provides:
* :class:`ONNXExporter` – Export PyTorch backbone models to ONNX format
  with validation and numerical equivalence checks.
* :class:`TensorRTOptimizer` – Build optimised TensorRT engines from
  ONNX models with FP16/INT8 precision support.  Gracefully degrades
  when TensorRT is not installed.

Usage:
    >>> from src.deployment.exporter import ONNXExporter, TensorRTOptimizer
    >>> exporter = ONNXExporter(model, device="cuda")
    >>> exporter.export("models/backbone.onnx")
    >>> trt = TensorRTOptimizer()
    >>> trt.build_engine("models/backbone.onnx", "models/backbone.engine")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ONNX Exporter
# ---------------------------------------------------------------------------

class ONNXExporter:
    """Export a PyTorch model to ONNX format.

    Handles the export process with:
    * Dynamic batch size axis support
    * ONNX model validation via ``onnx.checker``
    * Numerical equivalence verification between PyTorch and ONNX outputs

    Args:
        model: PyTorch model to export (typically the backbone).
        device: Source device for the model.
        input_shape: Expected input shape ``(C, H, W)`` without batch dim.
        opset_version: ONNX opset version (default 13).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        input_shape: Tuple[int, ...] = (3, 224, 224),
        opset_version: int = 13,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.input_shape = input_shape
        self.opset_version = opset_version

    def export(
        self,
        output_path: str,
        dynamic_batch: bool = True,
        verify: bool = True,
    ) -> str:
        """Export the model to ONNX format.

        Args:
            output_path: Destination ``.onnx`` file path.
            dynamic_batch: If True, the batch dimension is dynamic.
            verify: If True, validate the exported model and check
                numerical equivalence.

        Returns:
            The resolved output path as a string.

        Raises:
            RuntimeError: If ONNX validation or equivalence check fails.
        """
        import onnx

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self.model.to(self.device)

        # Create dummy input
        dummy_input = torch.randn(
            1, *self.input_shape, device=self.device
        )

        # Dynamic axes
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        # Export
        print(f"Exporting model to ONNX (opset {self.opset_version})...")
        torch.onnx.export(
            self.model,
            dummy_input,
            str(out),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        print(f"  → Saved to {out}")

        if verify:
            self._validate_onnx(str(out))
            self._verify_equivalence(str(out), dummy_input)

        return str(out.resolve())

    def _validate_onnx(self, onnx_path: str) -> None:
        """Validate the ONNX model structure.

        Args:
            onnx_path: Path to the ``.onnx`` file.

        Raises:
            RuntimeError: If validation fails.
        """
        import onnx

        print("  Validating ONNX model...")
        model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(model)
            print("  ✓ ONNX model validation passed")
        except onnx.checker.ValidationError as e:
            raise RuntimeError(f"ONNX validation failed: {e}") from e

    def _verify_equivalence(
        self,
        onnx_path: str,
        dummy_input: torch.Tensor,
        atol: float = 1e-5,
    ) -> None:
        """Verify numerical equivalence between PyTorch and ONNX outputs.

        Args:
            onnx_path: Path to the ONNX model.
            dummy_input: Input tensor used for comparison.
            atol: Absolute tolerance for the comparison.

        Raises:
            RuntimeError: If outputs differ beyond tolerance.
        """
        import onnxruntime as ort

        print("  Verifying numerical equivalence...")

        # PyTorch output
        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(dummy_input).cpu().numpy()

        # ONNX Runtime output
        session = ort.InferenceSession(onnx_path)
        ort_input = {"input": dummy_input.cpu().numpy()}
        ort_output = session.run(None, ort_input)[0]

        # Compare
        max_diff = np.max(np.abs(pytorch_output - ort_output))
        if max_diff < atol:
            print(f"  ✓ Numerical equivalence verified (max diff: {max_diff:.2e})")
        else:
            raise RuntimeError(
                f"ONNX output differs from PyTorch (max diff: {max_diff:.2e}, "
                f"tolerance: {atol:.2e})"
            )

    def benchmark_onnx(
        self,
        onnx_path: str,
        num_runs: int = 100,
        batch_size: int = 1,
    ) -> dict:
        """Benchmark ONNX Runtime inference latency.

        Args:
            onnx_path: Path to the ONNX model.
            num_runs: Number of inference iterations.
            batch_size: Batch size for benchmarking.

        Returns:
            Dictionary with latency statistics.
        """
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        dummy = np.random.randn(
            batch_size, *self.input_shape
        ).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {"input": dummy})

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {"input": dummy})
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "backend": "ONNX Runtime",
            "batch_size": batch_size,
            "num_runs": num_runs,
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }


# ---------------------------------------------------------------------------
# TensorRT Optimizer
# ---------------------------------------------------------------------------

class TensorRTOptimizer:
    """Build optimised TensorRT inference engines from ONNX models.

    Supports FP32, FP16, and INT8 precision modes.  Gracefully skips
    operations when TensorRT is not installed on the system.

    Args:
        max_workspace_size: Maximum GPU workspace in bytes (default 1 GB).
    """

    _TENSORRT_AVAILABLE: bool = False

    def __init__(
        self,
        max_workspace_size: int = 1 << 30,  # 1 GB
    ) -> None:
        self.max_workspace_size = max_workspace_size
        self._check_tensorrt()

    def _check_tensorrt(self) -> None:
        """Check if TensorRT is available."""
        try:
            import tensorrt  # noqa: F401
            TensorRTOptimizer._TENSORRT_AVAILABLE = True
            print(f"TensorRT {tensorrt.__version__} detected")
        except ImportError:
            TensorRTOptimizer._TENSORRT_AVAILABLE = False
            print(
                "⚠ TensorRT not available. "
                "TensorRT engine building will be skipped. "
                "Install with: pip install tensorrt"
            )

    @property
    def is_available(self) -> bool:
        """Whether TensorRT is installed and usable."""
        return self._TENSORRT_AVAILABLE

    def build_engine(
        self,
        onnx_path: str,
        engine_path: str,
        precision: str = "fp16",
        max_batch_size: int = 1,
    ) -> Optional[str]:
        """Build a TensorRT engine from an ONNX model.

        Args:
            onnx_path: Path to the source ONNX model.
            engine_path: Destination path for the serialised engine.
            precision: Precision mode (``'fp32'``, ``'fp16'``, ``'int8'``).
            max_batch_size: Maximum batch size for optimisation.

        Returns:
            The engine path if successful, ``None`` if TensorRT is unavailable.
        """
        if not self.is_available:
            print("TensorRT not available – skipping engine build.")
            return None

        import tensorrt as trt

        print(f"Building TensorRT engine ({precision})...")
        print(f"  Source ONNX: {onnx_path}")
        print(f"  Target:      {engine_path}")

        out = Path(engine_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Build network from ONNX
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for idx in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(idx)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, self.max_workspace_size,
        )

        # Set precision
        precision = precision.lower()
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  → FP16 precision enabled")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  → INT8 precision enabled")
        else:
            print("  → FP32 precision (default)")

        # Build engine
        print("  Building engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed")

        # Save engine
        with open(str(out), "wb") as f:
            f.write(serialized_engine)

        print(f"  ✓ TensorRT engine saved to {out}")
        return str(out.resolve())

    def benchmark_tensorrt(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_runs: int = 100,
    ) -> Optional[dict]:
        """Benchmark TensorRT inference latency.

        Args:
            engine_path: Path to the serialised TensorRT engine.
            input_shape: Input tensor shape (including batch dim).
            num_runs: Number of inference iterations.

        Returns:
            Latency statistics dictionary, or ``None`` if unavailable.
        """
        if not self.is_available:
            print("TensorRT not available – skipping benchmark.")
            return None

        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        input_size = int(np.prod(input_shape)) * 4  # float32
        output_shape = (input_shape[0], 512)  # embedding dim
        output_size = int(np.prod(output_shape)) * 4

        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)

        h_input = np.random.randn(*input_shape).astype(np.float32)
        h_output = np.empty(output_shape, dtype=np.float32)

        stream = cuda.Stream()

        # Warmup
        for _ in range(10):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle,
            )
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            start = time.perf_counter()
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle,
            )
            stream.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)

        return {
            "backend": "TensorRT",
            "batch_size": input_shape[0],
            "num_runs": num_runs,
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
