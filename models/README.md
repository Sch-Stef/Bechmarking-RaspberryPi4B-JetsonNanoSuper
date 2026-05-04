# Models Directory

Pre-trained MobileNetV2 models in TensorFlow Lite format ready for benchmarking.

## Available Models

### mobilenetv2_fp32.tflite

MobileNetV2 model in full precision FP32 format.

- **Size:** 13.99 MB
- **Input shape:** [1, 224, 224, 3] RGB image
- **Output shape:** [1, 1000] ImageNet classes
- **Precision:** FP32
- **Measured latency, Raspberry Pi 4B:** 43.27 ms
- **Measured latency, Jetson Orin Nano Super:** 8.82 ms

### mobilenetv2_fp16.tflite

MobileNetV2 model with FP16 weights.

- **Size:** 7.03 MB
- **Input shape:** [1, 224, 224, 3] RGB image
- **Output shape:** [1, 1000] ImageNet classes
- **Precision:** FP16 weights, float input/output
- **Measured latency, Raspberry Pi 4B:** 43.19 ms
- **Measured latency, Jetson Orin Nano Super:** 8.90 ms

### mobilenetv2_int8.tflite

MobileNetV2 model in INT8 quantized format.

- **Size:** 3.99 MB
- **Input shape:** [1, 224, 224, 3] RGB image
- **Output shape:** [1, 1000] ImageNet classes
- **Precision:** INT8 quantized
- **Measured latency, Raspberry Pi 4B:** 38.51 ms
- **Measured latency, Jetson Orin Nano Super:** 32.50 ms

## Size and Latency Summary

| Precision | File size | Raspberry Pi 4B latency | Jetson Orin Nano Super latency |
|-----------|-----------|-------------------------|--------------------------------|
| FP32 | 13.99 MB | 43.27 ms | 8.82 ms |
| FP16 | 7.03 MB | 43.19 ms | 8.90 ms |
| INT8 | 3.99 MB | 38.51 ms | 32.50 ms |

FP16 reduces file size by about half compared with FP32 while keeping nearly identical CPU latency. INT8 has the smallest file size, but its latency benefit depends on the runtime and platform. In the current CPU-only TFLite results, INT8 is modestly faster on Raspberry Pi and substantially slower than FP32/FP16 on Jetson Orin Nano Super.

## Model Information

All models are based on MobileNetV2:

- Lightweight architecture using depthwise separable convolutions
- ImageNet classification output with 1000 classes
- 224x224 RGB input
- Suitable for edge AI benchmarking and deployment experiments

## Quantization Details

INT8 quantization uses post-training quantization with calibration data:

- Calibration: ImageNet subset images
- Quantized input/output tensors for the INT8 model
- Largest storage reduction among the tested formats
- Latency must be measured on the target runtime because INT8 kernel support varies by platform

## Usage

Load models in Python:

```python
from tflite_runtime.interpreter import Interpreter

interpreter_fp32 = Interpreter(model_path="mobilenetv2_fp32.tflite")
interpreter_fp32.allocate_tensors()

interpreter_fp16 = Interpreter(model_path="mobilenetv2_fp16.tflite")
interpreter_fp16.allocate_tensors()

interpreter_int8 = Interpreter(model_path="mobilenetv2_int8.tflite")
interpreter_int8.allocate_tensors()
```

For detailed usage examples, see [docs/code_overview.md](../docs/code_overview.md).

## Benchmark Results

Model performance benchmarks are available in [benchmark_results.md](../benchmark_results.md).
