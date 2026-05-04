# Benchmark Results Summary

## Executive Summary

This document summarizes CPU-only TensorFlow Lite inference results for MobileNetV2 on Raspberry Pi 4B and NVIDIA Jetson Orin Nano Super. The benchmark covers FP32, FP16, and INT8 TFLite models using 224x224 RGB input.

The current results show that the Jetson Orin Nano Super is much faster than the Raspberry Pi for FP32 and FP16 CPU inference. INT8 reduces model size substantially, but it does not automatically improve latency in this setup. On Raspberry Pi it is only modestly faster than FP32, and on Jetson Orin Nano Super the INT8 TFLite CPU path is much slower and less consistent than FP32/FP16.

## Test Configuration

- **Test date**: May 3, 2026
- **Model architecture**: MobileNetV2
- **Precisions tested**: FP32, FP16, INT8
- **Input size**: 224x224x3 RGB
- **Thread count**: 4
- **Runtime**: TensorFlow Lite CPU inference
- **Result files**: `results/{platform}/mobilenetv2_{precision}.json`

All current result files were produced with 500 measured runs and 50 warmup runs. The tables below report the values stored in the current JSON files.

## Results Summary

### Latency and Throughput

| Platform | Precision | Runs | Mean latency | Median latency | P95 latency | P99 latency | Throughput |
|----------|-----------|------|--------------|----------------|-------------|-------------|------------|
| Raspberry Pi 4B | FP32 | 500 | 43.27 ms | 43.23 ms | 43.73 ms | 44.42 ms | 23.11 img/s |
| Raspberry Pi 4B | FP16 | 500 | 43.19 ms | 43.25 ms | 43.68 ms | 43.83 ms | 23.15 img/s |
| Raspberry Pi 4B | INT8 | 500 | 38.51 ms | 38.45 ms | 38.85 ms | 39.13 ms | 25.97 img/s |
| Jetson Orin Nano Super | FP32 | 500 | 8.82 ms | 8.81 ms | 8.85 ms | 9.11 ms | 113.40 img/s |
| Jetson Orin Nano Super | FP16 | 500 | 8.90 ms | 8.89 ms | 8.93 ms | 9.19 ms | 112.41 img/s |
| Jetson Orin Nano Super | INT8 | 500 | 32.50 ms | 30.42 ms | 51.27 ms | 58.13 ms | 30.77 img/s |

### Model File Size

| Precision | File | Size |
|-----------|------|------|
| FP32 | `models/mobilenetv2_fp32.tflite` | 13.99 MB |
| FP16 | `models/mobilenetv2_fp16.tflite` | 7.03 MB |
| INT8 | `models/mobilenetv2_int8.tflite` | 3.99 MB |

Compared with FP32, FP16 reduces file size by about 49.7% and INT8 reduces file size by about 71.5%.

## Key Findings

### 1. Throughput

Jetson Orin Nano Super has a clear throughput advantage for floating-point TFLite CPU inference:

- FP32: 113.40 img/s on Jetson vs 23.11 img/s on Raspberry Pi, about 4.9x higher throughput.
- FP16: 112.41 img/s on Jetson vs 23.15 img/s on Raspberry Pi, about 4.9x higher throughput.
- INT8: 30.77 img/s on Jetson vs 25.97 img/s on Raspberry Pi, about 1.2x higher throughput.

The INT8 throughput result is the exception. The Jetson remains faster than Raspberry Pi for INT8, but the gap is small because INT8 performs poorly on the Jetson TFLite CPU path compared with FP32/FP16.

### 2. Latency

For FP32 and FP16, Jetson Orin Nano Super is substantially lower latency:

- FP32: 8.82 ms on Jetson vs 43.27 ms on Raspberry Pi.
- FP16: 8.90 ms on Jetson vs 43.19 ms on Raspberry Pi.

For INT8, Jetson is still lower on mean latency, but the difference is much smaller:

- INT8: 32.50 ms on Jetson vs 38.51 ms on Raspberry Pi.

Raspberry Pi INT8 is about 11.0% lower latency than Raspberry Pi FP32. Jetson INT8 is about 3.7x slower than Jetson FP32 in these CPU-only TFLite measurements.

### 3. File Size

Model size scales as expected:

- FP32 is the largest model at 13.99 MB.
- FP16 cuts the model to 7.03 MB with nearly identical latency to FP32 on both devices.
- INT8 is the smallest model at 3.99 MB, but size reduction did not translate into a consistent latency win.

### 4. Consistency and Tail Latency

FP32 and FP16 are very stable on both platforms. Jetson INT8 is the least stable result:

- Jetson INT8 median latency is 30.42 ms, but P95 rises to 51.27 ms and P99 to 58.13 ms.
- Raspberry Pi INT8 is more consistent, with median 38.45 ms and P99 39.13 ms.

For real-time systems, Jetson FP32/FP16 are the strongest CPU-only options here. Jetson INT8 needs further investigation before it should be treated as a reliable real-time configuration.

## Detailed Analysis

### Within-Platform Precision Impact

| Platform | Comparison | Latency change | Throughput change | Interpretation |
|----------|------------|----------------|-------------------|----------------|
| Raspberry Pi 4B | FP16 vs FP32 | 0.2% faster | 0.2% higher | FP16 reduces file size while keeping effectively the same CPU latency. |
| Raspberry Pi 4B | INT8 vs FP32 | 11.0% faster | 12.4% higher | INT8 gives a modest CPU latency improvement and much smaller file size. |
| Jetson Orin Nano Super | FP16 vs FP32 | 0.9% slower | 0.9% lower | FP16 is effectively equal to FP32 for CPU inference while using less storage. |
| Jetson Orin Nano Super | INT8 vs FP32 | 268.5% slower | 72.9% lower | INT8 is not optimized in this CPU-only TFLite path. |

### Real-Time Target: 30 FPS

30 FPS requires latency below 33.3 ms.

| Platform | FP32 | FP16 | INT8 |
|----------|------|------|------|
| Raspberry Pi 4B | No, 43.27 ms | No, 43.19 ms | No, 38.51 ms |
| Jetson Orin Nano Super | Yes, 8.82 ms | Yes, 8.90 ms | Mean yes, but P95/P99 exceed target |

Using mean latency alone, Jetson INT8 meets 30 FPS. Using tail latency, it does not provide a stable 30 FPS profile because P95 and P99 are well above 33.3 ms.

## Practical Implications

- For Raspberry Pi 4B, INT8 is useful primarily for model size and a modest latency improvement, but it still does not reach 30 FPS in this test.
- For Jetson Orin Nano Super, FP32 and FP16 are the best CPU-only latency and throughput configurations.
- FP16 is attractive when storage size matters because it is about half the size of FP32 with nearly identical CPU latency.
- INT8 should not be assumed faster without measuring the actual runtime and hardware path. In this benchmark, Jetson INT8 is slower and has worse tail latency.
- GPU/TensorRT acceleration on Jetson is outside this benchmark and may change the best deployment choice.

## Reproducibility

Full benchmark details are available in:

- Methodology: [docs/benchmark_methodology.md](docs/benchmark_methodology.md)
- Code explanation: [docs/code_overview.md](docs/code_overview.md)
- Result interpretation: [docs/results_analysis.md](docs/results_analysis.md)

Raw results are stored in `results/{platform}/mobilenetv2_{precision}.json`.

## Conclusion

For CPU-only TensorFlow Lite inference, Jetson Orin Nano Super is clearly stronger for FP32 and FP16 MobileNetV2 inference. Raspberry Pi 4B benefits modestly from INT8 and gets the largest storage reduction from quantization, but not enough latency reduction to reach 30 FPS in the current measurements.

The most defensible deployment takeaway from these results is: use Jetson FP32/FP16 for CPU-only real-time latency, use FP16 when model size matters without sacrificing latency, and treat INT8 as a measured configuration rather than an automatic speed optimization.
