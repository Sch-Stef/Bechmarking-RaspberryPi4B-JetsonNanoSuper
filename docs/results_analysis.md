# Results Analysis

## Overview

This document explains how to interpret the benchmark results and summarizes the current MobileNetV2 FP32, FP16, and INT8 measurements for Raspberry Pi 4B and NVIDIA Jetson Orin Nano Super.

All current results are CPU-only TensorFlow Lite inference results. They do not use Jetson GPU acceleration, TensorRT, or batch processing.

## Current Results

| Platform | Precision | Runs | Mean latency | Median latency | P90 latency | P95 latency | P99 latency | Throughput |
|----------|-----------|------|--------------|----------------|-------------|-------------|-------------|------------|
| Raspberry Pi 4B | FP32 | 500 | 43.27 ms | 43.23 ms | 43.62 ms | 43.73 ms | 44.42 ms | 23.11 img/s |
| Raspberry Pi 4B | FP16 | 500 | 43.19 ms | 43.25 ms | 43.56 ms | 43.68 ms | 43.83 ms | 23.15 img/s |
| Raspberry Pi 4B | INT8 | 500 | 38.51 ms | 38.45 ms | 38.74 ms | 38.85 ms | 39.13 ms | 25.97 img/s |
| Jetson Orin Nano Super | FP32 | 500 | 8.82 ms | 8.81 ms | 8.83 ms | 8.85 ms | 9.11 ms | 113.40 img/s |
| Jetson Orin Nano Super | FP16 | 500 | 8.90 ms | 8.89 ms | 8.91 ms | 8.93 ms | 9.19 ms | 112.41 img/s |
| Jetson Orin Nano Super | INT8 | 500 | 32.50 ms | 30.42 ms | 46.44 ms | 51.27 ms | 58.13 ms | 30.77 img/s |

## Model Size

| Precision | File size | Change vs FP32 |
|-----------|-----------|----------------|
| FP32 | 13.99 MB | Baseline |
| FP16 | 7.03 MB | 49.7% smaller |
| INT8 | 3.99 MB | 71.5% smaller |

File size improves as precision is reduced, but latency does not necessarily improve. Runtime kernel support and hardware execution paths dominate latency.

## Key Metrics

### Latency

Latency is the time required for a single inference.

- **Mean**: Average latency across all measured runs.
- **Median**: 50th percentile latency. This is useful when outliers are present.
- **P95/P99**: Tail latency. These matter for real-time systems because occasional slow frames can still break timing requirements.

### Throughput

Throughput is calculated as:

```text
throughput_fps = 1000 / mean_latency_ms
```

It represents the maximum theoretical single-stream inference rate under the measured conditions.

### Thermal Data

Newer result files include a `thermal.summary` block. Use it to check whether temperature drift may have influenced long benchmark runs.

## Throughput Insights

Jetson Orin Nano Super has the strongest throughput for FP32 and FP16:

| Precision | Raspberry Pi 4B | Jetson Orin Nano Super | Jetson throughput advantage |
|-----------|-----------------|------------------------|-----------------------------|
| FP32 | 23.11 img/s | 113.40 img/s | 4.91x |
| FP16 | 23.15 img/s | 112.41 img/s | 4.86x |
| INT8 | 25.97 img/s | 30.77 img/s | 1.18x |

The INT8 result is not aligned with the common expectation that INT8 should be fastest. On Jetson, INT8 throughput is much lower than FP32/FP16 in the CPU-only TFLite setup. This likely reflects a less favorable CPU kernel/runtime path for this model and precision combination.

## Latency Insights

### Raspberry Pi 4B

| Comparison | Result |
|------------|--------|
| FP16 vs FP32 | FP16 is 0.2% faster. |
| INT8 vs FP32 | INT8 is 11.0% faster. |
| INT8 vs FP16 | INT8 is 10.8% faster. |

Raspberry Pi benefits modestly from INT8, but the improvement is much smaller than a typical 2x-4x quantization expectation. FP16 reduces file size while keeping CPU latency effectively unchanged.

### Jetson Orin Nano Super

| Comparison | Result |
|------------|--------|
| FP16 vs FP32 | FP16 is 0.9% slower. |
| INT8 vs FP32 | INT8 is 268.5% slower. |
| INT8 vs FP16 | INT8 is 265.3% slower. |

For CPU-only TFLite inference, Jetson FP32 and FP16 are the best latency configurations. INT8 is slower and has substantially worse tail latency.

## Real-Time Interpretation

A 30 FPS target requires latency below 33.3 ms.

| Platform | FP32 | FP16 | INT8 |
|----------|------|------|------|
| Raspberry Pi 4B | No | No | No |
| Jetson Orin Nano Super | Yes | Yes | Mean yes, tail no |

Jetson INT8 has a mean latency below 33.3 ms, but its P95 and P99 latencies exceed the target. For stable real-time behavior, FP32 or FP16 are better choices on Jetson in this benchmark.

## Comparative Analysis

### Precision Trade-Offs

- **FP32**: Largest model, strong baseline latency, best Jetson CPU result.
- **FP16**: About half the FP32 file size, nearly identical CPU latency to FP32 on both devices.
- **INT8**: Smallest model, modest Raspberry Pi latency improvement, poor Jetson CPU latency and tail behavior.

### Platform Trade-Offs

- Raspberry Pi 4B is slower overall but shows consistent latency across all three precisions.
- Jetson Orin Nano Super is much faster for FP32 and FP16 and still slightly faster for INT8 mean latency.
- Jetson INT8 has high tail latency, so mean throughput alone is misleading.

## Common Issues and Interpretations

| Observation | Likely cause | Action |
|-------------|--------------|--------|
| INT8 slower than FP32/FP16 | Runtime/kernel path is not optimized for this model or hardware | Benchmark the actual deployment runtime; consider XNNPACK, TensorRT, or another backend |
| P99 much higher than median | Outliers, scheduling, thermal effects, or runtime variability | Increase sample count and inspect thermal/system load |
| FP16 similar to FP32 | CPU path may dequantize or execute similarly to FP32 | Treat FP16 mainly as a file-size optimization unless measured otherwise |
| Mean meets FPS target but P95/P99 do not | Unstable real-time behavior | Use percentile latency for deployment decisions |

## Reporting Results

When presenting results:

1. Include mean, median, P95, and P99.
2. Include throughput and model file size.
3. State that the benchmark is CPU-only TensorFlow Lite inference.
4. Document run count, warmup count, thread count, and input size.
5. Avoid claiming quantization speedups unless the measured results show them.

## Next Steps

To extend the analysis:

1. Re-run all six tests with the same run count and warmup count for strict comparability.
2. Test TensorFlow Lite XNNPACK settings explicitly.
3. Test Jetson TensorRT/GPU execution separately from this CPU-only benchmark.
4. Measure power consumption for performance-per-watt comparison.
5. Use real image data to validate accuracy and input pipeline behavior.
