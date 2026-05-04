# Results Directory

Benchmark results are organized by target hardware platform. The current dataset contains MobileNetV2 FP32, FP16, and INT8 TensorFlow Lite CPU-only inference results for Raspberry Pi 4B and NVIDIA Jetson Orin Nano Super.

## Directory Structure

```text
results/
+-- raspberry_pi/
|   +-- mobilenetv2_fp32.json
|   +-- mobilenetv2_fp16.json
|   +-- mobilenetv2_int8.json
+-- jetson_orin_nano/
    +-- mobilenetv2_fp32.json
    +-- mobilenetv2_fp16.json
    +-- mobilenetv2_int8.json
```

## Current Result Summary

| Platform | Precision | Runs | Mean latency | Median latency | P95 latency | Throughput |
|----------|-----------|------|--------------|----------------|-------------|------------|
| Raspberry Pi 4B | FP32 | 500 | 43.27 ms | 43.23 ms | 43.73 ms | 23.11 img/s |
| Raspberry Pi 4B | FP16 | 500 | 43.19 ms | 43.25 ms | 43.68 ms | 23.15 img/s |
| Raspberry Pi 4B | INT8 | 500 | 38.51 ms | 38.45 ms | 38.85 ms | 25.97 img/s |
| Jetson Orin Nano Super | FP32 | 500 | 8.82 ms | 8.81 ms | 8.85 ms | 113.40 img/s |
| Jetson Orin Nano Super | FP16 | 500 | 8.90 ms | 8.89 ms | 8.93 ms | 112.41 img/s |
| Jetson Orin Nano Super | INT8 | 500 | 32.50 ms | 30.42 ms | 51.27 ms | 30.77 img/s |

## Model File Sizes

| Precision | File | Size |
|-----------|------|------|
| FP32 | `models/mobilenetv2_fp32.tflite` | 13.99 MB |
| FP16 | `models/mobilenetv2_fp16.tflite` | 7.03 MB |
| INT8 | `models/mobilenetv2_int8.tflite` | 3.99 MB |

## Result File Format

Each JSON file contains benchmark configuration, tensor metadata, latency statistics, throughput, system metadata, and, for newer runs, thermal measurements.

```json
{
  "model": "models/mobilenetv2_fp32.tflite",
  "runs": 500,
  "warmup_runs": 50,
  "num_threads": 4,
  "input_shape": [1, 224, 224, 3],
  "input_dtype": "<class 'numpy.float32'>",
  "output_shape": [1, 1000],
  "output_dtype": "float32",
  "latency_ms": {
    "mean": 8.82,
    "median": 8.81,
    "min": 8.78,
    "max": 9.15,
    "p90": 8.83,
    "p95": 8.85,
    "p99": 9.11
  },
  "throughput_fps": 113.40,
  "thermal": {
    "summary": {
      "available": true,
      "max_temperature_c": 47.47
    }
  }
}
```

## Interpreting Results

- **Mean latency**: Average inference latency and the basis for throughput.
- **Median latency**: More robust indicator when outliers are present.
- **P95/P99 latency**: Tail latency. Important for real-time behavior and service-level targets.
- **Throughput**: Maximum sustained inference rate, calculated as `1000 / mean_latency_ms`.
- **Thermal summary**: Available in newer result files and useful for identifying heat-related drift.

## Current Interpretation

- Jetson Orin Nano Super is much faster than Raspberry Pi 4B for FP32 and FP16 CPU inference.
- FP16 reduces file size by about half compared with FP32, but it does not improve CPU latency in these results.
- INT8 has the smallest file size. On Raspberry Pi it gives a modest latency improvement; on Jetson it is slower than FP32/FP16 and has much higher tail latency.
- Do not assume quantization improves latency. Validate the precision/runtime combination on the target hardware.

Detailed interpretation is available in [docs/results_analysis.md](../docs/results_analysis.md). The high-level summary is in [benchmark_results.md](../benchmark_results.md).

## Generating Results

To generate a benchmark result:

```bash
cd benchmark/pure_inference
python run.py \
  --model ../../models/mobilenetv2_fp32.tflite \
  --output ../../results/raspberry_pi/mobilenetv2_fp32.json
```

See [benchmark/README.md](../benchmark/README.md) for detailed usage instructions.

## Result Comparison

To compare results across platforms and models:

1. Open the relevant JSON files.
2. Compare `latency_ms.mean`, `latency_ms.median`, and `latency_ms.p95`.
3. Compare `throughput_fps`.
4. Check model size in `models/` when storage or transfer cost matters.

Example:

```python
import json

with open("results/raspberry_pi/mobilenetv2_fp32.json") as f:
    rpi = json.load(f)

with open("results/jetson_orin_nano/mobilenetv2_fp32.json") as f:
    jetson = json.load(f)

rpi_latency = rpi["latency_ms"]["mean"]
jetson_latency = jetson["latency_ms"]["mean"]
speedup = rpi_latency / jetson_latency

print(f"Jetson FP32 mean latency is {speedup:.1f}x faster than Raspberry Pi FP32")
```

## Archiving Results

For version control and reproducibility:

1. Tag results with date and configuration.
2. Document hardware state, including temperature and background processes.
3. Include software versions, OS version, and runtime version.
4. Save results alongside the benchmark code version.
