# Benchmark Methodology

## Objective

Provide a fair and reproducible comparison of neural network inference performance between Raspberry Pi 4B and NVIDIA Jetson Orin Nano Super using standardized benchmarking practices.

## Test Scenarios

### 1. Pure Inference Benchmark
- Measures raw inference latency
- Uses synthetic random input data
- Isolates model execution performance
- Excludes I/O overhead

### 2. Real-World Pipeline (Optional)
- Includes camera capture
- End-to-end latency measurement
- Reflects actual deployment scenarios

## Benchmark Configuration

### Models Tested
- **MobileNetV2 FP32**: Full precision floating-point
- **MobileNetV2 FP16**: Half precision floating-point weights
- **MobileNetV2 INT8**: Quantized 8-bit integer

### Hardware Specifications

#### Raspberry Pi 4 Model B
- **SoC:** Broadcom BCM2711  
- **CPU:** 4× ARM Cortex-A72 @ 1.5 GHz  
- **RAM:** 4 GB LPDDR4  
- **Storage:** MicroSD Card (Lexar 32 GB - UHS-1, U3, A1, V30, C10 )  
- **Operating System:** Debian GNU/Linux 12 (Bookworm)  
- **Inference Runtime:** TensorFlow Lite (CPU-only)

#### NVIDIA Jetson Orin Nano Super Developer Kit
- **SoC:** NVIDIA Tegra Orin  
- **CPU:** 6× Arm Cortex-A78AE @ up to 1.7 GHz  
- **GPU:** 1024-core NVIDIA Ampere GPU + 32 Tensor Cores  
- **RAM:** 8 GB LPDDR5  
- **Storage:** MicroSD Card (Lexar 32 GB - UHS-1, U3, A1, V30, C10 )  
- **Operating System:** Ubuntu 22.04-based Linux (NVIDIA Linux for Tegra / L4T)  
- **JetPack Version:** 6.2.1  
- **Inference Runtime:** TensorFlow Lite (CPU-only)

### Benchmark Parameters
- **Number of runs**: 500 inference iterations
- **Warmup runs**: 50, excluded from results
- **Input size**: 224x224x3 (ImageNet standard)
- **Thread count**: 4 (CPU cores utilized)

## Statistical Metrics

For each benchmark run, we collect:

| Metric | Unit | Description |
|--------|------|-------------|
| Mean | ms | Average inference latency |
| Median | ms | 50th percentile latency |
| Min | ms | Minimum latency |
| Max | ms | Maximum latency |
| P90 | ms | 90th percentile latency |
| P95 | ms | 95th percentile latency |
| P99 | ms | 99th percentile latency |
| Throughput | img/s | Images per second |

## Fairness Considerations

### System Configuration
1. Freshly booted system before benchmarking
2. Minimal background processes running
3. Device thermal state: cooled to room temperature
4. USB peripherals disconnected (except essentials)
5. No concurrent network activity

### Runtime Configuration
1. TensorFlow Lite runtime (optimized for mobile/embedded)
2. Consistent quantization scheme across platforms
3. Identical model architectures and weights
4. Same precision throughout benchmark (no mixed precision)

### Exclusions
1. Model loading time (only inference measured)
2. Memory allocation (pre-allocated before benchmark)
3. First inference cycle (warmup iterations)
4. I/O operations (pure computation focus)

## Result Interpretation

### Latency Analysis
- Lower latency indicates faster inference
- Median is more robust than mean for outlier detection
- Percentiles show performance consistency

### Throughput Calculation
- Throughput (img/s) = 1000 / mean_latency_ms
- Reflects maximum sustainable inference rate

### Quantization Benefits
- Compare FP32, FP16, and INT8 on the same platform
- Percentage improvement = (FP32_latency - INT8_latency) / FP32_latency * 100
- Do not assume INT8 is faster. The current CPU-only TFLite results show INT8 is modestly faster on Raspberry Pi but slower than FP32/FP16 on Jetson Orin Nano Super.

### Hardware Comparison
- Same model on different hardware
- Accounts for architectural differences
- Reveals bandwidth and compute limitations

## Reproducibility

### Requirements
1. Identical hardware specifications
2. Same model files and weights
3. Identical TensorFlow Lite runtime version
4. Consistent software environment

### Steps to Reproduce
1. Follow setup instructions in setup/HowTo.md
2. Install dependencies from requirements_target.txt
3. Run benchmark/pure_inference/run.py with specified parameters
4. Results saved to results/{device_name}/*.json

## Limitations

1. Synthetic input data (not real image distributions)
2. Thermal data is available only in newer result files
3. Single-threaded inference per sample
4. Homogeneous workload (no mixed precision)
5. CPU-only execution (no GPU acceleration on Raspberry Pi)

## Future Enhancements

- Multi-threaded batch processing benchmarks
- Thermal throttling analysis
- Memory bandwidth analysis
- Power consumption measurements
- Real image data testing
