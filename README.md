# Edge AI Benchmark: Raspberry Pi 4B vs Nvidia Jetson Orin Nano Super

Benchmarking neural network inference performance on embedded hardware using a reproducible and comparable methodology.

**Scope**: This benchmark measures CPU inference only. This ensures fair comparison between platforms and emphasizes reproducible methodology. Jetson Orin Nano Super GPU capabilities are noted as a future enhancement for optimization studies.

## Motivation

This project was developed as a learning initiative to deepen understanding of:
- Neural network inference optimization on embedded devices
- Quantization benefits and trade-offs in edge AI
- Fair benchmarking methodology and reproducibility standards
- Hardware-software interaction in deployment scenarios

## Quick Start

1. Review [benchmark_results.md](benchmark_results.md) for results summary
2. See [docs/benchmark_methodology.md](docs/benchmark_methodology.md) for methodology
3. Follow [setup/HowTo.md](setup/HowTo.md) to run benchmarks

## Overview

This project benchmarks neural network inference performance comparing:

- **Hardware**: Raspberry Pi 4 Model B vs Nvidia Jetson Orin Nano Super
- **Models**: MobileNetV2 in FP32, FP16, and INT8 formats
- **Focus**: Fair, reproducible performance comparison
- **Input**: 224x224 RGB images (ImageNet standard)

## Key Results

| Platform | Model | Latency | Throughput |
|----------|-------|---------|-----------|
| Raspberry Pi 4B | FP32 | 43.27 ms | 23.11 img/s |
| Raspberry Pi 4B | FP16 | 43.19 ms | 23.15 img/s |
| Raspberry Pi 4B | INT8 | 38.51 ms | 25.97 img/s |
| Jetson Orin Nano Super | FP32 | 8.82 ms | 113.40 img/s |
| Jetson Orin Nano Super | FP16 | 8.90 ms | 112.41 img/s |
| Jetson Orin Nano Super | INT8 | 32.50 ms | 30.77 img/s |

**Key Finding**: Jetson Orin Nano Super is about 4.9x faster than Raspberry Pi 4B for FP32/FP16 CPU inference. INT8 gives the smallest model file, but it is not automatically the fastest runtime path in these TensorFlow Lite CPU-only results.

See [benchmark_results.md](benchmark_results.md) for detailed analysis.

## Project Goals

- Fair hardware comparison using identical methodology
- Quantify precision and quantization trade-offs (FP32, FP16, INT8)
- Measure end-to-end performance overhead
- Provide reproducible baseline for edge AI deployment

## Documentation

### Getting Started
- [setup/README.md](setup/README.md) - Environment setup and configuration
- [setup/HowTo.md](setup/HowTo.md) - Detailed platform-specific instructions

### Understanding the Project
- [benchmark/README.md](benchmark/README.md) - Benchmarking code overview
- [models/README.md](models/README.md) - Model descriptions and specifications
- [results/README.md](results/README.md) - Results interpretation guide

### Deep Dives
- [docs/benchmark_methodology.md](docs/benchmark_methodology.md) - Detailed methodology and fairness considerations
- [docs/code_overview.md](docs/code_overview.md) - Code walkthrough and implementation details
- [docs/results_analysis.md](docs/results_analysis.md) - In-depth results interpretation
- [benchmark_results.md](benchmark_results.md) - Complete results summary with analysis

## Repository Structure

```
├── benchmark/                   Benchmarking code
│   ├── pure_inference/
│   │   └── run.py              Main benchmark script
│   └── README.md
├── calibration_data/            Model quantization calibration datasets
│   ├── imagenet_subset/         ImageNet subset for quantization
│   ├── raspi_camera_subset/     Real camera images
│   └── README.md
├── docs/                        Detailed documentation
│   ├── code_overview.md         Code walkthrough
│   ├── benchmark_methodology.md Methodology and fairness
│   └── results_analysis.md      Results interpretation
├── models/                      Pre-trained TFLite models
│   ├── mobilenetv2_fp32.tflite  FP32 model (13.99 MB)
│   ├── mobilenetv2_fp16.tflite  FP16 model (7.03 MB)
│   ├── mobilenetv2_int8.tflite  INT8 quantized model (3.99 MB)
│   └── README.md
├── results/                     Benchmark output results
│   ├── raspberry_pi/
│   │   ├── mobilenetv2_fp32.json
│   │   ├── mobilenetv2_fp16.json
│   │   ├── mobilenetv2_int8.json
│   │   └── README.md
│   ├── jetson_orin_nano/
│   │   ├── mobilenetv2_fp32.json
│   │   ├── mobilenetv2_fp16.json
│   │   ├── mobilenetv2_int8.json
│   │   └── README.md
│   └── README.md
├── scripts/                     Utility scripts
│   ├── prepare_models.py        Model conversion and quantization
│   └── README.md
├── setup/                       Environment setup and configuration
│   ├── HowTo.md                 Platform-specific setup instructions
│   ├── requirements_pc.txt      PC/workstation dependencies
│   ├── requirements_target.txt  Target device dependencies
│   ├── setup_venv.sh            Virtual environment setup script
│   └── README.md
├── LICENSE
├── README.md                    This file
├── benchmark_results.md         Results summary and analysis
└── quick_start.md               Quick start guide
```

## Hardware Specifications

### Raspberry Pi 4 Model B
- **SoC:** Broadcom BCM2711  
- **CPU:** 4× ARM Cortex-A72 @ 1.5 GHz  
- **RAM:** 4 GB LPDDR4  
- **Storage:** MicroSD Card (Lexar 32 GB - UHS-1, U3, A1, V30, C10 )  
- **Operating System:** Debian GNU/Linux 12 (Bookworm)  
- **Inference Runtime:** TensorFlow Lite (CPU-only)

### NVIDIA Jetson Orin Nano Super Developer Kit
- **SoC:** NVIDIA Tegra Orin  
- **CPU:** 6× Arm Cortex-A78AE @ up to 1.7 GHz  
- **GPU:** 1024-core NVIDIA Ampere GPU + 32 Tensor Cores  
- **RAM:** 8 GB LPDDR5  
- **Storage:** MicroSD Card (Lexar 32 GB - UHS-1, U3, A1, V30, C10 )  
- **Operating System:** Ubuntu 22.04-based Linux (NVIDIA Linux for Tegra / L4T)  
- **JetPack Version:** 6.2.1  
- **Inference Runtime:** TensorFlow Lite (CPU-only)

## Benchmark Note

Both platforms use **CPU-only inference** via TensorFlow Lite to ensure fair and comparable benchmark results independent of hardware-specific GPU acceleration (e.g. TensorRT on NVIDIA platforms).
## Benchmark Configuration

- **Model Architecture**: MobileNetV2
- **Input Size**: 224x224x3 RGB images
- **Precision formats**: FP32, FP16, and INT8
- **Runs**: 500 inference iterations
- **Warmup**: 50 runs, excluded from results
- **Threading**: 4 CPU cores
- **Framework**: TensorFlow Lite Runtime

## Running Benchmarks

### Pure Inference Benchmark

The simplest benchmark measuring raw inference performance:

```bash
cd benchmark/pure_inference
python run.py --model ../../models/mobilenetv2_fp32.tflite --runs 100
```

Output: JSON file with latency statistics and throughput metrics.

For detailed instructions, see [setup/HowTo.md](setup/HowTo.md)

### Models Tested

- MobileNetV2 FP32: Full precision (13.99 MB)
- MobileNetV2 FP16: Half precision weights (7.03 MB)
- MobileNetV2 INT8: Quantized 8-bit (3.99 MB)

For model details, see [models/README.md](models/README.md)

## Analysis and Interpretation

### Understanding Results

Raw results saved in `results/{platform}/` as JSON files.

Key metrics:
- **Latency (ms)**: Time per inference (lower is better)
- **Throughput (img/s)**: Maximum sustained rate
- **Std Dev**: Consistency metric (lower is better)
- **Percentiles**: Performance distribution (P95, P99)

See [benchmark_results.md](benchmark_results.md) for example analysis.

### Detailed Interpretation

For in-depth analysis guidance:
- [docs/results_analysis.md](docs/results_analysis.md) - How to interpret results
- [docs/benchmark_methodology.md](docs/benchmark_methodology.md) - Methodology details

## Code Structure

### Key Components

- **benchmark/pure_inference/run.py**: Main benchmarking script
- **scripts/prepare_models.py**: Model conversion and quantization
- **setup/**: Environment configuration
- **results/**: Benchmark output data

For detailed code walkthrough, see [docs/code_overview.md](docs/code_overview.md)

## Reproducibility

### To Reproduce Benchmarks

1. Set up environment (see [setup/HowTo.md](setup/HowTo.md))
2. Verify models in `models/` directory
3. Run benchmark script with same configuration
4. Compare results in JSON format

### System Information

Before benchmarking, document:
```bash
# Raspberry Pi
uname -a
cat /proc/cpuinfo
vcgencmd measure_temp

# Jetson Orin Nano Super
uname -a
tegrastats
```

For full reproducibility details, see [docs/benchmark_methodology.md](docs/benchmark_methodology.md)

## Future Enhancements

Potential extensions:

- GPU acceleration (Jetson TensorRT)
- XNNPACK optimization (Raspberry Pi)
- Real camera pipeline benchmarking
- Power consumption analysis
- Thermal throttling analysis
- Additional model architectures (EfficientNet, ResNet, YOLO)

## License

See [LICENSE](LICENSE) file
