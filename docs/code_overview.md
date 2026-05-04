# Code Overview

## Project Structure

### benchmark/pure_inference/run.py
Main benchmarking script that measures neural network inference performance.

#### Key Functions

**load_tflite_interpreter(model_path, num_threads)**
- Loads TFLite models with configurable thread count
- Supports both tflite_runtime and tensorflow.lite backends
- Automatically allocates tensors for inference

**create_dummy_input(input_details)**
- Generates random input data matching model specifications
- Handles FP32, INT8, and UINT8 data types
- Ensures shape compatibility with model expectations

**benchmark(model_path, runs, warmup_runs, num_threads)**
- Main benchmarking function
- Executes warmup runs to stabilize performance
- Measures latency, throughput, and statistical metrics

#### Workflow

1. Load model and create interpreter
2. Perform warmup runs (excluded from results)
3. Execute benchmark runs and record latencies
4. Calculate statistics (mean, median, min, max, percentiles)
5. Export results to JSON format

#### Metrics Collected

- **Latency** (ms): Time per inference
- **Throughput** (images/sec): Inferences per second
- **Percentiles** (p50, p90, p95, p99): Distribution analysis
- **Min/Max**: Performance bounds

### scripts/prepare_models.py
Prepares TensorFlow models for benchmarking.

#### Functionality

- Model conversion to TFLite format
- Optional precision conversion/quantization (FP32, FP16, or INT8)
- Model validation and shape verification

### setup/
Configuration and environment setup files.

- **requirements_pc.txt**: Dependencies for PC/development environment
- **requirements_target.txt**: Dependencies for target devices (Raspberry Pi 4B, Jetson Orin Nano Super)
- **setup_venv.sh**: Virtual environment setup script
- **HowTo.md**: Device-specific setup instructions

## Data Flow

```
Raw Model → prepare_models.py → TFLite Models
                                    ↓
                            run.py (Benchmark)
                                    ↓
                            results/*.json
```

## Configuration

Benchmarks are configured via command-line arguments:

- `--model`: Path to TFLite model file
- `--runs`: Number of benchmark iterations (default: 100)
- `--warmup-runs`: Warmup iterations (default: 10)
- `--num-threads`: Thread count for inference (platform-specific)

## Performance Considerations

### Warmup Runs
Initial runs are excluded because:
- JIT compilation may occur
- CPU frequency scaling adapts
- Cache states are not representative of steady-state

### Thread Configuration
- Raspberry Pi 4B: 4 cores (num_threads=4)
- Jetson Orin Nano Super: 4 benchmark threads (num_threads=4)
- Optimal values depend on model and system load

### Quantization Impact
- FP16 reduces model size by about half compared with FP32
- INT8 provides the smallest model file in the current model set
- Lower precision does not guarantee lower CPU latency; the current results show INT8 is only modestly faster on Raspberry Pi and slower than FP32/FP16 on Jetson Orin Nano Super
- Accuracy impact should be validated separately with representative image data
