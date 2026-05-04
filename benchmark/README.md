# Benchmark Directory

Contains benchmarking scripts and related code.

## pure_inference/

Measures raw neural network inference performance without additional overhead.

**run.py** - Main benchmarking executable
- Loads TFLite models
- Executes inference with configurable parameters
- Collects latency and throughput metrics
- Exports results to JSON format

**Usage Example:**
```bash
python run.py \
  --model ../models/mobilenetv2_int8.tflite \
  --runs 100 \
  --warmup-runs 10 \
  --num-threads 4
```

**Output:** JSON file containing latency statistics, throughput, and model configuration.

## Implementation Details

For detailed code explanation, see [docs/code_overview.md](../docs/code_overview.md)

## Results

Benchmark results are saved to `results/{platform}/` directory in JSON format. See [benchmark_results.md](../benchmark_results.md) for analysis and interpretation.
