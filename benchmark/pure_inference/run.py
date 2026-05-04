#!/usr/bin/env python3

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_tflite_interpreter(model_path: str, num_threads: int):
    """Loads a TFLite model and returns an interpreter instance."""
    # Try to import optimized tflite_runtime (faster on embedded devices)
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # Fallback to TensorFlow Lite if runtime not available
        from tensorflow.lite.python.interpreter import Interpreter

    # Create interpreter with specified model and thread configuration
    interpreter = Interpreter(
        model_path=model_path,
        num_threads=num_threads,
    )

    # Pre-allocate tensor buffers for inference
    interpreter.allocate_tensors()
    return interpreter


def create_dummy_input(input_details):
    """Creates a dummy input tensor based on the model's input details."""
    # Extract input specifications from model metadata
    detail = input_details[0]
    shape = detail["shape"]
    dtype = detail["dtype"]

    # Generate random input matching the expected data type
    if dtype == np.float32:
        # FP32: random values in [0, 1)
        return np.random.random_sample(shape).astype(np.float32)

    if dtype == np.uint8:
        # Unsigned 8-bit: random values in [0, 255]
        return np.random.randint(0, 255, size=shape, dtype=np.uint8)

    if dtype == np.int8:
        # Signed 8-bit: random values in [-128, 127]
        return np.random.randint(-128, 127, size=shape, dtype=np.int8)

    # Raise error if data type is not supported
    raise ValueError(f"Unsupported input dtype: {dtype}")


def percentile(values: List[float], p: float) -> float:
    """Computes the p-th percentile of a list of values."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
		
    # Sort values in ascending order
    values_sorted = sorted(values)

    # Calculate index for p-th percentile using linear interpolation
    index = int((p / 100.0) * (len(values_sorted) - 1))

    # Return value at percentile index
    return values_sorted[index]

def read_int_file(path: Path) -> Optional[int]:
    """Safely read an integer from a small text/sysfs file."""
    text = read_text_file(path)

    if text is None:
        return None

    try:
        return int(text)
    except ValueError:
        return None




def read_text_file(path: Path) -> Optional[str]:
    """Safely read a small text/sysfs file."""
    try:
        with open(path, "rb") as file:
            raw = file.read()

        if not raw:
            return None

        text = raw.decode("utf-8", errors="ignore").strip()
        return text or None

    except (OSError, TypeError):
        return None





def read_linux_thermal_zones() -> List[Dict[str, Any]]:
    """Read Linux thermal zones from /sys/class/thermal.

    Works on many Linux-based embedded platforms, including Raspberry Pi and
    NVIDIA Jetson. Some thermal zones may be unreadable or expose empty values;
    those are skipped.
    """
    thermal_root = Path("/sys/class/thermal")
    zones: List[Dict[str, Any]] = []

    if not thermal_root.exists():
        return zones

    for zone_path in sorted(thermal_root.glob("thermal_zone*")):
        temp_raw = read_int_file(zone_path / "temp")

        if temp_raw is None:
            continue

        zone_type = read_text_file(zone_path / "type") or zone_path.name

        # Most Linux thermal zones report milli-degree Celsius.
        temperature_c = temp_raw / 1000.0

        zones.append(
            {
                "zone": zone_path.name,
                "type": zone_type,
                "temperature_c": temperature_c,
            }
        )

    return zones


def read_raspberry_pi_temperature() -> Optional[float]:
    """Read Raspberry Pi SoC temperature through vcgencmd as fallback."""
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    # Typical output: temp=48.2'C
    output = result.stdout.strip()

    try:
        return float(output.split("=")[1].split("'")[0])
    except (IndexError, ValueError):
        return None


def read_raspberry_pi_throttling() -> Optional[Dict[str, Any]]:
    """Read Raspberry Pi throttling state through vcgencmd, if available."""
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    output = result.stdout.strip()

    try:
        value_str = output.split("=")[1]
        value_int = int(value_str, 16)
    except (IndexError, ValueError):
        return {"raw": output}

    return {
        "raw": output,
        "value_hex": hex(value_int),
        "currently_under_voltage": bool(value_int & 0x1),
        "currently_frequency_capped": bool(value_int & 0x2),
        "currently_throttled": bool(value_int & 0x4),
        "soft_temperature_limit_active": bool(value_int & 0x8),
        "under_voltage_occurred": bool(value_int & 0x10000),
        "frequency_capping_occurred": bool(value_int & 0x20000),
        "throttling_occurred": bool(value_int & 0x40000),
        "soft_temperature_limit_occurred": bool(value_int & 0x80000),
    }


def get_temperature_snapshot() -> Dict[str, Any]:
    """Return one temperature/throttling snapshot.

    The preferred source is /sys/class/thermal because it is portable across
    Raspberry Pi and Jetson. Raspberry Pi vcgencmd is used only as a fallback
    if no thermal zones are available.
    """
    linux_thermal_zones = read_linux_thermal_zones()
    all_temperatures = [
        zone["temperature_c"]
        for zone in linux_thermal_zones
        if zone.get("temperature_c") is not None
    ]

    snapshot: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "linux_thermal_zones" if linux_thermal_zones else "unavailable",
        "linux_thermal_zones": linux_thermal_zones,
        "max_temperature_c": max(all_temperatures) if all_temperatures else None,
    }

    if not linux_thermal_zones:
        raspberry_pi_soc_temp = read_raspberry_pi_temperature()

        if raspberry_pi_soc_temp is not None:
            snapshot["source"] = "raspberry_pi_vcgencmd"
            snapshot["raspberry_pi_soc_temperature_c"] = raspberry_pi_soc_temp
            snapshot["max_temperature_c"] = raspberry_pi_soc_temp

    raspberry_pi_throttling = read_raspberry_pi_throttling()

    if raspberry_pi_throttling is not None:
        snapshot["raspberry_pi_throttling"] = raspberry_pi_throttling

    return snapshot


def summarize_temperature_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a compact summary from temperature snapshots."""
    values = [
        entry.get("max_temperature_c")
        for entry in trace
        if entry.get("max_temperature_c") is not None
    ]

    if not values:
        return {
            "available": False,
            "start_temperature_c": None,
            "end_temperature_c": None,
            "max_temperature_c": None,
            "delta_temperature_c": None,
        }

    return {
        "available": True,
        "start_temperature_c": values[0],
        "end_temperature_c": values[-1],
        "max_temperature_c": max(values),
        "delta_temperature_c": values[-1] - values[0],
    }


def benchmark(
    model_path: str,
    runs: int,
    warmup_runs: int,
    num_threads: int,
    temp_sample_interval: int,
):
    """Benchmark the given TFLite model and return a result dictionary."""
    benchmark_started_utc = datetime.now(timezone.utc).isoformat()
    temperature_trace: List[Dict[str, Any]] = []

    temperature_trace.append(
        {
            "phase": "before_model_load",
            "run_index": None,
            **get_temperature_snapshot(),
        }
    )

    # Load TFLite model with specified thread count
    interpreter = load_tflite_interpreter(model_path, num_threads)

    # Retrieve model input/output tensor specifications
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create random input tensor matching model expectations
    input_tensor = create_dummy_input(input_details)
    input_index = input_details[0]["index"]

    # Capture pre-benchmark thermal/throttle state
    temperature_trace.append(
        {
            "phase": "before_warmup",
            "run_index": None,
            **get_temperature_snapshot(),
        }
    )

    # Warmup phase: runs excluded from measurement
    # Stabilizes CPU frequency and cache state before actual benchmarking
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

    temperature_trace.append(
        {
            "phase": "after_warmup",
            "run_index": None,
            **get_temperature_snapshot(),
        }
    )

    # Measured phase: collect latency measurements
    latencies_ms: List[float] = []

    for run_index in range(runs):
        # Set input tensor data
        interpreter.set_tensor(input_index, input_tensor)

        # Measure inference latency using high-resolution timer
        start = time.perf_counter()
        interpreter.invoke()  # Execute model inference
        end = time.perf_counter()

        # Convert latency to milliseconds and store
        latencies_ms.append((end - start) * 1000.0)

        # Sample temperature every N measured runs.
        # Avoid sampling every run because sysfs/subprocess calls can disturb timing.
        if temp_sample_interval > 0 and (run_index + 1) % temp_sample_interval == 0:
            temperature_trace.append(
                {
                    "phase": "measurement",
                    "run_index": run_index + 1,
                    **get_temperature_snapshot(),
                }
            )

    temperature_trace.append(
        {
            "phase": "after_measurement",
            "run_index": runs,
            **get_temperature_snapshot(),
        }
    )

    # Retrieve final output tensor for metadata extraction
    output = interpreter.get_tensor(output_details[0]["index"])
    mean_latency_ms = statistics.mean(latencies_ms)

    # Build comprehensive results dictionary with benchmark metadata and statistics
    result = {
        # Benchmark configuration
        "model": str(model_path),
        "runs": runs,
        "warmup_runs": warmup_runs,
        "num_threads": num_threads,
        "temp_sample_interval": temp_sample_interval,
        "timestamp_utc": benchmark_started_utc,
        # Input/output tensor specifications
        "input_shape": input_details[0]["shape"].tolist(),
        "input_dtype": str(input_details[0]["dtype"]),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        # Latency statistics in milliseconds
        "latency_ms": {
            "mean": mean_latency_ms,
            "median": statistics.median(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "p90": percentile(latencies_ms, 90),  # 90th percentile (tail latency)
            "p95": percentile(latencies_ms, 95),  # 95th percentile (common SLA target)
            "p99": percentile(latencies_ms, 99),  # 99th percentile (worst-case tail)
        },
        # Throughput calculated as inverse of mean latency
        "throughput_fps": 1000.0 / mean_latency_ms,
        # Thermal and throttling metadata for reproducibility
        "thermal": {
            "summary": summarize_temperature_trace(temperature_trace),
            "trace": temperature_trace,
        },
        # System information for reproducibility
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
    }

    return result


def main():
    # Initialize argument parser with description
    parser = argparse.ArgumentParser(
        description="TensorFlow Lite pure inference benchmark"
    )

    # Required argument: path to TFLite model file
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .tflite model",
    )

    # Benchmark configuration: number of measured runs (excludes warmup)
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of measured inference runs",
    )

    # Benchmark configuration: warmup runs to stabilize performance
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of warmup runs before measurement",
    )

    # Benchmark configuration: CPU threads for inference
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of CPU threads for TFLite interpreter",
    )

    # Thermal sampling configuration
    parser.add_argument(
        "--temp-sample-interval",
        type=int,
        default=50,
        help=(
            "Sample temperature every N measured runs. "
            "Use 0 to disable intermediate temperature samples."
        ),
    )

    # Optional argument: output JSON file path
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )

    # Parse all command-line arguments
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be greater than or equal to 0")

    if args.num_threads <= 0:
        raise ValueError("--num-threads must be greater than 0")

    if args.temp_sample_interval < 0:
        raise ValueError("--temp-sample-interval must be greater than or equal to 0")
	
	# Execute benchmark with specified configuration
    result = benchmark(
        model_path=args.model,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        num_threads=args.num_threads,
        temp_sample_interval=args.temp_sample_interval,
    )

    # Print results to console in formatted JSON
    print(json.dumps(result, indent=4))

    # Optionally save results to file if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4)

        print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()
