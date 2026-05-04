#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image


def build_mobilenetv2():
    """ Builds a MobileNetV2 model with pretrained ImageNet weights. """
    # Create MobileNetV2 model with ImageNet pretrained weights
    # - Lightweight architecture optimized for mobile/embedded devices
    # - Input shape: 224x224x3 (standard ImageNet size)
    # - Alpha: 1.0 (full width multiplier, no depth reduction)
    # - Include top: True (includes classification head for 1000 classes)
    # - Classes: 1000 (ImageNet dataset classes)
    return tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights="imagenet",
        classes=1000,
    )


def export_fp32(model, output_path: Path):
    """ Exports the given Keras model to FP32 TFLite format. """
    # Create TFLite converter from Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Convert model to TFLite format (FP32 by default)
    # No optimizations applied - full precision inference
    tflite_model = converter.convert()

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write binary TFLite model to file
    output_path.write_bytes(tflite_model)

    print(f"Saved FP32 model to: {output_path}")

def export_fp16(
    model,
    output_path: Path,
):
    """Exports the given Keras model to FP16 TFLite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Force FP16 weights
    converter.target_spec.supported_types = [tf.float16]

    # Explicit FP16 interface
    converter.inference_input_type = tf.float32 # tf.float16 is not supported for inputs
    converter.inference_output_type = tf.float32 # tf.float16 is not supported for outputs

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    print(f"Saved FP16 model to: {output_path}")


def representative_dataset_from_images(calibration_dir: Path, max_samples: int):
    """ Use real images from the specified directory for INT8 calibration. """
    # Collect all image files from calibration directory
    # Supports common image formats: JPG, JPEG, PNG
    image_paths = (
        list(calibration_dir.glob("*.jpg"))
        + list(calibration_dir.glob("*.jpeg"))
        + list(calibration_dir.glob("*.png"))
    )

    # Sort paths for reproducibility and limit to max_samples
    image_paths = sorted(image_paths)[:max_samples]

    # Validate that calibration images were found
    if not image_paths:
        raise RuntimeError(f"No calibration images found in: {calibration_dir}")

    print(f"Using {len(image_paths)} calibration images from: {calibration_dir}")

    # Generator: yields preprocessed images for quantization calibration
    for image_path in image_paths:
        # Load image and convert to RGB (handles RGBA, grayscale, etc.)
        img = Image.open(image_path).convert("RGB")
        
        # Resize to model input size (224x224)
        img = img.resize((224, 224))

        # Convert to numpy array and float32
        data = np.asarray(img).astype(np.float32)
        
        # Add batch dimension (required by TFLite)
        data = np.expand_dims(data, axis=0)

        # Apply MobileNetV2 preprocessing (normalization to [-1, 1])
        data = tf.keras.applications.mobilenet_v2.preprocess_input(data)

        # Yield as list for TFLite converter compatibility
        yield [data]


def representative_dataset_random(num_samples: int):
    """ Generates random data for INT8 calibration. """
    """ Note: This is not ideal but can be used when real calibration data is unavailable."""
    # WARNING: Random calibration data is a fallback approach
    # Quantization with real representative data produces better accuracy
    # This method helps when calibration dataset is not available
    print(f"Using {num_samples} random calibration samples")

    # Generator: yields random tensors matching model input specification
    for _ in range(num_samples):
        # Generate random FP32 values in [0, 1)
        data = np.random.random_sample((1, 224, 224, 3)).astype(np.float32)
        
        # Scale to typical image range [0, 255]
        data = data * 255.0
        
        # Apply MobileNetV2 preprocessing (normalization to [-1, 1])
        data = tf.keras.applications.mobilenet_v2.preprocess_input(data)
        
        # Yield as list for TFLite converter compatibility
        yield [data]


def export_int8(
    model,
    output_path: Path,
    calibration_samples: int,
    calibration_dir: Path | None = None,
):
    """ Exports the given Keras model to INT8 quantized TFLite format. """
    # Create TFLite converter from Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable default optimizations (includes quantization-aware optimizations)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Select calibration dataset source
    if calibration_dir is not None:
        # Use real calibration images if directory provided
        # Produces better quantization parameters and higher accuracy
        converter.representative_dataset = lambda: representative_dataset_from_images(
            calibration_dir,
            calibration_samples,
        )
    else:
        # Fallback to random data if no calibration directory specified
        # Results in lower accuracy but useful for testing/development
        converter.representative_dataset = lambda: representative_dataset_random(
            calibration_samples
        )

    # Specify that model should use INT8 operations
    # Restricts to operations supported by INT8 quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Set input/output types to INT8
    # Inputs will be int8 [-128, 127] instead of float32
    # Outputs will be int8 [-128, 127] instead of float32
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert model to quantized TFLite format
    # Results in ~4x smaller model
    tflite_model = converter.convert()

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write binary quantized TFLite model to file
    output_path.write_bytes(tflite_model)

    print(f"Saved INT8 model to: {output_path}")


def main():
    # Initialize argument parser with description of script purpose
    parser = argparse.ArgumentParser(
        description="Prepare FP32 and INT8 MobileNetV2 TFLite models"
    )
    
    # Argument: Output directory for generated models
    # Default: "models" directory in current working directory
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where models will be stored",
    )

    # Argument: Directory containing real calibration images
    # If provided: Uses real images for INT8 quantization (better accuracy)
    # If omitted: Uses random synthetic data (faster, lower accuracy)
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Directory with real calibration images. If omitted, random calibration data is used.",
    )

    # Argument: Number of samples for INT8 quantization calibration
    # More samples = better quantization parameters but slower conversion
    # Typical range: 50-500 samples
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of representative samples for INT8 calibration",
    )

    # Parse all command-line arguments
    args = parser.parse_args()

    # Set up output and calibration paths from arguments
    output_dir = Path(args.output_dir)
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else None

    # Define output file paths for both model variants (FP32 and INT8)
    fp32_path = output_dir / "mobilenetv2_fp32.tflite"
    fp16_path = output_dir / "mobilenetv2_fp16.tflite"
    int8_path = output_dir / "mobilenetv2_int8.tflite"

    # Step 1: Load pretrained MobileNetV2 model with ImageNet weights
    print("Building MobileNetV2 with ImageNet weights...")
    model = build_mobilenetv2()

    # Step 2: Export full-precision (FP32) model to TFLite format
    print("Exporting FP32 TFLite model...")
    export_fp32(model, fp32_path)

    # Step 3: Export half-precision (FP16) model to TFLite format
    print("Exporting FP16 TFLite model...")
    export_fp16(model, fp16_path)

    # Step 4: Export quantized (INT8) model using post-training quantization
    print("Exporting INT8 quantized TFLite model...")
    export_int8(
        model=model,
        output_path=int8_path,
        calibration_samples=args.calibration_samples,
        calibration_dir=calibration_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()