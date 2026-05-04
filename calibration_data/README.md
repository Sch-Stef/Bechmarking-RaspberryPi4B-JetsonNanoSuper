# Calibration Data Directory

Contains image datasets used for model calibration and quantization.

## imagenet_subset/

A subset of ImageNet dataset for representative calibration data.

- Used for INT8 quantization calibration
- Ensures quantization parameters accurately represent real-world image distributions
- Typical size: 100-500 representative images

## raspi_camera_subset/

Sample images captured from Raspberry Pi Camera Module v1.3.

- Real-world images from the camera used in benchmarks
- Helps validate quantization performance with actual camera hardware
- Useful for fine-tuning quantization if needed

## Usage

During model quantization in prepare_models.py:

```python
calibration_images = load_images_from_directory('calibration_data/imagenet_subset/')
quantize_model(model, calibration_images, output_type='int8')
```

## Best Practices

1. Use 100-500 representative images for calibration
2. Ensure good coverage of expected input scenarios
3. Store images in standard format (JPEG, PNG)
4. Maintain consistent image preprocessing pipeline
