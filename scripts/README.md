# Scripts Directory

Utility scripts for model preparation and benchmark setup.

## prepare_models.py

Prepares neural network models for benchmarking and deployment.

**Functionality:**
- Loads pre-trained TensorFlow models
- Converts to TensorFlow Lite format
- Applies precision conversion or quantization (FP32, FP16, or INT8)
- Validates model configuration
- Generates model metadata

**Usage:**
```bash
python prepare_models.py \
  --input model.pb \
  --output models/mobilenetv2_fp32.tflite \
  --quantization fp32

python prepare_models.py \
  --input model.pb \
  --output models/mobilenetv2_fp16.tflite \
  --quantization fp16

python prepare_models.py \
  --input model.pb \
  --output models/mobilenetv2_int8.tflite \
  --quantization int8 \
  --calibration-data calibration_data/imagenet_subset/
```

**Key Operations:**
1. Model conversion: TensorFlow → TFLite
2. Quantization: Reduces precision and model size
3. Validation: Ensures model correctness
4. Metadata extraction: Collects model specifications

## Setup Scripts

For environment configuration, see [setup/README.md](../setup/README.md)

## Model Pipeline

```
Original Model → prepare_models.py → TFLite Models (FP32, FP16 & INT8)
                                            ↓
                                   benchmark/run.py
                                            ↓
                                    results/
```

For detailed code explanation, see [docs/code_overview.md](../docs/code_overview.md)
