# Complete workflow

## PC (prepare models)
### Create venv:
./setup/setup_venv.sh

### Activate:
source .edge-ai-benchmark-env/bin/activate

### Install:
pip install -r setup/requirements_pc.txt

### Prepare models:
python scripts/prepare_models.py


## Raspberry Pi
### Create venv:
./setup/setup_venv.sh

### Activate:
source .edge-ai-benchmark-env/bin/activate

### Install:
pip install -r setup/requirements_target.txt

### Run benchmark:
python benchmark/pure_inference/run.py \
  --model models/mobilenetv2_fp32.tflite


## Jetson
### Create venv:
./setup/setup_venv.sh
source .edge-ai-benchmark-env/bin/activate
pip install -r setup/requirements_target.txt

### Run benchmark:
python benchmark/pure_inference/run.py \
  --model models/mobilenetv2_fp32.tflite


TensorRT later:
trtexec --onnx=model.onnx --fp16