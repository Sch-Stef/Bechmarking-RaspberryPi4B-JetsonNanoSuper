# Setup Directory

Configuration and setup instructions for development and target environments.

## Files

### requirements_pc.txt

Development dependencies for PC/workstation.

**Includes:**
- TensorFlow and TensorFlow Lite tools
- Model conversion utilities
- Python development tools
- Jupyter for analysis

**Installation:**
```bash
pip install -r requirements_pc.txt
```

### requirements_target.txt

Dependencies for target deployment hardware (Raspberry Pi 4B, Jetson Orin Nano Super).

**Includes:**
- TensorFlow Lite Runtime (optimized for embedded)
- NumPy for numerical computation
- Minimal dependencies for efficient deployment

**Installation on Raspberry Pi:**
```bash
pip install -r requirements_target.txt
```

### setup_venv.sh

Bash script to automate virtual environment setup.

**Features:**
- Creates Python virtual environment
- Installs appropriate dependencies
- Configures environment variables
- Validates installation

**Usage:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
source .edge-ai-benchmark-env/bin/activate
```

### HowTo.md

Detailed step-by-step setup instructions for each platform.

**Covers:**
- Hardware prerequisites
- Operating system configuration
- Python environment setup
- Dependency installation
- Troubleshooting

**Start here for first-time setup.**

## Setup Quick Start

### Raspberry Pi

1. Review [HowTo.md](HowTo.md) for hardware requirements
2. Install OS and update: `sudo apt update && sudo apt upgrade`
3. Run setup script: `./setup_venv.sh`
4. Install dependencies: `pip install -r requirements_target.txt`
5. Verify installation: `python -c "import tflite_runtime"`

### Jetson Orin Nano Super

1. Install NVIDIA JetPack
2. Verify CUDA and TensorRT installation
3. Run setup script: `./setup_venv.sh`
4. Install dependencies: `pip install -r requirements_target.txt`
5. Verify installation: `python -c "import tflite_runtime"`

### PC/Workstation

1. Create virtual environment: `python -m venv .env`
2. Activate: `source .env/bin/activate` (Linux/Mac) or `.env\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements_pc.txt`
4. Verify: `python -c "import tensorflow as tf"`

## Troubleshooting

### Common Issues

**ImportError: No module named tflite_runtime**
- Ensure TensorFlow Lite runtime is installed
- Verify Python version compatibility (3.7+)
- Check virtual environment activation

**Permission denied on setup_venv.sh**
- Make executable: `chmod +x setup_venv.sh`
- Run with correct shell: `bash setup_venv.sh`

**Dependency conflicts**
- Create fresh virtual environment
- Use Python 3.8 or 3.9 for best compatibility
- Check for system package conflicts

For detailed troubleshooting, see [HowTo.md](HowTo.md)

## Virtual Environment

Recommended approach for all platforms:

```bash
# Create
python -m venv .edge-ai-benchmark-env

# Activate (Linux/Mac)
source .edge-ai-benchmark-env/bin/activate

# Activate (Windows)
.edge-ai-benchmark-env\Scripts\activate

# Deactivate
deactivate
```

All subsequent Python commands will use the virtual environment.
