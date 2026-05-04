#!/usr/bin/env bash
set -e

ENV_NAME=".edge-ai-benchmark-env"

echo "Creating virtual environment..."

python3 -m venv ${ENV_NAME}

echo "Activating virtual environment..."

source ${ENV_NAME}/bin/activate

echo "Upgrading pip..."

python -m pip install --upgrade pip setuptools wheel

echo "Done."
echo ""
echo "Activate with:"
echo "source ${ENV_NAME}/bin/activate"