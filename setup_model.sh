#!/bin/bash
set -e

MODEL_DIR="./model"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2"

if [ -f "$MODEL_DIR/model_ready" ]; then
    echo "✅ Model already installed."
    exit 0
fi

echo "📥 Downloading Sherpa-ONNX Russian model..."
mkdir -p "$MODEL_DIR"
wget -q "$MODEL_URL" -O /tmp/model.tar.bz2
tar -xjf /tmp/model.tar.bz2 -C "$MODEL_DIR"
mv "$MODEL_DIR"/sherpa-onnx-zipformer-ru-2024-09-18/* "$MODEL_DIR"/
rm -rf /tmp/model.tar.bz2 "$MODEL_DIR"/sherpa-onnx-zipformer-ru-2024-09-18
touch "$MODEL_DIR/model_ready"

echo "✅ Model ready at $MODEL_DIR"