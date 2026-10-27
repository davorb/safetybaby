#!/bin/sh
set -eu
base=https://huggingface.co/audiomagic/yamnet-onnx/resolve/main
dir="$(dirname "$0")/../models"
mkdir -p "$dir"
curl -fL --retry 3 -o "$dir/yamnet.onnx" "$base/yamnet.onnx"
curl -fL --retry 3 -o "$dir/yamnet_class_map.csv" "$base/yamnet_class_map.csv"
