#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input="$1"
output="$2"

python "$SCRIPT_DIR/extract_embed.py" --input_path "$input" --output_path "$output"
python "$SCRIPT_DIR/embed-process.py" --input "$output/model.embed_tokens.weight.npy" --output "$output/model.embed_tokens.weight.float32.bin"
"$SCRIPT_DIR/fp32_to_bf16" "$output/model.embed_tokens.weight.float32.bin" "$output/model.embed_tokens.weight.bfloat16.bin"
