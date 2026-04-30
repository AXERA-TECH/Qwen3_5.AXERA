pulsar2 build --input qwen3_5_0.8b_vision.onnx \
                --config config.json \
                --output_dir build-output \
                --output_name qwen3_5_0.8b_vision.axmodel \
                --target_hardware AX650 \
                --compiler.check 0