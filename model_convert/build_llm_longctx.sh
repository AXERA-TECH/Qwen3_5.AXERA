set -e
export FLOAT_MATMUL_USE_CONV_EU=1 

# 编译长上下文模型

INPUT_DIR=../../Qwen/Qwen3.5-0.8B
OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-0.8B-AX650-C256-P6K-CTX8K

pulsar2 llm_build2 --input_path $INPUT_DIR \
                --output_path  $OUTPUT_DIR \
                --hidden_state_type bf16 \
                --prefill_len 6400 \
                --prefill_step_size 256 \
                --max_context 8192 \
                --chip AX650 \
                --parallel 8 

tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR