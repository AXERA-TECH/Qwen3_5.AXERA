set -e
export FLOAT_MATMUL_USE_CONV_EU=1 

# INPUT_DIR=../../Qwen/Qwen3.5-4B-GPTQ-Int4-EN-0
# OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-4B-AX650-GPTQ-Int4-C128-P1152-CTX2047-EN-0
# pulsar2 llm_build2 --input_path $INPUT_DIR \
#                 --output_path  $OUTPUT_DIR \
#                 --hidden_state_type bf16 \
#                 --prefill_len 1280 \
#                 --prefill_step_size 128 \
#                 --max_context 2048 \
#                 --chip AX650 \
#                 --parallel 8 

# tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR


INPUT_DIR=../../Qwen/Qwen3.5-4B-GPTQ-Int4-EN-1
OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-4B-AX650-GPTQ-Int4-C128-P1152-CTX2047-EN-1
pulsar2 llm_build2 --input_path $INPUT_DIR \
                --output_path  $OUTPUT_DIR \
                --hidden_state_type bf16 \
                --prefill_len 1280 \
                --prefill_step_size 128 \
                --max_context 2048 \
                --chip AX650 \
                --parallel 8 

tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR