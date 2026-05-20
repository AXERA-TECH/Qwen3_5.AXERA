set -e
export FLOAT_MATMUL_USE_CONV_EU=1 

# 编译长上下文模型

INPUT_DIR=../../Qwen/Qwen3.5-0.8B
OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-0.8B-AX650-C256-P6K-CTX8K
pulsar2 llm_build --input_path $INPUT_DIR \
                --output_path  $OUTPUT_DIR \
                --hidden_state_type bf16 \
                --prefill_len 256 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 512 \
                --last_kv_cache_len 768 \
                --last_kv_cache_len 1024 \
                --last_kv_cache_len 1280 \
                --last_kv_cache_len 1536 \
                --last_kv_cache_len 1792 \
                --last_kv_cache_len 2048 \
                --last_kv_cache_len 2304 \
                --last_kv_cache_len 2560 \
                --last_kv_cache_len 2816 \
                --last_kv_cache_len 3072 \
                --last_kv_cache_len 3328 \
                --last_kv_cache_len 3584 \
                --last_kv_cache_len 3840 \
                --last_kv_cache_len 4096 \
                --last_kv_cache_len 4352 \
                --last_kv_cache_len 4608 \
                --last_kv_cache_len 4864 \
                --last_kv_cache_len 5120 \
                --last_kv_cache_len 5376 \
                --last_kv_cache_len 5632 \
                --last_kv_cache_len 5888 \
                --last_kv_cache_len 6144 \
                --kv_cache_len 2047 \
                --kv_cache_len 4095 \
                --kv_cache_len 6143 \
                --kv_cache_len 8191 \
                --chip AX650 \
                --parallel 8 

tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR