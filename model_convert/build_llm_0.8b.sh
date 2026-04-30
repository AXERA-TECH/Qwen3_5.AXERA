set -e
export FLOAT_MATMUL_USE_CONV_EU=1 

INPUT_DIR=../../Qwen/Qwen3.5-0.8B-GPTQ-Int4-EN
OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-0.8B-GPTQ-Int4-AX650-C128-P1280-CTX2047-EN
pulsar2 llm_build --input_path $INPUT_DIR \
                --output_path  $OUTPUT_DIR \
                --kv_cache_len 2047 \
                --hidden_state_type bf16 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --last_kv_cache_len 640 \
                --last_kv_cache_len 768 \
                --last_kv_cache_len 896 \
                --last_kv_cache_len 1024 \
                --last_kv_cache_len 1152 \
                --chip AX650 \
                --parallel 8 

tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR
