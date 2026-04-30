set -e
export FLOAT_MATMUL_USE_CONV_EU=1 

# INPUT_DIR=/data/tmp/yongqiang/nfs/lhj/Qwen/Qwen3.5-2B-GPTQ-Int4-EN-0
# OUTPUT_DIR=/data/tmp/yongqiang/nfs/lhj/Qwen3_5.AXERA/ax-llm/Qwen3.5-2B-AX650-GPTQ-Int4-C128-P1152-CTX2047-EN-0
# pulsar2 llm_build --input_path $INPUT_DIR \
#                 --output_path  $OUTPUT_DIR \
#                 --kv_cache_len 2047 \
#                 --hidden_state_type bf16 \
#                 --prefill_len 128 \
#                 --last_kv_cache_len 128 \
#                 --last_kv_cache_len 256 \
#                 --last_kv_cache_len 384 \
#                 --last_kv_cache_len 512 \
#                 --last_kv_cache_len 640 \
#                 --last_kv_cache_len 768 \
#                 --last_kv_cache_len 896 \
#                 --last_kv_cache_len 1024 \
#                 --last_kv_cache_len 1152 \
#                 --chip AX650 \
#                 --parallel 8 

# tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR


INPUT_DIR=/data/tmp/yongqiang/nfs/lhj/Qwen/Qwen3.5-2B-GPTQ-Int4-EN-1
OUTPUT_DIR=/data/tmp/yongqiang/nfs/lhj/Qwen3_5.AXERA/ax-llm/Qwen3.5-2B-AX650-GPTQ-Int4-C128-P1152-CTX2047-EN-1
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