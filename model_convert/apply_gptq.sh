# need GPTQModel==6.0.3 , transformers==5.4.0


export CUDA_VISIBLE_DEVICES=1
export ENABLE_DYNAMIC_QUANT_CONFIG=1
export CALIBRATION_CACHE_PATH=cache/qwen3_5_0.8b_calib_1024_1.pt
export MODEL_ID=../../Qwen/Qwen3.5-0.8B/  
export QUANT_PATH=../../Qwen/Qwen3.5-0.8B-GPTQ-Int4-EN-1  
nohup python apply_gptq.py > apply_gptq.log &

# export CUDA_VISIBLE_DEVICES=1
# export ENABLE_DYNAMIC_QUANT_CONFIG=0
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_2b_calib_1024_0.pt
# export MODEL_ID=../../Qwen/Qwen3.5-2B/  
# export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-EN-0  
# nohup python apply_gptq.py > apply_gptq_2b_0.log &


# export CUDA_VISIBLE_DEVICES=1
# export ENABLE_DYNAMIC_QUANT_CONFIG=1
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_2b_calib_1024_1.pt
# export MODEL_ID=../../Qwen/Qwen3.5-2B/  
# export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-EN-1  
# nohup python apply_gptq.py > apply_gptq_2b_1.log &


# export CUDA_VISIBLE_DEVICES=2
# export ENABLE_DYNAMIC_QUANT_CONFIG=0
# export MODEL_ID=../../Qwen/Qwen3.5-4B/  
# export QUANT_PATH=../../Qwen/Qwen3.5-4B-GPTQ-Int4-EN-0  
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_4b_calib_1024_0.pt
# nohup python apply_gptq.py > apply_gptq_4b_0.log &

# export CUDA_VISIBLE_DEVICES=2
# export ENABLE_DYNAMIC_QUANT_CONFIG=1
# export MODEL_ID=../../Qwen/Qwen3.5-4B/  
# export QUANT_PATH=../../Qwen/Qwen3.5-4B-GPTQ-Int4-EN-1  
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_4b_calib_1024_1.pt
# nohup python apply_gptq.py > apply_gptq_4b_1.log &

