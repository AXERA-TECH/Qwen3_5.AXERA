# need GPTQModel==6.0.3 , transformers==5.4.0


export CUDA_VISIBLE_DEVICES=1,2
export ENABLE_DYNAMIC_QUANT_CONFIG=1
export TARGET_LANGUAGE=zh
export WIKI_ZH_CONFIG=20231101.zh
export TEXT_LANGUAGE_PATTERN=zh,zh,en
export COCO_PROMPT_PATTERN=zh,zh,en
export VQA_PROMPT_PATTERN=zh,zh,en
export LANGUAGE_FOLLOWING_PATTERN=zh,zh,en
export NUM_VQA_CALIB=640
export NUM_COCO_CALIB=256
export NUM_LANGUAGE_CALIB=64
export VQA_DATA_SOURCE=modelscope
export VQA_DATASET_ID=moonshotai/WorldVQA
export VQA_SPLIT=train
export VQA_IMAGE_FIELD=image_base64
export VQA_QUESTION_FIELD=question
export VQA_ANSWER_FIELD=answer
export VQA_LANG_FIELD=language
export VQA_ZH_QUESTION_FIELD=question
export VQA_ZH_ANSWER_FIELD=answer
export COCO_CAPTION_STRATEGY=grounded
export COCO_TRANSLATION_MODEL_ID=/data/tmp/yongqiang/nfs/lhj/Tencent-Hunyuan/Hunyuan-MT-7B
export COCO_TRANSLATION_DEVICE=cuda:1
export COCO_TRANSLATION_BATCH_SIZE=4
export MODEL_DEVICE=cuda:0
export CALIBRATION_DEVICE=cuda:0
export PRECOMPUTE_INPUTS_DEVICE=cuda:0
export RELEASE_TRANSLATION_MODEL=1
export CALIBRATION_CACHE_PATH=cache/qwen3_5_0.8b_calib_1024_vqa_zh_0.pt
export DATASETS_CACHE_DIR=/tmp/hf_datasets
export WIKI_DATA_SOURCE=modelscope
# export MODEL_ID=../../Qwen/Qwen3.5-0.8B/
# export QUANT_PATH=../../Qwen/Qwen3.5-0.8B-GPTQ-Int4-ZH-1
# nohup python apply_gptq.py > apply_gptq.log &

# export CUDA_VISIBLE_DEVICES=3,4
# export ENABLE_DYNAMIC_QUANT_CONFIG=0
# export MODEL_DEVICE=cuda:0
# export CALIBRATION_DEVICE=cuda:0
# export PRECOMPUTE_INPUTS_DEVICE=cuda:0
# export COCO_TRANSLATION_DEVICE=cuda:1
# export RELEASE_TRANSLATION_MODEL=1
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_2b_calib_1024_vqa_zh_0.pt
# export MODEL_ID=../../Qwen/Qwen3.5-2B/
# export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-ZH-0
# nohup python apply_gptq.py > apply_gptq_2b_0.log &


export CUDA_VISIBLE_DEVICES=5,6
export ENABLE_DYNAMIC_QUANT_CONFIG=1
export MODEL_DEVICE=cuda:0
export CALIBRATION_DEVICE=cuda:0
export PRECOMPUTE_INPUTS_DEVICE=cuda:0
export COCO_TRANSLATION_DEVICE=cuda:1
export RELEASE_TRANSLATION_MODEL=1
export CALIBRATION_CACHE_PATH=cache/qwen3_5_2b_calib_1024_vqa_zh_1.pt
export MODEL_ID=../../Qwen/Qwen3.5-2B/
export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-ZH-1
nohup python apply_gptq.py > apply_gptq_2b_1.log &


# export CUDA_VISIBLE_DEVICES=3
# export ENABLE_DYNAMIC_QUANT_CONFIG=0
# export MODEL_ID=../../Qwen/Qwen3.5-4B/
# export QUANT_PATH=../../Qwen/Qwen3.5-4B-GPTQ-Int4-ZH-0
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_4b_calib_1024_zh_0.pt
# nohup python apply_gptq.py > apply_gptq_4b_0.log &

# export CUDA_VISIBLE_DEVICES=3
# export ENABLE_DYNAMIC_QUANT_CONFIG=1
# export MODEL_ID=../../Qwen/Qwen3.5-4B/
# export QUANT_PATH=../../Qwen/Qwen3.5-4B-GPTQ-Int4-ZH-1
# export CALIBRATION_CACHE_PATH=cache/qwen3_5_4b_calib_1024_zh_1.pt
# nohup python apply_gptq.py > apply_gptq_4b_1.log &
