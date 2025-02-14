#!/usr/bin/env bash

set -x
set -e

MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-11B-Vision"
# MODEL_NAME_OR_PATH="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME_OR_PATH="llava-hf/llava-v1.6-mistral-7b-hf"

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./checkpoint/ft_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/"
fi

DS_CONFIG_PATH="ds_config.json"

if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=4
fi

if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="mllama"
#   MODEL_BACKBONE="phi35v"
#   MODEL_BACKBONE="llava_next"
fi

if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="meta-llama/Llama-3.2-11B-Vision"
#   PROCESSOR_NAME="microsoft/Phi-3.5-vision-instruct"
#   PROCESSOR_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
fi

deepspeed --master_port 18271 train.py --deepspeed "${DS_CONFIG_PATH}" \
    --dataset_name "intfloat/mmE5-MMEB-hardneg" \
    --subset_name TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    --synthetic_dataset_name "intfloat/mmE5-synthetic" \
    --synthetic_subset_name Classification Retrieval VQA \
    --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
    --num_sample_per_subset 50000 \
    --dataloader_num_workers 4 \
    --image_dir "images/MMEB-train" \
    --gradient_checkpointing True --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --lora --lora_r 8 \
    --max_len 256 --output_dir "${OUTPUT_DIR}" --logging_steps 5 \
    --lr_scheduler_type linear --learning_rate 1e-5 --max_grad_norm 5.0 \
    --warmup_ratio 0.05 --save_steps 100 --save_total_limit 3 --normalize True \
    --temperature 0.02 --per_device_train_batch_size ${BATCH_SIZE} \
    --model_backbone "${MODEL_BACKBONE}" \
    --processor_name "${PROCESSOR_NAME}" \
    --resume_from_checkpoint "${OUTPUT_DIR}" \
    --negative_ratio 2 \
    --report_to none "$@"
