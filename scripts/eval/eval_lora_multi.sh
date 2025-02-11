#!/usr/bin/env bash

set -x
set -e

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="meta-llama/Llama-3.2-11B-Vision"
#   MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
    # MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
fi
if [ -z "$CKP_PATH" ]; then
    CKP_PATH="./checkpoint/ft_xxx/checkpoint-xxx"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./outputs/multilingual"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/"
fi
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=16
fi
if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="mllama"
#   MODEL_BACKBONE="phi35v"
#   MODEL_BACKBONE="llava_next"
fi

PYTHONPATH=src/ python eval_multi.py --lora \
  --model_name "${MODEL_NAME}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --checkpoint_path "${CKP_PATH}" \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_path "/home/v-chenhaonan/multimodal/VLM2Vec/data/XTD-datasets" \
  --subset_name ar_t2i de_t2i en_t2i es_t2i fr_t2i it_t2i jp_t2i ko_t2i pl_t2i ru_t2i tr_t2i zh_t2i \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --image_dir "images/XTD10_dataset/" \
  --model_backbone "${MODEL_BACKBONE}" \

echo "done"
