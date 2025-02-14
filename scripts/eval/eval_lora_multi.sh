#!/usr/bin/env bash

set -x
set -e

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="meta-llama/Llama-3.2-11B-Vision"
#   MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
    # MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
fi
if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="meta-llama/Llama-3.2-11B-Vision"
fi
if [ -z "$CKP_PATH" ]; then
    CKP_PATH="./checkpoint/ft_xxx/checkpoint-xxx"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./outputs/multilingual"
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
  --processor_name "${PROCESSOR_NAME}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --checkpoint_path "${CKP_PATH}" \
  --max_len 256 \
  --dataloader_num_workers 4 \
  --pooling last --normalize True \
  --dataset_name "Haon-Chen/XTD-10" \
  --subset_name es it ko pl ru tr zh \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --image_dir "images/XTD10_dataset/" \
  --model_backbone "${MODEL_BACKBONE}"

echo "done"
