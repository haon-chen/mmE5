#!/usr/bin/env bash

set -x
set -e

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="intfloat/mmE5-mllama-11b-instruct"
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

PYTHONPATH=src/ python eval_multi.py --model_name "${MODEL_NAME}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --max_len 256 \
  --dataloader_num_workers 4 \
  --pooling last --normalize True \
  --dataset_name "Haon-Chen/XTD-10" \
  --subset_name es it ko pl ru tr zh \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --image_dir "images/XTD10_dataset/" \
  --model_backbone "${MODEL_BACKBONE}"

echo "done"
