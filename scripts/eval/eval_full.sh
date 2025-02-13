#!/usr/bin/env bash

set -x
set -e

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="intfloat/mmE5-mllama-11b-instruct"
fi
if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="meta-llama/Llama-3.2-11B-Vision"
fi
if [ -z "$CKP_PATH" ]; then
    CKP_PATH="./checkpoint/ft_xxx/checkpoint-xxx"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./outputs"
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


PYTHONPATH=src/ python ./eval.py --model_name "${MODEL_NAME}" \
  --processor_name "${PROCESSOR_NAME}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataloader_num_workers 4 \
  --dataset_name "TIGER-Lab/MMEB-eval" \
  --subset_name Wiki-SS-NQ Visual7W-Pointing RefCOCO RefCOCO-Matching ImageNet-1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 ScienceQA VizWiz GQA TextVQA OVEN FashionIQ EDIS \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --image_dir "images/eval_images/" \
  --model_backbone "${MODEL_BACKBONE}" \
