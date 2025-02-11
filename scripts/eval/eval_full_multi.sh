#!/usr/bin/env bash

set -x
set -e

DIR="/home/v-chenhaonan/teamdrive"

# MODEL_NAME_OR_PATH="TIGER-Lab/VLM2Vec-Full"
# MODEL_NAME_OR_PATH="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
# MODEL_NAME_OR_PATH="jinaai/jina-clip-v2"
# MODEL_NAME_OR_PATH="nvidia/MM-Embed"
MODEL_NAME_OR_PATH="TIGER-Lab/VLM2Vec-LLaVa-Next"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
#   OUTPUT_DIR="outputs/MM-Embed/"
#   OUTPUT_DIR="outputs/jina/"
  OUTPUT_DIR="outputs/VLM2Vec-LLaVa-Next1/"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/multimodal/data"
fi
#   --dataset_path "${DATA_DIR}/MMEB-eval-datasets" \
#   --subset_name ar_t2i de_t2i en_t2i es_t2i fr_t2i it_t2i jp_t2i ko_t2i pl_t2i ru_t2i tr_t2i zh_t2i \
#   --subset_name en_t2i \

if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="mllama"
fi

MODEL_BACKBONE="llava_next"
# MODEL_BACKBONE="mmembed"
# MODEL_BACKBONE="jina"
OUTPUT_DIR="$OUTPUT_DIR/multilingual/"

PYTHONPATH=src/ python ./eval_multi.py --model_name "${MODEL_NAME_OR_PATH}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --model_backbone "${MODEL_BACKBONE}" \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_path "/home/v-chenhaonan/multimodal/VLM2Vec/data/XTD-datasets" \
  --subset_name tr_t2i \
  --dataset_split test --per_device_eval_batch_size 8 \
  --image_dir "${DATA_DIR}/xtd_11/photos/XTD10_dataset/" \

echo "done"

# screen -S 2004471.mteb -X quit