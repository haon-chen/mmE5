#!/usr/bin/env bash

set -x
set -e

DIR="/home/v-chenhaonan/teamdrive"

# MODEL_NAME_OR_PATH="TIGER-Lab/VLM2Vec-Full"
MODEL_NAME_OR_PATH="nvidia/MM-Embed"
# MODEL_NAME_OR_PATH="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="/home/v-chenhaonan/teamdrive/multimodal/outputs/MM-Embed/"
#   OUTPUT_DIR="/home/v-chenhaonan/teamdrive/multimodal/outputs/VLM2Vec/gme"
#   OUTPUT_DIR="${DIR}/multimodal/outputs/VLM2Vec/GME/"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/"
fi
if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="mllama"
fi

# MODEL_BACKBONE="llava_next"
MODEL_BACKBONE="mmembed"
# MODEL_BACKBONE="gme"

PYTHONPATH=src/ python ./eval.py --model_name "${MODEL_NAME_OR_PATH}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --model_backbone "${MODEL_BACKBONE}" \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_name "TIGER-Lab/MMEB-eval" \
  --subset_name Wiki-SS-NQ Visual7W-Pointing RefCOCO RefCOCO-Matching ImageNet-1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 ScienceQA VizWiz GQA TextVQA OVEN FashionIQ EDIS \
  --dataset_split test --per_device_eval_batch_size 3 \
  --image_dir "images/eval_images/"

echo "done"
