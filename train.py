# Adapted from Tevatron code
import logging
import sys
import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from transformers import AutoProcessor, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from src.dataset import TrainDataset
from src.collator import TrainCollator, LlamaCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.trainer import MMEBTrainer
from src.vlm_backbone.llava_next.processing_llava_next import LlavaNextProcessor
from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor

logger = logging.getLogger(__name__)

def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if model_args.model_backbone == "llava_next":
        processor = LlavaNextProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True)
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == "phi35v":
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == "qwen":
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "right"
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"

    train_dataset = TrainDataset(data_args, model_args)
    
    if 'Llama' in model_args.model_name or model_args.model_backbone == "mllama":
        collator = LlamaCollator(data_args, processor)
    else:
        collator = TrainCollator(data_args, model_args, processor)

    model = MMEBModel.build(model_args, training_args)
    trainer = MMEBTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataset.trainer = trainer

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    if training_args.resume_from_checkpoint is not None:
        if os.path.exists(training_args.resume_from_checkpoint):
            resume_from_checkpoint = get_last_checkpoint(training_args.resume_from_checkpoint)
            print(f"Restarting from LoRA checkpoint {resume_from_checkpoint}")
        else:
            print(f"Checkpoint {training_args.resume_from_checkpoint} not found")
            resume_from_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
