import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator, LlamaEvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset, load_from_disk
from evaluation.eval_utils import get_pred, precision_at_k, recall_at_k
from src.vlm_backbone.llava_next.processing_llava_next import LlavaNextProcessor

xtd_10_data_group = ["it", "es", "ru", "zh", "pl", "tr", "ko"]

def main():
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

    if model_args.checkpoint_path:
        base_name = os.path.basename(model_args.checkpoint_path)
        if base_name.startswith("checkpoint"):
            dir_name = os.path.basename(os.path.dirname(model_args.checkpoint_path)).split('-')[-1]
        else:
            dir_name = os.path.basename(model_args.checkpoint_path).split('-')[-1]
        output_path = f"{data_args.encode_output_path}/{dir_name}/{base_name}/" if base_name.startswith("checkpoint") else f"{data_args.encode_output_path}/{dir_name}/"
    else:
        output_path = data_args.encode_output_path

    print(output_path)
    os.makedirs(output_path, exist_ok=True)

    if model_args.model_backbone == "llava_next":
        processor = LlavaNextProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )

    processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    if 'Llama' in model_args.model_name or model_args.model_backbone == "mllama":
        eval_collator = LlamaEvalCollator(
            data_args=data_args,
            model_args=model_args,
            processor=processor,
        )
    else:
        eval_collator = EvalCollator(
            data_args=data_args,
            model_args=model_args,
            processor=processor,
        )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            continue

        dataset_func = EvalDataset

        eval_qry_dataset = dataset_func(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = dataset_func(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                batch = {key: value.to(training_args.device) for key, value in batch.items()}
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=batch)
                encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                batch = {key: value.to(training_args.device) for key, value in batch.items()}
                output = model(tgt=batch)
                encoded_tensor.append(output["tgt_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    accuracy_t2i = []
    result_per_lang = {}
    recall_per_lang = {}
    xtd_10_res = {
        "precision": [],
        "recall": [],
    }

    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t
        
        if data_args.dataset_name:
            eval_data = load_dataset(
                data_args.dataset_name,
                subset,
                split=data_args.dataset_split
            )
        elif data_args.dataset_path:
            subset_path = os.path.join(data_args.dataset_path, subset) 
            eval_data = load_from_disk(subset_path)
        n_correct = 0
        all_pred = []
        all_precisions = []
        all_recalls = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_precisions.append(precision_at_k(scores, [0], 1))
            all_recalls.append(recall_at_k(scores, [0], 10))
            all_pred.append(all_candidates[pred])
        with open(os.path.join(output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        score_path = os.path.join(output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        accuracy_t2i.append(n_correct/len(eval_data))
        
        if subset.split("_")[0] not in result_per_lang:
            result_per_lang[subset.split("_")[0]] = [(n_correct, len(eval_data))]
            recall_per_lang[subset.split("_")[0]] = np.mean(all_recalls)
        else:
            result_per_lang[subset.split("_")[0]].append((n_correct, len(eval_data)))
            recall_per_lang[subset.split("_")[0]] += np.mean(all_recalls)
        if subset.split("_")[0] in xtd_10_data_group:
            xtd_10_res["precision"].append(np.mean(all_precisions))
            xtd_10_res["recall"].append(np.mean(all_recalls))
        
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)*100}\033[0m")
        print(f"Precision@1: {np.mean(all_precisions)*100}")
        print(f"Recall@10: {np.mean(all_recalls)*100}")
    print(f"All accuracy: {np.mean(accuracy_t2i)*100}, dataset num: {len(accuracy_t2i)}")
    print(f"All recall: {np.mean([recall_per_lang[lang]/len(result) for lang, result in result_per_lang.items()])*100}")
    print(f"xtd-10 precision: {np.mean(xtd_10_res['precision'])*100}")
    print(f"xtd-10 recall: {np.mean(xtd_10_res['recall'])*100}")

    for lang, result in result_per_lang.items():
        total_correct = 0
        total = 0
        for idx, (correct, num) in enumerate(result):
            total_correct += correct
            total += num
        print(f"{lang} accuracy: {total_correct/total*100}, dataset num: {len(result)}")
        print(f"{lang} recall: {recall_per_lang[lang]/len(result)*100}")

if __name__ == "__main__":
    main()
