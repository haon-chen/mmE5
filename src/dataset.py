import os
import random
import datasets

from typing import List, Tuple
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

Phi_Image_token = "<|image_1|>"
Llava_Image_token = "<image>"
class TrainDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        self.negative_ratio = self.data_args.negative_ratio
        train_data = []
        if self.data_args.synthetic_dataset_name or self.data_args.synthetic_dataset_path:
            print(f"Loading {len(data_args.synthetic_subset_name)} synthetic datasets: {data_args.synthetic_subset_name}")
            for subset in data_args.synthetic_subset_name:
                num_sample = -1
                if self.data_args.synthetic_dataset_name:
                    subset_data = load_dataset(
                        self.data_args.synthetic_dataset_name,
                        subset,
                        split=f"{self.data_args.dataset_split}[:{num_sample}]",
                    )
                elif self.data_args.synthetic_dataset_path:
                    subset_path = os.path.join(self.data_args.synthetic_dataset_path, subset) 
                    subset_data = load_from_disk(subset_path)
                    if len(subset_data) > num_sample and num_sample != -1:
                        subset_data = subset_data.select(range(num_sample))
                train_data.append(subset_data)
        if self.data_args.dataset_name or self.data_args.dataset_path:
            print(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
            for subset in data_args.subset_name:
                num_sample = data_args.num_sample_per_subset
                if self.data_args.dataset_name:
                    subset_data = load_dataset(
                        self.data_args.dataset_name,
                        subset,
                        split=f"{self.data_args.dataset_split}[:{num_sample}]",
                    )
                elif self.data_args.dataset_path:
                    subset_path = os.path.join(self.data_args.dataset_path, subset) 
                    subset_data = load_from_disk(subset_path)
                    if len(subset_data) > num_sample and num_sample != -1:
                        subset_data = subset_data.select(range(num_sample))
                
                train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)

        print(f"Number of samples: {len(self.train_data)}")

    def __len__(self):
        return len(self.train_data)

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((512, 512))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if 'Llama' in self.model_args.model_name or self.model_args.model_backbone == "mllama":
            if image.size[1] == 1:
                # print(f"Failed Image: {image}.")
                image = image.resize((image.size[0], 2))
        if self.model_args.model_backbone == "llava_next":
            # TODO: make it configurable
            return self._process_image(image, "high")
        else:
            return image

    def filter_hard_negtives(self, negs, pos, negative_ratio):
        negs = eval(negs)
        if not isinstance(negs, list):
            negs = [negs]

        if len(negs) < negative_ratio and len(negs) > 0:
            negs += [negs[-1]] * (negative_ratio - len(negs))

        negs = negs[:negative_ratio]
        return negs

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        qry_text, qry_image_path, pos_text, pos_image_path = (
            self.train_data[item]["qry"], self.train_data[item]["qry_image_path"],
            self.train_data[item]["pos_text"], self.train_data[item]["pos_image_path"],
        )
        neg_texts, neg_image_paths, neg_images = [], [], []
        if self.negative_ratio > 0:
            neg_text_list, neg_image_path_list = (
                self.train_data[item]["neg_text"], self.train_data[item]["neg_image_path"],
            )
            neg_texts = self.filter_hard_negtives(neg_text_list, pos_text, self.negative_ratio)
            neg_image_paths = self.filter_hard_negtives(neg_image_path_list, pos_image_path, self.negative_ratio)

        for ind, neg in enumerate(neg_texts):
            if neg == '':
                if len(set(eval(neg_text_list))) == 1:
                    neg_texts[ind] = pos_text
                else:
                    neg_texts[ind] = random.choice([text for text in eval(neg_text_list) if text != ""])
        if self.model_args.model_backbone == "llava_next":
            # Update image token
            qry_text = qry_text.replace(Phi_Image_token, Llava_Image_token)
            pos_text = pos_text.replace(Phi_Image_token, Llava_Image_token)
            for ind, neg in enumerate(neg_texts):
                neg_texts[ind] = neg.replace(Phi_Image_token, Llava_Image_token)
        for neg_img in neg_image_paths:
            neg_images.append(self._get_image(neg_img))
        return (qry_text, self._get_image(qry_image_path),
                pos_text, self._get_image(pos_image_path),
                neg_texts, neg_images)


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args

        if self.data_args.dataset_name:
            self.eval_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=self.data_args.dataset_split,
                download_mode="force_redownload"
            )
        elif self.data_args.dataset_path:
            subset_path = os.path.join(self.data_args.dataset_path, subset) 
            self.eval_data = load_from_disk(subset_path)
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.model_args.model_backbone == "llava_next":
            # Update llava image token
            text = text.replace(Phi_Image_token, Llava_Image_token)
        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((512, 512))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "llava_next":
            return self._process_image(image, "high")
        else:
            return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data
