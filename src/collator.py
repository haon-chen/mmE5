import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch
from IPython import embed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

@dataclass
class TrainCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        neg_inputs = self._get_batch_inputs(examples, 4, 5)

        return qry_inputs, pos_inputs, neg_inputs
    
    def _process_sigle_data(self, text, image, input_ids, pixel_values, image_sizes, has_image, image_exist):
        if image is None:
            if self.model_args.model_backbone == "llava_next":
                inputs = self.processor(images=None, text=text, return_tensors="pt")
            elif self.model_args.model_backbone == "qwen":
                inputs = self.processor(text=[text], images=None, return_tensors="pt",
                                        max_length=self.data_args.max_len, truncation=True)
            else:
                inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                        truncation=True)
            input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            if self.model_args.model_backbone == "llava_next":
                pixel_values.append(None)
                image_sizes.append(None)
            if not image_exist:
                return False
            else:
                return image_exist
        else:
            if self.model_args.model_backbone == "llava_next":
                inputs = self.processor(images=image, text=text, return_tensors="pt")
            else:
                inputs = self.processor(text, [image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
            pixel_values.append(inputs['pixel_values'])
            input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            image_sizes.append(inputs['image_sizes'])
            has_image.append(True)
            return True

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        input_ids, pixel_values, image_sizes, has_image = [], [], [], []
        image_exist = False
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            if isinstance(image, List):
                for t, img in zip(text, image):
                    image_exist = self._process_sigle_data(t, img, input_ids, pixel_values, image_sizes, has_image, image_exist)
            else:
                image_exist = self._process_sigle_data(text, image, input_ids, pixel_values, image_sizes, has_image, image_exist)
        if len(input_ids)==0:
            return None
        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        if not image_exist:
            dummy_pixel_values = torch.zeros(input_ids.shape[0], 1)
            dummy_image_sizes = torch.ones(input_ids.shape[0], 1)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': dummy_pixel_values,
                'image_sizes': dummy_image_sizes,
            }
        else:
            if self.model_args.model_backbone == "llava_next":
                pixel_values_shape = list(set(v.shape for v in pixel_values if v is not None))[0]
                pixel_values = [v if v is not None else torch.zeros(pixel_values_shape) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            if self.model_args.model_backbone == "llava_next":
                image_sizes_shape = list(set(v.shape for v in image_sizes if v is not None))[0]
                image_sizes = [v if v is not None else torch.ones(image_sizes_shape) for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }
        return inputs


    
@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, image_sizes = [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image is None:
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=None, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                            truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                image_exist = True
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=image, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, [image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                image_sizes.append(inputs['image_sizes'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            if self.model_args.model_backbone == "llava_next":
                pixel_values_shape = list(set(v.shape for v in pixel_values if v is not None))[0]
                pixel_values_list = [v if v is not None else torch.zeros(pixel_values_shape) for v in pixel_values]
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = torch.cat(pixel_values, dim=0)
            if self.model_args.model_backbone == "llava_next":
                image_sizes_shape = list(set(v.shape for v in image_sizes if v is not None))[0]
                image_sizes = [v if v is not None else torch.ones(image_sizes_shape) for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }

        return inputs


def first_non_int_element(lst):
    for element in lst:
        if not isinstance(element, int):
            return element
    return None

def convert_zero_tensor(tensor_list, batch_size, need_zero=False, seq_len=None):
    
    if not tensor_list:
        raise ValueError("The tensor_list is empty. Cannot infer tensor properties.")
    
    first_tensor = first_non_int_element(tensor_list)
    tensor_shape = first_tensor.shape
    if seq_len is not None:
        tensor_shape = torch.Size([seq_len, *tensor_shape[1:]])
    dtype = first_tensor.dtype
    device = first_tensor.device

    if need_zero:
        zero_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
    else:
        zero_tensor = torch.ones(tensor_shape, dtype=dtype, device=device)
    
    return zero_tensor


@dataclass
class LlamaCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """

        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        neg_inputs = self._get_batch_inputs(examples, 4, 5)

        return qry_inputs, pos_inputs, neg_inputs
    
    def _process_sigle_data(self, text, image, input_ids, pixel_values, aspect_ratio_ids, aspect_ratio_mask, batch_cross_attention_mask, image_exist):
        if image == None:
            text = str(text)
            text = text.replace("<|image_1|>\n", "<|begin_of_text|>")
        else:
            text = str(text)
            text = text.replace("<|image_1|>\n", "<|image|><|begin_of_text|>")
        
        if image is None:
            inputs = self.processor(text=text, images=None, return_tensors="pt", max_length=self.data_args.max_len,
                                    truncation=True)
            input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))

            pixel_values.append(input_ids[-1].shape[0])
            aspect_ratio_ids.append(input_ids[-1].shape[0])
            aspect_ratio_mask.append(input_ids[-1].shape[0])
            batch_cross_attention_mask.append(input_ids[-1].shape[0])
            if not image_exist:
                return False
            else:
                return image_exist
        else:
            try:
                inputs = self.processor(text=text, images=[image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
            except Exception as e:
                print(f"Error: {image}")
                print(f"Error: {text}")
                return False
            input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            pixel_values.append(inputs['pixel_values'])
            aspect_ratio_ids.append(inputs["aspect_ratio_ids"])
            aspect_ratio_mask.append(inputs["aspect_ratio_mask"])
            batch_cross_attention_mask.append(inputs["cross_attention_mask"].squeeze(0))
            return True

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        input_ids, pixel_values, invalid_indices, aspect_ratio_ids, aspect_ratio_mask, batch_cross_attention_mask = [], [], [], [], [], []
        image_exist = False
        for idx, example in enumerate(examples):
            text, image = example[text_idx], example[image_idx]
            if isinstance(image, List):
                for t, img in zip(text, image):
                    image_exist = self._process_sigle_data(t, img, input_ids, pixel_values, aspect_ratio_ids, aspect_ratio_mask, batch_cross_attention_mask, image_exist)

            else:
                image_exist = self._process_sigle_data(text, image, input_ids, pixel_values, aspect_ratio_ids, aspect_ratio_mask, batch_cross_attention_mask, image_exist)
        if len(input_ids)==0:
            return None
        if image_exist:
            batch_size = len(input_ids)
            
            for ind, input_id in enumerate(input_ids):
                if not isinstance(pixel_values[ind], int):
                    continue
                pixel_values[ind] = convert_zero_tensor(pixel_values, batch_size)
                aspect_ratio_ids[ind] = convert_zero_tensor(aspect_ratio_ids, batch_size)
                aspect_ratio_mask[ind] = convert_zero_tensor(aspect_ratio_mask, batch_size, need_zero=True)
                batch_cross_attention_mask[ind] = convert_zero_tensor(batch_cross_attention_mask, batch_size, need_zero=True, seq_len=input_id.shape[0])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            aspect_ratio_ids = torch.cat(aspect_ratio_ids, dim=0)
            aspect_ratio_mask = torch.cat(aspect_ratio_mask, dim=0)
            cross_attention_mask = torch._C._nn.pad_sequence(
                batch_cross_attention_mask, batch_first=True, padding_value=0
            )
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'aspect_ratio_ids': aspect_ratio_ids,
                'aspect_ratio_mask': aspect_ratio_mask,
                'cross_attention_mask': cross_attention_mask,
            }

        return inputs

@dataclass
class LlamaEvalCollator:
    data_args: DataArguments
    model_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, aspect_ratio_ids, aspect_ratio_mask, batch_cross_attention_mask = [], [], [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image == None:
                text = text.replace("<|image_1|>\n", "<|begin_of_text|>")
            else:
                text = text.replace("<|image_1|>\n", "<|image|><|begin_of_text|>")
            if image is None:
                inputs = self.processor(text=text, images=None, return_tensors="pt", max_length=self.data_args.max_len,
                                        truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                image_exist = True
                inputs = self.processor(text=text, images=[image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                aspect_ratio_ids.append(inputs["aspect_ratio_ids"])
                aspect_ratio_mask.append(inputs["aspect_ratio_mask"])
                batch_cross_attention_mask.append(inputs["cross_attention_mask"].squeeze(0))

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            aspect_ratio_ids = torch.cat(aspect_ratio_ids, dim=0)
            aspect_ratio_mask = torch.cat(aspect_ratio_mask, dim=0)
            cross_attention_mask = torch._C._nn.pad_sequence(
                batch_cross_attention_mask, batch_first=True, padding_value=0
            )
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'aspect_ratio_ids': aspect_ratio_ids,
                'aspect_ratio_mask': aspect_ratio_mask,
                'cross_attention_mask': cross_attention_mask,
            }

        return inputs
