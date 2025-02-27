from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig
import torch
from PIL import Image

# Pooling and Normalization
def last_pooling(last_hidden_state, attention_mask, normalize=True):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

def compute_similarity(q_reps, p_reps):
    return torch.matmul(q_reps, p_reps.transpose(0, 1))

model_name = "intfloat/mmE5-mllama-11b-instruct"

# Load Processor and Model
processor = AutoProcessor.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = MllamaForConditionalGeneration.from_pretrained(
    model_name, config=config, 
    torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

# Image + Text -> Text
inputs = processor(text='<|image|><|begin_of_text|>Represent the given image with the following question: What is in the image\n', images=[Image.open(
    'figures/example.jpg')], return_tensors="pt").to("cuda")
qry_output = last_pooling(model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])

string = 'A cat and a dog'
text_inputs = processor(text=string, return_tensors="pt").to("cuda")
tgt_output = last_pooling(model(**text_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], text_inputs['attention_mask'])
print(string, '=', compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.4219]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
text_inputs = processor(text=string, return_tensors="pt").to("cuda")
tgt_output = last_pooling(model(**text_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], text_inputs['attention_mask'])
print(string, '=', compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.3184]], device='cuda:0', dtype=torch.bfloat16)

# Text -> Image
inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a dog.\n', return_tensors="pt").to("cuda")
qry_output = last_pooling(model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])

string = '<|image|><|begin_of_text|>Represent the given image.\n'
tgt_inputs = processor(text=string, images=[Image.open('figures/example.jpg')], return_tensors="pt").to("cuda")
tgt_output = last_pooling(model(**tgt_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], tgt_inputs['attention_mask'])
print(string, '=', compute_similarity(qry_output, tgt_output))
## <|image|><|begin_of_text|>Represent the given image. = tensor([[0.4414]], device='cuda:0', dtype=torch.bfloat16)

inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a tiger.\n', return_tensors="pt").to("cuda")
qry_output = last_pooling(model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])

string = '<|image|><|begin_of_text|>Represent the given image.\n'
tgt_inputs = processor(text=string, images=[Image.open('figures/example.jpg')], return_tensors="pt").to("cuda")
tgt_output = last_pooling(model(**tgt_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], tgt_inputs['attention_mask'])
print(string, '=', compute_similarity(qry_output, tgt_output))
## <|image|><|begin_of_text|>Represent the given image. = tensor([[0.3730]], device='cuda:0', dtype=torch.bfloat16)
