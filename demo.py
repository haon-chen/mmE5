from src.model import MMEBModel
from src.arguments import ModelArguments
from src.utils import load_processor
import torch
from PIL import Image

model_args = ModelArguments(
    model_name='intfloat/mmE5-mllama-11b-instruct',
    pooling='last',
    normalize=True,
    model_backbone='mllama')

processor = load_processor(model_args)

model = MMEBModel.load(model_args)
model.eval()
model = model.to('cuda', dtype=torch.bfloat16)


# Image + Text -> Text
inputs = processor(text='<|image|><|begin_of_text|> Represent the given image with the following question: What is in the image', images=[Image.open(
    'figures/example.jpg')], return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(text=string, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.3965]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(text=string, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.3105]], device='cuda:0', dtype=torch.bfloat16)

# Text -> Image
inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a dog.', return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<|image|><|begin_of_text|> Represent the given image.'
inputs = processor(text=string, images=[Image.open('figures/example.jpg')], return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## <|image|><|begin_of_text|> Represent the given image. = tensor([[0.4219]], device='cuda:0', dtype=torch.bfloat16)

inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a tiger.', return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<|image|><|begin_of_text|> Represent the given image.'
inputs = processor(text=string, images=[Image.open('figures/example.jpg')], return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## <|image|><|begin_of_text|> Represent the given image. = tensor([[0.3887]], device='cuda:0', dtype=torch.bfloat16)
