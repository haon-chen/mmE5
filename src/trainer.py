import os
import torch
import torch.distributed as dist

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
from typing import Optional


class MMEBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MMEBTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, *args, **kwargs):
        qry_inputs, tgt_inputs, neg_inputs = inputs
        return model(qry=qry_inputs, tgt=tgt_inputs, neg=neg_inputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
