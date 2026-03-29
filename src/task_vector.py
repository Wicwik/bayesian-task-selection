# Adopted from https://github.com/danielm1405/iso-merging/blob/main/src/models/task_vectors.py

import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open


class TaskVector:
    def __init__(self, model_name, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        self.model_name = model_name


        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )

            with torch.no_grad():
                # load pretrained weights
                pretrained_state_dict = self._safe_load(pretrained_checkpoint)

                # load finetuned weights
                finetuned_state_dict = self._safe_load(finetuned_checkpoint)


            print(pretrained_state_dict.keys())
            print(finetuned_state_dict.keys())


    def _safe_load(self, checkpoint_path):
        try:
            tensors = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

            return tensors
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    pretrained_checkpoint = "saves_bts_preliminary/lora/llama-3.2-1b-instruct/train_sst2_42_1773148417/adapter_model.safetensors"
    finetuned_checkpoint = "saves_bts_preliminary/lora/llama-3.2-1b-instruct/train_mnli_42_1773148411/adapter_model.safetensors"
    task_vector = TaskVector(model_name="example_model", pretrained_checkpoint=pretrained_checkpoint, finetuned_checkpoint=finetuned_checkpoint)