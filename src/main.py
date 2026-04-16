from task_vector import TaskVector

from iso import iso_c, iso_cts


if __name__ == "__main__":
    pretrained_checkpoint = "saves_pretrained_weights/lora/llama-3.2-1b-instruct/train_mnli_42_1775733638/adapter_model.safetensors"
    finetuned_checkpoint = "saves_bts_preliminary/lora/llama-3.2-1b-instruct/train_mnli_42_1773148411/adapter_model.safetensors"
    task_vector = TaskVector(
        model_name="example_model",
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        target_modules=["lora_A", "lora_B"],
    )

    pretrained_state_dict = task_vector._safe_load(pretrained_checkpoint)
    pretrained_state_dict2 = task_vector._safe_load("saves_pretrained_weights/lora/llama-3.2-1b-instruct/train_mnli_42_1775736849/adapter_model.safetensors")
    print(pretrained_state_dict["base_model.model.model.layers.9.self_attn.k_proj.lora_A.weight"])

    print(pretrained_state_dict2["base_model.model.model.layers.9.self_attn.k_proj.lora_A.weight"])   
