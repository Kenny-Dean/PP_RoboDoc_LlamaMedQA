# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel
import torch
import json
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported
from huggingface_hub import HfApi, HfFolder

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Modell und Tokenizer laden
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Modell mit PEFT konfigurieren
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

medqa_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = medqa_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset, concatenate_datasets
dataset1 = load_dataset("medalpaca/medical_meadow_medqa", split = "train")
dataset4 = load_dataset("medalpaca/medical_meadow_medical_flashcards", split = "train")

dataset1 = dataset1.map(formatting_prompts_func, batched = True)
dataset4 = dataset4.map(formatting_prompts_func, batched = True)

dataset_comp = concatenate_datasets([dataset1, dataset4])

# change the padding tokenizer value
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
model.config.pad_token_id = tokenizer.pad_token_id # updating model config
tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)

# Training-Parameter festlegen
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_comp,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=6,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=114433,
        output_dir="outputs",
    ),
)

# Training durchführen
trainer_stats = trainer.train()

model.save_pretrained("unsloth_Llama3_8B_bnb_4bit_RoboDoc_K")
tokenizer.save_pretrained("unsloth_Llama3_8B_bnb_4bit_RoboDoc_K_Tokenizer")
