# -*- coding: utf-8 -*-
  
  import torch
  from datasets import load_dataset
  from transformers import TrainingArguments
  from ollama import OllamaLanguageModel, OllamaTrainer, TextStreamer

max_seq_length = 2048

# Model and Tokenizer loading using ollama
model_name = "ollama/llama-3-8b-bnb-4bit"
model = OllamaLanguageModel.from_pretrained(model_name, max_seq_length=max_seq_length)

# Configure PEFT for the model
model.configure_peft(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Formatting prompts function for dataset
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
"""
        texts.append(text.strip() + tokenizer.eos_token)
    return {"text": texts}

# Load and format dataset
dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Training parameters
trainer = OllamaTrainer(
    model=model,
    train_dataset=dataset["train"],
    text_field="text",
    max_seq_length=max_seq_length,
    num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=7,
        learning_rate=2e-4,
        fp16=False,  # Ollama uses 8-bit training, not bfloat16
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Perform training
trainer.train()

# Saving the model and tokenizer to Hugging Face Hub
hf_token = ""  # Your Hugging Face API token

model.save_pretrained("Llama3_8B_bnb_4bit_RoboDoc_MedQA", push_to_hub=True, use_auth_token=hf_token)
tokenizer.save_pretrained("Llama3_8B_bnb_4bit_RoboDoc_MedQA_Tokenizer", push_to_hub=True, use_auth_token=hf_token)
