import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoTokenizer
)

class cfg:
    load_in_4bit = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_compute_dtype = torch.bfloat16
    bnb_4bit_use_double_quant=True
    output_dir = "./Merak-Kurikulum-TIF-7B-lora"
    save_strategy = "steps"
    gradient_accumulation_steps = 2
    gradient_checkpointing = False
    optim = "paged_adamw_32bit"
    save_steps = 500
    num_train_epochs = 3
    logging_steps = 1
    learning_rate = 2e-4
    fp16 = True
    warmup_ratio = 0.03
    lr_scheduler_type = "linear"
    batch_size = 4
    max_seq_length = 1024
    lora_alpha = 16
    lora_dropout = 0.1
    r = 64
    bias = "none"
    task_type = "CAUSAL_LM"
    target_modules = ["q_proj", "v_proj"]

wandb.login()
wandb.init(project="merak-kurikulum-TIF-7B", name="lora-v1")
dataset = load_dataset("csv", data_files="./qa-kurikulum-v2.csv")["train"]
dataset = dataset.map(lambda x: {'text': x['instruction'] + " " + x['output']})
checkpoint = "Ichsan2895/Merak-7B-v2"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=cfg.load_in_4bit,
    bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
    bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
)

peft_config = LoraConfig(
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    r=cfg.r,
    bias=cfg.bias,
    task_type=cfg.task_type,
    target_modules=cfg.target_modules,
)

training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    save_strategy=cfg.save_strategy,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    gradient_checkpointing=cfg.gradient_checkpointing,
    # max_grad_norm=cfg.max_grad_norm,
    per_device_train_batch_size=cfg.batch_size,
    optim=cfg.optim,
    num_train_epochs=cfg.num_train_epochs,
    save_steps=cfg.save_steps,
    logging_steps=cfg.logging_steps,
    learning_rate=cfg.learning_rate,
    fp16=cfg.fp16,
    warmup_ratio=cfg.warmup_ratio,
    lr_scheduler_type=cfg.lr_scheduler_type,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_cfg
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=cfg.max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model(cfg.output_dir)