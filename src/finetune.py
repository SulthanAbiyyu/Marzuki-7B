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

class MarzukiTrainer:
    def __init__(self, checkpoint, dataset_path, project_name, run_name):
        self.checkpoint = checkpoint
        self.dataset_path = dataset_path
        self.project_name = project_name
        self.run_name = run_name
        self.dataset = None
        self.cfg = None
        self.bnb_cfg = None
        self.peft_config = None
        self.training_args = None
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_dataset(self):
        self.dataset = load_dataset("json", data_files=self.dataset_path)
        self.dataset = self.dataset.map(lambda x: {'text': x['instruction'] + x['output']})

    def set_cfg(self):
        self.cfg = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "output_dir": "./Marzuki-7B-lora",
            "save_strategy": "steps",
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "max_grad_norm": 0.3,
            "optim": "paged_adamw_32bit",
            "save_steps": 500,
            "num_train_epochs": 3,
            "logging_steps": 1,
            "learning_rate": 2e-4,
            "fp16": True,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "linear",
            "batch_size": 4,
            "max_seq_length": 2048,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "r": 64,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"]
        }

    def set_bnb_cfg(self):
        self.bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=self.cfg["load_in_4bit"],
            bnb_4bit_quant_type=self.cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=self.cfg["bnb_4bit_compute_dtype"],
            bnb_4bit_use_double_quant=self.cfg["bnb_4bit_use_double_quant"],
        )

    def set_peft_config(self):
        self.peft_config = LoraConfig(
            lora_alpha=self.cfg["lora_alpha"],
            lora_dropout=self.cfg["lora_dropout"],
            r=self.cfg["r"],
            bias=self.cfg["bias"],
            task_type=self.cfg["task_type"],
            target_modules=self.cfg["target_modules"],
        )

    def set_training_args(self):
        self.training_args = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            save_strategy=self.cfg["save_strategy"],
            gradient_accumulation_steps=self.cfg["gradient_accumulation_steps"],
            gradient_checkpointing=self.cfg["gradient_checkpointing"],
            max_grad_norm=self.cfg["max_grad_norm"],
            per_device_train_batch_size=self.cfg["batch_size"],
            optim=self.cfg["optim"],
            num_train_epochs=self.cfg["num_train_epochs"],
            save_steps=self.cfg["save_steps"],
            logging_steps=self.cfg["logging_steps"],
            learning_rate=self.cfg["learning_rate"],
            fp16=self.cfg["fp16"],
            warmup_ratio=self.cfg["warmup_ratio"],
            lr_scheduler_type=self.cfg["lr_scheduler_type"],
        )

    def set_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            quantization_config=self.bnb_cfg
        )

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def set_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.cfg["max_seq_length"],
            tokenizer=self.tokenizer,
            args=self.training_args,
        )

    def train(self, resume_from_checkpoint=True):
        wandb.login()
        wandb.init(project=self.project_name, name=self.run_name)
        self.load_dataset()
        self.set_cfg()
        self.set_bnb_cfg()
        self.set_peft_config()
        self.set_training_args()
        self.set_model()
        self.set_tokenizer()
        self.set_trainer()
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.trainer.save_model(self.cfg["output_dir"])
