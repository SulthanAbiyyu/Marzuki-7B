import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import login
from getpass import getpass

hf_token = getpass("Enter your huggingface token: ")
login(token=hf_token)

def run(checkpoint, adapters, hub_name):
    checkpoint = checkpoint
    adapters = adapters

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = PeftModel.from_pretrained(model, adapters)
    model = model.merge_and_unload()
    model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(adapters)
    model.push_to_hub(hub_name)

    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint", type=str, required=True)
    args.add_argument("--adapters", type=str, required=True)
    args = args.parse_args()
    
    run(args.checkpoint, args.adapters, args.prompt)