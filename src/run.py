import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

def run(checkpoint, adapters, prompt):

    checkpoint = checkpoint
    adapters = adapters

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = PeftModel.from_pretrained(model, adapters)
    model = model.merge_and_unload()
    model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(adapters)
    prompt = prompt

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_length=512, num_beams=5, num_return_sequences=5, temperature=1.0)
    print(tok.decode(out[0], skip_special_tokens=True))
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint", type=str, required=True)
    args.add_argument("--adapters", type=str, required=True)
    args.add_argument("--prompt", type=str, required=True)
    args = args.parse_args()
    
    run(args.checkpoint, args.adapters, args.prompt)
    