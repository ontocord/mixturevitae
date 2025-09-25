from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

model_name = "sample"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(model)
print(tokenizer)
