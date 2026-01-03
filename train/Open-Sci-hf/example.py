import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


model_name = "sample"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(model)
print(tokenizer)
