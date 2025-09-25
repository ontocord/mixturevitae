from datasets import load_dataset
import json
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/Nemotron-PrismMath", streaming=True)

with open("prism.jsonl", "w") as outf:
    for idx, data in enumerate(ds['train']):
        text = data['problem']+"\n### Aurora: "+data['solution']
        del data['problem']
        del data['solution']
        data = {'text': text, 'metadata': data}
        outf.write(json.dumps(data)+"\n")
