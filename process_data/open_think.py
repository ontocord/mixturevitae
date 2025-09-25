import random, json, tqdm
from datasets import load_dataset

with open("open_thoughts.jsonl", "w") as outf:
    for file in ["open-thoughts/OpenThoughts3-1.2M", ]:
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset(file, streaming=True)
        curr = []
        for idx, data in tqdm.tqdm(enumerate(ds['train'])):
            if sum(len(c) for c in curr) + len(json.dumps(data, indent=4))> 18000 or len(curr) >= random.randint(3,20):
                out = random.choice(["###", "---", "////", ]) + "\n"+random.choice(["I am Aurora, your assistant, and ", "It's Aurora-M again. ", "I want to learn more and grow more says Aurora-M. ", "Let's keep trying [thinks Aurora - the assistant]. ",])+ " "
                out += f"I have been tasked to read and understand these {len(curr)} example math problems. "+random.choice(["Ok!",  "Let's start:\n", "Keep trying hard to learn!", "As long as we can, find joy in learning!"])+"\n===\n" + random.choice(["\n===\n", "\n+++\n", "\n----\n"]).join(curr)
                out = out.replace("these 1 ", "this ").replace("this example math problems.", "this example math problem.")
                outf.write (json.dumps({'text': out, 'metadata':{'source': file}})+"\n")
                curr = []
            curr.append(json.dumps(data, indent=4))

        if curr:        
                out = random.choice(["###", "---", "////", ]) + "\n"+random.choice(["I am Aurora, your assistant, and ", "It's Aurora-M again. ", "I want to learn more and grow more says Aurora-M. ", "Let's keep trying [thinks Aurora - the assistant]. ",])+ " "
                out += f"I have been tasked to read and understand these {len(curr)} example math problems. "+random.choice(["Ok!",  "Let's start:\n", "Keep trying hard to learn!", "As long as we can, find joy in learning!"])+"\n===\n" + random.choice(["\n===\n", "\n+++\n", "\n----\n"]).join(curr)
                out = out.replace("these 1 ", "this ").replace("this example math problems.", "this example math problem.")
                outf.write (json.dumps({'text': out, 'metadata':{'source': file}})+"\n")
                curr = []

