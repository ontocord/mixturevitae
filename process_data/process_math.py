import json, random
curr = 0
i0 = 0
prev_data = None
outf =  open(f"math_word_problems-{i0}.jsonl", "w")
for l in open("math_generated.jsonl"):
    data = json.loads(l)
    text = data['text']
    if len(text) > 10000:
        continue
    if "He" in text or "he" in text or "ok" in text or " a " in text or " in " in text or " of " in text or " for " in text or " on " in text or "step" in text or "Step" in text:
        prob = ""
        if " / " in text:
            prob += " division, "
        if " - " in text:
            prob += " subtraction, "
        if " * " in text:
            prob += " multiplication, "
        if " + " in text:
            prob += " addition, "
        if prob:
            prob = prob.strip(" ,").split(", ")
            random.shuffle(prob)
            prob = ", ".join(prob)
        prefix =  f"Here is a set of {prob} math problems:\n"
        prefix = prefix.replace("Here is a ", random.choice(["Here is a ", "Here are ", "Here's ", "Think deeply about this ", "Please examine the ", "Study this "]))
        prefix = prefix.replace("a set of", random.choice(["a set of", "some exercises for", "examples of"]))
        prefix = prefix.replace("problems", random.choice(["problems", "guide", "exercises", "examples", "insights"]))
        prefix = prefix.replace("math problems", random.choice(["math problems", "arithmaic examples", "fundamental math study guide"]))
        text ="### "+prefix+text.replace("<|im_end|><|im_start|>", "\n###\n")
        data['text'] = text
        data['metadata'] = [{'source': 'ontocord-synthetic-math'}]
        if prev_data:
            if len(data['text'])+ len(prev_data['text']) > 10000:
                prev_data['metadata'] = json.dumps(prev_data['metadata'])
                outf.write(json.dumps(prev_data)+"\n")
                curr += len(prev_data['text'])
                if curr > 20000000000:
                    curr = 0
                    i0+=1
                    outf =  open(f"math_word_problems-{i0}.jsonl", "w")
                prev_data = data
            else:
                prev_data['text'] += "<|endoftext|>"+data['text']
                prev_data['metadata'].extend(data['metadata'])
                
        else:
            prev_data = data
        
if prev_data:
    prev_data['metadata'] = json.dumps(prev_data['metadata'])
    outf.write(json.dumps(prev_data)+"\n")
    
