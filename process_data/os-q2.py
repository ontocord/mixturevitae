import glob, os, json
import random
import tqdm
#from shared import *
import multiprocessing


def process(l_file):
    l, file = l_file
    batch = []
    file2 = file.split("/")[-1]
    if True:
            try:
                data = json.loads(l)
            except:
                print (l)
                return None
            if "72B" not in file:
                text = random.choice(["### User:", "### Instruction:", "Q:", ])+"\n"+data['input']+"\n"+random.choice(["### Assistant:", "### Output:", "A:", ]) +"\n"+data['output']
            else:
                text = data['output']
            if len(text) < 100:
                return None
            text = " ".join(a.replace(a.strip("\"~!@#$%^&*()-_+="), "[URL]") if ("http:" in a or "https:" in a) else a for a in text.replace("<EMAIL_ADDRESS>", "").split(" "))
            data = {"text": text, "metadata": [{'source': file2}]}
            data['metadata'][0]['source'] = file.split("/")[-1]
            if len(text) < 200: None
            
            return data
def yield_file(file):
    for l in tqdm.tqdm(open(file, "rb")):
        yield(l, file)
        
if __name__ == "__main__":
    all_files = []
    prev_data = None    
    
    #list(glob.glob('../peS2o/*.jsonl', recursive=True)) +  +  list(glob.glob('../pubmed_common_pile/*.jsonl', recursive=True))
    all_files.extend(list(set(  list(glob.glob('OS-Q2*.jsonl', recursive=True)))))
    random.shuffle(all_files)
    with multiprocessing.Pool(5) as pool:
        curr = 0
        i0 = 0
        outf = open(f"../mixture-vitae-200BT/synthetic_instruct/os-q2-{i0}.jsonl", "w")
        for file in all_files:
            for data in pool.imap_unordered(process, yield_file(file)):
                if not data: continue
                if prev_data:
                    if len(prev_data['text'])+ len(data['text']) > 10000:
                        prev_data['metadata'] = json.dumps(prev_data['metadata'])                
                        curr += len(prev_data['text'])
                        outf.write(json.dumps(prev_data)+"\n")        
                        if curr > 20000000000:
                            curr = 0
                            i0 +=1
                            outf = open(f"../mixture-vitae-200BT/synthetic_instruct/os-q2-{i0}.jsonl", "w")
                        prev_data = data
                    else:
                        prev_data['metadata'].extend(data['metadata'])
                        prev_data['text'] += "<|endoftext|>"+ data['text']
                else:
                    prev_data = data
    if prev_data:
        prev_data['metadata'] = json.dumps(prev_data['metadata'])                
        outf.write(json.dumps(prev_data)+"\n")        
                        
