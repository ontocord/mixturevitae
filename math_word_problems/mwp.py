import random
import sympy
import mpmath
import mathfunctions
import inspect
import itertools
import random
import argparse
import tqdm

# Get all global functions from the mathfunctions module
global_functions = [
    name
    for name, obj in inspect.getmembers(mathfunctions)
    if inspect.isfunction(obj) and inspect.getmodule(obj) == mathfunctions
]

def execute_functions(function_dict):
    import mathfunctions
    
    results = []
    
    for function_name in tqdm.tqdm(function_dict):
        function = getattr(mathfunctions, function_name, None)
        
        if function and callable(function):
            result = function(function_dict[function_name])
            results.extend(result)
            print (len(result))
        else:
            print("Invalid function:", function_name)
    return results

def generate_functions(num_samples=100000):
    # Create a dictionary of function names and assign the desired number of samples to each function
    functions = {}
    num_samples_per_function = int (num_samples / len(global_functions))
    for function_name in global_functions:
        functions[function_name] = num_samples_per_function

    # Get the list of results
    return execute_functions(functions)

import json
with open("math_test.jsonl", "w") as outf:
    for _ in tqdm.tqdm(range(1)):
        curr = ""
        problems = generate_functions()
        random.shuffle(problems)
        for problem in problems:
            print ((problem,))
            
            if "\nSo, " in problem:
                a = problem.split("\nSo, ")[-1]
                a = a.split("=")[0]+"="
                if curr:
                    curr += "<|im_end|><|im_start|>" +a+"\\n####\n"+problem
                else:
                    curr = a+"\\n####\n"+problem
            else:
                if curr:
                    curr += "<|im_end|><|im_start|>" +problem
                else:
                    curr = problem
            if len(curr) > 2000:
                outf.write(json.dumps({'text':curr})+"\n")
                curr = ""
        if curr:
            outf.write(json.dumps({'text':curr})+"\n")        
