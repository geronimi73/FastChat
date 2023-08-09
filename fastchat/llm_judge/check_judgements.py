
import os
import json
import random

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)

def read_jsonl(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line, strict=False))
        f.close()
        return json_list

def write_jsonl(file_path, data):
    with open(os.path.expanduser(file_path), "w") as f:
        for o in data:
            f.write(json.dumps(o) + "\n")

def get_error_judgements(input_fn):
    judgements=read_jsonl(input_fn)
    empty_count=0
    error_dist={}
    error_judgements=[]
    for c in [0,1,2]:
        error_dist[c]=0

    for judgement in judgements:
        if judgement["g1_winner"] == "error" and judgement["g2_winner"] == "error":
            c=2
        elif judgement["g1_winner"] == "error" or judgement["g2_winner"] == "error":
            c=1
        else:
            c=0
        if judgement["g1_judgment"] == "" or judgement["g2_judgment"] == "":
            empty_count+=1

        error_dist[c]+=1
        if c>0:
            error_judgements.append(judgement)

    return error_dist, error_judgements, empty_count

jfiles=[
    "data/mt_bench/model_judgment/gpt-4_pair_fixed.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_temp-0.1-exllama_fixed.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_temp-0.1_fixed.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_temp-0.7_fixed.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_fixed.jsonl",
    ]
output_erroneous_judgements = True
model_path = "/home/g/models/Upstage-Llama-2-70B-instruct-v2-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)


for input_fn in jfiles:
    error_dist, error_judgements, empty_count=get_error_judgements(input_fn)

    for c in [0,1,2]:
        print(f"{os.path.basename(input_fn)}, {error_dist[c]} judgements with {c} errors")
    print(f"{os.path.basename(input_fn)}, {empty_count} judgements with empty judgement")

    for ej in error_judgements:
        for key in ["g1_user_prompt", "g2_user_prompt"]:
            input_ids = tokenizer(ej[key]).input_ids
            ej[key+"_length"] = len(input_ids)


    write_jsonl(os.path.splitext(os.path.basename(input_fn))[0]+"_errors.jsonl", error_judgements)
    os.system(f"cat {os.path.splitext(os.path.basename(input_fn))[0]}_errors.jsonl | jq > {os.path.splitext(os.path.basename(input_fn))[0]}_errors_nice.jsonl")

