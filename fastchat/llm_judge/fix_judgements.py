
import os
import json
import random


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

def read_and_fix_judgements(input_fn):
    judgements=read_jsonl(input_fn)
    empty_count=0
    error_dist={}
    error_judgements=[]
    for c in [0,1,2]:
        error_dist[c]=0
    num_changes=0
    num_total=0

    for judgement in judgements:
        for k in ["g1_judgment","g2_judgment"]:
            num_total+=1
            judgement_text=judgement[k]

            if "[[A]]" in judgement_text and "[[B]]" in judgement_text:
                winner = "error"
            elif "[[A]]" in judgement_text:
                winner = "A"
            elif "[[B]]" in judgement_text:
                winner = "B"
            elif "[[C]]" in judgement_text:
                winner = "tie"
            elif "[[Tie]]" in judgement_text:
                winner = "tie"
            else:
                if "[[Assistant A]]" in judgement_text:
                    winner = "A"
                elif "[[Assistant B]]" in judgement_text:
                    winner = "B"
                else:
                    winner = "error"

            if k == "g1_judgment":
                if winner =="A" or winner =="B":
                    winner_mapped="model_1" if winner=="A" else "model_2"
                else:                    
                    winner_mapped=winner

                if judgement["g1_winner"] != winner_mapped:
                    # print(f"\nold: " + judgement["g1_winner"] + f", new: {winner_mapped}")
                    # print(f"## {judgement_text}")
                    # print(judgement)
                    judgement["g1_winner"]=winner_mapped
                    num_changes+=1
            else:
                if winner =="A" or winner =="B":
                    winner_mapped="model_2" if winner=="A" else "model_1"
                else:                    
                    winner_mapped=winner
                if judgement["g2_winner"] != winner_mapped:
                    # print(f"\nold: " + judgement["g2_winner"] + f", new: {winner_mapped}")
                    # print(f"## {judgement_text}")
                    # print(judgement)
                    judgement["g2_winner"]=winner_mapped
                    num_changes+=1

    return num_total, num_changes,judgements

jfiles=[
    "data/mt_bench/model_judgment/gpt-4_pair.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_temp-0.1-exllama.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_temp-0.1.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair_temp-0.7.jsonl",
    "data/mt_bench/model_judgment/text-generation-webui-api_pair.jsonl",
    ]


for input_fn in jfiles:
    # print(f"------ {input_fn}")
    num_total,num_changes,judgements=read_and_fix_judgements(input_fn)

    print(f"{os.path.basename(input_fn)}: {num_changes} changes, {num_total} total")

    write_jsonl(os.path.splitext(input_fn)[0]+"_fixed.jsonl", judgements)

