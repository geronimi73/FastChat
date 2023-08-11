
import os
import json
import random
from copy import deepcopy

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

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def checkErrors(judgements):
    empty_count=0

    stats= {
        'total': len(judgements), 
        'one error': 0, 
        'two errors': 0, 
        'empty judgements': 0,
        'error judgements': []
    }

    for judgement in judgements:
        if judgement["g1_winner"] == "error" and judgement["g2_winner"] == "error":
            stats["two errors"]+=1
            stats["error judgements"].append({
                    'question_id': judgement['question_id'],
                    'model': judgement['model_1'],
                    'text': judgement['g1_judgment'],
                })
            stats["error judgements"].append({
                    'question_id': judgement['question_id'],
                    'model': judgement['model_2'],
                    'text': judgement['g2_judgment'],
                })
        elif judgement["g1_winner"] == "error" or judgement["g2_winner"] == "error":
            stats["one error"]+=1
            if judgement["g1_winner"] == "error":
                stats["error judgements"].append({
                        'question_id': judgement['question_id'],
                        'model': judgement['model_1'],
                        'text': judgement['g1_judgment'],
                    })
            else:
                stats["error judgements"].append({
                        'question_id': judgement['question_id'],
                        'model': judgement['model_2'],
                        'text': judgement['g2_judgment'],
                    })
        if judgement["g1_judgment"] == "" or judgement["g2_judgment"] == "":
            stats["empty judgements'"]+=1
    return stats

def parse_winner(judgement_text):
    if "[[A]]" in judgement_text and "[[B]]" in judgement_text:
        if (judgement_text.find("[[A]]")<judgement_text.find("[[B]]")):
            winner = "A"
        else:
            winner = "B"
    elif "[[A]]" in judgement_text:
        winner = "A"
    elif "[[B]]" in judgement_text:
        winner = "B"
    elif "[[C]]" in judgement_text:
        winner = "tie"
    elif "[[tie]]" in judgement_text.lower():
        winner = "tie"
    else:
        if "[[Assistant A]]" in judgement_text:
            winner = "A"
        elif "[[Assistant B]]" in judgement_text:
            winner = "B"
        else:
            winner = "error"
    return winner

def fix_judgements(judgements):
    for judgement in judgements:
        for k in ["g1_judgment","g2_judgment"]:
            judgement_text=judgement[k]

            winner = parse_winner(judgement_text)

            if k == "g1_judgment":
                if winner =="A" or winner =="B":
                    winner_mapped="model_1" if winner=="A" else "model_2"
                else:                    
                    winner_mapped=winner

                if judgement["g1_winner"] != winner_mapped:
                    judgement["g1_winner"]=winner_mapped
            else:
                if winner =="A" or winner =="B":
                    winner_mapped="model_2" if winner=="A" else "model_1"
                else:                    
                    winner_mapped=winner
                if judgement["g2_winner"] != winner_mapped:
                    judgement["g2_winner"]=winner_mapped
    return judgements

files= [
    "text-generation-webui-api_pair.jsonl",
    ]

for file_path in files:
    fixed=False
    print(file_path)
    judgments=read_jsonl(file_path)
    for j in [judgments, fix_judgements(deepcopy(judgments))]:
        print("before" if not fixed else "after")
        stats=checkErrors(j)
        for key in stats:
            if key=="error judgements":
                if fixed:
                    out_fn=file_path.split(".")[0]+"_errors.json"
                    write_pretty_json(out_fn, stats[key])
                    print(f" Wrote remaining error judgements to {out_fn}")
            else:
                print(f" {key}: {json.dumps(stats[key],indent=2)}")
        if fixed:
            out_fn=file_path.split(".")[0]+"_fixed.jsonl"
            write_jsonl(out_fn, j)
            print(f" Wrote fixed judgements to {out_fn}")

        fixed=True
        print("")

