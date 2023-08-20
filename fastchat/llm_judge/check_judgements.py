
import os
import json
import random
import fire
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

def stripJudgement(j):
    newJ={}
    keys=[ "question_id", "model_1", "model_2", "g1_winner", "g2_winner", "g1_judgment", "g2_judgment", "turn" ]

    for k in keys:
        newJ[k]=j[k]
    return newJ

def getJudgementStats(judgements):
    qstats={}

    qstat_params=["total", "errors_zero", "errors_any", "errors_one", "errors_both", "errors_st", "errors_mt", "empty", "judge_consistent", "judge_inconsistent", "ties"]
    for k in qstat_params:
        qstats[k]=0

    judgements_error = []

    for judgement in judgements:
        qstats['total']+=1

        if judgement["g1_winner"] == "error" or judgement["g2_winner"] == "error":
            qstats['errors_any']+=1
            judgements_error.append(stripJudgement(judgement))

            if judgement["g1_winner"] == "error" and judgement["g2_winner"] == "error":
                qstats['errors_both']+=1
            else:
                qstats['errors_one']+=1
                
            if judgement["g1_judgment"] == "" or judgement["g2_judgment"] == "":
                qstats['empty']+=1

            if judgement["turn"]==1:
                qstats['errors_st']+=1
            else:
                qstats['errors_mt']+=1
        else:
            qstats['errors_zero']+=1
            if judgement["g1_winner"] == judgement["g2_winner"]:
                qstats['judge_consistent']+=1
                if judgement["g1_winner"] == "tie":
                    qstats['ties']+=1
            else:
                qstats['judge_inconsistent']+=1

    return qstats, judgements_error

def parse_winner(judgement_text):
    look_for=[]
    look_for.append([ "[[A]]", "[[B]]", ["[[C]]", "[[Tie]]", "[[tie]]", "[[TIE]]" ], False ])
    look_for.append([ "[[Assistant A]]", "[[Assistant B]]", [], False ])
    look_for.append([ "[Assistant A]", "[Assistant B]", [], False ])
    look_for.append([ "[A]", "[B]", ["[C]"], False ])
    look_for.append([ "A", "B", [], True ])

    for callsign_modelA, callsign_modelB, callsign_ties, exact_match_only in look_for:
        if exact_match_only:
            if judgement_text==callsign_modelA:
                return "A"
            elif judgement_text==callsign_modelB:
                return "B"
        else:
            if callsign_modelA in judgement_text and callsign_modelB in judgement_text:
                if (judgement_text.find(callsign_modelA)<judgement_text.find(callsign_modelB)):
                    return "A"
                else:
                    return "B"
            elif callsign_modelA in judgement_text:
                return "A"
            elif callsign_modelB in judgement_text:
                return "B"
            else:
                for callsign_tie in callsign_ties:
                    if callsign_tie in judgement_text:
                        return "tie"
    return "error"

def fix_judgements(judgements):
    changes=[]

    for judgement in judgements:
        for k in ["g1_judgment","g2_judgment"]:
            judgement_text=judgement[k]

            winner = parse_winner(judgement_text)
            winner_old = judgement["g1_winner"] if k == "g1_judgment" else judgement["g2_winner"]

            new_winner=False
            if k == "g1_judgment":
                if winner =="A" or winner =="B":
                    winner_mapped="model_1" if winner=="A" else "model_2"
                else:                    
                    winner_mapped=winner

                if judgement["g1_winner"] != winner_mapped:
                    judgement["g1_winner"]=winner_mapped
                    new_winner=True
            else:
                if winner =="A" or winner =="B":
                    winner_mapped="model_2" if winner=="A" else "model_1"
                else:                    
                    winner_mapped=winner
                if judgement["g2_winner"] != winner_mapped:
                    judgement["g2_winner"]=winner_mapped
                    new_winner=True

            if new_winner:
                changes.append(
                {
                    'judgement': judgement_text,
                    'winner_old': winner_old,
                    'winner_new': winner,
                })

    return judgements, changes

def prepare_for_mtbench_agreement(judgements):
    jcopy=deepcopy(judgements)

    for judgement in jcopy:
        judgement["model_a"]=judgement["model_1"]
        judgement["model_b"]=judgement["model_2"]
        judgement["judge"]=judgement["judge"][0]

        if judgement["g1_winner"] == "error" or judgement["g2_winner"] == "error":
            judgement["winner"]="tie (gt-0-errors)"
        elif judgement["g1_winner"] == "Tie" and judgement["g2_winner"] == "Tie":
            judgement["winner"]="tie"
        elif judgement["g1_winner"] == "Tie" or judgement["g2_winner"] == "Tie":
            judgement["winner"]="tie (one tie, one not)"
        elif judgement["g1_winner"] != judgement["g2_winner"]:
            judgement["winner"]="tie"
        elif judgement["g1_winner"] == judgement["g2_winner"]:
            judgement["winner"]="model_a" if judgement["g1_winner"]=="model_1" else "model_b"

        del judgement["g1_user_prompt"]
        del judgement["g2_user_prompt"]
        del judgement["g1_winner"]
        del judgement["g2_winner"]
        del judgement["model_1"]
        del judgement["model_2"]
        del judgement["g1_judgment"]
        del judgement["g2_judgment"]
        for k in ["orig_prompt1","orig_prompt2"]:
            if k in judgement:
                del judgement[k]              

    return jcopy

def filter_judgments(judgments, model_list):
    def getkey(j):
        k=j["model_1"]+"-"+j["model_2"] if j["model_1"]<j["model_2"] else j["model_2"]+"-"+j["model_1"]
        k=str(j["question_id"])+"-"+k+"-"+str(j["turn"])
        return k

    seen={}
    for j in judgments:
        k=getkey(j)
        if j["model_1"] in model_list and j["model_2"] in model_list and not k in seen:
            seen[k]=1
            yield(j)

def main(judgement_file=None, fix=False, write_errors=False, prepare_for_agreement=False, judgement_dir="data/mt_bench/model_judgment/" , filter_models="gpt-4 vicuna-13b-v1.2 claude-v1 alpaca-13b llama-13b gpt-3.5-turbo"):
    input_file = judgement_dir + "/" + judgement_file 
    assert(os.path.exists(input_file))

    filter_models_list=filter_models.split(" ")
    print(f"filtering for models:\n {json.dumps(filter_models_list, indent=2)}")

    print(f"Reading {input_file}")
    judgments=read_jsonl(input_file)
    judgments=list(filter_judgments(judgments, filter_models_list))

    judgments_fixed, changes = fix_judgements(deepcopy(judgments))

    fixed=False
    for j in [judgments, judgments_fixed]:
        print("before fix" if not fixed else "after fix")

        stats, judgements_error=getJudgementStats(j)
        for key in stats:
                print(f" {key}: {json.dumps(stats[key],indent=2)} ({round(stats[key]/stats['total']*100,2)}%)")
        print("")

        if fix:
            out_fn=input_file.split(".")[0]+"_fixed.jsonl"
            write_jsonl(out_fn, j)
            print(f"\nWrote fixed judgements to {out_fn}")

        if prepare_for_agreement:
            jprepared=prepare_for_mtbench_agreement(j)
            out_fn=input_file.split(".")[0]+"_agreement-ready.json"
            write_jsonl(out_fn, jprepared)
            print(f"Wrote judgements prepared for MT-bench agreement analysis to {out_fn}")

        if write_errors:
            out_fn=input_file.split(".")[0]+"_errors.json"
            write_pretty_json(out_fn, judgements_error)
            print(f"Wrote remaining error judgements to {out_fn}")

        fixed=True

    # for i in range(len(changes)):
    #     c=changes[i]
    #     print(f"\nChange #{i}")
    #     for key in c:
    #         print(f" {key}: {json.dumps(c[key],indent=2)}")
if __name__ == '__main__':
    fire.Fire(main)
