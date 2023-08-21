# Original FastChat

| [**Demo**](https://chat.lmsys.org/) | [**Chatbot Arena**](https://arena.lmsys.org) | [**Discord**](https://discord.gg/HSWAKCrnFx) | [**Twitter**](https://twitter.com/lmsysorg) |

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. The core features include:

- The weights, training code, and evaluation code for state-of-the-art models (e.g., Vicuna).
- A distributed multi-model serving system with web UI and OpenAI-compatible RESTful APIs.

# This fork

### 5 models 
* gpt-4
* vicuna-13b-v1.2
* claude-v1
* alpaca-13b
* llama-13b 

## Setup
```bash
git clone https://github.com/geronimi73/FastChat

# Download model answers 

# Start oobabooga/text-generation-webui and load the judge model upstage_Llama-2-70b-instruct-v2 

# Generate judgments by judge upstage_Llama-2-70b-instruct-v2
# Note: no need to do this again, data is already in the fork: data/mt_bench/model_judgment/llama2_pair.jsonl

# download GPT-4 judgements
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment

wget https://huggingface.co/spaces/lmsys/mt-bench/resolve/89039fbb04009aa54b2ce88a294402fe9d756913/data/mt_bench/model_judgment/gpt-4_pair.jsonl


```

## Figure 1: Agreement GPT-4 and llama 2 with human evaluation 
see https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#agreement-computation

```bash
# download human judgements
# https://colab.research.google.com/drive/1ctgygDRJhVGUJTQy8-bRZCl1WNcT8De6?usp=sharing
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/

python3 -c "\
from datasets import load_dataset; \
dataset = load_dataset(\"lmsys/mt_bench_human_judgments\");\
dataset[\"human\"].to_json(\"human_judgments.json\");"

# transform the GPT-4 and llama2 judgments to agreement-ready format
# (output format of gen_judgment.py is slightly different than the input format needed for compute_agreement.py)

## GPT-4: additionally filter GPT-4 judgment file to the models used in this study
cd ../../../
python3 check_judgements.py llama2_pair.jsonl --prepare_for_agreement=True --filter_models "gpt-4 vicuna-13b-v1.2 claude-v1 alpaca-13b llama-13b gpt-3.5-turbo"

## Llama 2 (upstage_Llama-2-70b-instruct-v2):
python3 check_judgements.py gpt-4_pair.jsonl --prepare_for_agreement=True

# Show agreement

## gpt4/human
python3 compute_agreement.py --judges gpt-4 human --votefiles data/mt_bench/model_judgment/human_judgments.json data/mt_bench/model_judgment/gpt-4_pair_agreement-ready.json

## llama2/human
python3 compute_agreement.py --judges upstage_Llama-2-70b-instruct-v2 human --votefiles data/mt_bench/model_judgment/human_judgments.json data/mt_bench/model_judgment/llama2_pair_agreement-ready.json

## human/human
python3 compute_agreement.py --judges human human --votefiles data/mt_bench/model_judgment/human_judgments.json 
```

## Figure 1, panel on the right: Correlation GPT-4 and Llama2 all-against-all
```bash
# GPT-4
python3 show_result.py --mode pairwise-all --input-file data/mt_bench/model_judgment/gpt-4_pair.jsonl --model-list gpt-4 vicuna-13b-v1.2 claude-v1 alpaca-13b llama-13b gpt-3.5-turbo

# Llama 2 (upstage_Llama-2-70b-instruct-v2):
python3 show_result.py --mode pairwise-all --input-file data/mt_bench/model_judgment/llama2_pair.jsonl
```

## Figure 2: Judgment quality
```bash
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment

# GPT-4
python3 check_judgements.py gpt-4_pair.jsonl

## Llama 2 (upstage_Llama-2-70b-instruct-v2):
python3 check_judgements.py llama2_pair.jsonl
```

## Figure 3: Winrate against GPT-3.5 of Guanaco 7B checkpoints
* download the checkpoints from https://huggingface.co/g-ronimo/llama-2-7b-guanaco-2023-08-04_misunderstood-lion
* merge them with the base model, obtained here: https://huggingface.co/meta-llama/Llama-2-7b

```bash
# generate model answers
dir="/models/loras"
modelprefixes="llama-2-7b-guanaco-2023-08-04"

IFS=$'\n'
for model in $(cd $dir;ls -1 | grep ${modelprefixes} | sed "s/ /\\\ /g")
do
    echo ${model}
    python gen_model_answer.py --model-path ${dir}/${model} --model-id ${model}
done

# generate GPT-4 judgments
export OPENAI_API_KEY=NOT_A_REAL_KEY
models="alpaca-13b \
gpt-4 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3500"

python3 gen_judgment.py --mode pairwise-baseline --model-list alpaca-13b gpt-4 --parallel 4

# start oobabooga/text-generation-webui with a 4 bit quantized version of upstage_Llama-2-70b-instruct-v2
cd text-generation-webui
python3 server.py --model Upstage-Llama-2-70B-instruct-v2-GPTQ --api --listen --chat --no_inject_fused_attention
python3 server.py --model Upstage-Llama-2-70B-instruct-v2-GPTQ --api --listen --chat --no_inject_fused_attention 

# add reference answers

# create instruct prompt format

# generate Llama2 judgments
models="alpaca-13b \
gpt-4 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3500"

python3 gen_judgment.py --mode pairwise-baseline --model-list alpaca-13b gpt-4 --parallel 1 --use-api True --judge-model upstage_Llama-2-70b-instruct-v2

```
