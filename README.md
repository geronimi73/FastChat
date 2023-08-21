# Original FastChat

| [**Demo**](https://chat.lmsys.org/) | [**Chatbot Arena**](https://arena.lmsys.org) | [**Discord**](https://discord.gg/HSWAKCrnFx) | [**Twitter**](https://twitter.com/lmsysorg) |

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. The core features include:

- The weights, training code, and evaluation code for state-of-the-art models (e.g., Vicuna).
- A distributed multi-model serving system with web UI and OpenAI-compatible RESTful APIs.

# This fork

### 5 models

* gpt-4
* gpt-3.5-turbo
* vicuna-13b-v1.2
* claude-v1
* alpaca-13b
* llama-13b 

## Figure 1: Agreement of judges GPT-4 and llama 2 with human evaluation

```bash
# 1: download human judgements
# https://colab.research.google.com/drive/1ctgygDRJhVGUJTQy8-bRZCl1WNcT8De6?usp=sharing
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/

python3 -c "\
from datasets import load_dataset; \
dataset = load_dataset(\"lmsys/mt_bench_human_judgments\");\
dataset[\"human\"].to_json(\"human_judgments.json\");"

# 2: download GPT-4 judgments
wget https://huggingface.co/spaces/lmsys/mt-bench/resolve/89039fbb04009aa54b2ce88a294402fe9d756913/data/mt_bench/model_judgment/gpt-4_pair.jsonl

# 3: transform the GPT-4 and llama2 judgments to agreement-ready format (output format of gen_judgment.py is slightly different than the input format needed for compute_agreement.py)

## GPT-4: transform, and additionally filter GPT-4 judgment file for the models used in this study
cd ../../../
python3 check_judgements.py llama2_pair.jsonl --prepare_for_agreement=True --filter_models "gpt-4 vicuna-13b-v1.2 claude-v1 alpaca-13b llama-13b gpt-3.5-turbo"

## Llama 2 (upstage_Llama-2-70b-instruct-v2):
python3 check_judgements.py gpt-4_pair.jsonl --prepare_for_agreement=True

# 4: Show agreement

## gpt4/human
python3 compute_agreement.py --judges gpt-4 human --votefiles data/mt_bench/model_judgment/human_judgments.json data/mt_bench/model_judgment/gpt-4_pair_agreement-ready.json

## llama2/human
python3 compute_agreement.py --judges upstage_Llama-2-70b-instruct-v2 human --votefiles data/mt_bench/model_judgment/human_judgments.json data/mt_bench/model_judgment/llama2_pair_agreement-ready.json

## human/human
python3 compute_agreement.py --judges human human --votefiles data/mt_bench/model_judgment/human_judgments.json
```

## Figure 1, blue panel on the right: Correlation GPT-4 and Llama2 all-against-all

```bash
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/

# Show results for GPT-4
python3 show_result.py --mode pairwise-all --input-file data/mt_bench/model_judgment/gpt-4_pair.jsonl --model-list gpt-4 vicuna-13b-v1.2 claude-v1 alpaca-13b llama-13b gpt-3.5-turbo

# Show results for Llama 2 (upstage_Llama-2-70b-instruct-v2):
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

## Figure 3: Winrate against GPT-3.5 for Guanaco 7B checkpoints

```bash
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment

# GPT-4
python3 show_result.py --mode pairwise-baseline --input-file data/mt_bench/model_judgment/guanaco_gpt-4_pair.jsonl

## Llama 2 (upstage_Llama-2-70b-instruct-v2):
python3 show_result.py --exclude-ties-and-errors --mode pairwise-baseline --judge-model upstage_Llama-2-70b-instruct-v2 --input-file data/mt_bench/model_judgment/guanaco_llama2_pair.jsonl
```

## Run judgements

- ***Note***: everything that follows below is only needed if you want to rerun the judgments using another judge or other models' answers; the judgment data to produce the figures are already included in this fork

### Setup FastChat

```bash
# Download model answers
cd FastChat/fastchat/llm_judge
python3 download_mt_bench_pregenerated.py  # downloads everything except the vicuna-13b-v1.2 data
wget https://huggingface.co/spaces/lmsys/mt-bench/raw/eb60c015e9c4dad0cbdb01c905067ec8b0973fd7/data/mt_bench/model_answer/vicuna-13b-v1.2.jsonl -O data/mt_bench/model_answer/vicuna-13b-v1.2.jsonl        # deleted, old commit

# Create reference answers for upstage_Llama-2-70b-instruct-v2 (same as for GPT-4)
cd FastChat/fastchat/llm_judge/data/mt_bench/model_judgment
cp gpt-4.jsonl upstage_Llama-2-70b-instruct-v2.jsonl
```

### Generate model answers

#### All-against-all analysis (Figures 1 and 2)

```bash
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
```

#### Guanaco analysis (Figure 3)

- download the checkpoints from [g-ronimo/llama-2-7b-guanaco-2023-08-04_misunderstood-lion · Hugging Face](https://huggingface.co/g-ronimo/llama-2-7b-guanaco-2023-08-04_misunderstood-lion)
- merge them with the base model, obtained here: [meta-llama/Llama-2-7b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b)
- generate answers from each checkpoint:

```bash
dir="models"
modelprefixes="llama-2-7b-guanaco-2023-08-04"

IFS=$'\n'
for model in $(cd $dir;ls -1 | grep ${modelprefixes} | sed "s/ /\\\ /g")
do
    echo ${model}
    python gen_model_answer.py --model-path ${dir}/${model} --model-id ${model}
done
```

### Setup oobabooga/text-generation-webui to serve the judge via API

- Setup text-generation-webui as shown below

- Download judge model, I used [TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ at gptq-4bit-32g-actorder_True](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ/tree/gptq-4bit-32g-actorder_True)

```bash
# clone oobabooga/text-generation-webui
git clone https://github.com/oobabooga/text-generation-webui

# create new instruction template: Orca-Hashes
echo """user: \"### User:\"
bot: \"### Assistant:\"
turn_template: \"<|user|>\n<|user-message|>\n\n<|bot|>\n<|bot-message|>\n\n\"
context: \"### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n\"""" > Orca-Hashes.yaml

# copy Orca-Hashes.yaml to text-generation-webui/instruction-templates or characters/instruction-following, depending on your version of text-generation-webui

# Start oobabooga/text-generation-webui, load the judge model upstage_Llama-2-70b-instruct-v2 
python3 server.py --model Upstage-Llama-2-70B-instruct-v2-GPTQ --public-api --api --no_inject_fused_attention 
```

### Generate judgments

#### All-against-all analysis (Figures 1 and 2)

```bash
models="gpt-4 vicuna-13b-v1.2 claude-v1 llama-13b gpt-3.5-turbo alpaca-13b"

python3 gen_judgment.py --mode pairwise-baseline --model-list $models --parallel 1 --use-api True --judge-model upstage_Llama-2-70b-instruct-v2
```

#### Guanaco analysis (Figure 3)

```bash
models="alpaca-13b \
gpt-4 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2500 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3000 \
llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3500"

python3 gen_judgment.py --mode pairwise-baseline --model-list $models --parallel 1 --use-api True --judge-model upstage_Llama-2-70b-instruct-v2
```

### 
