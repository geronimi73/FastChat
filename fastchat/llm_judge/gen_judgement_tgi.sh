# models="alpaca-13b \
#  gpt-4 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500"

# python3 gen_judgment.py --mode pairwise-baseline --model-list ${models} --judge-model text-generation-webui-api --parallel 1

# models="llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1000 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1500 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2000 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2500"

# python3 gen_judgment.py --mode pairwise-baseline --model-list ${models} --judge-model text-generation-webui-api --parallel 1

models="alpaca-13b \
 gpt-4 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1000 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1500 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2000 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2500 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3000 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3500 \
 llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500"

python3 gen_judgment.py --mode pairwise-baseline --model-list ${models} --judge-model text-generation-webui-api --parallel 1


# a="alpaca-13b \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500 \
#  gpt-4 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1000 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-1500 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2000 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-2500 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3000 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-3500 \
#  llama-2-7b-guanaco-2023-08-04_misunderstood-lion-checkpoint-500"

# echo ${a}