dir="/home/g/models/loras"
modelprefixes="llama-2-7b-guanaco-2023-08-04"

IFS=$'\n'
for model in $(cd $dir;ls -1 | grep ${modelprefixes} | sed "s/ /\\\ /g")
do
	echo ${model}
	python gen_model_answer.py --model-path ${dir}/${model} --model-id ${model}
done