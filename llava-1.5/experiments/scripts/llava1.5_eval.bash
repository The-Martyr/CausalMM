seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"random"}
model_path=${4:-"/your_path/llava-v1.5-7b"}
gamma=${5:-1.0}
epsilon=${6:-0.6}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/your_path/coco/val2014
else
  image_folder=/your_path/gqa/images
fi

python /your_path/llava-1.5/experiments/eval/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file /your_path/llava-1.5/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file /your_path/llava-1.5/experiments/output/llava15_${dataset_name}_pope_${type}_answers.jsonl \
--gamma $gamma \
--epsilon $epsilon \
--seed ${seed}

python /your_path/llava-1.5/experiments/eval/eval_pope.py \
--gt_files /your_path/llava-1.5/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--gen_files /your_path/llava-1.5/experiments/output/llava15_${dataset_name}_pope_${type}_answers.jsonl 


