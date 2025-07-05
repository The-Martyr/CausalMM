import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

from transformers import set_seed
from causalmm_cf.causalmm_sm import evolve_causalmm_sampling
evolve_causalmm_sampling()

def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    output_dir = os.path.join(args.output_folder, "your_results") #change to your output directory
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(args.data_folder):
        file_path = os.path.join(args.data_folder, filename)
        output_file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'r') as fin, open(output_file_path, 'w') as fout:
            lines = fin.read().splitlines()
            for line in tqdm(lines, desc=f"Processing {filename}"):
                img, question, gt_answer = line.strip().split('\t')
                img_path = os.path.join(args.image_folder, filename.replace('.txt', ''), img)
                assert os.path.exists(img_path), f"Image not found: {img_path}"

                if model.config.mm_use_im_start_end:
                    question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                else:
                    question = DEFAULT_IMAGE_TOKEN + '\n' + question

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], question + " Please answer this question with one word.")
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt', padding=True)['pixel_values'][0]

                # Set up stopping criteria and generation
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        gamma = args.gamma,
                        epsilon = args.epsilon,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=1024,
                        use_cache=True)

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                
                print(img, question, gt_answer, outputs, sep='\t', file=fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="MME_Benchmark_release_version/images")
    parser.add_argument("--data-folder", type=str, default="MME_Benchmark_release_version/Your_Results")
    parser.add_argument("--output-folder", type=str, default="MME_Benchmark_release_version/eval_tool")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
