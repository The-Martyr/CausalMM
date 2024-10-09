# CausalMM: Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality
<p align="center" width="100%">
<a target="_blank"><img src="imgs/pipeline.png" alt="CausalMM" style="width: 35%; min-width: 200px; display: block; margin: auto;"></a>
</p>

The office repo for CausalMM, a plug-and-play method for deciphering attention causality in MLLMs. 
Full paper can be found at: [https://arxiv.org/abs/2410.04780](https://arxiv.org/abs/2410.04780).

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2410.04780'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='LICENCE'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
</div>

## Update
* [2024-10] Detailed instructions coming soon.
* [2024-10] Key code for editing attention released.

## Structural Causal Model
<p align="center" width="100%">
<a target="_blank"><img src="imgs/causal_graph.png" alt="SCM" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Environment Setup
```
cd env
conda env create -f causalmm_llava.yml
conda activate causalmm_llava.yml
```

## Counterfactual Attention
Four methods for generating counterfactual attention (an example):
```
def edit_attention(self, attention_maps, method='shuffle'):
      batch_size, num_heads, height, width = attention_maps.shape　#depends on how the vision encoder extracts attention

      if method == 'random':
      edited_attention_maps = torch.rand(batch_size, num_heads, height, width, device=attention_maps.device) * 2

      elif method == 'uniform':
      avg_value = torch.mean(attention_maps, dim=(2, 3), keepdim=True)
      edited_attention_maps = avg_value.expand(batch_size, num_heads, height, width)

      elif method == 'reversed':
      max_value_height, _ = torch.max(attention_maps, dim=2, keepdim=True)
      
      max_value, _ = torch.max(max_value_height, dim=3, keepdim=True)

      edited_attention_maps = max_value - attention_maps

      elif method == 'shuffle':
      edited_attention_maps = attention_maps.clone()
      for i in range(num_heads):
            edited_attention_maps[:, i] = edited_attention_maps[:, i].view(batch_size, -1).gather(1, torch.randperm(height * width, device=attention_maps.device).expand(batch_size, -1)).view(batch_size, height, width)

      else:
      raise ValueError("Invalid method. Choose from ['random', 'uniform', 'reversed', 'shuffle']")

      return edited_attention_maps
```
The complete experimental code can be found in [cf_encoder](llava-1.5/cf_encoder.py).

## Citation
Welcome to star our repo and cite our work:
```
@misc{zhou2024mitigatingmodalitypriorinducedhallucinations,
      title={Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality}, 
      author={Guanyu Zhou and Yibo Yan and Xin Zou and Kun Wang and Aiwei Liu and Xuming Hu},
      year={2024},
      eprint={2410.04780},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.04780}, 
}
```

## Acknowledgement
* [VCD](https://github.com/DAMO-NLP-SG/VCD)
* [OPEAR](https://github.com/shikiw/OPERA?tab=readme-ov-file)
* [LLaVA](https://github.com/haotian-liu/LLaVA)
* [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)

