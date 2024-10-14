# CausalMM: Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality
<p align="center" width="100%">
<a target="_blank"><img src="imgs/pipeline.png" alt="CausalMM" style="width: 35%; min-width: 200px; display: block; margin: auto;"></a>
</p>

The official repo for CausalMM, a plug-and-play method for deciphering attention causality in MLLMs. 
Full paper can be found at: [https://arxiv.org/abs/2410.04780](https://arxiv.org/abs/2410.04780).

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2410.04780'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='LICENCE'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
</div>

## Introdection
Multimodal Large Language Models (MLLMs) have emerged as a central focus in both industry and academia, but often suffer from biases introduced by visual and language priors, which can lead to multimodal hallucination. These biases arise from the visual encoder and the Large Language Model (LLM) backbone, affecting the attention mechanism responsible for aligning multimodal inputs. Existing decoding-based mitigation methods focus on statistical correlations and overlook the causal relationships between attention mechanisms and model output, limiting their effectiveness in addressing these biases. To tackle this issue, we propose a causal inference framework termed CausalMM that applies structural causal modeling to MLLMs, treating modality priors as a confounder between attention mechanisms and output. Specifically, by employing backdoor adjustment and counterfactual reasoning at both the visual and language attention levels, our method mitigates the negative effects of modality priors and enhances the alignment of MLLM's inputs and outputs, with a maximum score improvement of 65.3% on 6 VLind-Bench indicators and 164 points on MME Benchmark compared to conventional methods. Extensive experiments validate the effectiveness of our approach while being a plug-and-play solution.

## Update
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
@article{zhou2024mitigating,
  title={Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality},
  author={Zhou, Guanyu and Yan, Yibo and Zou, Xin and Wang, Kun and Liu, Aiwei and Hu, Xuming},
  journal={arXiv preprint arXiv:2410.04780},
  year={2024}
}
```

## Acknowledgement
* [VCD](https://github.com/DAMO-NLP-SG/VCD)
* [OPEAR](https://github.com/shikiw/OPERA?tab=readme-ov-file)
* [LLaVA](https://github.com/haotian-liu/LLaVA)
* [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)

