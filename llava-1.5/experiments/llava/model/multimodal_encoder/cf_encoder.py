import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import math

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    
    #counterfactual attn
    def edit_attention(self, attention_maps, method='shuffle'):
        batch_size, num_heads, height, width = attention_maps.shape

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

    def get_attentions(self, image_forward_outs):
        """
        Get attention maps from the model.
        
        Parameters:
        - image_forward_outs: the output of the vision model forward pass

        Returns:
        - attentions: List of attention maps from each layer
        """
        attentions = image_forward_outs.attentions  # Directly access the attentions from the forward output
        
        # (batch_size, num_heads, seq_length)
        if len(attentions[0].shape) == 3:
            seq_length = attentions[0].shape[-1]
            grid_size = int(seq_length ** 0.5)  # Assuming it's a square grid
            attentions = [attn.view(attn.shape[0], attn.shape[1], grid_size, grid_size) for attn in attentions]
        
        return attentions

    def apply_attention(self, image_features, attention_maps):
        """
        Apply attention maps to image features by averaging over multiple heads.
        
        Parameters:
        - image_features: torch.Tensor of shape (batch_size, height_img, width_img, hidden_dim)
        - attention_maps: torch.Tensor of shape (batch_size, num_heads, height_attn, width_attn)
        
        Returns:
        - weighted_features: torch.Tensor of shape (batch_size, height_img, width_img, hidden_dim)
        """
        # Resize
        height_img, width_img = int(math.sqrt(image_features.shape[1])), int(math.sqrt(image_features.shape[1]))
        attention_maps_resized = F.interpolate(attention_maps, size=(height_img, width_img), mode='bilinear', align_corners=False)
        
        # Average over all attention heads to get a single attention map
        averaged_attention_map = torch.mean(attention_maps_resized, dim=1) 
        
        #match the features dimension
        averaged_attention_map = averaged_attention_map.view(averaged_attention_map.size(0), -1, 1) 
        averaged_attention_map = averaged_attention_map.expand(-1, -1, image_features.shape[-1])  

        # Apply attention map to image features
        weighted_features = image_features * averaged_attention_map
        weighted_features = weighted_features.to(torch.float16)

        return weighted_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            weighted_features = []
            for i, image in enumerate(images):
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                
                attention_maps = image_forward_out.attentions[self.select_layer]  # attn
                edited_attention_maps = self.edit_attention(attention_maps)  # edit
                
                weighted_feature = self.apply_attention(image_feature, edited_attention_maps)  # apply
                
                weighted_features.append(weighted_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            
            image_features = image_forward_outs.hidden_states[self.select_layer]       

            image_features = image_features[:, 1:].to(images.dtype)
    
            attention_maps = self.get_attentions(image_forward_outs)
            edited_attention_maps = self.edit_attention(attention_maps[self.select_layer])  # select attn
            weighted_features = self.apply_attention(image_features, edited_attention_maps)  # apply attn
        
        return image_features, weighted_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    


    