""" LRP-based method devised for ViT introduced in 
"Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 782-791)."
The implementation is from "https://github.com/hila-chefer/Transformer-Explainability" by Hila Chefer.

"""

import argparse
import torch
import numpy as np
from numpy import *


class LRP:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device=device

    def generate(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        # print(output)
        _, prediction = torch.max(output, 1)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

    
        mask = self.model.relprop(torch.tensor(one_hot_vector).to(input.to(self.device)), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width)
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return prediction, mask
                    
