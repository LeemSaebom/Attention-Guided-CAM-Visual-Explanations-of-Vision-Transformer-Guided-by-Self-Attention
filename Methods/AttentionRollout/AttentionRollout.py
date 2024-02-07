""" Attention rollout introduced in "Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. arXiv preprint arXiv:2005.00928."
The implementation is from "https://github.com/jacobgil/vit-explain" by Jacob Gildenblat.
"""

import torch



def rollout(attentions, discard_ratio, head_fusion, device='cpu'):
    result = torch.eye(attentions[0].size(-1)).to(device)
    with torch.no_grad():
        for attention in attentions: 
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1) #[1, 38809] = [1, (197x197)]
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            
            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    mask = result[0, 0 , 1 :]    
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width)

    # modified to have the same shape as the result of other methods. 
    mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(0)
    return mask

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.0, device='cpu'):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention) 

        self.attentions = []
        self.device=device

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        return prediction, rollout(self.attentions, self.discard_ratio, self.head_fusion)
    
    def generate(self, input, label=None):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input)
        _, prediction = torch.max(output, 1)
        return prediction, rollout(self.attentions, self.discard_ratio, self.head_fusion, device=self.device)