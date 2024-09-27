from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed.nn
import torch.distributed as dist
from typing import Sequence



class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


#???
# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # print(f"x type: {type(x)}, dtype: {x.dtype}")
        # print(f"normalized_shape type: {type(self.normalized_shape)}")
        # print(f"weight type: {type(self.weight)}, dtype: {self.weight.dtype if self.weight is not None else 'None'}")
        # print(f"bias type: {type(self.bias)}, dtype: {self.bias.dtype if self.bias is not None else 'None'}")
        # print(f"eps type: {type(self.eps)}")
        x = F.layer_norm(x.to(self.weight.dtype), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


    
class MoEResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,MoE_args=None):
        super().__init__()
        self.num_experts=MoE_args[0]
        self.top_k=MoE_args[1]
        self.dropout=MoE_args[2]
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.experts=nn.ModuleList([nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(p=self.dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)  )
        ])) for _ in range(self.num_experts)])
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.gate=nn.Linear(d_model, self.num_experts, bias=False)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        hidden_states=self.ln_2(x)
        router_logits = self.gate(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # if self.training and self.jitter_noise > 0:
        #     hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        #selected_experts batch_size * sequence_length, top_k
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        #expert_mask num_experts, top_k, batch_size * sequence_length

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            #idx: which top. i.e.,top0 or top1 or top2
            #top_x: which sample does this expert need to process

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
      
            #current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_state = hidden_states[top_x].reshape(-1, hidden_dim)
            #routing_weights: batch_size * sequence_length, top_k
            #current_state: num_tokens_for_expert, hidden_dim
            #routing_weights[top_x, idx]: num_tokens_for_expert,
            #routing_weights[top_x, idx, None]: num_tokens_for_expert, 1
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        #return final_hidden_states, router_logits
        #x = x + self.mlp(self.ln_2(x))
        return x+final_hidden_states,router_logits



def load_balancing_loss_func(gate_logits: torch.Tensor, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits :
            Logits from the `gate`, should be a tensors of
            shape [model.config.num_hidden_layers,batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
   

    # if isinstance(gate_logits, tuple):
    #     compute_device = gate_logits[0].device
    #     concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    num_experts=gate_logits.shape[-1]
    concatenated_gate_logits=gate_logits.view(-1, num_experts)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class MoETransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,MoE_args=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.moe_layers=MoE_args[-1]
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers-self.moe_layers)]
            +[MoEResidualAttentionBlock(width, heads, attn_mask,MoE_args) for _ in range(self.moe_layers)])

    def forward(self, x: torch.Tensor):
        router_logits=[]
        for i,block in enumerate(self.resblocks):
            if i<self.layers-self.moe_layers:
                x=block(x)
            else:
                x,router_logit = block(x)
                router_logits.append(router_logit) 
       
        
        return x,torch.stack(router_logits)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,MoE_args=0):
        super().__init__()
        self.MoE_args=MoE_args
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        if MoE_args:
            self.transformer = MoETransformer(width, layers, heads,MoE_args=MoE_args)
        else:
            self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        router_logits=None
        if self.MoE_args:
            x,router_logits=self.transformer(x)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        if self.MoE_args:
            return x,router_logits
        else:
            return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 load_from_clip: bool,
                 MoE_args=None,
                 use_short_text=False
                 ):
        super().__init__()
        self.MoE_args=MoE_args

        self.context_length = 248
        self.use_short_text=use_short_text

    
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            MoE_args=MoE_args
        )

        if self.MoE_args:
            if self.MoE_args[-1]>=transformer_layers:
                self.MoE_args[-1]=transformer_layers
            self.transformer = MoETransformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                MoE_args=MoE_args
            )
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        if load_from_clip == False:
            self.positional_embedding = nn.Parameter(torch.empty(248, transformer_width))
            self.positional_embedding_res = nn.Parameter(torch.empty(248, transformer_width))

        else:
            self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1

    def lock_text_tower(self):
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        # x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device) 
        # router_logits=None
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # if self.MoE_args:
        #     x,router_logits=self.transformer(x)
        # else:
        #     x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)

        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # 冻结 text_projection 层的所有参数
        self.text_projection.requires_grad = False
        # 冻结 token_embedding 层的所有参数
        self.token_embedding.requires_grad = False

        # 冻结 positional_embedding 层的所有参数
        self.positional_embedding.requires_grad = False
        # 冻结 transformer 层的所有参数
        for param in self.transformer.parameters():
            param.requires_grad = False

        # 冻结 ln_final 层的所有参数
        for param in self.ln_final.parameters():
            param.requires_grad = False
  
        

    def lock_except_mlp(self,unlocked_groups=0):

        for name, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.visual.transformer.named_parameters():
            layer_id = int(name.split('resblocks.')[-1].split('.')[0])
            if layer_id >=unlocked_groups:
                if 'mlp' in name:
                    param.requires_grad=True
        for name, param in self.transformer.named_parameters():
            if 'mlp' in name:
                param.requires_grad=True
         

        
            

    def lock_except_gate(self):
        for name, param in self.named_parameters():
            if 'gate' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        if not self.MoE_args:
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image,router_output=False):
        if self.MoE_args:
            x,router_logits=self.visual(image.type(self.dtype))
        else:
            return self.visual(image.type(self.dtype))

        if self.training or router_output:
            return x,router_logits
        else:
            return x

    def encode_text(self, text,router_output=False): 
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device) 
        router_logits=None
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.MoE_args:
            x,router_logits=self.transformer(x)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if self.MoE_args:
            if self.training or router_output:
                return x,router_logits
            else:
                return x
        else:
            return x


    def forward(self, image, text_long,text_short,rank):
        if self.MoE_args:
            image_features_long,image_routerLogits_long = self.encode_image(image)
            text_features_long,text_routerLogits_long = self.encode_text(text_long)
        else:
            image_features_long = self.encode_image(image)
            text_features_long = self.encode_text(text_long)
            

        # normalized features
        image_features_long = image_features_long / image_features_long.norm(dim=1, keepdim=True)
        text_features_long = text_features_long / text_features_long.norm(dim=1, keepdim=True)
            
        image_feat_all_long = torch.cat(torch.distributed.nn.all_gather(image_features_long), dim=0)#gather with grad
        text_feat_all_long = torch.cat(torch.distributed.nn.all_gather(text_features_long), dim=0)

        
        sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
        sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
        sim_tl2i = sim_tl2i.T

        sim_i2tl = self.logit_scale.exp() * sim_i2tl
        sim_tl2i = self.logit_scale.exp() * sim_tl2i

        bs = image.size(0)
        targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)
        
        loss_itcl = (
                F.cross_entropy(sim_i2tl, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.1)
            ) / 2
        #todo-------
        loss_itcs=0
        if self.use_short_text:
            if self.MoE_args:
                text_features_short,text_routerLogits_short = self.encode_text(text_short)
            else:
                text_features_short = self.encode_text(text_short)
            text_features_short = text_features_short / text_features_short.norm(dim=1, keepdim=True)
            text_feat_all_short = torch.cat(torch.distributed.nn.all_gather(text_features_short), dim=0)
            sim_i2ts = torch.matmul(image_features_long, text_feat_all_short.T)
            sim_ts2i = torch.matmul(image_feat_all_long, text_features_short.T)
            sim_ts2i = sim_ts2i.T
            sim_i2ts = self.logit_scale.exp() * sim_i2ts
            sim_ts2i = self.logit_scale.exp() * sim_ts2i
            loss_itcs = (
                    F.cross_entropy(sim_i2ts, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.1)
                ) / 2
        if self.MoE_args:
            #todo
            image_routerLoss_long=load_balancing_loss_func(image_routerLogits_long, top_k=self.MoE_args[1])
            image_routerLoss_long /= dist.get_world_size()
            text_routerLoss_long=load_balancing_loss_func(text_routerLogits_long, top_k=self.MoE_args[1])
            text_routerLoss_long /= dist.get_world_size()
            text_routerLoss_short=0
            if self.use_short_text:
                text_routerLoss_short=load_balancing_loss_func(text_routerLogits_short, top_k=self.MoE_args[1])
                text_routerLoss_short /= dist.get_world_size()
            return loss_itcl, loss_itcs,image_routerLoss_long,text_routerLoss_long,text_routerLoss_short
        else:
            return loss_itcl, loss_itcs
        


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, load_from_clip: bool, MoE_args=None):
    #MoE_args: None for standard non-MoE model. [num_MoE_experts,top_k]
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, load_from_clip,MoE_args
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if not MoE_args:
        convert_weights(model)
        model.load_state_dict(state_dict)
        return model.eval()
    else:
        #only build the MoE model, do not load state dict
        convert_weights(model)
        return model.eval()
