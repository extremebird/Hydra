import torch
from torch import nn
import timm
import random
import torch.nn.functional as F

    
def forward_vit_hydra_ffn(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    par = self.hydra_mlp_par(x)
    x = self.fc2(x)
    seq = self.hydra_mlp_seq(x)
    x = self.drop2(x + par + seq)
    return x

def forward_vit_hydra_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    par = self.hydra_proj_par(x)
    x = self.proj(x)
    seq = self.hydra_proj_seq(x)
    x = self.proj_drop(x+par+seq)
    return x

class Hydra_layer(nn.Module):

    def __init__(
            self, in_features=768, hidden_dim=8, out_features=768, scale=1, do=0.0
    ):
        super().__init__()
        self.down_proj = nn.Linear(in_features, hidden_dim, bias=True)
        self.up_proj = nn.Linear(hidden_dim, out_features, bias=True)
        self.dropout=nn.Dropout(do)
        self.scale=scale

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        x=self.up_proj(self.dropout(self.down_proj(x)))*self.scale
        return x



def reparameterize(W0, b0, Wp, bp, Ws, bs, scale=1):
    weight = W0 + scale*(Wp + W0@Ws)
    bias = 0
    if b0 is not None:
        bias += b0 + scale*(b0@Ws)
    if bp is not None:
        bias += scale*bp
    if bs is not None:
        bias += scale*bs
    return weight.T, bias if isinstance(bias, torch.Tensor) else None


def set_hydra(model, method, configs, dim=2, set_forward=True):
    if method == 'hydra_both':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.mlp.hydra_mlp_par = Hydra_layer(in_features=4*768, hidden_dim=dim, scale=configs['scale'], do=configs['dropout'])
                _.mlp.hydra_mlp_seq = Hydra_layer(hidden_dim=dim,scale=configs['scale'], do=configs['dropout'])
                _.attn.hydra_proj_par = Hydra_layer(in_features=768, hidden_dim=dim, scale=configs['scale'], do=configs['dropout'])
                _.attn.hydra_proj_seq = Hydra_layer(in_features=768, hidden_dim=dim, scale=configs['scale'], do=configs['dropout'])
                bound_method_ffn = forward_vit_hydra_ffn.__get__(_.mlp, _.mlp.__class__)
                bound_method_attn = forward_vit_hydra_attn.__get__(_.attn, _.attn.__class__)
                if set_forward:
                    setattr(_.mlp, 'forward', bound_method_ffn)
                    setattr(_.attn, 'forward', bound_method_attn)
            elif len(list(_.children())) != 0:
                set_hydra(_, method, configs, dim, set_forward=set_forward)

    else:
        raise NotImplementedError()


def set_hydraWeight(model, method, configs, dim=2):
    if method == 'hydra_both':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                weight_parallel = _.mlp.hydra_mlp_par.down_proj.weight.T @  _.mlp.hydra_mlp_par.up_proj.weight.T
                bias_parallel = _.mlp.hydra_mlp_par.down_proj.bias @  _.mlp.hydra_mlp_par.up_proj.weight.T + _.mlp.hydra_mlp_par.up_proj.bias
                weight_sequential = _.mlp.hydra_mlp_seq.down_proj.weight.T @  _.mlp.hydra_mlp_seq.up_proj.weight.T
                bias_sequential = _.mlp.hydra_mlp_seq.down_proj.bias @  _.mlp.hydra_mlp_seq.up_proj.weight.T + _.mlp.hydra_mlp_seq.up_proj.bias
                hydra_scale = _.mlp.hydra_mlp_seq.scale
                ffn_weight, ffn_bias=reparameterize(_.mlp.fc2.weight.T,_.mlp.fc2.bias, weight_parallel, bias_parallel, 
                                                    weight_sequential, bias_sequential, hydra_scale)

                with torch.no_grad():
                    _.mlp.fc2.weight.copy_(ffn_weight)
                    _.mlp.fc2.bias.copy_(ffn_bias)
                
                weight_parallel = _.attn.hydra_proj_par.down_proj.weight.T @  _.attn.hydra_proj_par.up_proj.weight.T
                bias_parallel = _.attn.hydra_proj_par.down_proj.bias @  _.attn.hydra_proj_par.up_proj.weight.T + _.attn.hydra_proj_par.up_proj.bias
                weight_sequential = _.attn.hydra_proj_seq.down_proj.weight.T @  _.attn.hydra_proj_seq.up_proj.weight.T
                bias_sequential = _.attn.hydra_proj_seq.down_proj.bias @  _.attn.hydra_proj_seq.up_proj.weight.T + _.attn.hydra_proj_seq.up_proj.bias
                hydra_scale = _.attn.hydra_proj_seq.scale
                attn_weight, attn_bias=reparameterize(_.mlp.fc2.weight,_.mlp.fc2.bias, weight_parallel.T, bias_parallel, 
                                                    weight_sequential.T, bias_sequential, hydra_scale)

                with torch.no_grad():
                    _.attn.proj.weight.copy_(attn_weight)
                    _.attn.proj.bias.copy_(attn_bias)

            elif len(list(_.children())) != 0:
                set_hydraWeight(_, method, configs, dim)
    else:
        raise NotImplementedError()