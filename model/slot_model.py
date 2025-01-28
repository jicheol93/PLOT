import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum

import pdb



class PositionEmbeddingLearned1D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, w, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)

    def forward(self, x):
        w = x.shape[-2]
        ##print(w)
        i = torch.arange(w, device=x.device)
        y_emb = self.row_embed(i)
        pos = y_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        pos = rearrange(pos, 'b ... d -> b (...) d')

        return pos

class PositionEmbeddingLearned2D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, w, h, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(h, num_pos_feats)
        ##self.row_embed = nn.Embedding(193, num_pos_feats)
        self.col_embed = nn.Embedding(w, num_pos_feats)
        ##self.col_embed = nn.Embedding(1, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-3:-1]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        pos = rearrange(pos, 'b ... d -> b (...) d')

        return pos



def build_position_encoding(dim, pos_type, w, h, axis):
    N_steps = dim // 2
    if pos_type in ('sine'):
        position_embedding = PositionEmbeddingSine1D(dim, normalize=True) if axis == 1 else PositionEmbeddingSine2D(N_steps, normalize=True)
    elif pos_type in ('learned'):
        position_embedding = PositionEmbeddingLearned1D(w,dim) if axis == 1 else PositionEmbeddingLearned2D(w, h, N_steps)
    elif pos_type in ('none'):
        position_embedding = lambda x : None
    else:
        raise ValueError(f"not supported {pos_type}")

    return position_embedding



class MlpDecoder(nn.Module):
    def __init__(self, num_patches_h, num_patches_w, slot_dim, feat_dim, normalizer='softmax', self_attn='False', pos_enc='learned') -> None:
        super().__init__()
        self.width = num_patches_w
        self.height = num_patches_h
        self.pos_emb = build_position_encoding(slot_dim, pos_enc, num_patches_w, num_patches_h, axis=2)
        self.alpha_holder = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, feat_dim+1)
        )
        self.normalizer = {
            'softmax': F.softmax,
        }[normalizer]

        self.self_attn = None
        """
        if self_attn:
            self.self_attn = TransformerLayer(
                query_dim=slot_dim,
                ff_dim=slot_dim,
                context_dim=slot_dim,
                heads=4,
                ff_activation='gelu',
                last_norm=True
            )
        """

    def forward(self, slots):
        if self.self_attn is not None:
            slots = self.self_attn(slots, context=slots)
            
        slots = repeat(slots, 'b s d -> b s h w d', h=self.height, w=self.width)
        slots = rearrange(slots, 'b s h w d -> b s (h w) d') + self.pos_emb(slots[:, 0, :, :, :]).unsqueeze(1)
        feat_decode = self.mlp(slots)
        feat, alpha = feat_decode[:, :, :, :-1], feat_decode[:, :, :, -1]
        alpha = self.alpha_holder(self.normalizer(alpha, dim=1))
        ##JC select top1 slot
        """
        idx = torch.argmax(alpha,dim=1, keepdims=True)
        alpha_mask = torch.zeros_like(alpha).scatter_(1,idx,1.0)
        recon = einsum(feat, alpha_mask, 'b s hw d, b s hw -> b hw d')
        """
        recon = einsum(feat, alpha, 'b s hw d, b s hw -> b hw d')
        return recon


class MlpDecoder_woAlpha(nn.Module):
    def __init__(self, num_patches_h, num_patches_w, slot_dim, feat_dim, normalizer='softmax', pos_enc='learned') -> None:
        super().__init__()
        self.width = num_patches_w
        self.height = num_patches_h
        self.pos_emb = build_position_encoding(slot_dim, pos_enc, num_patches_w, num_patches_h, axis=2)
        self.alpha_holder = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, feat_dim)
        )
        self.normalizer = {
            'softmax': F.softmax,
        }[normalizer]

    def forward(self, slots, attn):
            
        slots = repeat(slots, 'b s d -> b s h w d', h=self.height, w=self.width)
        slots = rearrange(slots, 'b s h w d -> b s (h w) d') + self.pos_emb(slots[:, 0, :, :, :]).unsqueeze(1)
        feat_decode = self.mlp(slots)
        recon = einsum(feat_decode, attn, 'b s hw d, b hw s -> b hw d')
        ##recon = einsum(feat, alpha, 'b s hw d, b s hw -> b hw d')
        return recon


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m



def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m



class SlotAttention_inSlate_backup(nn.Module):
    def __init__(self, num_iter, num_slots,
                 input_size, slot_size, mlp_hidden_size, heads=8,
                 epsilon=1e-8):
        super().__init__()

        self.num_iter = num_iter
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        # Slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

    def forward(self, slots, inputs, mask=None):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        B, N_kv, D_inp = inputs.size()
        N_q, D_slot = slots.size()

        slots = slots.expand(B, -1, -1)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k

        # Multiple rounds of attention.
        for idx in range(self.num_iter):
            """
            if idx == self.num_iter-1:
                slots = slots.detach()
            """
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                             # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            if mask is not None:
                attn_mask = ~mask.unsqueeze(1).repeat(1, attn.size(1), 1)
                attn_mask = attn_mask.unsqueeze(-1).repeat(1,1,1,self.num_slots)
                attn = attn * attn_mask

            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn / (torch.sum(attn, dim=-2, keepdim=True) + self.epsilon)
            updates = torch.matmul(attn.transpose(-1, -2), v)       # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                          # Shape: [batch_size, num_slots, slot_size].

            # Slot update.
            """
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)
            """

            slots = self.gru(updates.reshape(-1, self.slot_size),
                             slots_prev.reshape(-1, self.slot_size))
            slots = slots.reshape(-1, self.num_slots, self.slot_size)

            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_vis


class SlotAttention_LearnableSlots(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = linear(dim, dim)
        self.to_k = linear(dim, dim)
        self.to_v = linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, slots, inputs, num_slots = None, mask = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        slots = slots.expand(b, -1, -1)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        slot_list = []

        for i in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1)

            if mask is not None:
                attn_mask = ~mask.unsqueeze(1).repeat(1, attn.size(1), 1)
                attn = attn * attn_mask

            ##if i != self.iters-1:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            attn_vis = attn

            ##attn_vis = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            ## simple replace
            ##slots = updates

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            slot_list.append(slots)

        ##return slots, attn_vis.permute(0,2,1)
        ##return slot_list[-1], attn_vis.permute(0,2,1)
        return slot_list, attn_vis.permute(0,2,1)


class SlotAttention_LearnableSlots_withG(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = linear(dim, dim)
        self.to_k = linear(dim, dim)
        self.to_v = linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, slots, inputs, inputs_g, num_slots = None, mask = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        slots = slots.expand(b, -1, -1)
        
        slots = slots + inputs_g.unsqueeze(1)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        slot_list = []

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1)

            if mask is not None:
                attn_mask = ~mask.unsqueeze(1).repeat(1, attn.size(1), 1)
                attn = attn * attn_mask

            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            attn_vis = attn

            ##attn_vis = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            slot_list.append(slots)

        ##return slots, attn_vis.permute(0,2,1)
        ##return slot_list[-1], attn_vis.permute(0,2,1)
        return slot_list, attn_vis.permute(0,2,1)


class SlotAttention_LearnableSlots_withConcatG(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_slots = linear(dim*2, dim)
        self.to_q = linear(dim, dim)
        self.to_k = linear(dim, dim)
        self.to_v = linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots_init  = nn.LayerNorm(dim*2)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, slots, inputs, inputs_g, num_slots = None, mask = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        slots = slots.expand(b, -1, -1)
        
        slots = torch.cat([slots,inputs_g.unsqueeze(1).expand(-1, slots.size(1), -1)],dim=-1)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        slots = self.norm_slots_init(slots)
        slots = self.to_slots(slots)

        slot_list = []

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1)

            if mask is not None:
                attn_mask = ~mask.unsqueeze(1).repeat(1, attn.size(1), 1)
                attn = attn * attn_mask

            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            attn_vis = attn

            ##attn_vis = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            slot_list.append(slots)

        ##return slots, attn_vis.permute(0,2,1)
        ##return slot_list[-1], attn_vis.permute(0,2,1)
        return slot_list, attn_vis.permute(0,2,1)




class SlotAttention_LearnableSlots_attnPool(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = linear(dim, dim)
        self.to_k = linear(dim, dim)
        self.to_v = linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)


        self.k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.num_heads = 8

    def forward(self, slots, inputs, num_slots = None, mask = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        slots = slots.expand(b, -1, -1)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            ##attn = dots.softmax(dim=1) + self.eps
            attn = dots.softmax(dim=1)

            if mask is not None:
                attn_mask = ~mask.unsqueeze(1).repeat(1, attn.size(1), 1)
                attn = attn * attn_mask

            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            attn_vis = attn

            updates = torch.einsum('bjd,bij->bid', v, attn)

            """
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            updates = updates.permute(1,0,2)
            slots_prev = slots_prev.permute(1,0,2)

            slots, _ = F.multi_head_attention_forward(
                    query=updates,
                    key=slots_prev,
                    value=slots_prev,
                    embed_dim_to_check=updates.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat(
                        [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                    ),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                    )
            ##slots +=  updates
            slots = slots.permute(1,0,2)
            """

            ##slots = slots.reshape(b, -1, d)
            slots = slots_prev + self.mlp(self.norm_pre_ff(updates))

        return slots, attn_vis.permute(0,2,1)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

