import torch
import pickle

import numpy as np
import torch.nn as nn

from einops import rearrange, repeat


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AnalogistAttentionEdit(AttentionBase):
    def __init__(self, start_step=0, end_step=50, 
                  sac_start_layer=0, sac_end_layer=16, sac_layer_idx=None, 
                  cam_start_layer=0, cam_end_layer=16, cam_layer_idx=None, 
                  scale_sac=1.0,
                  step_idx=None):
        super().__init__()
        with open("index_sa.pkl", "rb") as f:
            self.index_sa = pickle.load(f)
        self.sa_layer_idx = sac_layer_idx if sac_layer_idx is not None else list(range(sac_start_layer, sac_end_layer))
        self.ca_layer_idx = cam_layer_idx if cam_layer_idx is not None else list(range(cam_start_layer, cam_end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.scale_sac = scale_sac
  
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):

        if (self.cur_step not in self.step_idx) \
            or ((self.cur_att_layer // 2 not in self.sa_layer_idx) and (self.cur_att_layer // 2 not in self.ca_layer_idx)):
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        cur_res = np.sqrt(attn.shape[1]).astype(np.int32)
        ids = self.index_sa[int(cur_res)]

        if is_cross and (self.cur_att_layer // 2 in self.ca_layer_idx):
            # Cross Attention Masking
            attn[:,ids["id_A"],:] =  0.0
            attn[:,ids["id_Aprime"],:] =  0.0
            attn[:,ids["id_B"],:] =  0.0
        if (not is_cross) and (self.cur_att_layer // 2 in self.sa_layer_idx):
            # Self Attention Cloning
            grid_a, grid_b = torch.meshgrid(ids["id_A"], ids["id_B"], indexing='xy')
            grid_ap, grid_bp = torch.meshgrid(ids["id_Aprime"], ids["id_Bprime"], indexing='xy')
            sim[:, grid_ap, grid_bp] = sim[:, grid_a, grid_b] * self.scale_sac
            sim[:, grid_bp, grid_ap] = sim[:, grid_b, grid_a] * self.scale_sac
            attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)

        return out


def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count

