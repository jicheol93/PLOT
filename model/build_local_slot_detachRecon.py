from model import objectives
from .clip_model_extAtt import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict

from .slot_model import SlotAttention_LearnableSlots, SlotAttention_LearnableSlots_withG, SlotAttention_LearnableSlots_withConcatG, SlotAttention_LearnableSlots_attnPool, MlpDecoder, MlpDecoder_woAlpha

import math
import pdb


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


class conv(nn.Module):
    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x



class PLOT(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.features = []

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)
            nn.init.normal_(self.classifier.weight.data, std=0.001)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim // 64,
                                                       num_layers=None)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'recon' in args.loss_names:
            self.recon_decoder = MlpDecoder(24, 8, 512, 512)
            self.recon_decoder_t = MlpDecoder(1, 75, 512, 512)



        ##JC-local
        if 'local' in args.loss_names:
            self.text_slot_classifier = nn.Sequential(
                nn.Linear(self.embed_dim, int(self.embed_dim/8)),
                nn.ReLU(inplace = True),
                nn.Linear(int(self.embed_dim/8), args.num_slots)
            )

            nn.init.normal_(self.text_slot_classifier[0].weight.data, std=0.001)
            nn.init.constant_(self.text_slot_classifier[0].bias.data, val=0.0)
            nn.init.normal_(self.text_slot_classifier[2].weight.data, std=0.001)
            nn.init.constant_(self.text_slot_classifier[2].bias.data, val=0.0)

            self.slots =  nn.Parameter(torch.randn(args.num_slots, 512, device='cuda')/ 512 ** 0.5)
            nn.init.orthogonal_(self.slots)
           
            self.slot_attention = SlotAttention_LearnableSlots(args.num_slots, 512, iters=args.num_iters)
            self.slot_attention_t = SlotAttention_LearnableSlots(args.num_slots, 512, iters=args.num_iters)

            if 'id' in args.loss_names:
                self.local_classifier = nn.Linear(self.embed_dim*args.num_slots*2, self.num_classes, bias=False)
                nn.init.normal_(self.local_classifier.weight.data, std=0.001)


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    ##JC-shared decoding
    def shared_decoding(self, q, k, v, image_flag=True):
        q = q.unsqueeze(0)
        q = q.expand(k.size(0),-1,-1).half()
        if image_flag:
            x = self.cross_attn_local(
                    self.ln_pre_q_local(q),
                    self.ln_pre_i_local(k),
                    self.ln_pre_i_local(v),
                    need_weights=False)[0]
        else:
            x = self.cross_attn_local(
                    self.ln_pre_q_local(q),
                    self.ln_pre_t_local(k),
                    self.ln_pre_t_local(v),
                    need_weights=False)[0]

        x = x.permute(1, 0, 2)
        x = self.shared_decoder(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post_local(x)
        return x
    
    
    def cross_former(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        return x


    def encode_image(self, image):
        x, weights = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_image_local(self, image):
        x, weights = self.base_model.encode_image(image)
        x_i = x[:,1:,:]

        ##vit cls attn with images
        if weights is not None:
            image_attn = weights[-1][:,0,1:]

        image_local, image_attn = self.slot_attention(self.slots, x_i)
        image_local = image_local[-1]


        return x[:, 0, :], weights, image_local, image_attn

    def encode_text(self, text):
        x, weights = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_text_local(self, text):
        x, weights = self.base_model.encode_text(text)
        x_t = []
        mask = []
        for i in range(x.size(0)):
            split_idx = text.argmax(dim=-1)[i]
            if split_idx == 76:
                x_t.append(x[i,1:split_idx,:].unsqueeze(dim=0))
                mask.append(text[i,1:split_idx].unsqueeze(dim=0))
            else:
                x_t.append(torch.cat([x[i,1:split_idx,:],x[i,split_idx+1:,:]],dim=0).unsqueeze(dim=0))
                mask.append(torch.cat([text[i,1:split_idx],text[i,split_idx+1:]],dim=0).unsqueeze(dim=0))
 
        x_t = torch.cat(x_t,dim=0)
        mask = torch.cat(mask,dim=0)
        mask = mask.eq(0)

        text_local, text_attn = self.slot_attention_t(self.slots,x_t,mask=mask)
        text_local = text_local[-1]

        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)],  weights, text_local, text_attn

    def forward(self, batch, epoch):
        ret = dict()

        epoch_ratio = (epoch - self.args.warmup_epochs) / (self.args.num_epoch - self.args.warmup_epochs)
        global_lr_factor = 0.5 * (1 + math.cos(math.pi * epoch_ratio))

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, image_attn, text_feats, text_attn = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :]
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)]
        t_cls = text_feats[:, 0, :]

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
       
        if 'id' in self.current_task:
            norm_i_feats = i_feats / i_feats.norm(dim=-1, keepdim=True)
            image_logits = self.classifier(norm_i_feats).float()
            norm_t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            text_logits = self.classifier(norm_t_feats).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'metric' in self.current_task:
            ret.update({'metric_loss':objectives.compute_nce(i_feats, t_feats, batch['pids'], logit_scale, epoch)})
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats, _ = self.base_model.encode_text(mlm_ids)
            mlm_feats = mlm_feats

            ###cat mlm
            x = torch.cat([mlm_feats,image_feats],dim=1)
            x = self.cross_former(x)
            x = self.mlm_head(x[:,:77,:])
            ###\cat mlm

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        ##JC
        if 'local' in self.current_task:
            x_i = image_feats[:,1:,:]

            x_t = []
            mask = []
            for i in range(text_feats.size(0)):
                split_idx = caption_ids.argmax(dim=-1)[i]
                if split_idx == 76:
                    x_t.append(text_feats[i,1:split_idx,:].unsqueeze(dim=0))
                    mask.append(caption_ids[i,1:split_idx].unsqueeze(dim=0))
                else:
                    x_t.append(torch.cat([text_feats[i,1:split_idx,:],text_feats[i,split_idx+1:,:]],dim=0).unsqueeze(dim=0))
                    mask.append(torch.cat([caption_ids[i,1:split_idx],caption_ids[i,split_idx+1:]],dim=0).unsqueeze(dim=0))

            x_t = torch.cat(x_t,dim=0)
            mask = torch.cat(mask,dim=0)
            cap = mask
            mask = mask.eq(0)

            if  epoch > self.args.start_metric or epoch > self.args.start_recon:

                ##vit cls attn with images
                if image_attn is not None:
                    image_attn = image_attn[-1][:,0,1:]

                if self.args.fg:
                    attn_rank = torch.argsort(image_attn)
                    attn_mask = torch.where(attn_rank>=144, True, False)

                    image_local, attn = self.slot_attention(self.slots, x_i, mask=attn_mask)
                else:
                    image_local, image_attn = self.slot_attention(self.slots, x_i)
                    image_local = image_local[-1]

                text_local, text_attn = self.slot_attention_t(self.slots, x_t,mask=mask)
                text_local = text_local[-1]


            if 'recon' in self.current_task and epoch > self.args.start_recon:
                recon_i = self.recon_decoder(image_local)
                recon_t = self.recon_decoder_t(text_local)

                if self.args.fg:
                    mse = ((x_i.detach().float() - recon_i.float())**2).sum() / x_i.size(0)
                    mse += ((x_t.detach() - recon_t)**2).sum() / x_t.size(0)
                else:
                    mse = ((x_i.detach() - recon_i)**2).sum() / x_i.size(0)
                    mse += ((x_t.detach() - recon_t)**2).sum() / x_t.size(0)
                ret.update({'recon_loss': mse*0.01})

            ###local id###
            if 'id' in self.current_task and epoch > self.args.start_metric:
                image_local_cat = image_local.reshape(image_local.size(0),-1)
                text_local_cat = text_local.reshape(text_local.size(0),-1)

              
                ###cross cat###
                local_cat = torch.cat([image_local_cat, text_local_cat],dim=1)
                local_cat = local_cat / local_cat.norm(dim=-1, keepdim=True)

                local_logits = self.local_classifier(local_cat)
                ret.update({f'id_local_loss':objectives.compute_id(local_logits, None, batch['pids'], local_flag=True)})
                text_pred_local = torch.argmax(local_logits, dim=1)
                text_precision_local = (text_pred_local == batch['pids']).float().mean()
                ret.update({f'txt_local_acc': text_precision_local})
                ###cross cat###


            if 'metric' in self.current_task and epoch > self.args.start_metric:
                slot_agg_score = self.text_slot_classifier(t_feats)
                ret.update({f'metric_local_loss': objectives.compute_nce(image_local, text_local, batch['pids'], logit_scale*5, local_flag=True, part_w=slot_agg_score)})
        return ret


def build_model_local_slot_detachRecon(args, num_classes=11003):
    model = PLOT(args, num_classes)
    return model
