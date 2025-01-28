from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging

import torchvision.utils as vutils

import pdb


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader, args):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("PLOT.eval")
        
        self.num_slots = args.num_slots
        self.args = args


    def _compute_embedding(self, model, tb_writer, epoch, f_idx=None):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        qfeats_local, gfeats_local = [], []
        # text
        vis_attns, txt_attns, txt_cls = [], [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                ##text_feat = model.encode_text(caption)
                text_feat, vit_attn_t, text_feat_local, attn_t = model.encode_text_local(caption)
                ##txt cls
                ##text_feat, text_cls, vit_attn_t, text_feat_local, attn_t = model.encode_text_local(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
            qfeats_local.append(text_feat_local)
            ##txt cls
            ##txt_cls.append(text_cls)

            if self.args.vis_f:
                attn_t = attn_t.reshape(attn_t.size(0),1,75,attn_t.size(2)).permute(0,3,2,1)
                ##attn_t = attn_t.reshape(attn_t.size(0),1,76,attn_t.size(2)).permute(0,3,2,1)
                attn_t = attn_t.unsqueeze(dim=2)
                txt_attns.append(attn_t)

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        qfeats_local = torch.cat(qfeats_local,0)
        ##txt cls
        ##txt_cls = torch.cat(txt_cls,0)
        # image
        vis_cnt = 0
        imgs, captions = [], []
        
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                ##img_feat = model.encode_image(img)
                img_feat, vit_attn_i, img_feat_local, attn_i = model.encode_image_local(img)

            ##tb_writer.close()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
            gfeats_local.append(img_feat_local)
            vis_cnt += 1
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        gfeats_local = torch.cat(gfeats_local,0)

        return qfeats, gfeats, qids, gids, qfeats_local, gfeats_local, imgs, vis_attns, txt_attns
    
    def eval(self, model, tb_writer, i2t_metric=False, epoch=None):
        qfeats, gfeats, qids, gids, qfeats_local, gfeats_local, imgs, vis_attns, txt_attns  = self._compute_embedding(model, tb_writer, epoch)

        qfeats_tmp = qfeats
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        

        similarity = qfeats @ gfeats.t()
        similarity = similarity.cpu()

        ###shift norm###
        ##similarity = (1+similarity)/2
        ###shift norm###

        ###
        if epoch is not None and epoch > 0:
            similarity_zero = torch.zeros_like(similarity)
            image_norm = gfeats_local / gfeats_local.norm(dim=2, keepdim=True)
            text_norm = qfeats_local / qfeats_local.norm(dim=2, keepdim=True)
            t2i_part_sim = []
            for i in range(self.num_slots):
                ##if not i==7: continue;
                part_sim = text_norm[:,i,:] @ image_norm[:,i,:].t()
                ###shift&norm###
                ##part_sim = (1+part_sim)/2
                ###shift&norm###

                t2i_part_sim.append(part_sim.cpu())

            if 'part_w'  in  [name for name,_ in model.named_parameters()]:
                part_w = F.softmax(model.part_w,dim=0)
                t2i_part_sim = torch.cat(t2i_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),image_norm.size(0))
                similarity_p = torch.einsum('bid,ij -> bd', t2i_part_sim, part_w.cpu())
                ##similarity_p = similarity_p/(self.num_slots)
                ###rerank###
                """
                for i in range(qfeats.size(0)):
                    topk_sim, topk_idx = similarity[i].topk(16, dim=0)
                    similarity_zero[i,topk_idx] += (topk_sim + similarity_p[i, topk_idx])
                """
                ###rerank###
            elif 'text_slot_classifier.0.weight'  in  [name for name,_ in model.named_parameters()]:
                part_w = model.text_slot_classifier(qfeats_tmp)
                part_w = F.softmax(part_w,dim=1) * 3
                t2i_part_sim = torch.cat(t2i_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),image_norm.size(0))
                similarity_p = torch.einsum('bid,bi -> bd', t2i_part_sim, part_w.cpu())
                ###rerank###
                """
                for i in range(qfeats.size(0)):
                    topk_sim, topk_idx = similarity[i].topk(16, dim=0)
                    similarity_zero[i,topk_idx] += (topk_sim + similarity_p[i, topk_idx])
                """
                ###rerank###
            else:
                similarity_p = torch.sigmoid(t2i_part_sim[0])
                for i in range(1, self.num_slots):
                    similarity_p += torch.sigmoid(t2i_part_sim[i])

            if epoch > self.args.start_metric:
                similarity += similarity_p

        t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        pred_labels = gids[indices.cpu()]  # q * k
        matches = pred_labels.eq(qids.view(-1, 1))  # q * k
        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        if epoch is not None:
            tb_writer.add_scalar('R1', t2i_cmc[0], epoch)
        
        return t2i_cmc[0]
