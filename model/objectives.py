import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

def compute_nce(image_features, text_features, pid, logit_scale, epoch=0, local_flag=False, part_w=None, factor=0.3, epsilon=1e-8, scale=5.0):
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

 
    if not local_flag:
        image_norm = image_features / image_features.norm(dim=1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        ###sim entropy###
        if epoch < 0:
            with torch.no_grad():
                sim_ent = -1 * torch.sum(F.softmax(t2i_cosine_theta*logit_scale,dim=1)*torch.log(F.softmax(t2i_cosine_theta*logit_scale,dim=1)+epsilon),dim=1) / torch.log(torch.tensor(t2i_cosine_theta.size(1)))
            t2i = logit_scale * t2i_cosine_theta * (0.25+sim_ent)
            i2t = logit_scale * i2t_cosine_theta * (0.25+sim_ent.t())
            ###sim entropy###
        else:
            t2i = logit_scale * t2i_cosine_theta
            i2t = logit_scale * i2t_cosine_theta

        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(i2t, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(i2t, dim=1) - torch.log(labels_distribute + epsilon))
        ### clipping neg ###
        ##i2t_loss = torch.where(i2t_loss<0,torch.tensor(0.0).cuda(),i2t_loss)
        t2i_pred = F.softmax(t2i, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(t2i, dim=1) - torch.log(labels_distribute + epsilon))
        ### clipping neg ###
        ##t2i_loss = torch.where(t2i_loss<0,torch.tensor(0.0).cuda(),t2i_loss)

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    else:
        image_norm = image_features / image_features.norm(dim=2, keepdim=True)
        text_norm = text_features / text_features.norm(dim=2, keepdim=True)

        t2i_part_sim, t2t_part_sim, i2i_part_sim, i2t_part_sim = [], [], [], []

        cumsim_i2t = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        cumsim_t2i = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        cumsim_i2i = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        cumsim_t2t = torch.zeros(text_features.size(0),text_features.size(0)).to(text_features)

        similarity_p = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        for i in range(image_features.size(1)):
            part_sim = text_norm[:,i,:] @ image_norm[:,i,:].t()
            t2t_sim = text_norm[:,i,:] @ text_norm[:,i,:].t()
            i2i_sim = image_norm[:,i,:] @ image_norm[:,i,:].t()

            similarity_p += torch.sigmoid(part_sim)
            t2i_part_sim.append(part_sim)
            i2t_part_sim.append(part_sim.t())

            i2i_part_sim.append(i2i_sim)
            t2t_part_sim.append(t2t_sim)

            cumsim_i2t += torch.sigmoid(part_sim)
            cumsim_t2i += torch.sigmoid(part_sim.t())
            cumsim_i2i += i2i_sim
            cumsim_t2t += t2t_sim

        if part_w is not None:
            t2i_part_sim = torch.cat(t2i_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),text_norm.size(0))
            i2t_part_sim = torch.cat(i2t_part_sim,dim=1).view(image_norm.size(0),image_norm.size(1),image_norm.size(0))
            i2i_part_sim = torch.cat(i2i_part_sim,dim=1).view(image_norm.size(0),image_norm.size(1),image_norm.size(0))
            t2t_part_sim = torch.cat(t2t_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),text_norm.size(0))

            if part_w.shape[-1] == 1:
                part_w = F.softmax(part_w,dim=0)
                t2i_cosine_theta = torch.einsum('bid,ij -> bd', t2i_part_sim, part_w)
                i2t_cosine_theta = torch.einsum('bid,ij -> bd', i2t_part_sim, part_w)
                i2i_cosine_theta = torch.einsum('bid,ij -> bd', i2i_part_sim, part_w)
                t2t_cosine_theta = torch.einsum('bid,ij -> bd', t2t_part_sim, part_w)
            else:
                part_w = F.softmax(part_w,dim=1)
                t2i_cosine_theta = torch.einsum('bid,bi -> bd', t2i_part_sim, part_w)
                i2t_cosine_theta = torch.einsum('bid,bi -> bd', i2t_part_sim, part_w)
                i2i_cosine_theta = torch.einsum('bid,bi -> bd', i2i_part_sim, part_w)
                t2t_cosine_theta = torch.einsum('bid,bi -> bd', t2t_part_sim, part_w)
        else:
            """
            ##top5
            t2i_cosine_theta = torch.sum(t2i_part_sim.topk(5,dim=1)[0],dim=1)
            i2t_cosine_theta = torch.sum(i2t_part_sim.topk(5,dim=1)[0],dim=1)
            """
            t2i_cosine_theta = similarity_p
            i2t_cosine_theta = similarity_p.t()

        t2i = logit_scale * t2i_cosine_theta
        i2t = logit_scale * i2t_cosine_theta
        t2t = logit_scale * t2t_cosine_theta
        i2i = logit_scale * i2i_cosine_theta

        loss = F.cross_entropy(t2t, labels, reduction='mean') +  F.cross_entropy(i2i, labels, reduction='mean') + F.cross_entropy(i2t, labels, reduction='mean') + F.cross_entropy(t2i, labels, reduction='mean')

        reg_i = torch.einsum('bid, bdj -> bij', image_norm, image_norm.permute(0,2,1))
        reg_i -= torch.eye(8).cuda()
        reg_i = reg_i.sum() / 56

        reg_t = torch.einsum('bid, bdj -> bij', text_norm, text_norm.permute(0,2,1))
        reg_t -= torch.eye(8).cuda()
        reg_t = reg_t.sum() / 56

        reg = (reg_i + reg_t) / 128

        loss += reg*scale


        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(i2t, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(i2t, dim=1) - torch.log(labels_distribute + epsilon))
        ### clipping neg ###
        ##i2t_loss = torch.where(i2t_loss<0,torch.tensor(0.0).cuda(),i2t_loss)
        ### clipping neg ###
        t2i_pred = F.softmax(t2i, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(t2i, dim=1) - torch.log(labels_distribute + epsilon))
        ### clipping neg ###
        ##t2i_loss = torch.where(t2i_loss<0,torch.tensor(0.0).cuda(),t2i_loss)
        ### clipping neg ###

        loss += (torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1)))
        if torch.isnan(loss) or torch.isinf(loss):
            pdb.set_trace()

        ### part kl_div cross modal ###
        """
        ii = torch.einsum('bid,bjd -> bij', image_norm, image_norm)
        ##ii = ii.view(ii.size(0), -1)
        ##ii = (ii + 1) / 2
        ##ii = F.softmax(ii*333,dim=2)
        tt = torch.einsum('bid,bjd -> bij', text_norm, text_norm)
        ##tt = tt.view(tt.size(0), -1)
        ##tt = (tt + 1) / 2
        ##tt = F.softmax(tt*333,dim=2)
        ##part_kl = F.kl_div(F.log_softmax(tt), F.softmax(ii), reduction='batchmean') * 100
        part_kl = 0
        for i in range(image_features.size(1)):
            part_kl += torch.sum(F.softmax(tt[:,i,:]*333,dim=1) * (F.log_softmax(tt[:,i,:]*333, dim=1) - (F.log_softmax(ii[:,i,:]*333, dim=1) + epsilon)))
        loss += (part_kl)/image_features.size(1)
        """
        ### part kl_div cross modal ###

    return loss


def compute_cl(image_features, text_features, pid, logit_scale, local_flag=False, part_w=None, factor=0.3, epsilon=1e-8, scale=5.0):
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()


    if not local_flag:
        image_norm = image_features / image_features.norm(dim=1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        t2t_cosine_theta = text_norm @ text_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()
        i2i_cosine_theta = image_norm @ image_norm.t()

        t2i = logit_scale * t2i_cosine_theta
        i2t = logit_scale * i2t_cosine_theta
        t2t = t2t_cosine_theta
        i2i = i2i_cosine_theta

        loss = F.cross_entropy(t2t, labels, reduction='mean') +  F.cross_entropy(i2i, labels, reduction='mean') + F.cross_entropy(i2t, labels, reduction='mean') + F.cross_entropy(t2i, labels, reduction='mean')
    else:
        ### rand mix with p partw
        ##p = part_w.cumsum(0)
        ##idx = torch.searchsorted(p, torch.rand(1))
        ### rand mix with p partw

        image_norm = image_features / image_features.norm(dim=2, keepdim=True)
        text_norm = text_features / text_features.norm(dim=2, keepdim=True)

        t2i_part_sim, t2t_part_sim, i2i_part_sim, i2t_part_sim = [], [], [], []
        cumsim_i2t = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        cumsim_t2i = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        cumsim_i2i = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
        cumsim_t2t = torch.zeros(text_features.size(0),text_features.size(0)).to(text_features)

        for i in range(image_features.size(1)):
            ##part_sum_sim += text_norm[:,i,:] @ image_norm[:,i,:].t()
            part_sim = text_norm[:,i,:] @ image_norm[:,i,:].t()
            t2t_sim = text_norm[:,i,:] @ text_norm[:,i,:].t()
            i2i_sim = image_norm[:,i,:] @ image_norm[:,i,:].t()
            ###part sim entropy###
            """
            part_sim_ent = -1 * torch.sum(F.softmax(part_sim,dim=1)*torch.log(F.softmax(part_sim,dim=1)),dim=1) / torch.log(torch.tensor(part_sim.size(1)))
            ##part_sim_ent = torch.where(part_sim_ent>torch.tensor(0.2),part_sim_ent,torch.tensor(0.2).cuda())
            ##part_sim =part_sim * (2 - part_sim_ent) * logit_scale
            part_sim =part_sim * part_sim_ent * logit_scale
            """
            ###part sim entropy###

            t2i_part_sim.append(part_sim)
            i2t_part_sim.append(part_sim.t())
            i2i_part_sim.append(i2i_sim)
            t2t_part_sim.append(t2t_sim)

            cumsim_i2t += torch.sigmoid(part_sim)
            cumsim_t2i += torch.sigmoid(part_sim.t())
            cumsim_i2i += i2i_sim
            cumsim_t2t += t2t_sim

        t2i_part_sim = torch.cat(t2i_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),text_norm.size(0))
        i2t_part_sim = torch.cat(i2t_part_sim,dim=1).view(image_norm.size(0),image_norm.size(1),image_norm.size(0))
        i2i_part_sim = torch.cat(i2i_part_sim,dim=1).view(image_norm.size(0),image_norm.size(1),image_norm.size(0))
        t2t_part_sim = torch.cat(t2t_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),text_norm.size(0))

        if part_w is not None:
            if part_w.shape[-1] == 1:
                part_w = F.softmax(part_w,dim=0)
                t2i_cosine_theta = torch.einsum('bid,ij -> bd', t2i_part_sim, part_w)
                i2t_cosine_theta = torch.einsum('bid,ij -> bd', i2t_part_sim, part_w)
                i2i_cosine_theta = torch.einsum('bid,ij -> bd', i2i_part_sim, part_w)
                t2t_cosine_theta = torch.einsum('bid,ij -> bd', t2t_part_sim, part_w)
            else:
                part_w = F.softmax(part_w,dim=1)
                t2i_cosine_theta = torch.einsum('bid,bi -> bd', t2i_part_sim, part_w)
                i2t_cosine_theta = torch.einsum('bid,bi -> bd', i2t_part_sim, part_w)
                i2i_cosine_theta = torch.einsum('bid,bi -> bd', i2i_part_sim, part_w)
                t2t_cosine_theta = torch.einsum('bid,bi -> bd', t2t_part_sim, part_w)
        else:
            t2i_cosine_theta = cumsim_t2i
            i2t_cosine_theta = cumsim_i2t
            i2i_cosine_theta = cumsim_i2i
            t2t_cosine_theta = cumsim_t2t

        t2i = logit_scale * t2i_cosine_theta
        i2t = logit_scale * i2t_cosine_theta
        t2t = logit_scale * t2t_cosine_theta
        i2i = logit_scale * i2i_cosine_theta

        loss = F.cross_entropy(t2t, labels, reduction='mean') +  F.cross_entropy(i2i, labels, reduction='mean') + F.cross_entropy(i2t, labels, reduction='mean') + F.cross_entropy(t2i, labels, reduction='mean')
        ##loss = F.cross_entropy(t2t, labels, reduction='mean') + F.cross_entropy(i2i, labels, reduction='mean')
        ##loss = F.cross_entropy(t2i, labels, reduction='mean') + F.cross_entropy(i2t, labels, reduction='mean')

        ##div_reg##
        reg_i = torch.einsum('bid, bdj -> bij', image_norm, image_norm.permute(0,2,1))
        reg_i -= torch.eye(8).cuda()
        reg_i = reg_i.sum() / 56

        reg_t = torch.einsum('bid, bdj -> bij', text_norm, text_norm.permute(0,2,1))
        reg_t -= torch.eye(8).cuda()
        reg_t = reg_t.sum() / 56

        reg = (reg_i + reg_t) / 128

        loss += reg*scale
        ##div_reg##

    return loss


def compute_cl_part_negative(image_features, text_features, pid, logit_scale, factor=0.3, epsilon=1e-8, part_w=None):
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    image_norm = image_features / image_features.norm(dim=2, keepdim=True)
    text_norm = text_features / text_features.norm(dim=2, keepdim=True)

    t2i_part_sim, t2t_part_sim, i2i_part_sim, i2t_part_sim = [], [], [], []

    if part_w is not None:
        part_w = F.softmax(part_w,dim=0)

        image_neg_locals = []
        text_neg_locals = []

        for j in range(image_features.size(0)):
            ##TODO use args
            num_neg = 3
            p = part_w.cumsum(0).cpu()
            idx = torch.searchsorted(p[:,0], torch.rand(num_neg))

            rand_idx = torch.randperm(image_features.size(0))
            rand_idx = rand_idx[rand_idx != j]
            if len(rand_idx) >= num_neg:
                rand_idx = rand_idx[:num_neg]
                
            image_neg_parts = image_norm[rand_idx,idx,:]
            text_neg_parts = text_norm[rand_idx,idx,:]
            for k in range(len(rand_idx)):
                neg_image_local =  image_norm.clone()
                neg_image_local[j,idx[k],:] = image_neg_parts[k]
                neg_text_local =  text_norm.clone()
                neg_text_local[j,idx[k],:] = text_neg_parts[k]

                image_neg_locals.append(neg_image_local[j].unsqueeze(0))
                text_neg_locals.append(neg_text_local[j].unsqueeze(0))
        image_neg_locals = torch.cat(image_neg_locals,0)
        text_neg_locals = torch.cat(text_neg_locals,0)

        if sum(sum(labels == labels.t())) != labels.size(0)**2:
            pdb.set_trace()

        for i in range(image_features.size(1)):
            ###part_neg###
            idx = [x for x in range(image_features.size(1)) if x != i]
            image_norm_partNeg = image_norm[:,idx,:].view(-1, image_features.size(-1))
            text_norm_partNeg = text_norm[:,idx,:].view(-1 ,text_features.size(-1))

            t2i_sim = text_norm[:,i,:] @ torch.cat([image_norm[:,i,:],image_neg_locals[:,i,:]],dim=0).t()
            i2t_sim = image_norm[:,i,:] @ torch.cat([text_norm[:,i,:],text_neg_locals[:,i,:]],dim=0).t()
            t2t_sim = text_norm[:,i,:] @ text_norm[:,i,:].t()
            i2i_sim = image_norm[:,i,:] @ image_norm[:,i,:].t()
            t2i_part_sim.append(t2i_sim)
            i2t_part_sim.append(i2t_sim)
            i2i_part_sim.append(i2i_sim)
            t2t_part_sim.append(t2t_sim)
            ###part_neg###

        ###part neg###
        part_negative_labels = torch.zeros(labels.size(0), image_neg_locals.size(0)).to(labels)
        labels_with_neg = torch.cat([labels, part_negative_labels],dim=1)
        ###part neg###

        t2i_part_sim = torch.cat(t2i_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),labels_with_neg.size(1))
        i2t_part_sim = torch.cat(i2t_part_sim,dim=1).view(image_norm.size(0),image_norm.size(1),labels_with_neg.size(1))
        i2i_part_sim = torch.cat(i2i_part_sim,dim=1).view(image_norm.size(0),image_norm.size(1),labels.size(1))
        t2t_part_sim = torch.cat(t2t_part_sim,dim=1).view(text_norm.size(0),text_norm.size(1),labels.size(1))

        t2i_cosine_theta = torch.einsum('bid,ij -> bd', t2i_part_sim, part_w)
        i2t_cosine_theta = torch.einsum('bid,ij -> bd', i2t_part_sim, part_w)
        i2i_cosine_theta = torch.einsum('bid,ij -> bd', i2i_part_sim, part_w)
        t2t_cosine_theta = torch.einsum('bid,ij -> bd', t2t_part_sim, part_w)

        ##t2i = logit_scale * t2i_cosine_theta
        ##i2t = logit_scale * i2t_cosine_theta
        t2i = t2i_cosine_theta
        i2t = i2t_cosine_theta
        t2t = t2t_cosine_theta
        i2i = i2i_cosine_theta

        loss = F.cross_entropy(t2t, labels, reduction='mean') +  F.cross_entropy(i2i, labels, reduction='mean') + F.cross_entropy(i2t, labels_with_neg, reduction='mean') + F.cross_entropy(t2i, labels_with_neg, reduction='mean')

        return loss
    else:
        loss = 0
        for i in range(image_features.size(1)):
            ##part_sum_sim += text_norm[:,i,:] @ image_norm[:,i,:].t()
            idx = [x for x in range(image_features.size(1)) if x != i]
            image_norm_partNeg = image_norm[:,idx,:].view(-1, image_features.size(-1))
            text_norm_partNeg = text_norm[:,idx,:].view(-1 ,text_features.size(-1))

            t2i_sim = text_norm[:,i,:] @ torch.cat([image_norm[:,i,:],image_norm_partNeg],dim=0).t()
            i2t_sim = image_norm[:,i,:] @ torch.cat([text_norm[:,i,:],text_norm_partNeg],dim=0).t()
            ##t2t_sim = text_norm[:,i,:] @ torch.cat([text_norm[:,i,:],text_norm_partNeg],dim=0).t()
            ##i2i_sim = image_norm[:,i,:] @ torch.cat([image_norm[:,i,:],image_norm_partNeg],dim=0).t()

            part_negative_labels = torch.zeros(labels.size(0), image_norm_partNeg.size(0)).to(labels)
            part_labels = torch.cat([labels, part_negative_labels],dim=1)

            t2i = logit_scale * t2i_sim
            i2t = logit_scale * i2t_sim
            ##t2t = t2t_sim
            ##i2i = i2i_sim

            ##loss += (F.cross_entropy(t2t, part_labels, reduction='mean') +  F.cross_entropy(i2i, part_labels, reduction='mean') + F.cross_entropy(i2t, part_labels, reduction='mean') + F.cross_entropy(t2i, part_labels, reduction='mean'))
            loss += F.cross_entropy(i2t, part_labels, reduction='mean') + F.cross_entropy(t2i, part_labels, reduction='mean')

        return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)

def compute_pmlm(scores, labels):
    ce = nn.CrossEntropyLoss()
    ##ce = nn.MultiLabelSoftMarginLoss()
    return ce(scores, labels)


def compute_id(image_logits, text_logits, labels, local_flag=False):
    criterion = nn.CrossEntropyLoss(reduction="mean")
    if not local_flag:

        loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        
        return loss / 2
    else:
        loss = criterion(image_logits, labels)

        return loss


##JC- for pmlm loss
"""
def compute_pmlm(image_part, text_recon_part, idx_list):
    pmlm_loss = 0
    for i in range(image_part.size(0)):
        ##pmlm_loss += ((image_part[i,idx_list[i],:] - text_recon_part[i,idx_list[i],:])**2).mean() / image_part.size(0)
        pmlm_loss += ((image_part[i,idx_list[i],:].float() - text_recon_part[i,idx_list[i],:].float())**2).sum() / image_part.size(0)
        ##pdb.set_trace()
    ##return pmlm_loss*10
    return pmlm_loss
"""


##JC for ranking loss
def calculate_similarity(image_embedding, text_embedding):
    image_embedding = image_embedding.view(image_embedding.size(0), -1) 
    text_embedding = text_embedding.view(text_embedding.size(0), -1) 
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    similarity_match = torch.sum(image_embedding_norm * text_embedding_norm, dim=1)

    return similarity, similarity_match

def semi_hard_negative(loss, margin):
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

def get_triplets(similarity, labels, auto_margin_flag, margin):

    similarity = similarity.cpu().data.numpy()

    labels = labels.cpu().data.numpy()
    triplets = []

    for idx, label in enumerate(labels):  # same class calculate together
        if margin[idx] >= 0.16 or auto_margin_flag is False:
            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]

            loss = similarity[idx, negative] - ap_sim + margin[idx]

            negetive_index = semi_hard_negative(loss, margin[idx])

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

    if len(triplets) == 0:
        triplets.append([idx, idx, negative[0]])

    triplets = torch.LongTensor(np.array(triplets))

    return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).cuda()

    return triplets, return_margin

def get_triplets_local(similarity, labels, auto_margin_flag, margin, score_list):

        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []
        logit = score_list
        scores = []
        ##pdb.set_trace()

        for idx, label in enumerate(labels):  # same class calculate together
            if margin[idx] >= 0.16 or auto_margin_flag is False:
                negative = np.where(labels != label)[0]

                ap_sim = similarity[idx, idx]

                loss = similarity[idx, negative] - ap_sim + margin[idx]

                negetive_index = semi_hard_negative(loss, margin[idx])

                if negetive_index is not None:
                    triplets.append([idx, idx, negative[negetive_index]])
                    scores.append(logit[idx])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])
            scores.append(logit[idx])

        triplets = torch.LongTensor(np.array(triplets))

        return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).cuda()

        return triplets, return_margin, scores


def calculate_loss(similarity, label, auto_margin_flag, margin):

        image_triplets, img_margin = get_triplets(similarity, label, auto_margin_flag, margin)
        text_triplets, txt_margin = get_triplets(similarity.t(), label, auto_margin_flag, margin)

        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        similarity = similarity.t()
        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss

def calculate_loss_local(similarity, label, auto_margin_flag, margin, img_score, txt_score):

        image_triplets, img_margin, img_scores = get_triplets_local(similarity, label, auto_margin_flag, margin, img_score)
        text_triplets, txt_margin, txt_scores = get_triplets_local(similarity.t(), label, auto_margin_flag, margin, txt_score)

        img_scores = torch.tensor(img_scores).cuda()
        txt_scores = torch.tensor(txt_scores).cuda()

        img_margin = img_margin * img_scores
        img_margin = torch.where(img_margin>0.05,img_margin,torch.tensor(0.05).cuda())

        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        if image_anchor_loss.shape != img_scores.shape:
            pdb.set_trace()

        similarity = similarity.t()

        txt_margin = txt_margin * txt_scores
        txt_margin = torch.where(txt_margin>0.05,txt_margin,torch.tensor(0.05).cuda())

        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        if text_anchor_loss.shape != txt_scores.shape:
            pdb.set_trace()

        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss


def compute_ranking(image_embeddings, text_embeddings, labels, margin, img_score=[], txt_score=[], local_flag=False, auto_margin_flag=False):

    ##similarity, similarity_match = calculate_similarity(image_embeddings, text_embeddings)

    r_loss = 0
    if local_flag:
        pdb.set_trace()
        for i in range(image_embeddings.size(1)):
            similarity, similarity_match = calculate_similarity(image_embeddings, text_embeddings)
        ##r_loss = calculate_loss_local(similarity, labels, auto_margin_flag, margin, img_score, txt_score)
        r_loss += calculate_loss(similarity, labels, auto_margin_flag, margin)
    else:
        similarity, similarity_match = calculate_similarity(image_embeddings, text_embeddings)
        r_loss = calculate_loss(similarity, labels, auto_margin_flag, margin)

    return r_loss

def compute_ranking_local(image_features, text_features, labels, margin, auto_margin_flag=False):
    ##similarity, similarity_match = calculate_similarity(image_embeddings, text_embeddings)
    image_norm = image_features / image_features.norm(dim=2, keepdim=True)
    text_norm = text_features / text_features.norm(dim=2, keepdim=True)

    similarity_p = torch.zeros(image_features.size(0),image_features.size(0)).to(image_features)
    for i in range(image_features.size(1)):
        ##part_sum_sim += text_norm[:,i,:] @ image_norm[:,i,:].t()
        part_sim = text_norm[:,i,:] @ image_norm[:,i,:].t()
        similarity_p += part_sim
    similarity = similarity_p / image_features.size(1)

    r_loss = calculate_loss(similarity.t(), labels, auto_margin_flag, margin)

    return r_loss




def compute_vsepp(image_features, text_features, margin, max_violation=True):
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)
    scores = image_norm @ text_norm.t()
    diagonal = scores.diag().view(image_features.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    margin = torch.ones_like(scores) * margin 
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    cost_s = cost_s.masked_fill_(mask.cuda(), 0)
    cost_im = cost_im.masked_fill_(mask.cuda(), 0)

    # keep the maximum violating negative for each query
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()

