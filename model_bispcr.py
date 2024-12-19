import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import knn, batch_choice
from scipy.spatial.transform import Rotation
from conv_bispcr import Conv1DBNReLU, Conv1DBlock, Conv2DBNReLU, Conv2DBlock


class BIFH(nn.Module):
    def __init__(self, emb_dims=64):
        super(BIFH, self).__init__()
        self.harvester1 = Harvester(3, 64)
        self.harvester2 = Harvester(64, 64)
        self.harvester3 = Harvester(64, 64)
        self.harvester4 = Harvester(64, 64)
        self.harvester5 = Harvester(64, emb_dims)       
        

    def forward(self, x):
        nn_idx = knn(x, k=16)
        x1 = self.harvester1(x, nn_idx, 1)        
        x2 = self.harvester2(x1, nn_idx, 0)
        x2 = x1 + x2
        x3 = self.harvester3(x2, nn_idx,0)
        x3 = x1 + x2 + x3
        x4 = self.harvester4(x3, nn_idx,0)
        x4 = x1 + x2 + x3 + x4
        x5 = self.harvester5(x4, nn_idx,0)
        x5 = x1 + x2 + x3 + x4 + x5
        
        return x5


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, tgt, weights):

        var_eps = 1e-3
        weights = weights.unsqueeze(2)  # 16 128 1
        src = src.transpose(1, 2)  # b 768 3
        srcm = torch.matmul(weights.transpose(1, 2), src)  # b 1 3
        src_centered = src - srcm  # b 128 3
        src_centered = src_centered.transpose(1, 2)
        tgt = tgt.transpose(1, 2)  # b 768 3
        tgtm = torch.matmul(weights.transpose(1, 2), tgt)  # b 1 3
        tgt_centered = tgt - tgtm  # b 128 3
        tgt_centered = tgt_centered.transpose(1, 2)

        weight_matrix = torch.diag_embed(weights.squeeze(2))
        H = torch.matmul(src_centered, torch.matmul(weight_matrix, tgt_centered.transpose(2, 1).contiguous()))

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, srcm.transpose(1, 2)) + tgtm.transpose(1, 2)
        return R, t.view(src.size(0), 3)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


class fft_loss(nn.Module):
    def __init__(self, dim):
        super(fft_loss, self).__init__()
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x, y):
        x_req = torch.fft.fft2(x)
        y_req = torch.fft.fft2(y)
        loss_real = self.criterion(x_req.real, y_req.real)
        loss_imag = self.criterion(x_req.imag, y_req.imag)        
        loss = loss_real + loss_imag
        return loss
        

class BIISI(nn.Module):
    def __init__(self, args):
        super(BIISI, self).__init__()
        self.emb_dims = args.emb_dims
        self.num_iter = args.num_iter

        self.sim_mat_conv_hd = nn.ModuleList(
            [Conv2DBlock((self.emb_dims * 3 + 1, 32, 32), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv_td = nn.ModuleList([Conv2DBlock((9 + 1, 32, 32), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv = nn.ModuleList([Conv2DBlock((32, 32, 16), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv2 = nn.ModuleList([Conv2DBlock((16, 16, 1), 1) for _ in range(self.num_iter)])
        self.weight_fc = nn.ModuleList([Conv1DBlock((16, 16, 1), 1) for _ in range(self.num_iter)])
        self.weight_fc_rc = nn.ModuleList([Conv1DBlock((16, 16, 1), 1) for _ in range(self.num_iter)])
        self.frerc = FRERC(16)
        self.head = SVDHead(args=args)

        self.tah = nn.Tanh()
        self.fft_loss = fft_loss(1) 
        

    def forward(self, src, tgt, src_embedding, tgt_embedding, 
                gt_src, src_sig_score, tgt_sig_score, match_labels):
        ##### initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        ##### initialize #####
        loss = 0.
        num_iter = self.num_iter

        for i in range(num_iter):

            batch_size, num_dims, num_points = src_embedding.size()

            ##### similarity td matrix convolution to get features #####
            diff_td = src.unsqueeze(-1) - tgt.unsqueeze(-2)
            dist_td = (diff_td ** 2).sum(1, keepdim=True)
            dist_td = torch.sqrt(dist_td)
            diff_td = diff_td / (dist_td + 1e-8)    
            
            _src = src.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt = tgt.unsqueeze(-2).repeat(1, 1, num_points, 1)
            similarity_matrix_td = torch.cat([_src, _tgt, diff_td, dist_td], 1)
            similarity_matrix_td = self.sim_mat_conv_td[i](similarity_matrix_td)
            ##### similarity td matrix convolution to get features #####

            ##### similarity hd matrix convolution to get features #####
            diff_hd = src_embedding.unsqueeze(-1) - tgt_embedding.unsqueeze(-2)
            dist_hd = (diff_hd ** 2).sum(1, keepdim=True)
            dist_hd = torch.sqrt(dist_hd)
            diff_hd = diff_hd / (dist_hd + 1e-8)

            _src_emb = src_embedding.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt_emb = tgt_embedding.unsqueeze(-2).repeat(1, 1, num_points, 1)
            similarity_matrix = torch.cat([_src_emb, _tgt_emb], 1)
            similarity_matrix_hd = torch.cat((similarity_matrix, diff_hd, dist_hd), 1)
            similarity_matrix_hd = self.sim_mat_conv_hd[i](similarity_matrix_hd)
            ##### similarity hd matrix convolution to get features #####

            ##### similarity matrix convolution #####
            similarity_matrix = similarity_matrix_hd * similarity_matrix_td
            similarity_matrix = self.sim_mat_conv[i](similarity_matrix)
            ##### similarity matrix convolution#####

            ##### get weights #####
            weights = similarity_matrix.max(-1)[0]
            weights = self.weight_fc[i](weights)            
            if i == 2:
                wrc = self.frerc(src, tgt) 
                wrc = self.weight_fc_rc[i](wrc)                  
                wrc = 1 - self.tah(torch.abs(wrc))
                weights = wrc * weights   
            weights = weights.squeeze(1)    
            ##### get weights  #####
            
            ##### frequency consensus measure loss #####
            if self.training:
                src_knn_idx = knn(src, 20)
                src_batch_idx = np.arange(src.size(0)).reshape(src.size(0), 1, 1)
                src_nn_feat = src[src_batch_idx, :, src_knn_idx].permute(0, 3, 1, 2)
                src_graph = src_nn_feat - src.unsqueeze(-1)
                
                tgt_knn_idx = knn(tgt, 20)
                tgt_batch_idx = np.arange(tgt.size(0)).reshape(tgt.size(0), 1, 1)
                tgt_nn_feat = tgt[tgt_batch_idx, :, tgt_knn_idx].permute(0, 3, 1, 2)
                tgt_graph = tgt_nn_feat - tgt.unsqueeze(-1)       

                fft_loss = self.fft_loss(src_graph, tgt_graph)  #               
                loss = loss + 0.05 * fft_loss
            ##### frequency consensus measure loss #####                 

            ##### similarity matrix convolution to get similarities #####
            similarity_matrix = self.sim_mat_conv2[i](similarity_matrix)
            similarity_matrix = similarity_matrix.squeeze(1)
            similarity_matrix = similarity_matrix.clamp(min=-20, max=20)
            ##### similarity matrix convolution to get similarities #####

            ##### spatial domain loss #####
            if self.training and i == 0:
                src_neg_ent = torch.softmax(similarity_matrix, dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(similarity_matrix, dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                neloss = F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score,
                                                                                      tgt_neg_ent.detach())
                loss = loss + 0.5 * neloss
            ##### spatial domain loss #####

            ##### spatial domain loss #####
            if self.training:
                temp = torch.softmax(similarity_matrix, dim=-1)
                temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
                temp = - torch.log(temp)
                match_loss = (temp * match_labels).sum() / match_labels.sum()
                loss = loss + match_loss
            ##### spatial domain loss #####

            ##### finding correspondences #####
            corr_idx = similarity_matrix.max(-1)[1]
            src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)
            ##### finding correspondences #####

            ##### spatial domain loss #####
            if self.training:
                weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).cuda().unsqueeze(0)).float()
                num_pos = torch.relu(torch.sum(weight_labels) - 1) + 1
                num_neg = torch.relu(torch.sum(1 - weight_labels) - 1) + 1
                weight_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg * 1.0 / num_pos, reduction='mean')(weights,
                                                                                                         weight_labels.float())
                loss = loss + weight_loss               
            ##### spatial domain loss #####

            weights = torch.sigmoid(weights)
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

            ##### get transformation #####
            rotation_ab, translation_ab = self.head(src, src_corr, weights)
            rotation_ab = rotation_ab.detach()  # prevent backprop through svd
            translation_ab = translation_ab.detach()  # prevent backprop through svd
            src = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(-1)
            R = torch.matmul(rotation_ab, R)
            t = torch.matmul(rotation_ab, t.unsqueeze(-1)).squeeze() + translation_ab
            ##### get transformation #####
        euler_ab = npmat2euler(R.detach().cpu().numpy())
        return R, t, euler_ab, loss
    
class frequency_process(nn.Module):
    def __init__(self, dim):
        super(frequency_process, self).__init__()
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)        
        self.bn2 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.conv1 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True) 
        self.bn1 = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
    def forward(self,x):
        mag1 = torch.abs(x)
        pha1 = torch.angle(x)

        mag = self.relu1(self.bn1(self.conv1(mag1)))
        mag = mag1 + mag
        pha = self.relu2(self.bn2(self.conv2(pha1)))
        pha = pha1 + pha

        real_part = mag * torch.cos(pha)
        img_part = mag * torch.sin(pha)
        x = torch.complex(real_part, img_part)
        return x

class spatial(nn.Module):
    def __init__(self, dim):
        super(spatial, self).__init__() 
        self.conv1 = torch.nn.Conv2d(dim, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = torch.nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    def forward(self,x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        return x + x2
        
class Harvester(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Harvester, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)
        self.conv3d = Conv1DBlock((2*in_channel, emb_dims, emb_dims), 1)
        self.conv4d = Conv1DBlock((in_channel + emb_dims, emb_dims), 1)
        self.fft = FFTBlock2d(64)

    def forward(self, x, idx, first):
        x_ori = x
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        x = self.fft(x)
        x = x.max(-1)[0]
        x = self.conv1d(x)
        x = torch.cat((x, x_ori), dim=1)
        if first==0:
            x = self.conv3d(x)
        else:
            x = self.conv4d(x)
        return x

class FFTBlock2d(nn.Module):
    def __init__(self, dim):
        super(FFTBlock2d, self).__init__()
        self.frequency_process = frequency_process(dim)
        self.conv = torch.nn.Conv2d(2*dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.spatial = spatial(dim)

    def forward(self,x):
        x_spatial = self.spatial(x)
        x_req = torch.fft.rfft2(x, norm='backward')
        x_req = self.frequency_process(x_req)
        x_back_freq = torch.fft.irfft2(x_req, norm='backward')
        x_out = torch.cat((x_spatial, x_back_freq), dim=1)
        x_out = self.relu(self.bn(self.conv(x_out)))
        return x_out + x

        
class Propagate_sig(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate_sig, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        return x
        
class FRERC(nn.Module):
    def __init__(self, emb_dims):
        super(FRERC, self).__init__()
        self.propogate = Propagate_sig(64, 64)
                
        self.conv1 = torch.nn.Conv2d(3, emb_dims, kernel_size=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)        
        self.conv2 = torch.nn.Conv2d(emb_dims, emb_dims, kernel_size=1, bias=False)
        
        self.conv3 = torch.nn.Conv2d(3, emb_dims, kernel_size=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)        
        self.conv4 = torch.nn.Conv2d(emb_dims, emb_dims, kernel_size=1, bias=False)
        
        self.conv5 = torch.nn.Conv2d(emb_dims, emb_dims, kernel_size=1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)        
        self.conv6 = torch.nn.Conv2d(emb_dims, emb_dims, kernel_size=1, bias=False)

        self.tah = nn.Tanh()
        self.conv = Conv2DBlock((emb_dims, emb_dims),1)
        self.sigma_spat = 1e-5
    def forward(self, src, tgt): 

        src_graph = src.unsqueeze(-1) - src.unsqueeze(-2)
        
        src_nergibor_fft = torch.fft.fft2(src_graph)
        tgt_graph = tgt.unsqueeze(-1) - tgt.unsqueeze(-2)
        
        tgt_nergibor_fft = torch.fft.fft2(tgt_graph)
        out_imag = src_nergibor_fft.imag - tgt_nergibor_fft.imag       
        out_real = src_nergibor_fft.real - tgt_nergibor_fft.real 
        fft_imag = self.conv4(self.relu2(self.conv3(out_imag)))
        fft_real = self.conv4(self.relu2(self.conv3(out_real)))
        out = fft_imag + fft_real
        out = self.conv6(self.relu3(self.conv5(out)))
        out = out.max(-1)[0] 
        return out    
    
class BISPCR(nn.Module):
    def __init__(self, args):
        super(BISPCR, self).__init__()
        self.emb_dims = args.emb_dims
        self.num_iter = args.num_iter
        self.significance_fc = Conv1DBlock((self.emb_dims, 64, 32, 1), 1)

        self.num_point_preserved = args.num_point_preserved
        self.spe = Conv1DBlock((self.emb_dims, 64, 64), 1)
        self.ope = Conv1DBlock((3, 16, 3), 1)
        self.emb_nn = BIFH(args.emb_dims)
        self.biisi = BIISI(args=args)

        self.fft_conv_src = Conv1DBlock((self.emb_dims * 2, 64), 1)
        self.fft_conv_tgt = Conv1DBlock((self.emb_dims * 2, 64), 1)

        self.fft_conv_src_cross = Conv1DBlock((self.emb_dims * 2, 64), 1)
        self.fft_conv_tgt_cross = Conv1DBlock((self.emb_dims * 2, 64), 1)
        self.conv = Conv1DBlock((64*2,64), 1)

    def forward(self, src, tgt, training, R_gt, t_gt):
        ##### initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        ##### initialize #####
        loss = 0.
        num_iter = self.num_iter
        
        ##### getting ground truth correspondences #####
        if training:
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
            min_dist, min_idx = (dist ** 2).sum(1).min(-1)  # [B, npoint], [B, npoint]
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy()  # drop to cpu for numpy
            match_labels_real = (min_dist < 0.05).float()
            indicator = match_labels_real.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)
        batch_idx = np.arange(src.size(0))[:, np.newaxis]
        ##### getting ground truth correspondences #####

        ##### keypoint detection #####
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt) 
        src_sig_score = self.significance_fc(src_embedding).squeeze(1)
        tgt_sig_score = self.significance_fc(tgt_embedding).squeeze(1)

        num_point_preserved = self.num_point_preserved
        if training:
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_point_preserved // 2, p=pos_probs)
            neg_idx = batch_choice(candidates, num_point_preserved - num_point_preserved // 2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        else:
            src_idx = src_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()
            tgt_idx = tgt_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy()
        if training:
            match_labels = match_labels_real[batch_idx, src_idx]
        src = src[batch_idx, :, src_idx].transpose(1, 2)
        src_embedding = src_embedding[batch_idx, :, src_idx].transpose(1, 2)
        src_sig_score = src_sig_score[batch_idx, src_idx]
        tgt = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_embedding = tgt_embedding[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_sig_score = tgt_sig_score[batch_idx, tgt_idx]
        if training:
            gt_src = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
        ##### keypoint detection #####
        
        if self.training:
            R, t, euler_ab, loss = self.biisi(src, tgt, src_embedding,
                                             tgt_embedding, gt_src, 
                                             src_sig_score, tgt_sig_score, match_labels)
        else:
            R, t, euler_ab, loss = self.biisi(src, tgt, src_embedding, 
                                             tgt_embedding, None, 
                                             src_sig_score, tgt_sig_score, None)

        return R, t, euler_ab, loss

        

class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        self.emb_dims = args.emb_dims
        
        self.bispcr = BISPCR(args=args)

    def forward(self, src, tgt, R_gt=None, t_gt=None):

        ##### only pass ground truth while training #####
        if not (self.training or (R_gt is None and t_gt is None)):
            raise Exception('Passing ground truth while testing')
        ##### only pass ground truth while training #####        
        R, t, euler_ab, loss = self.bispcr(src, tgt, self.training, R_gt, t_gt)
       
        return R, t, loss