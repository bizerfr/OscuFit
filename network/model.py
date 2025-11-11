import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)

class GraphLayer(nn.Module):
    def __init__(self, dim=64, k1=40, k2=20):
        super(GraphLayer, self).__init__()
        self.dim = dim
        self.k1 = k1
        self.k2 = k2

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn4 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm2d(self.dim)

        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                    self.bn4,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_knn1 = get_graph_feature(x, k=self.k1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x_knn1 = self.conv1(x_knn1)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn1 = self.conv2(x_knn1)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_knn1.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x_knn2 = get_graph_feature(x, self.k2)
        x_knn2 = self.conv3(x_knn2)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn2 = self.conv4(x_knn2)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_k1.unsqueeze(-1).repeat(1, 1, 1, self.k2)

        out = torch.cat([x_knn2, x_k1], dim=1)

        out = self.conv5(out)
        out = out.max(dim=-1, keepdim=False)[0]

        return out

class AdaptiveLayer(nn.Module):
    def __init__(self, C, r=8):
        super(AdaptiveLayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(C, C // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // r, C, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        fea = x[0] + x[1]
        b, C, _ = fea.shape
        out = self.squeeze(fea).view(b, C)
        out = self.excitation(out).view(b, C, 1)
        attention_vectors = out.expand_as(fea)
        fea_v = attention_vectors * x[0] + (1 - attention_vectors) * x[1]
        return fea_v

class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x        

class PointNetFeatures(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', k1=40, k2=20):
        super(PointNetFeatures, self).__init__()
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.k1 = k1
        self.k2 = k2
        self.graph_layer = GraphLayer(dim=64, k1=self.k1,  k2=self.k2)


        self.ada_layer = AdaptiveLayer(C=64)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))# 128,64,128
        x = F.relu(self.bn2(self.conv2(x)))

        x_k = self.graph_layer(x)
        x = self.ada_layer([x, x_k])

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)#(128,64,64)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)#128,64,128
        else:
            trans2 = None

        return x

class PointNetEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', k1=40, k2=20):
        super(PointNetEncoder, self).__init__()
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op, k1=k1, k2=k2)
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256+128+64, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        self.k1 = k1
        self.k2 = k2

        self.graph_layer = GraphLayer(dim=128, k1=self.k1, k2=self.k2)

        self.ada_layer = AdaptiveLayer(C=128)


    def forward(self, points):
        n_pts = points.size()[2]
        pointfeat = self.pointfeat(points)#pointfeat 128,64,128

        x1 = F.relu(self.bn2(self.conv2(pointfeat)))#(128,128,128)
        x_k = self.graph_layer(x1)
        x1 = self.ada_layer([x1, x_k])

        x2 = F.relu(self.bn3(self.conv3(x1)))  # (128,128,128)
        x = torch.cat([pointfeat, x1, x2], 1)
        x = self.bn4(self.conv4(x))#(128,1024,128)
        global_feature = torch.max(x, 2, keepdim=True)[0]#(128,1024,1)
        x = global_feature.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat, x1, x2], 1), global_feature.squeeze()
    

class GraphCNN(nn.Module):
    def __init__(self, k=1, num_points=500, use_point_stn=False,  use_feat_stn=False, point_tuple=1,
                 sym_op='max',  k1=40, k2=20):
        super(GraphCNN, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.num_points=num_points
        self.point_tuple = point_tuple

        self.feat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                            point_tuple=point_tuple, sym_op=sym_op, k1=k1, k2=k2)

        feature_dim = 1024 + 64 + 128 + 256

        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Attention & Position Encoding
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.ffn = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.1)

        self.conv_offset = nn.Conv1d(128, 3, 1)

    def forward(self, points):
        batch_size = points.size(0)
        x, global_feat = self.feat(points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x0 = F.relu(self.bn3(self.conv3(x))) # B*128*N

        points = points.permute(0, 2, 1)         # (B, N, 3)
        x0 = x0.permute(0, 2, 1)                 # (B, N, 128)
        pos_enc = self.pos_mlp(points)           # (B, N, 128)
        x0 = x0 + pos_enc
        # Multihead Attention + Residual + FFN
        x_attn, _ = self.attn(x0, x0, x0)        # (B, N, 128)
        x0 = x0 + x_attn                         # Residual connection
        x0 = x0 + self.dropout(self.ffn(x0))     # Residual + FFN
        x0 = self.norm(x0)                       # LayerNorm
        # Back to (B, 128, N)
        x0 = x0.permute(0, 2, 1)
        
        x0 = F.normalize(x0, p=2.0, dim=1)
        x = x0 - x0[:, :, 0].unsqueeze(-1)
        d = torch.sum( x * x, dim=1)
        weights = torch.softmax(-d, dim=1)
        
        offsets = self.conv_offset(x0)
        offsets[:, :, 0] = 0 #adafit
        
        return weights, offsets

    def get_loss(self, n_est, n_gt, 
                kg_est, km_est, kg_gt, km_gt,
                u_est, u_gt):      
        
        # fitting loss
        loss_fit = torch.mean(1.0 - torch.abs(F.cosine_similarity(u_est, u_gt, dim=1)))

        # query normal loss
        loss_query_normal = torch.mean(1.0 - torch.abs(torch.sum(n_est * n_gt, dim=1))) 
        
        loss_query_kg = F.l1_loss(torch.tanh(kg_est), torch.tanh(kg_gt))

        loss_query_km = F.l1_loss(torch.tanh(km_est).unsqueeze(-1) * n_est, torch.tanh(km_gt).unsqueeze(-1) * n_gt)

        loss = loss_fit + loss_query_normal + loss_query_kg + loss_query_km 

        loss_dict = {
                    "loss_fit" : loss_fit,
                    "loss_query_normal" : loss_query_normal, 
                    "loss_query_kg" : loss_query_kg,
                    "loss_query_km" : loss_query_km
                    }

        return loss, loss_dict    

