from models.modules import TSGM, BSINF, ITE
from models.pytorch_pwc.extract_flow import extract_flow_torch, get_flow_2frames
from models.pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from utils.core import warp

class TimeEmbedding(nn.Module):
    def __init__(self, in_features=1, hidden_features=16, out_features=8):
        super(TimeEmbedding, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = torch.sin

    def forward(self, t):
        x = self.fc1(t)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    

class SINF(nn.Module):
    def __init__(self, img_channels=3, num_resblocks=5, num_channels=64):
        super(SINF, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()

        self.bsinf_forward = BSINF(in_ch=img_channels*2 + num_channels+8, out_ch=num_channels, base_ch=num_channels, num_module=num_resblocks)
        self.bsinf_backward = BSINF(in_ch=img_channels*2 + num_channels+8, out_ch=num_channels, base_ch=num_channels, num_module=num_resblocks)
        self.ite_fusion = ITE(in_channels=num_channels*2, mid_channels=num_channels * 2, out_channels=num_channels, num_res=num_resblocks)

        self.tsgm = TSGM(in_channels=num_channels)
        self.recon_head = ITE(in_channels=num_channels, mid_channels=num_channels, out_channels=img_channels+6, num_res=num_resblocks//2)

        self.time_embedding = TimeEmbedding(in_features=1, hidden_features=16, out_features=8)

    def trainable_parameters(self):
        return [{'params': self.recon_head.parameters()}, {'params': self.bsinf_forward.parameters()}, {'params': self.ite_fusion.parameters()},
                {'params': self.bsinf_backward.parameters()}, {'params': self.tsgm.parameters()}]  # , 'lr': 3e-5

    def forward(self, seqn, noise_level_map=None):
        if self.training:
            return self.forward_training(seqn, noise_level_map)
        else:
            return self.forward_testing(seqn, noise_level_map)

    def forward_testing(self, seqn, noise_level_map=None):
        feature_device = torch.device('cpu')
        N, T, C, H, W = seqn.shape
        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty(N, T, C+6, H, W, device='cuda')

        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        t_norm = torch.full((N, 1), 0 / float(T-1), device=seqn.device)
        t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
        forward_input = torch.cat((seqn[:, 0], init_forward_h, noise_level_map[:, 0], t_emb), dim=1)
        forward_h = self.bsinf_forward(forward_input)
        #forward_h = self.bsinf_forward(torch.cat((seqn[:, 0], init_forward_h, noise_level_map[:, 0]), dim=1))
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            flow = extract_flow_torch(self.pwcnet, seqn[:, i], seqn[:, i-1])
            aligned_forward_h, _ = warp(forward_h, flow)
            t_norm = torch.full((N, 1), i / float(T-1), device=seqn.device)
            t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
            forward_input = torch.cat((seqn[:, i], aligned_forward_h, noise_level_map[:, i], t_emb), dim=1)
            forward_h = self.bsinf_forward(forward_input)
            forward_h = self.ite_fusion(torch.cat((forward_h, aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        t_norm = torch.full((N, 1), (T-1) / float(T-1), device=seqn.device)
        t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
        backward_input = torch.cat((seqn[:, -1], init_backward_h, noise_level_map[:, -1], t_emb), dim=1)
        backward_h = self.bsinf_backward(backward_input)
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            flow = extract_flow_torch(self.pwcnet, seqn[:, T-i], seqn[:, T-i+1])
            aligned_backward_h, _ = warp(backward_h, flow)
            t_norm = torch.full((N, 1), (T-i) / float(T-1), device=seqn.device)
            t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
            backward_input = torch.cat((seqn[:, T-i], aligned_backward_h, noise_level_map[:, T-i], t_emb), dim=1)
            backward_h = self.bsinf_backward(backward_input)
            backward_h = self.ite_fusion(torch.cat((backward_h, aligned_backward_h), dim=1))

            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        # for i in tqdm(range(T)):
        for i in range(T):
            # seqdn[:, i] = self.tsgm(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))
            temp = self.tsgm(forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device))
            seqdn[:, i] = self.recon_head(temp)

        return seqdn, None

    def forward_training(self, seqn, noise_level_map):
        feature_device = torch.device('cuda')
        N, T, C, H, W = seqn.shape

        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty(N, T, C+6, H, W, device='cuda')

        # extract optical flow
        flows_backward, flows_forward = get_flow_2frames(self.pwcnet, seqn)

        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        t_norm = torch.full((N, 1), 0 / float(T-1), device=seqn.device)
        t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
        forward_input = torch.cat((seqn[:, 0], init_forward_h, noise_level_map[:, 0], t_emb), dim=1)
        forward_h = self.bsinf_forward(forward_input)
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            aligned_forward_h, _ = warp(forward_h, flows_forward[:, i-1].cuda())
            t_norm = torch.full((N, 1), i / float(T-1), device=seqn.device)
            t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
            forward_input = torch.cat((seqn[:, i], aligned_forward_h, noise_level_map[:, i], t_emb), dim=1)
            forward_h = self.bsinf_forward(forward_input)
            forward_h = self.ite_fusion(torch.cat((forward_h, aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        t_norm = torch.full((N, 1), (T-1) / float(T-1), device=seqn.device)
        t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
        backward_input = torch.cat((seqn[:, -1], init_backward_h, noise_level_map[:, -1], t_emb), dim=1)
        backward_h = self.bsinf_backward(backward_input)
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            aligned_backward_h, _ = warp(backward_h, flows_backward[:, T-i].cuda())
            t_norm = torch.full((N, 1), (T-i) / float(T-1), device=seqn.device)
            t_emb = self.time_embedding(t_norm).view(N, -1, 1, 1).expand(-1, -1, H, W)
            backward_input = torch.cat((seqn[:, T-i], aligned_backward_h, noise_level_map[:, T-i], t_emb), dim=1)
            backward_h = self.bsinf_backward(backward_input)
            backward_h = self.ite_fusion(torch.cat((backward_h, aligned_backward_h), dim=1))

            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        # iterate = if self.training else tqdm(range(T))
        for i in range(T):
            # seqdn[:, i] = self.tsgm(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))
            temp = self.tsgm(forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device))
            seqdn[:, i] = self.recon_head(temp)
        return seqdn, None
