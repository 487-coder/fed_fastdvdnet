import numpy as np
import torch

from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from fastdvd_model import denoise_seq_fastdvdnet
from utils import batch_psnr


class Client(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """

    def __init__(self, args, model, trainloader,noise_level,noise_type,testloader, idx, logger, device):
        '''    def __init__(self, args, model, trainloader, noise_level,noise_type,testloader, idx, logger, device):
        self.LocalModels.append(ClientFedAvg(self.args, copy.deepcopy(self.global_model), self.client_dataloader[idx],
                                                 self.client_noise_level, self.client_noise_type,
                                                 self.local_test_dataloader[idx], idx=idx,logger=self.logger,
                                                 device=self.device))#这里后面要加noise_level,noise_type'''
        self.args = args
        self.logger = logger
        self.trainloader = trainloader
        self.testloader = testloader
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.idx = idx
        self.criterion = nn.MSELoss(reduction='sum')

        self.device = device
        #self.kld = nn.KLDivLoss()
        #self.mse = nn.MSELoss()
        self.model = copy.deepcopy(model)

    def test_psnr(self):
        self.model.eval()
        total_psnr = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, (seq,) in enumerate(self.testloader):
                seq = seq.to(self.device)
                noise = torch.empty_like(seq).normal_(mean=0, std=self.noise_level).to(self.device)
                noisy_seq = seq + noise
                noise_map = torch.tensor([self.noise_level], dtype=torch.float32).to(self.device)

                denoised_seq = denoise_seq_fastdvdnet(
                    seq=noisy_seq,
                    noise_std=noise_map,
                    temporal_window=self.args.temp_psz,
                    model=self.model
                )

                psnr = batch_psnr(denoised_seq, seq, data_range=1.0)
                total_psnr += psnr
                cnt += 1
        return total_psnr / cnt

    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)