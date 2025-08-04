
import torch
import copy
import torch.nn as nn

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
        self.ctrl_fr_idx = (self.args.temp_psz - 1) // 2
        self.device = device
        #self.kld = nn.KLDivLoss()
        #self.mse = nn.MSELoss()
        self.model = copy.deepcopy(model)

    def test_psnr(self):
        self.model.eval()
        total_psnr = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, seq in enumerate(self.testloader):
                seq = seq.to(self.device)  # [1, T, C, H, W]
                if seq.dim() == 5 and seq.shape[0] == 1:
                    seq = seq.squeeze(0)  # [T, C, H, W]

                noise = torch.empty_like(seq).normal_(mean=0, std=self.args.test_noise/255.0).to(self.device)
                noisy_seq = seq + noise
                noisy_seq = torch.clamp(noisy_seq, 0.0, 1.0)
                noise_map = torch.tensor([self.args.test_noise/255.0], dtype=torch.float32).to(self.device)

                denoised_seq = denoise_seq_fastdvdnet(
                    seq=noisy_seq,
                    noise_std=noise_map,
                    temporal_window=self.args.temp_psz,
                    model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                )

                # 只评估中心帧
                gt = seq[self.ctrl_fr_idx].unsqueeze(0)
                pred = denoised_seq[self.ctrl_fr_idx].unsqueeze(0)
                #print("GT min/max:", gt.min().item(), gt.max().item())
                #print("Pred min/max:", pred.min().item(), pred.max().item())

                psnr = batch_psnr(pred, gt, data_range=1.0)
                total_psnr += psnr
                cnt += 1
                print(f"[Client {self.idx}] PSNR on sample {cnt}: {psnr:.2f}")
        return total_psnr / cnt

    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)