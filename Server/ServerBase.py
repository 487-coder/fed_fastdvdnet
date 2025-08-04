
import torch
from torch import nn

from utils import batch_psnr
from fastdvd_model import denoise_seq_fastdvdnet


class Server(object):
    def __init__(self,args, global_model, client_dataloader, client_noise_level, client_noise_type,local_test_dataloader,
                 global_test_dataloader, logger, device):
        '''class ServerFedAvg(Server):
    def __init__(self, args, global_model, client_dataloader, client_noise_level, client_noise_type,local_test_dataloader, global_test_dataloader, logger, device):
        super().__init__(args, global_model, client_dataloader,client_noise_level, client_noise_type, local_test_dataloader, global_test_dataloader, logger, device)
'''
        self.global_model = global_model
        self.args = args
        self.client_dataloader = client_dataloader
        self.client_noise_level = client_noise_level
        self.client_noise_type = client_noise_type
        self.local_test_dataloader = local_test_dataloader
        self.global_test_dataloader = global_test_dataloader
        self.ctrl_fr_idx = (self.args.temp_psz - 1) // 2
        self.logger = logger
        self.device = device
        self.LocalModels = []
#
    def global_test_psnr(self):
        self.global_model.eval()
        total_psnr = 0.0
        total_noisy_psnr = 0.0
        cnt = 0

        for idx, (seq,) in enumerate(self.global_test_dataloader):
            seq = seq.to(self.device)  # [1, T, C, H, W]
            if seq.dim() == 5 and seq.shape[0] == 1:
                seq = seq.squeeze(0)  # [T, C, H, W]

            noise = torch.empty_like(seq).normal_(mean=0, std=self.args.test_noise/255.0).to(self.device)
            noisy_seq = seq + noise
            noisy_seq = torch.clamp(noisy_seq, 0.0, 1.0)

            noise_map = torch.tensor([self.args.test_noise/255.0], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                denoised_seq = denoise_seq_fastdvdnet(
                    seq=noisy_seq,
                    noise_std=noise_map,
                    temporal_window=self.args.temp_psz,
                    model=self.global_model.module if isinstance(self.global_model,
                                                                 nn.DataParallel) else self.global_model
                )

            # 中心帧比较
            gt = seq[self.ctrl_fr_idx].unsqueeze(0)
            pred = denoised_seq[self.ctrl_fr_idx].unsqueeze(0)
            noisy_center = noisy_seq[self.ctrl_fr_idx].unsqueeze(0)
            #print("GT min/max:", gt.min().item(), gt.max().item())
            #print("Pred min/max:", pred.min().item(), pred.max().item())

            psnr_clean = batch_psnr(pred, gt, data_range=1.0)
            psnr_noisy = batch_psnr(noisy_center, gt, data_range=1.0)

            total_psnr += psnr_clean
            total_noisy_psnr += psnr_noisy
            cnt += 1

            print(f"[{idx}] PSNR_noisy: {psnr_noisy:.2f} dB | PSNR_denoised: {psnr_clean:.2f} dB")

        avg_psnr = total_psnr / cnt
        avg_psnr_noisy = total_noisy_psnr / cnt

        print(f"\n[Global Test PSNR] Clean: {avg_psnr:.4f} dB | Noisy: {avg_psnr_noisy:.4f} dB\n")

        return avg_psnr, avg_psnr_noisy


    def Save_CheckPoint(self, save_path):
        torch.save(self.global_model.state_dict(), save_path)

