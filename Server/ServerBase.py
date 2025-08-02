
import torch

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
            seq = seq.to(self.device)
            noise = torch.empty_like(seq).normal_(mean=0, std=self.args.test_noise).to(self.device)
            noisy_seq = seq + noise
            noise_map = torch.tensor([self.args.test_noise], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                denoised_seq = denoise_seq_fastdvdnet(seq = denoised_seq,
                                                       noise_std = noise_map,
                                                       temporal_window=self.args.temp_psz,  # e.g. 5
                                                       model=self.global_model
                                                       )
            psnr_clean = batch_psnr(denoised_seq, seq, data_range=1.0)
            psnr_noisy = batch_psnr(noisy_seq.squeeze(), seq, data_range=1.0)

            total_psnr += psnr_clean
            total_noisy_psnr += psnr_noisy
            cnt += 1

            self.logger.info(f"[{idx}] PSNR_noisy: {psnr_noisy:.4f} dB | PSNR_denoised: {psnr_clean:.4f} dB")

        avg_psnr = total_psnr / cnt
        avg_psnr_noisy = total_noisy_psnr / cnt

        print(f"\n[Global Test PSNR] Clean: {avg_psnr:.4f} dB | Noisy: {avg_psnr_noisy:.4f} dB\n")
        self.logger.info(f"[Global Test PSNR] Clean: {avg_psnr:.4f} dB | Noisy: {avg_psnr_noisy:.4f} dB")

    def Save_CheckPoint(self, save_path):
        torch.save(self.global_model.state_dict(), save_path)

