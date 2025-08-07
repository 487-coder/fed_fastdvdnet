import copy

from Server.ServerBase import Server
from Client.fedavg_client import ClientFedAvg
from tqdm import tqdm
import numpy as np
from utils import average_weights
from utils import MemReporter
import time
from utils import save_model_checkpoint

class ServerFedAvg(Server):
    def __init__(self, args, global_model, client_dataloader, client_noise_level, client_noise_type,local_test_dataloader, global_test_dataloader, logger, device):
        super().__init__(args, global_model, client_dataloader,client_noise_level, client_noise_type, local_test_dataloader, global_test_dataloader, logger, device)

    def Create_Clients(self):
        for idx in range(self.args.client_numbers):
            self.LocalModels.append(ClientFedAvg(self.args, copy.deepcopy(self.global_model), self.client_dataloader[idx],
                                                 self.client_noise_level, self.client_noise_type,
                                                 self.local_test_dataloader[idx], idx=idx,logger=self.logger,
                                                 device=self.device))#这里后面要加noise_level,noise_type

    def train(self):

        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.num_epochs)):
            current_lr = self.args.lr * (0.1 ** (epoch // 2))
            #test_accuracy = 0
            global_psnr = 0
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch + 1} |\n')
            m = max(int(self.args.sampling_rate * self.args.client_numbers), 1)
            idxs_users = np.random.choice(range(self.args.client_numbers), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch,lr=current_lr)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                local_psnr = self.LocalModels[idx].test_psnr()
                global_psnr += local_psnr
                self.logger.add_scalar(f'Client_{idx}/Loss', loss, epoch)
                self.logger.add_scalar(f'Client_{idx}/PSNR', local_psnr, epoch)

            # update global weights
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average test psnr:', global_psnr / m)
            self.logger.add_scalar('Global/Average_Loss', loss_avg, epoch)
            self.logger.add_scalar('Global/Average_PSNR', global_psnr / m, epoch)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                global_test_psnr, global_psnr_noisy = self.global_test_psnr()
                self.logger.add_scalar('Global/Test_PSNR', global_test_psnr, epoch)
                self.logger.add_scalar('Global/Test_PSNR_Noisy', global_psnr_noisy, epoch)
            save_model_checkpoint(
                model=self.global_model,
                config={
                    'log_dir': self.args.save_dir,
                    'save_every_epochs': 5  # 每轮都保存，也可以换成比如5
                },
                optimizer=None,  # 没有使用server端optimizer就设为None
                train_pars={
                    'epoch_loss': loss_avg,
                    'epoch': epoch
                },
                epoch=epoch,
                role='global'
            )
        print('Training is completed.')
        end_time = time.time()

