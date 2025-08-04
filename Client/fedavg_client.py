import copy
import torch
import torch.nn as nn
import torch.optim as optim
from utils import save_model_checkpoint


from utils import normalize_augment
from Client.ClientBase import Client


class ClientFedAvg(Client):
    def __init__(self, args, model, trainloader, noise_level,noise_type,testloader, idx, logger, device):
        '''self.LocalModels.append(ClientFedAvg(self.args, copy.deepcopy(self.global_model), self.client_dataloader[idx],
                                                 self.client_noise_level, self.client_noise_type,
                                                 self.local_test_dataloader[idx], idx=idx,logger=self.logger,
                                                 device=self.device))#这里后面要加noise_level,noise_type'''
        self.args = args
        self.model = copy.deepcopy(model)   # 本地副本
        self.trainloader = trainloader
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.testloader = testloader
        self.idx = idx
        self.logger = logger
        self.device = device

        self.ctrl_fr_idx = (self.args.temp_psz - 1) // 2
        self.criterion = nn.MSELoss(reduction='sum').to(device)

    def update_weights(self, global_round):
        self.model.to(self.device)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = self.criterion.to(self.device)

        epoch_loss = []

        for epoch in range(self.args.local_ep):
           batch_loss = []
           for batch_idx,(seq,gt) in enumerate(self.trainloader):
               self.model.train()
               optimizer.zero_grad()
               img_train, gt_train = normalize_augment(seq, self.ctrl_fr_idx)
               img_train = img_train.to(self.device, non_blocking=True)
               gt_train = gt_train.to(self.device, non_blocking=True)
               N, _, H, W = img_train.size()
               stdn = torch.empty((N, 1, 1, 1), device=self.device).uniform_(
                   self.noise_level [0]/255.0, self.noise_level[1]/255.0
               )
               noise = torch.normal(mean=0.0, std=stdn.expand_as(img_train))
               #print("testValue")
               #print(img_train.max(), img_train.min())
               imgn_train = img_train + noise
               imgn_train = torch.clamp(imgn_train, 0.0, 1.0)
               noise_map = stdn.expand((N, 1, H, W)).to(self.device)
               out_train = self.model(imgn_train, noise_map)
               loss = criterion(out_train, gt_train) / (N * 2)
               loss.backward()
               if self.args.clip_grad is not None:
                   nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
               optimizer.step()


               if batch_idx % 10 == 0:
                   print(f"| Global Round: {global_round} | Client: {self.idx} | Local Epoch: {epoch} | "
                         f"[{batch_idx * len(img_train)}/{len(self.trainloader.dataset)} ({100. * batch_idx / len(self.trainloader):.0f}%)] "
                         f"\tLoss: {loss.item():.6f}")
                   self.logger.add_scalar('loss', loss.item())

               batch_loss.append(loss.item())

           epoch_loss.append(sum(batch_loss) / len(batch_loss))

           save_model_checkpoint(
                model=self.model,
                config={
                   'log_dir': self.args.save_dir,
                   'save_every_epochs': 1  # 每轮都保存，或自定义
                },
                optimizer=optimizer,
                train_pars={'epoch_losses': epoch_loss[-1], 'epoch': global_round},
                epoch=global_round,
                role="client",
                client_id=self.idx
               )

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

