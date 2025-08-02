import torch
import os.path
from torch.utils.data import DataLoader

import pickle
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from fastdvd_model import FastDVDnet
from Server.fedavg_server import ServerFedAvg
from new_dataset import partition,partition_test_dataset
from new_dataset import LocalDataset,TestDataset
from options import args_parser







print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#np.set_printoptions(threshold=np.inf)


args = args_parser()

client_dataset_path = partition(args.train_dataset,args.client_numbers)
global_testset = TestDataset(data_dir=args.test_dataset)
global_testloader = DataLoader(global_testset, batch_size=1)

client_dataset = {}
client_testset = {}
client_dataloader = {}
client_test_dataloader = {}
for i in range(args.client_numbers):
    client_dataset[i] = LocalDataset(client_dataset_path[i],sequence_length= 5, crop_size=args.patch_size)
    client_testset[i] = partition_test_dataset(global_testset,args.client_numbers)
    client_dataloader[i] = DataLoader(client_dataset[i],batch_size=args.batch_size,shuffle=False,num_workers=4)
    client_test_dataloader[i] = DataLoader(client_testset[i],batch_size=1)



logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+ args.train_dataset + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
print('Checkpoint dir:', checkpoint_dir)


global_model = FastDVDnet()
global_model = nn.DataParallel(global_model)
global_model.to(device)

server = ServerFedAvg(args,global_model,client_dataloader,args.client_noise_level,args.client_noise_type,client_test_dataloader,global_testloader,logger,device)
'''class ServerFedAvg(Server):
    def __init__(self, args, global_model, client_dataloader, client_noise_level, client_noise_type,local_test_dataloader, global_test_dataloader, logger, device):
        super().__init__(args, global_model, client_dataloader,client_noise_level, client_noise_type, local_test_dataloader, global_test_dataloader, logger, device)
'''
server.Create_Clients()
server.train()
server.global_test_psnr()

save_path = checkpoint_dir + '.pth'
if args.upload_model == True:
    server.Save_CheckPoint(save_path)
    print('Model is saved on: ')
    print(save_path)

