import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset,DataLoader,DistributedSampler,BatchSampler
import torch.multiprocessing as mp
from random import shuffle
from random import seed


class MultiData(Dataset):
    def __init__(self):
        self.idxes = torch.tensor([ [i]*5 for i in range(40) ])
    def __getitem__(self,idx):
        return self.idxes[idx]
    def __len__(self):
        return len(self.idxes)


def run(idx,datag,numWorkers):

    dist.init_process_group("nccl", rank=idx, world_size=numWorkers)
    sampler = DistributedSampler(datag,drop_last=True)
    loader  = DataLoader(datag,sampler=BatchSampler(sampler, batch_size=5, drop_last=True))
    for e in range(2):
        sampler.set_epoch(e)
        print('-'*10)
        for batch in loader:
            print('%d / %s\n'%(dist.get_rank(),' '.join([str(elt) for elt in batch])))



if __name__ == '__main__':

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

    num_gpus = 3
    datag = MultiData()

    mp.spawn(run, args=(datag,num_gpus),
             nprocs=num_gpus,
             join=True)
