import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.aligned_dataset import AlignedDataset_Detection, collate_func
import torch.distributed as dist


def CreateDataset(opt):
    dataset = None
    dataset = AlignedDataset_Detection()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        if opt.isTrain and len(opt.gpu_ids)>0 and not opt.fp16:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, num_replicas=dist.get_world_size(), rank=opt.local_rank)
            self.train_batch_sampler = torch.utils.data.BatchSampler(self.sampler, opt.batchSize, drop_last=True)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                num_workers=int(opt.nThreads),
                collate_fn=collate_func,
                batch_sampler=self.train_batch_sampler)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads),
                collate_fn=collate_func)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

