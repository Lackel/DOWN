import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class MemoryBank(object):
    
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.coarse_labels = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True, rank=False):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        # index = faiss.IndexFlatL2(dim) 
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        # print(indices[191,:])
        # print(distances[191,:])
        # data = np.array(distances).mean(axis=0)
        # np.savez('./dis.npz', acc=data)
        # distances = torch.tensor(distances)
        # print(distances.mean(dim=0))
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            # np.savez('./acc.npz', acc=(neighbor_targets == anchor_targets).mean(axis=0))
            # print(str())
            return indices, accuracy

        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets, coarse_labels):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.coarse_labels[self.ptr:self.ptr+b].copy_(coarse_labels.detach())
        self.ptr += b

    def up(self, feature, index, label, coarse_labels):
        for index, item in enumerate(index):
            self.features[item].copy_(feature[index].detach())
            self.targets[item].copy_(label[index].detach())
            self.coarse_labels[item].copy_(coarse_labels[index].detach()) 

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):

        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, coarse_labels, label_ids = batch
        X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
        feature = model(X)

        memory_bank.update(feature, label_ids, coarse_labels)
        # if i % 10 == 0:
        #     print('Fill Memory Bank [%d/%d]' %(i, len(loader)))