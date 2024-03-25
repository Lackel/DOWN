import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import get_linear_schedule_with_warmup, logging, WEIGHTS_NAME, AutoTokenizer
from transformers.optimization import AdamW
from sklearn.cluster import KMeans
from model import MCNBert
from utils.util import clustering_score, view_generator
from utils.memory import MemoryBank, fill_memory_bank
from init_parameter import init_model
from data import Data, NeighborsDataset
from pretrain import PretrainModelManager
from torch.utils.data import DataLoader
import math

class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        self.args = args
        self.data = data
        self.set_seed()
        self.model = MCNBert(args, data.n_fine, data.n_coarse)
        self.model_m = MCNBert(args, data.n_fine, data.n_coarse)
        if pretrained_model is None:
            pretrained_model = PretrainModelManager(args, data)
            if os.path.exists(args.save_model_path):
                pretrained_model = self.restore_model(args, pretrained_model.model)
            pretrained_dict = pretrained_model.backbone.state_dict()
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)
            self.model_m.backbone.load_state_dict(pretrained_dict, strict=False)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_m.to(self.device)
        self.freeze_parameters_m(self.model_m)
        self.optimizer = self.get_optimizer(args)
        self.num_training_steps = int(
            len(data.train_examples) / args.train_batch_size) * 100
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps) 
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        self.m = args.momentum_factor

    def set_seed(self):
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def get_features_labels(self, dataloader, model, args, visual=False):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        if visual:
            coarse_labels = torch.empty(0,dtype=torch.long).to(self.device)
            for _, batch in enumerate(dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, coarse_ids, label_ids = batch
                X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                with torch.no_grad():
                    feature = model(X)
                coarse_labels = torch.cat((coarse_labels, coarse_ids))
                total_features = torch.cat((total_features, feature))
                total_labels = torch.cat((total_labels, label_ids))

            return total_features, coarse_labels, total_labels

        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, _, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def momentum_update_encoder_m(self):
        """
        Updating the Momentum BERT.
        We only update the last four layers by default.
        """
        for (_, param_q), (_, param_m) in zip(self.model.backbone.named_parameters(), self.model_m.backbone.named_parameters()):
                param_m.data = param_m.data * self.m + param_q.data * (1. - self.m)

    def freeze_parameters_m(self, model):
        """
        Freeze all the weights of Momentum BERT.
        """
        for _, param in model.named_parameters():
            param.requires_grad = False
    
    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(data.train_dataset, indices)
        self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    def get_neighbor_inds(self, args, data, rank=False):
        """get indices of neighbors"""
        self.memory_bank = MemoryBank(len(data.train_dataset), args.feat_dim, data.n_coarse, 0.1)
        fill_memory_bank(data.train_dataloader, self.model, self.memory_bank)
        print("mine nearest neighbors")
        indices, acc = self.memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=True, rank=rank)

        return indices
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                # if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                #     adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj

    def get_mask(self, inds, neighbors, epoch):
        """get adjacency matrix"""
        mask = torch.zeros(inds.shape[0], self.memory_bank.features.shape[0])
        
        bases = [150, 10, 5, 2]
        base = bases[math.floor(epoch/5)]
        for b1, n in enumerate(inds):
            rank = 0
            for b2 in range(neighbors.shape[1]):
                mask[b1][neighbors[b1][b2]] = np.power(base, -rank/self.args.topk)
                rank += 1
        return mask

    def get_optimizer(self, args):
        """
        Setting the optimizer with weight decay for BERT.
        """
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        return optimizer

    def train(self, args, data):
        # load neighbors for the first epoch
        indices = self.get_neighbor_inds(args, data, rank=True)
        self.get_neighbor_dataset(args, data, indices)

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for _, batch in enumerate(self.train_dataloader):
                # load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"]) # anchor data
                fine_label = batch["target"].to(self.device)
                coarse_label = batch["coarse_label"].to(self.device)
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # data ind

                mask = self.get_mask(data_inds, pos_neighbors, epoch).cuda()
                X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                with torch.set_grad_enabled(True):
                    hidden_states, logits = self.model(X_an, output_logits=True)
                    loss_coarse = self.model.coarse_loss(logits, coarse_label)
                    loss_fine = self.model.fine_loss(hidden_states, self.memory_bank.features.detach().cuda(), mask, 0.07)
                    hidden_states_m = self.model_m(X_an)
                    self.memory_bank.up(hidden_states_m, data_inds, fine_label, coarse_label)
                    loss = loss_fine + loss_coarse
                    tr_loss += loss.item()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.momentum_update_encoder_m()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1
            torch.cuda.empty_cache()
            loss = tr_loss / nb_tr_steps
            print('Epoch ' + str(epoch) + ' loss:' + str(loss))
            indices = self.get_neighbor_inds(args, data)
            self.get_neighbor_dataset(args, data, indices)

    def test(self):
        """
        Testing trained model on the test sets by clustering.
        """
        self.model.eval()

        feats, coarse, labels = self.get_features_labels(self.data.test_dataloader, self.model, self.args, visual=True)
        feats = F.normalize(feats, dim=1)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = self.data.n_fine, n_init=20, random_state=self.args.seed).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results_all = clustering_score(y_true, y_pred)
        print(results_all)
        return results_all

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.save_model_path, WEIGHTS_NAME)
        model.backbone.load_state_dict(torch.load(output_model_file))
        return model

    def pairwise_cosine_sim(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.matmul(x, y.T)

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.set_verbosity_error()
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    data = Data(args)
    
    pretrain = PretrainModelManager(args, data)
    pretrain.train()
    
    manager = ModelManager(args, data)
    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')

    
    print('Evaluation begin...')
    manager.test()
    print('Evaluation finished!')