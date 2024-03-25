import numpy as np
from tqdm import tqdm
import random
from model import PretrainBert
import torch
from sklearn.metrics import accuracy_score
import copy
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch.nn as nn

class PretrainModelManager:  

    def __init__(self, args, data):
        self.set_seed(args.seed)
        self.args = args
        self.data = data
        self.model = PretrainBert(args, data)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.optimizer = self.get_optimizer(args)
        
        self.optimization_steps = int(len(data.train_examples) / args.pretrain_batch_size) * args.num_pretrain_epochs
        self.num_warmup_steps = int(args.warmup_proportion * self.optimization_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.optimization_steps)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        return optimizer

    def save_model(self, save_path):
        self.model.save_backbone(save_path)

    def train(self):

        wait = 0
        best_model = None
        mlm_iter = iter(self.data.pretrain_dataloader)

        for epoch in range(self.args.num_pretrain_epochs):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(self.data.pretrain_dataloader):
                # 1. load data
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_coarse, _ = batch
                X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                try:
                    batch = mlm_iter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, _, _ = batch
                except StopIteration:
                    mlm_iter = iter(self.data.pretrain_dataloader)
                    batch = mlm_iter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, _, _ = batch
                X_mlm = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                # 2. get masked data
                mask_ids, mask_lb = self.mask_tokens(input_ids.cpu(), self.tokenizer)
                X_mlm["input_ids"] = mask_ids.to(self.device)

                # 3. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    logits = self.model(X)["logits"]
                    loss_src = self.model.loss_ce(logits, label_coarse)
                    loss_mlm = self.model.mlmForward(X_mlm, mask_lb.to(self.device))
                    lossTOT = loss_src + loss_mlm
                    lossTOT.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    tr_loss += lossTOT.item()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.eval()
            print('score', eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= self.args.wait_patient:
                    break
                
        self.model = best_model
        if self.args.save_model:
            self.save_model(self.args.save_model_path)

    def eval(self):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.data.n_coarse)).to(self.device)

        for batch in self.data.eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_coarse, _ = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                logits = self.model(X)['logits']
                total_labels = torch.cat((total_labels, label_coarse))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc

    def mask_tokens(self, inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.25):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels