import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F

class PretrainBert(nn.Module):
    def __init__(self, args, data):
        super(PretrainBert, self).__init__()
        self.num_labels = data.n_coarse
        self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

class MCNBert(nn.Module):
    
    def __init__(self, args, n_fine, n_coarse, feat_dim=128):
        super(MCNBert, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_fine = n_fine
        self.n_coarse = n_coarse
        self.args = args
        self.model_name = args.model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.temperature = args.temperature
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.classifier = nn.Linear(feat_dim, n_coarse)
        self.backbone.to(self.device)
        self.head.to(self.device)
        
    def forward(self, X, output_logits=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = self.head(cls_embed)
        logits = self.classifier(features)

        cls_embed = F.normalize(cls_embed, dim=1)
        
        if output_logits:
            return cls_embed, logits
        return cls_embed

    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

    def fine_loss(self, features, k, mask, temperature):
        logits = F.cosine_similarity(features.unsqueeze(1), k.unsqueeze(0), dim=2) / temperature
        exp_logits = torch.exp(logits)
        log_prob =  logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss     

    def coarse_loss(self, logits, labels):
        loss_ce = nn.CrossEntropyLoss()(logits, labels)
        return loss_ce


