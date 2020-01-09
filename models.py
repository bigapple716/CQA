# -*- coding: utf-8 -*-

import torch.nn as nn
from pytorch_pretrained import BertModel


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained('./bert_pretrain')
        for param in self.bert.parameters():
            param.requires_grad = True
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, q, a):
        _, pooled_q = self.bert(q, output_all_encoded_layers=False)
        _, pooled_a = self.bert(a, output_all_encoded_layers=False)
        return self.cos_sim(pooled_q, pooled_a)


class Ernie(nn.Module):
    def __init__(self):
        super(Ernie, self).__init__()

        self.ernie = BertModel.from_pretrained('./ERNIE_pretrain')
        for param in self.ernie.parameters():
            param.requires_grad = True
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, q, a):
        _, pooled_q = self.ernie(q, output_all_encoded_layers=False)
        _, pooled_a = self.ernie(a, output_all_encoded_layers=False)
        return self.cos_sim(pooled_q, pooled_a)
