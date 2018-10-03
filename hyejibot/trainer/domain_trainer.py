import torch
from torch import optim
import torch.nn as nn

class DomainTrainer:
    def __init__(self, embedding, model,
                 config, data_loader, valid_data_loader=None):
        self.embedding = embedding
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(embedding.parameters())+list(model.parameters()))
        self.config = config
        self.batch_size = config.batch
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

    def _eval_metrics(self, output, target):
        pred = torch.topk(output, 1)[1].squeeze(-1)
        correct = torch.eq(pred, target).sum()
        return correct.item()/len(target)

    def _train_epoch(self):
        self.model.train()

        total_loss, correct = 0, 0
        for i, (src, src_len, trg) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            embed_src = self.embedding(src)
            output = self.model(embed_src)
            loss = self.loss(output, trg)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += self._eval_metrics(output, trg)
        log = {
            'loss' : total_loss / len(self.data_loader),
            'acc' : correct / len(self.data_loader)
        }
        return log

    def _valid_epoch(self):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for i, (src, src_len, trg) in enumerate(self.valid_data_loader):
                embed_src = self.embedding(src)
                output = self.model(embed_src)
                loss = self.loss(output, trg)
                total_loss += loss.item()
                correct += self._eval_metrics(output, trg)
        log = {
            'acc' : correct / len(self.valid_data_loader),
            'loss' : total_loss / len(self.valid_data_loader)
        }
        return log

    def train(self):
        for i in range(self.config.epoch):
            train_log = self._train_epoch()
            valid_log = self._valid_epoch()
            print("[{}] train loss {:.04f} train acc {:02f}  valid loss {:.04f}  acc {:02f}".format(i, train_log["loss"], train_log["acc"], valid_log["loss"],valid_log["acc"]))

            # early stop
            if train_log["acc"] == 1.0:
                break