import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import _LRScheduler


class FindLR(_LRScheduler):
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):
        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in
                self.base_lrs]


class LRFinder(object):
    def __init__(self, args, data_loader, model, criterion, optimizer):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def start(self):
        loader = self.data_loader['train']
        lr_scheduler = FindLR(self.optimizer, num_iter=self.args.epochs)
        epoches = int(self.args.epochs / len(loader)) + 1

        n = 0

        learning_rate = []
        losses = []
        for epoch in range(epoches):
            self.model.train()

            for batch_idx, (inputs, target) in enumerate(loader, 0):
                if n > self.args.epochs:
                    break

                lr_scheduler.step()

                target = target.cuda()
                inputs_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)

                output = self.model(inputs_var)
                loss = self.criterion(output, target_var.long())

                self.optimizer.zero_grad()
                if torch.isnan(loss).any():
                    n += 1e8
                    break
                loss.backward()
                self.optimizer.step()

                print('Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'.format(
                    loss.item(),
                    self.optimizer.param_groups[0]['lr'],
                    iter_num=n,
                    trained_samples=batch_idx * self.args.batch_size + len(inputs),
                    total_samples=len(loader),
                ))

                learning_rate.append(self.optimizer.param_groups[0]['lr'])
                losses.append(loss.item())
                n += 1

        learning_rate = learning_rate[10:-5]
        losses = losses[10:-5]

        fig, ax = plt.subplots(1, 1)
        ax.plot(learning_rate, losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('losses')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

        fig.savefig('result.jpg')