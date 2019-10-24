import os
import time

import torch

from utils.logger import AverageMeter


class Trainer(object):
    def __init__(self, args, writer, data_loader, model, evaluator, criterion, scheduler, optimizer):
        self.args = args
        self.writer = writer
        self.data_loader = data_loader
        self.model = model
        self.evaluator = evaluator
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer

    def start(self):
        best = 0.0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.scheduler.step()
            self.train(epoch)
            if (epoch + 1) % self.args.eval_freq == 0 or epoch == self.args.epochs - 1:
                accuracy = self.validate(epoch)
                is_best = accuracy > best
                if is_best:
                    best = accuracy
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict()
                    })

    def train(self, epoch):
        self._print_log('Training epoch: {}'.format(epoch + 1), 'train')

        self.model.train()
        loader = self.data_loader['train']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for batch_idx, (inputs, target) in enumerate(loader, 0):
            data_time.update(time.time() - end)

            target = target.cuda()
            inputs = inputs.cuda()
            inputs_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)

            output = self.model(inputs_var)
            loss = self.criterion(output, target_var.long())

            prec1, prec5 = self.evaluator.accuracy(output.data, target.long(), topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            n_iter = epoch * len(loader) + batch_idx + 1

            batch_time.update(time.time() - end)
            end = time.time()

            self.writer.add_scalar('Train/loss', loss.item(), n_iter)
            self.writer.add_scalar('Train/prec@1', prec1.item(), n_iter)
            self.writer.add_scalar('Train/prec@5', prec5.item(), n_iter)

            if batch_idx % self.args.print_freq == 0:
                message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    epoch, batch_idx, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=self.optimizer.param_groups[-1]['lr'])
                self._print_log(message, 'train')

        self._print_log(('Training Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
                         .format(top1=top1, top5=top5, loss=losses)), 'train')

        self.writer.add_scalar('Train/Loss', losses.avg, epoch)
        self.writer.add_scalar('Train/Prec@1', top1.avg, epoch)
        self.writer.add_scalar('Train/Prec@5', top5.avg, epoch)

        for name, param in self.model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            self.writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    def validate(self, epoch):
        self._print_log('Validating epoch: {}'.format(epoch + 1), 'validate')

        self.model.eval()
        loader = self.data_loader['test']
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for batch_idx, (inputs, target) in enumerate(loader, 0):
            target = target.cuda()
            inputs = inputs.cuda()
            with torch.no_grad():
                inputs_var = torch.autograd.Variable(inputs)
                target_var = torch.autograd.Variable(target)
                output = self.model(inputs_var)

            loss = self.criterion(output, target_var.long())

            prec1, prec5 = self.evaluator.accuracy(output.data, target.long(), topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.args.print_freq == 0:
                message = ('Time: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    batch_idx, len(loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5)
                self._print_log(message, 'validate')

        self._print_log(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
                         .format(top1=top1, top5=top5, loss=losses)), 'validate')

        self.writer.add_scalar('Validate/Loss', losses.avg, epoch)
        self.writer.add_scalar('Validate/Prec@1', top1.avg, epoch)
        self.writer.add_scalar('Validate/Prec@5', top5.avg, epoch)

        return top1.avg

    def _print_log(self, s, file_name):
        print(s)
        if self.args.print_log:
            with open('{}/{}.txt'.format(self.args.work_dir, file_name), 'a') as f:
                print(s, file=f)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filename = '_'.join((self.args.snapshot_pref, filename))
        torch.save(state, filename)