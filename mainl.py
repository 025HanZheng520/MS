﻿import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.final_model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_DFEW import train_data_loader, test_data_loader
import random
import  numpy
from sklearn.metrics import accuracy_score
import confusion_matrix
import TSNE
from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--noise_sigma', type=float, default=1.)
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=6, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=8, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=2)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
project_path = './'
log_txt_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
log_curve_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
pretrained_checkpoint_path='./pretrain_DFEW.pth'


def main():
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    #with open(log_txt_path, 'a') as f:
        #f.write('The training set: set ' + str(args.data_set) + '\n')
    model = GenerateModel()
    # create model and load pre_trained parameters
    #if len(args.gpu.split(',')) > 1:
    model = nn.DataParallel(model).cuda()
   
    #model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer

    criterion = LargeMarginInSoftmaxLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=62, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args,best_acc, epoch)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            print('save_model')
            torch.save(model.state_dict(),'./pretrain_DFEW.pth')
        best_acc = max(val_acc, best_acc)
        """save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)"""

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target,index) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()
        
        

        # compute output
        output,x_FER = model(images)
        
        

        loss = criterion(output, target)
        

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg



def validate(val_loader, model, criterion, args,best_acc, epoch):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        pres_tr, trues_tr = [], [] #预测pre 真实true
        scatter_x,scatter_y=[],[]

        for i, (images, target,index) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            model = model.cuda()
            
            

            # compute output
            #flops, params = profile(model, images)
            output,x_FER= model(images)
            loss = criterion(output, target)

            #TSNE
            scatter_x.extend(torch.softmax(output, dim=1).cpu().numpy())
            scatter_y.extend(target.cpu().numpy()) #可以视为labels列表

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))


            # confusion matrix
            _, pre_tr = torch.max(output, 1)
            # condusion Y_test=target
            pres_tr += pre_tr.cpu().numpy().tolist()
            trues_tr += target.cpu().numpy().tolist()

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))


            if i % args.print_freq == 0:
                progress.display(i)


        if top1.avg > best_acc:  # and top1.avg>=66:
            confusion_matrix.plot_confusion_matrix_2(pres_tr,trues_tr, epoch)
            if True:  # top1.avg>=66:
                tSNE_x = np.array(scatter_x)
                tSNE_y = np.array(scatter_y)
                TSNE.tsne(tSNE_x, tSNE_y, 1, 1)
        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        print('Current UAR: '+ str(confusion_matrix.plot_confusion_matrix_2(pres_tr,trues_tr, epoch)))
        #print("GFLOPs :{:.5f}, Params : {:.5f}".format(flops / 1e9, params / 1e6))  # flops单位G，para单位M
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
            f.write('Current UAR: '+ str(confusion_matrix.plot_confusion_matrix_2(pres_tr,trues_tr, epoch)) + '\n')
            #f.write("GFLOPs :{:.5f}, Params : {:.5f}".format(flops / 1e9, params / 1e6))  # flops单位G，para单位M
    return top1.avg, losses.avg



def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_WAR(trues_te, pres_te):
    WAR  = accuracy_score(trues_te, pres_te)
    return WAR


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


import torch.nn.functional as F
class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    def __init__(self, deg_logit=None,
                 weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginInSoftmaxLoss, self).__init__(weight=weight, size_average=size_average,
                                                       ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.deg_logit = deg_logit

    def forward(self, input, target):
        C = input.size(1)  # number of classes
        
        Mask = torch.zeros_like(input).scatter_(1, target.long().unsqueeze(1), 1)
        

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        logprobs = F.log_softmax(input, dim=-1)
        hard_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        hard_loss = hard_loss.squeeze(1)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        smooth_loss = -1.0/(C-1) * (F.log_softmax(X, dim=1) * (1.0 - Mask)).sum(dim=1)

        loss = 0.9*hard_loss + 0.1 * smooth_loss
        
        return loss.mean()


if __name__ == '__main__':
    main()
