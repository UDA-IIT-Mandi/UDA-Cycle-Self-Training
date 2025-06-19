import random
import time
import warnings
import sys
import argparse
import shutil
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging Infrastructure ---
class CompleteLogger:
    def __init__(self, root, phase):
        self.root = root
        self.visualize_directory = osp.join(root, "visualize")
        os.makedirs(self.visualize_directory, exist_ok=True)
        self.checkpoint_directory = osp.join(root, "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        self.log_file = open(osp.join(root, f"log_{phase}.txt"), "a")
        self.phase = phase

    def get_checkpoint_path(self, name):
        return osp.join(self.checkpoint_directory, f"{name}.pth")

    def log(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# --- Analysis Utilities (t-SNE and A-distance) ---
def collect_feature(data_loader, feature_extractor, device):
    feature_extractor.eval()
    features = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            feat = feature_extractor(x)
            features.append(feat.cpu())
    return torch.cat(features, dim=0).numpy()

def tsne_visualize(source_feature, target_feature, filename):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    X = np.concatenate([source_feature, target_feature], axis=0)
    y = np.array([0]*len(source_feature) + [1]*len(target_feature))
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(8, 8))
    plt.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], label='Source', alpha=0.5)
    plt.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], label='Target', alpha=0.5)
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig(filename)
    plt.close()

def a_distance(source_feature, target_feature):
    # Simple A-distance estimation using a linear classifier
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([source_feature, target_feature], axis=0)
    y = np.array([0]*len(source_feature) + [1]*len(target_feature))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y)
    return 2 * (1 - 2 * abs(acc - 0.5))

# --- Per-class Evaluation ---
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, targets, preds):
        targets = targets.cpu().numpy()
        preds = preds.cpu().numpy()
        for t, p in zip(targets, preds):
            self.mat[t, p] += 1

    def format(self, classes):
        lines = ["Confusion Matrix:"]
        lines.append("\t" + "\t".join(classes))
        for i, row in enumerate(self.mat):
            lines.append(f"{classes[i]}\t" + "\t".join(map(str, row)))
        per_class_acc = self.mat.diagonal() / (self.mat.sum(axis=1) + 1e-8)
        lines.append("Per-class accuracy: " + ", ".join([f"{a*100:.2f}%" for a in per_class_acc]))
        return "\n".join(lines)

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class TsallisEntropy(nn.Module):
    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
        return 1 / (self.alpha - 1) * torch.sum((1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1)))

class TransformFixMatch(object):
    def __init__(self, size=32):
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.weak = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.strong = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong

class ImageClassifier(nn.Module):
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck = nn.Sequential(
            nn.Linear(backbone.fc.in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(bottleneck_dim, num_classes)
        backbone.fc = nn.Identity()

    def forward(self, x):
        f = self.backbone(x)
        f = self.bottleneck(f)
        y = self.head(f)
        return y, f

    def get_parameters(self):
        return self.parameters()

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm

class ForeverDataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_svhn_mnist_data(root_dir, batch_size, workers=2):
    svhn_transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    mnist_transform_single = T.Compose([
        T.Resize(32),
        T.Lambda(lambda x: x.convert('RGB')),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    mnist_transform_fixmatch = TransformFixMatch(size=32)
    train_source_dataset = datasets.SVHN(root=root_dir, split='train', download=True, transform=svhn_transform)
    train_target_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=mnist_transform_fixmatch)
    val_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=mnist_transform_single)
    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_source_loader, train_target_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='CST for SVHN→MNIST')
    parser.add_argument('--root', type=str, default='./cst_svhnmnist_log', help='Root path for logs and data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--bottleneck-dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=1.9)
    parser.add_argument('--trade-off', type=float, default=0.08)
    parser.add_argument('--trade-off1', type=float, default=0.5)
    parser.add_argument('--trade-off3', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.97)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-gamma', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=float, default=0.75)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--iters-per-epoch', type=int, default=500)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'])
    parser.add_argument('--per-class-eval', action='store_true', help='Output per-class accuracy during evaluation')
    args = parser.parse_args()

    logger = CompleteLogger(args.root, args.phase)
    logger.log("Starting SVHN→MNIST CST Training")
    logger.log(f"Device: {device}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    train_source_loader, train_target_loader, val_loader = load_svhn_mnist_data(args.root, args.batch_size, args.workers)
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    logger.log(f"=> using pre-trained model '{args.arch}'")
    backbone = models.__dict__[args.arch](pretrained=True)
    if backbone.conv1.in_channels != 3:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_classes = 10
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)

    base_optimizer = SGD
    optimizer = SAM(classifier.get_parameters(), base_optimizer, lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay,
                    adaptive=True, rho=args.rho)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    ts_loss = TsallisEntropy(temperature=args.temperature, alpha=args.alpha)

    # --- Checkpoint Management ---
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # --- Analysis Mode ---
    if args.phase == 'analysis':
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne_visualize(source_feature, target_feature, tSNE_filename)
        logger.log(f"Saving t-SNE to {tSNE_filename}")
        A_dist = a_distance(source_feature, target_feature)
        logger.log(f"A-distance = {A_dist}")
        logger.close()
        return

    if args.phase == 'test':
        acc1 = validate(val_loader, classifier, args, logger)
        logger.log(f"Test Acc@1: {acc1:.2f}")
        logger.close()
        return

    best_acc1 = 0.
    for epoch in range(args.epochs):
        logger.log(f"Epoch {epoch+1}/{args.epochs}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        train(train_source_iter, train_target_iter, classifier, ts_loss, optimizer, lr_scheduler, epoch, args, logger)
        acc1 = validate(val_loader, classifier, args, logger)
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        logger.log(f"Best accuracy so far: {best_acc1:.2f}%")
    logger.log(f"Final best accuracy: {best_acc1:.2f}%")
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(val_loader, classifier, args, logger)
    logger.log(f"Test Acc@1: {acc1:.2f}")
    logger.close()

def train(train_source_iter, train_target_iter, model, ts, optimizer, lr_scheduler, epoch, args, logger):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    rev_losses = AverageMeter('CST Loss', ':3.2f')
    fix_losses = AverageMeter('Fix Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, rev_losses, fix_losses, cls_accs],
        prefix=f"Epoch: [{epoch}]")
    model.train()
    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        target_data = next(train_target_iter)
        if isinstance(target_data[0], tuple):
            (x_t, x_t_u), _ = target_data
        else:
            x_t, _ = target_data
            x_t_u = x_t
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)
        data_time.update(time.time() - end)
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_t_u, _ = model(x_t_u)
        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)
        max_prob, pred_u = torch.max(F.softmax(y_t, dim=-1), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u, reduction='none') *
              max_prob.ge(args.threshold).float().detach()).mean()
        target_data_train_r = f_t / torch.norm(f_t, dim=-1, keepdim=True)
        target_data_test_r = f_s / torch.norm(f_s, dim=-1, keepdim=True)
        target_gram_r = torch.clamp(target_data_train_r @ target_data_train_r.T, -0.99999999, 0.99999999)
        test_gram_r = torch.clamp(target_data_test_r @ target_data_train_r.T, -0.99999999, 0.99999999)
        target_train_label_r = F.one_hot(pred_u, 10).float() - 1.0/10.0
        target_test_pred_r = test_gram_r @ torch.inverse(target_gram_r + 0.001 * torch.eye(args.batch_size, device=device)) @ target_train_label_r
        reverse_loss = F.mse_loss(target_test_pred_r, F.one_hot(labels_s, 10).float() - 1.0/10.0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts(y_t)
        if Lu > 0:
            loss = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1 + Lu * args.trade_off3
        else:
            loss = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1
        cls_acc = accuracy(y_s, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        rev_losses.update(reverse_loss.item(), x_s.size(0))
        fix_losses.update(Lu.item(), x_s.size(0))
        loss.backward()
        optimizer.first_step(zero_grad=True)
        lr_scheduler.step()
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_t_u, _ = model(x_t_u)
        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)
        max_prob, pred_u = torch.max(F.softmax(y_t, dim=-1), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u, reduction='none') *
              max_prob.ge(args.threshold).float().detach()).mean()
        target_data_train_r = f_t / torch.norm(f_t, dim=-1, keepdim=True)
        target_data_test_r = f_s / torch.norm(f_s, dim=-1, keepdim=True)
        target_gram_r = torch.clamp(target_data_train_r @ target_data_train_r.T, -0.99999999, 0.99999999)
        test_gram_r = torch.clamp(target_data_test_r @ target_data_train_r.T, -0.99999999, 0.99999999)
        target_train_label_r = F.one_hot(pred_u, 10).float() - 1.0/10.0
        target_test_pred_r = test_gram_r @ torch.inverse(target_gram_r + 0.001 * torch.eye(args.batch_size, device=device)) @ target_train_label_r
        reverse_loss = F.mse_loss(target_test_pred_r, F.one_hot(labels_s, 10).float() - 1.0/10.0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts(y_t)
        if Lu > 0:
            loss1 = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1 + Lu * args.trade_off3
        else:
            loss1 = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1
        loss1.backward()
        optimizer.second_step(zero_grad=True)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if args.per_class_eval:
        classes = [str(i) for i in range(10)]
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            output, _ = model(images)
            loss = F.cross_entropy(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        logger.log(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
        if confmat:
            logger.log(confmat.format(classes))
    return top1.avg

if __name__ == '__main__':
    main()