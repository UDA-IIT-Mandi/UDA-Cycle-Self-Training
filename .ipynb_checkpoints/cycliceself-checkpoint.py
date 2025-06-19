import random
import time
import warnings
import sys
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cell 2: Define Utility Functions
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# Cell 3: Define Loss Functions
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

# Cell 4: Define SAM Optimizer
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
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    dtype=torch.float32
               )
        return norm

# Cell 5: Define Data Transforms for FixMatch
class TransformFixMatch(object):
    def __init__(self, mean=(0.5,), std=(0.5,)):
        normalize = transforms.Normalize(mean=mean, std=std)
        
        # Weak augmentation
        self.weak = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        # Strong augmentation
        self.strong = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
        ])
        
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong

# Cell 6: Define Image Classifier Model
class ImageClassifier(nn.Module):
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.num_features = backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the final classification layer
        
        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        
        self.head = nn.Linear(bottleneck_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        bottleneck_output = self.bottleneck(features)
        predictions = self.head(bottleneck_output)
        return predictions, bottleneck_output
    
    def get_parameters(self):
        """Get parameters for different learning rates"""
        return [
            {'params': self.backbone.parameters()},
            {'params': self.bottleneck.parameters()},
            {'params': self.head.parameters()}
        ]

# Cell 7: Data Loading Setup
def get_data_loaders(batch_size=32, num_workers=2):
    # Define transforms
    source_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # SVHN normalization
    ])
    
    target_unlabeled_transform = TransformFixMatch(mean=(0.5,), std=(0.5,))  # MNIST normalization
    
    target_test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST normalization
    ])
    
    # Load datasets
    train_source_dataset = datasets.SVHN(
        root='./data', split='train', download=True, transform=source_transform
    )
    
    # For target domain, we use MNIST training set as unlabeled data
    train_target_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=target_unlabeled_transform
    )
    
    # MNIST test set for evaluation
    test_target_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=target_test_transform
    )
    
    # Create data loaders
    train_source_loader = DataLoader(
        train_source_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    
    train_target_loader = DataLoader(
        train_target_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    
    test_loader = DataLoader(
        test_target_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )
    
    return train_source_loader, train_target_loader, test_loader

# Cell 8: Forever Data Iterator
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

# Cell 9: Training Function
def train_epoch(train_source_iter, train_target_iter, model, ts_loss, optimizer, 
                lr_scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    rev_losses = AverageMeter('CST Loss', ':3.2f')
    fix_losses = AverageMeter('Fix Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args['iters_per_epoch'],
        [batch_time, data_time, losses, trans_losses, rev_losses, fix_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    
    for i in range(args['iters_per_epoch']):
        x_s, labels_s = next(train_source_iter)
        (x_t, x_t_u), _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)

        data_time.update(time.time() - end)

        # Forward pass
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_t_u, _ = model(x_t_u)

        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)

        # Generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t, dim=-1), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u, reduction='none') * 
              max_prob.ge(args['threshold']).float().detach()).mean()

        # Compute CST loss
        target_data_train_r = f_t / torch.norm(f_t, dim=-1, keepdim=True)
        target_data_test_r = f_s / torch.norm(f_s, dim=-1, keepdim=True)
        
        target_gram_r = torch.clamp(
            target_data_train_r.mm(target_data_train_r.transpose(0, 1)), 
            -0.99999999, 0.99999999
        )
        test_gram_r = torch.clamp(
            target_data_test_r.mm(target_data_train_r.transpose(0, 1)), 
            -0.99999999, 0.99999999
        )
        
        target_train_label_r = (F.one_hot(pred_u, args['num_classes']).float() - 
                               1.0 / args['num_classes'])
        
        eye_matrix = torch.eye(args['batch_size']).to(device)
        target_test_pred_r = test_gram_r.mm(
            torch.inverse(target_gram_r + 0.001 * eye_matrix)
        ).mm(target_train_label_r)
        
        source_labels_one_hot = (F.one_hot(labels_s, args['num_classes']).float() - 
                                1.0 / args['num_classes'])
        reverse_loss = F.mse_loss(target_test_pred_r, source_labels_one_hot)

        # Compute losses
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts_loss(y_t)

        if Lu != 0:
            loss = (cls_loss + 
                   transfer_loss * args['trade_off'] + 
                   reverse_loss * args['trade_off1'] + 
                   Lu * args['trade_off3'])
        else:
            loss = (cls_loss + 
                   transfer_loss * args['trade_off'] + 
                   reverse_loss * args['trade_off1'])

        cls_acc = accuracy(y_s, labels_s)[0]

        # Update meters
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        rev_losses.update(reverse_loss.item(), x_s.size(0))
        fix_losses.update(Lu.item(), x_s.size(0))

        # SAM first step
        loss.backward()
        optimizer.first_step(zero_grad=True)
        lr_scheduler.step()

        # SAM second step - recompute loss
        y, f = model(x)
        y_t_u, _ = model(x_t_u)
        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)

        max_prob, pred_u = torch.max(F.softmax(y_t, dim=-1), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u, reduction='none') * 
              max_prob.ge(args['threshold']).float().detach()).mean()

        target_data_train_r = f_t / torch.norm(f_t, dim=-1, keepdim=True)
        target_data_test_r = f_s / torch.norm(f_s, dim=-1, keepdim=True)
        
        target_gram_r = torch.clamp(
            target_data_train_r.mm(target_data_train_r.transpose(0, 1)), 
            -0.99999999, 0.99999999
        )
        test_gram_r = torch.clamp(
            target_data_test_r.mm(target_data_train_r.transpose(0, 1)), 
            -0.99999999, 0.99999999
        )
        
        target_train_label_r = (F.one_hot(pred_u, args['num_classes']).float() - 
                               1.0 / args['num_classes'])
        
        target_test_pred_r = test_gram_r.mm(
            torch.inverse(target_gram_r + 0.001 * eye_matrix)
        ).mm(target_train_label_r)
        
        reverse_loss = F.mse_loss(target_test_pred_r, source_labels_one_hot)
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts_loss(y_t)

        if Lu != 0:
            loss1 = (cls_loss + 
                    transfer_loss * args['trade_off'] + 
                    reverse_loss * args['trade_off1'] + 
                    Lu * args['trade_off3'])
        else:
            loss1 = (cls_loss + 
                    transfer_loss * args['trade_off'] + 
                    reverse_loss * args['trade_off1'])

        loss1.backward()
        optimizer.second_step(zero_grad=True)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            progress.display(i)

# Cell 10: Validation Function
def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args['print_freq'] == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

# Cell 11: Main Training Loop
def main():
    # Set hyperparameters
    args = {
        'batch_size': 32,
        'lr': 0.005,
        'momentum': 0.9,
        'weight_decay': 1e-3,
        'epochs': 20,
        'iters_per_epoch': 500,
        'print_freq': 50,
        'temperature': 2.0,
        'alpha': 1.9,
        'trade_off': 0.08,
        'trade_off1': 0.5,
        'trade_off3': 0.5,
        'threshold': 0.97,
        'rho': 0.5,
        'lr_gamma': 0.001,
        'lr_decay': 0.75,
        'num_classes': 10,
        'seed': 42
    }
    
    # Set seed for reproducibility
    if args['seed'] is not None:
        random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        cudnn.deterministic = True
        cudnn.benchmark = True
    
    # Load data
    train_source_loader, train_target_loader, test_loader = get_data_loaders(
        batch_size=args['batch_size']
    )
    
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    
    # Create model
    backbone = resnet18(pretrained=True)
    # Modify first conv layer for single channel input (MNIST compatibility)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model = ImageClassifier(backbone, args['num_classes'], bottleneck_dim=256).to(device)
    
    # Create optimizer and scheduler
    base_optimizer = SGD
    optimizer = SAM(
        model.get_parameters(), 
        base_optimizer, 
        lr=args['lr'],
        momentum=args['momentum'],
        weight_decay=args['weight_decay'],
        adaptive=True,
        rho=args['rho']
    )
    
    lr_scheduler = LambdaLR(
        optimizer, 
        lambda x: args['lr'] * (1. + args['lr_gamma'] * float(x)) ** (-args['lr_decay'])
    )
    
    # Create loss function
    ts_loss = TsallisEntropy(temperature=args['temperature'], alpha=args['alpha'])
    
    # Training loop
    best_acc1 = 0.
    print("Starting training...")
    
    for epoch in range(args['epochs']):
        print(f"\nEpoch {epoch+1}/{args['epochs']}")
        print("Learning rate:", lr_scheduler.get_last_lr()[0])
        
        # Train for one epoch
        train_epoch(train_source_iter, train_target_iter, model, ts_loss, 
                   optimizer, lr_scheduler, epoch, args)
        
        # Evaluate on test set
        acc1 = validate(test_loader, model, args)
        
        # Save best model
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best accuracy: {best_acc1:.3f}")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc1:.3f}")
    
    return model, best_acc1

# Cell 12: Run Training
if __name__ == "__main__":
    model, best_acc = main()

# Cell 13: Visualization and Analysis (Optional)
def visualize_features(model, source_loader, target_loader, num_samples=1000):
    """Visualize features using t-SNE"""
    model.eval()
    
    source_features = []
    source_labels = []
    target_features = []
    target_labels = []
    
    with torch.no_grad():
        # Collect source features
        for i, (images, labels) in enumerate(source_loader):
            if len(source_features) >= num_samples:
                break
            images = images.to(device)
            _, features = model(images)
            source_features.append(features.cpu())
            source_labels.append(labels)
        
        # Collect target features
        for i, (images, labels) in enumerate(target_loader):
            if len(target_features) >= num_samples:
                break
            if isinstance(images, tuple):  # Handle FixMatch transform
                images = images[0]
            images = images.to(device)
            _, features = model(images)
            target_features.append(features.cpu())
            target_labels.append(labels)
    
    # Concatenate features
    source_features = torch.cat(source_features[:num_samples//len(source_features[0])])
    target_features = torch.cat(target_features[:num_samples//len(target_features[0])])
    source_labels = torch.cat(source_labels[:num_samples//len(source_labels[0])])
    target_labels = torch.cat(target_labels[:num_samples//len(target_labels[0])])
    
    # Combine features
    all_features = torch.cat([source_features, target_features]).numpy()
    all_labels = torch.cat([source_labels, target_labels]).numpy()
    domain_labels = ['Source'] * len(source_features) + ['Target'] * len(target_features)
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Plot by domain
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue']
    for i, domain in enumerate(['Source', 'Target']):
        mask = [d == domain for d in domain_labels]
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=colors[i], alpha=0.6, label=domain)
    plt.title('Features by Domain')
    plt.legend()
    
    # Plot by class
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Features by Class')
    
    plt.tight_layout()
    plt.show()

# Cell 14: Test the trained model
def test_model():
    # Load data
    _, _, test_loader = get_data_loaders(batch_size=32)
    
    # Create model
    backbone = resnet18(pretrained=True)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = ImageClassifier(backbone, 10, bottleneck_dim=256).to(device)
    
    # Load best model
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model")
    else:
        print("No saved model found. Please run training first.")
        return
    
    # Test
    args = {'print_freq': 50}
    test_acc = validate(test_loader, model, args)
    print(f"Final test accuracy: {test_acc:.3f}")
    
    return model

# Uncomment to run testing
# test_model()