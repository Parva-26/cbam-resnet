
import os
import torch


class AverageMeter:

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val, n=1):
        
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def accuracy(outputs, targets, topk=(1,)):
    
    with torch.no_grad():
        max_k     = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
        pred    = pred.t()  

        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_size).item())

        return results


def count_parameters(model):
    
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(state, filepath):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint
