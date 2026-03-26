
import os
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet50
from models.cbam_resnet import CBAMResNet50
from utils.metrics import AverageMeter, accuracy, count_parameters, save_checkpoint, load_checkpoint

CONFIG = {
    'model'        : 'baseline',    
    'num_epochs'   : 100,
    'batch_size'   : 128,
    'lr'           : 0.1,
    'momentum'     : 0.9,
    'weight_decay' : 5e-4,
    'warmup_epochs': 5,
    'num_classes'  : 10,
    'num_workers'  : 2,
    'checkpoint_dir': 'checkpoints',
    'results_dir'  : 'results',
    'resume'       : False,
    'resume_path'  : '',
}


def get_data_loaders(batch_size, num_workers):
    
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def build_model(model_type, num_classes):
    if model_type == 'baseline':
        return ResNet50(num_classes=num_classes)
    elif model_type == 'cbam':
        return CBAMResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"model must be 'baseline' or 'cbam', got '{model_type}'")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    
    model.train()

    loss_meter = AverageMeter('Loss')
    acc_meter  = AverageMeter('Acc')

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top1 = accuracy(outputs, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Epoch [{epoch+1}] "
                f"Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Acc: {acc_meter.avg:.2f}%"
            )

    return loss_meter.avg, acc_meter.avg


def validate(model, loader, criterion, device):
    
    model.eval()

    loss_meter = AverageMeter('Loss')
    acc_meter  = AverageMeter('Acc')

    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, targets)

            top1 = accuracy(outputs, targets, topk=(1,))[0]
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(top1, images.size(0))

    return loss_meter.avg, acc_meter.avg


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=0
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    train_loader, val_loader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])

    model = build_model(CONFIG['model'], CONFIG['num_classes']).to(device)

    total_p, trainable_p = count_parameters(model)
    print(f"Model             : {CONFIG['model']}")
    print(f"Total params      : {total_p:,}")
    print(f"Trainable params  : {trainable_p:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG['lr'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler = build_scheduler(optimizer, CONFIG['warmup_epochs'], CONFIG['num_epochs'])

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_acc    = 0.0
    start_epoch = 0

    if CONFIG['resume'] and CONFIG['resume_path']:
        print(f"Resuming from: {CONFIG['resume_path']}")
        ckpt        = load_checkpoint(CONFIG['resume_path'], model, optimizer, scheduler)
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['best_acc']
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses',   [])
        train_accs   = ckpt.get('train_accs',   [])
        val_accs     = ckpt.get('val_accs',     [])
        print(f"Resumed at epoch {start_epoch}, best acc so far: {best_acc:.2f}%")

    print(f"\nTraining for {CONFIG['num_epochs']} epochs...\n")

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"--- Epoch {epoch+1}/{CONFIG['num_epochs']}  LR: {current_lr:.6f} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%\n"
        )

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(
                CONFIG['checkpoint_dir'], f"{CONFIG['model']}_epoch{epoch+1}.pth"
            )
            save_checkpoint({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc'            : best_acc,
                'train_losses'        : train_losses,
                'val_losses'          : val_losses,
                'train_accs'          : train_accs,
                'val_accs'            : val_accs,
            }, filepath=ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if val_acc > best_acc:
            best_acc  = val_acc
            best_path = os.path.join(CONFIG['checkpoint_dir'], f"{CONFIG['model']}_best.pth")
            save_checkpoint({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc'            : best_acc,
                'train_losses'        : train_losses,
                'val_losses'          : val_losses,
                'train_accs'          : train_accs,
                'val_accs'            : val_accs,
            }, filepath=best_path)
            print(f"New best accuracy: {best_acc:.2f}% — saved to {best_path}\n")

    history = {
        'model'        : CONFIG['model'],
        'train_losses' : train_losses,
        'val_losses'   : val_losses,
        'train_accs'   : train_accs,
        'val_accs'     : val_accs,
        'best_val_acc' : best_acc,
    }
    history_path = os.path.join(CONFIG['results_dir'], f"{CONFIG['model']}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val accuracy: {best_acc:.2f}%")
    print(f"History saved to {history_path}")


if __name__ == '__main__':
    main()
