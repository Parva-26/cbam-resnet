
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from models.resnet import ResNet50
from models.cbam_resnet import CBAMResNet50
from utils.metrics import count_parameters, load_checkpoint


def get_test_loader(batch_size=128, num_workers=2):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


def evaluate_model(model, loader, device):

    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs  = model(images)
            _, preds = outputs.max(dim=1)  

            correct += preds.eq(targets).sum().item()
            total   += targets.size(0)

    top1_acc = 100.0 * correct / total
    return top1_acc


def plot_training_curves(results_dir='results'):
    
    paths = {
        'baseline': os.path.join(results_dir, 'baseline_history.json'),
        'cbam'    : os.path.join(results_dir, 'cbam_history.json'),
    }

    histories = {}
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                histories[name] = json.load(f)

    if not histories:
        print("No history JSON files found in results/ — train both models first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    style = {
        'baseline': {'color': '#2563eb', 'label': 'ResNet-50 (Baseline)'},
        'cbam'    : {'color': '#dc2626', 'label': 'ResNet-50 + CBAM'},
    }

    for model_name, history in histories.items():
        epochs = range(1, len(history['train_losses']) + 1)
        c      = style[model_name]['color']
        lbl    = style[model_name]['label']

        axes[0].plot(epochs, history['train_losses'], color=c, linestyle='--',
                     alpha=0.5, label=f'{lbl} Train')
        axes[0].plot(epochs, history['val_losses'],   color=c, linestyle='-',
                     label=f'{lbl} Val')

        axes[1].plot(epochs, history['train_accs'], color=c, linestyle='--',
                     alpha=0.5, label=f'{lbl} Train')
        axes[1].plot(epochs, history['val_accs'],   color=c, linestyle='-',
                     label=f'{lbl} Val')

    for ax, title, ylabel in zip(
        axes,
        ['Loss vs Epoch', 'Top-1 Accuracy vs Epoch'],
        ['Cross-Entropy Loss', 'Accuracy (%)']
    ):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs('results', exist_ok=True)

    test_loader = get_test_loader()

    checkpoints = {
        'baseline': ('checkpoints/baseline_best.pth', ResNet50),
        'cbam'    : ('checkpoints/cbam_best.pth',     CBAMResNet50),
    }

    results = {}

    for model_name, (ckpt_path, model_class) in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for '{model_name}': {ckpt_path} — skipping")
            continue

        model = model_class(num_classes=10).to(device)
        load_checkpoint(ckpt_path, model)

        total_params, _ = count_parameters(model)
        top1_acc        = evaluate_model(model, test_loader, device)

        results[model_name] = {
            'top1_accuracy': round(top1_acc, 2),
            'total_params' : total_params,
        }

        label = 'ResNet-50 Baseline' if model_name == 'baseline' else 'ResNet-50 + CBAM'
        print(f"Model      : {label}")
        print(f"Parameters : {total_params:,}")
        print(f"Top-1 Acc  : {top1_acc:.2f}%")
        print()

    if len(results) == 2:
        acc_delta   = results['cbam']['top1_accuracy'] - results['baseline']['top1_accuracy']
        param_delta = results['cbam']['total_params']  - results['baseline']['total_params']
        print("--- Comparison ---")
        print(f"Accuracy gain from CBAM : +{acc_delta:.2f}%")
        print(f"Extra parameters (CBAM) : +{param_delta:,}")

    results_path = 'results/eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_training_curves()


if __name__ == '__main__':
    main()
