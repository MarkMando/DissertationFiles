"""
=============================================================================
Breast Cancer Detection — Synthetic External Image Module (Local Machine)
Model   : MobileNetV3-Large (Transfer Learning, PyTorch)
Dataset : StyleGAN-XL Synthetic (350 Normal / 350 Abnormal — balanced)
Output  : Best model checkpoint + evaluation plots
Author  : Mark Sthembiso Mando | Mulungushi University MSc Data Science
=============================================================================

SETUP — install dependencies once:
    pip install torch torchvision scikit-learn matplotlib seaborn tqdm

HOW TO RUN:
    python train_synthetic_external.py

EXPECTED DATASET STRUCTURE:
    final_synthetic_dataset/
    ├── train/
    │   ├── normal/       (350 images)
    │   └── abnormal/     (350 images)
    ├── val/
    │   ├── normal/
    │   └── abnormal/
    └── test/
        ├── normal/
        └── abnormal/

NOTE ON H1 THRESHOLD:
    The primary H1 target is ROC-AUC ≥ 0.85 on the test split.
    Bootstrapped 95% CIs are reported alongside the point estimate;
    a marginal miss should be interpreted in the context of the CI
    and the 75-sample test set size (Section 5.x, domain shift discussion).

FIXES APPLIED (v2):
    BUG 1-6 — All sklearn metric calls (roc_auc_score, average_precision_score,
               roc_curve, precision_recall_curve) were missing pos_label=ABNORMAL_CLS_IDX.
               ImageFolder assigns abnormal=0, normal=1 alphabetically; sklearn
               treats label 1 as positive by default. Passing P(abnormal) scores
               against a flipped label convention produced AUC=0.0 on a model
               that was actually performing well.
    BUG 7  — save_gradcam_grid() used int(prob >= 0.5) for pred_label comparison,
               which always returns 0 or 1 — but abnormal label is 0 and normal is 1,
               so ✓ and ✗ tick marks were systematically inverted.
=============================================================================
"""

import os
import random
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION  ← Only edit this section
# =============================================================================

DATASET_DIR = r'C:\Users\mmando_Adm\Documents\final_synthetic_dataset'
OUTPUT_DIR  = r'C:\Users\mmando_Adm\Documents\external_image_output'

# ── Training hyperparameters (proposal Section 4.4.2) ────────────────────────
IMG_SIZE        = 224       # MobileNetV3 standard input
BATCH_SIZE      = 16
EPOCHS_FROZEN   = 30        # Phase 1: frozen convolutional base
EPOCHS_FINETUNE = 20        # Phase 2: unfreeze last N feature layers
LEARNING_RATE   = 0.001     # Adam initial LR (Phase 1)
FINETUNE_LR     = 0.0001    # Adam LR for Phase 2 (10× lower)
PATIENCE_STOP   = 10        # Early stopping patience (epochs without improvement)
PATIENCE_LR     = 5         # ReduceLROnPlateau patience
UNFREEZE_LAYERS = 20        # Number of feature sub-layers to unfreeze in Phase 2
SEED            = 42

# ── Hypothesis target (proposal Section 4.4.3) ───────────────────────────────
H1_AUC_THRESHOLD = 0.85

# ── Class index constants ─────────────────────────────────────────────────────
# ImageFolder assigns indices alphabetically: 'abnormal'=0, 'normal'=1.
# ALL sklearn metric calls must pass pos_label=ABNORMAL_CLS_IDX so that
# P(abnormal) scores are correctly interpreted against label 0 as positive.
ABNORMAL_CLS_IDX = 0
NORMAL_CLS_IDX   = 1

# =============================================================================
# Derived paths & setup — do not edit below this line
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR           = os.path.join(OUTPUT_DIR, "plots")
CHECKPOINT_PATH_P1 = os.path.join(OUTPUT_DIR, "best_ext_mobilenetv3_phase1.pth")
CHECKPOINT_PATH_P2 = os.path.join(OUTPUT_DIR, "best_ext_mobilenetv3_phase2.pth")
CHECKPOINT_PATH    = CHECKPOINT_PATH_P2   # alias used downstream
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"\n{'='*62}")
print("  StyleGAN-XL Synthetic External  |  MobileNetV3-Large  |  PyTorch")
print(f"{'='*62}\n")
print(f"  PyTorch    : {torch.__version__}")
print(f"  Device     : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING    : No GPU detected — training will be slow.")
print(f"  Dataset    : {DATASET_DIR}")
print(f"  Output     : {OUTPUT_DIR}")
print()


# =============================================================================
# 2. DATA TRANSFORMS (proposal Section 4.4.2)
#    Training augmentation mirrors the CBIS-DDSM pipeline for consistency.
#    Val / Test: resize + normalise only (no augmentation).
# =============================================================================

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =============================================================================
# 3. DATA LOADING
# =============================================================================

def load_datasets():
    """
    Loads the pre-split synthetic dataset from disk using ImageFolder.
    Expects train/, val/, test/ subdirectories each containing
    class-named subfolders (normal/ and abnormal/).

    Returns train_loader, val_loader, test_loader, class_names.
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'train'), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'val'),   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'test'),  transform=eval_transforms)

    # num_workers=0 avoids DataLoader multiprocessing issues on Windows
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(
        test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = train_dataset.classes

    print(f"  Classes    : {class_names}")
    print(f"  Train      : {len(train_dataset):4d} images")
    print(f"  Val        : {len(val_dataset):4d} images")
    print(f"  Test       : {len(test_dataset):4d} images\n")

    # ── Class distribution check ──────────────────────────────────────────────
    targets = np.array(train_dataset.targets)
    for idx, name in enumerate(class_names):
        count = (targets == idx).sum()
        print(f"    Train [{name}] : {count} images  ({count/len(targets)*100:.1f}%)")
    print()

    return train_loader, val_loader, test_loader, class_names


# =============================================================================
# 4. MODEL — MobileNetV3-Large (proposal Section 4.4.1)
#    ImageNet pre-trained weights; classifier head replaced for binary output.
# =============================================================================

def build_model(freeze_base: bool = True) -> nn.Module:
    """
    Constructs MobileNetV3-Large with a 2-class output head.

    Phase 1 (freeze_base=True) : Only the classifier trains.
                                  Prevents feature distortion on small data.
    Phase 2 (freeze_base=False): Last UNFREEZE_LAYERS feature sub-layers
                                  are unfrozen for domain adaptation.
    """
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')

    # ── Replace classifier head ───────────────────────────────────────────────
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)

    if freeze_base:
        # Freeze entire feature extractor — only classifier trains
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Phase 1 — frozen base  | Trainable params : {trainable:,}")
    else:
        # Unfreeze last N sub-layers of features for fine-tuning
        feature_children = list(model.features.children())
        for child in feature_children[:-UNFREEZE_LAYERS]:
            for param in child.parameters():
                param.requires_grad = False
        for child in feature_children[-UNFREEZE_LAYERS:]:
            for param in child.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Phase 2 — last {UNFREEZE_LAYERS} feature layers unfrozen"
              f" | Trainable params : {trainable:,}")

    return model.to(DEVICE)


# =============================================================================
# 5. TRAINING UTILITIES
#    Single epoch train / validate functions used by both phases.
# =============================================================================

def run_epoch(model, loader, criterion, optimizer=None, phase='train'):
    """
    Runs one full train or validation epoch.
    Returns (loss, accuracy, probability scores, true labels).
    """
    is_train = (phase == 'train')
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(outputs, dim=1)[:, ABNORMAL_CLS_IDX]   # P(abnormal)
            _, predicted = outputs.max(1)

            running_loss += loss.item()
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc  = 100.0 * correct / total

    return epoch_loss, epoch_acc, np.array(all_probs), np.array(all_labels)


# =============================================================================
# 6. TWO-PHASE TRAINING (proposal Section 4.4.2)
#    Phase 1 : Frozen convolutional base — trains classifier head only.
#    Phase 2 : Last UNFREEZE_LAYERS feature sub-layers unfrozen — fine-tunes
#              domain-relevant representations at reduced learning rate.
# =============================================================================

def train(train_loader, val_loader, skip_phase1: bool = False):
    """
    Runs the two-phase training pipeline.

    Returns model, h_frozen (or None if skipped), h_finetune.
    h_frozen / h_finetune are dicts with keys:
        train_loss, val_loss, train_acc, val_acc, val_auc
    """
    criterion = nn.CrossEntropyLoss()

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Frozen base
    # ─────────────────────────────────────────────────────────────────────────
    model = build_model(freeze_base=True)

    if skip_phase1:
        print("\n─── Phase 1: Skipped — loading saved checkpoint ───")
        print(f"  Loading weights from : {CHECKPOINT_PATH_P1}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH_P1, map_location=DEVICE))
        h_frozen = None
    else:
        print("\n─── Phase 1: Frozen base ───")
        print(f"  Max epochs     : {EPOCHS_FROZEN}")
        print(f"  Early stopping : patience={PATIENCE_STOP} on val ROC-AUC")
        print(f"  Checkpoint     : {CHECKPOINT_PATH_P1}\n")

        optimizer1  = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
        )
        scheduler1  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, mode='max', factor=0.5,
            patience=PATIENCE_LR, min_lr=1e-7, verbose=True,
        )

        h_frozen = _run_phase(
            model, train_loader, val_loader,
            criterion, optimizer1, scheduler1,
            EPOCHS_FROZEN, CHECKPOINT_PATH_P1,
        )

        print(f"\n  Reloading best Phase 1 weights from : {CHECKPOINT_PATH_P1}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH_P1, map_location=DEVICE))

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Fine-tune last UNFREEZE_LAYERS feature layers
    # ─────────────────────────────────────────────────────────────────────────
    print("\n─── Phase 2: Fine-tuning last {} feature layers ───".format(
        UNFREEZE_LAYERS))
    print(f"  Checkpoint : {CHECKPOINT_PATH_P2}\n")

    # Rebuild from checkpoint with frozen base then selectively unfreeze
    feature_children = list(model.features.children())
    for child in feature_children[:-UNFREEZE_LAYERS]:
        for param in child.parameters():
            param.requires_grad = False
    for child in feature_children[-UNFREEZE_LAYERS:]:
        for param in child.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {trainable:,}")
    print(f"  Learning rate    : {FINETUNE_LR} (Phase 1 was {LEARNING_RATE})\n")

    optimizer2  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
    )
    scheduler2  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='max', factor=0.5,
        patience=PATIENCE_LR, min_lr=1e-7, verbose=True,
    )

    h_finetune = _run_phase(
        model, train_loader, val_loader,
        criterion, optimizer2, scheduler2,
        EPOCHS_FINETUNE, CHECKPOINT_PATH_P2,
    )

    print(f"\n  Reloading best Phase 2 weights from : {CHECKPOINT_PATH_P2}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH_P2, map_location=DEVICE))

    return model, h_frozen, h_finetune


def _run_phase(model, train_loader, val_loader, criterion,
               optimizer, scheduler, max_epochs, ckpt_path):
    """
    Inner training loop for one phase. Implements early stopping on val_auc.
    Returns history dict.
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'val_auc':    [],
    }
    best_val_auc = 0.0
    no_improve   = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, _,        _      = run_epoch(
            model, train_loader, criterion, optimizer, phase='train')
        va_loss, va_acc, va_probs, va_lbl = run_epoch(
            model, val_loader,   criterion,             phase='val')

        # FIX 1: roc_auc_score has no pos_label parameter. Instead convert labels
        # to boolean (True=abnormal) so sklearn treats abnormal(0) as positive.
        try:
            va_auc = roc_auc_score(va_lbl == ABNORMAL_CLS_IDX, va_probs)
        except ValueError:
            va_auc = 0.0    # only one class present in tiny val batch

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_auc'].append(va_auc)

        scheduler.step(va_auc)
        elapsed = time.time() - t0

        print(f"  Epoch [{epoch:3d}/{max_epochs}]  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%  |  "
              f"Val Loss: {va_loss:.4f}  Acc: {va_acc:.2f}%  "
              f"AUC: {va_auc:.4f}  ({elapsed:.1f}s)")

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✅ Best model saved  (val AUC = {va_auc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE_STOP:
                print(f"\n  Early stopping triggered — no improvement for "
                      f"{PATIENCE_STOP} epochs.")
                break

    return history


# =============================================================================
# 7. TRAINING HISTORY PLOT
# =============================================================================

def plot_history(h_frozen, h_finetune) -> None:
    """
    Plots Loss, Accuracy, and Val ROC-AUC across both phases.
    A vertical dashed line marks the Phase 1 → Phase 2 boundary.
    """
    phase1_skipped = h_frozen is None
    metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc',  'val_acc',  'Accuracy (%)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, (tr_key, va_key, title) in zip(axes[:2], metrics):
        if phase1_skipped:
            ax.plot(h_finetune[tr_key], label='Train (Phase 2)')
            ax.plot(h_finetune[va_key], label='Val (Phase 2)')
        else:
            tr_vals = h_frozen[tr_key] + h_finetune[tr_key]
            va_vals = h_frozen[va_key] + h_finetune[va_key]
            split   = len(h_frozen[tr_key]) - 1
            ax.plot(tr_vals, label='Train')
            ax.plot(va_vals, label='Validation')
            ax.axvline(split, color='gray', linestyle='--',
                       alpha=0.7, label='Fine-tune start')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)

    # ── Val ROC-AUC subplot ───────────────────────────────────────────────────
    ax_auc = axes[2]
    if phase1_skipped:
        ax_auc.plot(h_finetune['val_auc'], label='Val AUC (Phase 2)')
    else:
        auc_vals = h_frozen['val_auc'] + h_finetune['val_auc']
        split    = len(h_frozen['val_auc']) - 1
        ax_auc.plot(auc_vals, label='Val ROC-AUC')
        ax_auc.axvline(split, color='gray', linestyle='--',
                       alpha=0.7, label='Fine-tune start')
    ax_auc.axhline(H1_AUC_THRESHOLD, color='red', linestyle=':',
                   alpha=0.8, label=f'H1 target ({H1_AUC_THRESHOLD})')
    ax_auc.set_title('Val ROC-AUC', fontsize=12)
    ax_auc.set_xlabel('Epoch')
    ax_auc.set_ylim(0, 1.05)
    ax_auc.legend(fontsize=8)

    title_str = (
        "MobileNetV3 — Training History (Phase 2 only)"
        if phase1_skipped else
        "MobileNetV3 — Training History (Phase 1 + 2)"
    )
    plt.suptitle(title_str, fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# 8. EVALUATION (proposal Section 4.4.3)
#    Reports ROC-AUC + AUPRC with bootstrapped 95% CIs (1,000 iterations).
#    Plots ROC curve, PR curve, and confusion matrix in a single figure.
# =============================================================================

def evaluate(model, test_loader, class_names) -> dict:
    """
    Runs inference on the test set and computes all metrics.
    Returns results dict consumed by main() for H1 verdict.
    """
    print("\n─── Test Set Evaluation ───")
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, ABNORMAL_CLS_IDX]   # P(abnormal)
            _, predicted = outputs.max(1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # FIX 2 & 3: roc_auc_score has no pos_label — convert labels to boolean
    # (True=abnormal) so sklearn treats abnormal(0) as the positive class.
    # average_precision_score does support pos_label so that is used there.
    roc_auc = roc_auc_score(y_true == ABNORMAL_CLS_IDX, y_prob)
    auprc   = average_precision_score(y_true, y_prob, pos_label=ABNORMAL_CLS_IDX)
    report  = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)

    print(f"\n  ROC-AUC : {roc_auc:.4f}  (H1 target ≥ {H1_AUC_THRESHOLD})")
    print(f"  AUPRC   : {auprc:.4f}  (imbalance-aware supplement)")
    print(f"\n{report}")

    # ── Bootstrapped 95% CI (1,000 iterations) ───────────────────────────────
    print("  Computing bootstrapped 95% CIs (1,000 iterations)...")
    rng = np.random.default_rng(SEED)
    boot_roc, boot_prc = [], []
    for _ in range(1000):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        # FIX 4: same label conversion for bootstrap iterations.
        boot_roc.append(roc_auc_score(y_true[idx] == ABNORMAL_CLS_IDX, y_prob[idx]))
        boot_prc.append(average_precision_score(y_true[idx], y_prob[idx], pos_label=ABNORMAL_CLS_IDX))

    roc_lo, roc_hi = np.percentile(boot_roc, [2.5, 97.5])
    prc_lo, prc_hi = np.percentile(boot_prc, [2.5, 97.5])
    print(f"  ROC-AUC 95% CI : [{roc_lo:.4f}, {roc_hi:.4f}]")
    print(f"  AUPRC   95% CI : [{prc_lo:.4f}, {prc_hi:.4f}]")

    # ── Plots: ROC | PR | Confusion Matrix ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # FIX 5: pos_label=ABNORMAL_CLS_IDX on roc_curve so the curve reflects
    # true positive rate for abnormal detection, not normal detection.
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=ABNORMAL_CLS_IDX)
    axes[0].plot(fpr, tpr,
                 label=f"MobileNetV3 (AUC={roc_auc:.3f})")
    axes[0].fill_between(fpr, tpr, alpha=0.1)
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # FIX 6: pos_label=ABNORMAL_CLS_IDX on precision_recall_curve.
    prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=ABNORMAL_CLS_IDX)
    axes[1].plot(rec, prec,
                 label=f"MobileNetV3 (AUPRC={auprc:.3f})")
    axes[1].fill_between(rec, prec, alpha=0.1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[2], cmap="Blues")
    axes[2].set_title("Confusion Matrix")

    plt.suptitle(
        "MobileNetV3 — Synthetic External Image Module Evaluation", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "evaluation.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Saved → {save_path}")

    return {
        "roc_auc": roc_auc, "roc_lo": roc_lo, "roc_hi": roc_hi,
        "auprc":   auprc,   "prc_lo": prc_lo, "prc_hi": prc_hi,
    }


# =============================================================================
# 9. GRAD-CAM (proposal Section 4.4.3)
#    Hook-based implementation for PyTorch (no GradientTape).
#    Targets the last Conv2d in model.features — found by reverse search.
# =============================================================================

class GradCAM:
    """
    Computes Grad-CAM heatmaps by registering forward and backward hooks
    on the deepest Conv2d layer in model.features.

    Usage:
        gcam = GradCAM(model)
        heatmap = gcam(img_tensor)   # img_tensor: (1, 3, H, W) on DEVICE
        gcam.remove()
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._handles    = []

        target_layer = self._find_last_conv(model.features)
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found in model.features.")

        self._handles.append(
            target_layer.register_forward_hook(self._save_activations))
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradients))

    @staticmethod
    def _find_last_conv(module: nn.Module):
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _save_activations(self, _, __, output):
        self.activations = output.detach()

    def _save_gradients(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE).requires_grad_(True)

        output = self.model(img_tensor)
        class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients → channel weights
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        heatmap  = (weights * self.activations).sum(dim=1).squeeze()
        heatmap  = torch.relu(heatmap)

        # Normalise to [0, 1]
        heatmap -= heatmap.min()
        denom    = heatmap.max()
        if denom > 1e-8:
            heatmap /= denom

        return heatmap.cpu().numpy()

    def remove(self):
        for h in self._handles:
            h.remove()


def overlay_gradcam(img_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Overlays a Grad-CAM heatmap (JET colormap) on a normalised image."""
    import cv2
    h = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    h = h[..., ::-1]   # BGR → RGB
    return np.uint8(img_np * 255 * (1 - alpha) + h * alpha)


def save_gradcam_grid(model, test_loader, class_names) -> None:
    """
    Saves a 2×4 grid of Grad-CAM overlays — 2 normal, 2 abnormal samples.
    Row 0: original images. Row 1: Grad-CAM overlays.
    """
    abnormal_cls = ABNORMAL_CLS_IDX
    normal_cls   = NORMAL_CLS_IDX

    normal_imgs, abnormal_imgs = [], []
    model.eval()
    gcam = GradCAM(model)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denormalise(t):
        img = t.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        return np.clip(img, 0, 1)

    for images, labels in test_loader:
        for img_t, lbl in zip(images, labels):
            if lbl.item() == normal_cls   and len(normal_imgs)   < 2:
                normal_imgs.append((img_t, lbl.item()))
            if lbl.item() == abnormal_cls and len(abnormal_imgs) < 2:
                abnormal_imgs.append((img_t, lbl.item()))
        if len(normal_imgs) == 2 and len(abnormal_imgs) == 2:
            break

    # col 1-2: normal  |  col 3-4: abnormal  — matches subtitle
    samples = normal_imgs + abnormal_imgs

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (img_t, true_label) in enumerate(samples):
        img_np = denormalise(img_t)
        inp    = img_t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp)
            prob = torch.softmax(out, dim=1)[0, abnormal_cls].item()  # P(abnormal)

        # FIX 7: map prob threshold back to the correct class index.
        # int(prob >= 0.5) always returns 0 or 1, but abnormal=0 and normal=1,
        # so the tick mark was inverted for every sample.
        pred_label = ABNORMAL_CLS_IDX if prob >= 0.5 else NORMAL_CLS_IDX
        tick       = "✓" if pred_label == true_label else "✗"
        label_str  = class_names[true_label]

        try:
            hm = gcam(img_t.unsqueeze(0))
            ov = overlay_gradcam(img_np, hm)

            axes[0, col].imshow(img_np)
            axes[0, col].set_title(f"Original\n({label_str})", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(ov)
            axes[1, col].set_title(
                f"Grad-CAM  {tick}\nP(abnormal)={prob:.2f}", fontsize=9)
            axes[1, col].axis("off")

        except Exception as e:
            print(f"  Grad-CAM failed for sample (col {col}): {e}")

    gcam.remove()
    plt.suptitle(
        "Grad-CAM — Synthetic External Image Module  "
        "(col 1-2: normal, col 3-4: abnormal)",
        fontsize=12,
    )
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "gradcam.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# 10. INT8 QUANTIZATION + TORCHSCRIPT EXPORT (proposal Section 4.5.1)
# =============================================================================

TORCHSCRIPT_PATH   = os.path.join(OUTPUT_DIR, "ext_mobilenetv3_int8.ptl")
ONNX_PATH          = os.path.join(OUTPUT_DIR, "ext_mobilenetv3.onnx")
SIZE_THRESHOLD_MB  = 15.0
LATENCY_TARGET_MS  = 100.0
ACC_DROP_THRESHOLD = 2.0


def quantize_and_export(model: nn.Module, test_loader, class_names) -> tuple:
    """
    Applies INT8 dynamic quantization to the classifier head, exports to
    TorchScript (.ptl) and ONNX (.onnx), then benchmarks size, latency,
    and accuracy drop against the float32 baseline.

    Returns (size_mb, mean_ms).
    """
    print("\n─── INT8 Quantization + TorchScript Export ───")

    model_cpu = model.to('cpu')
    model_cpu.eval()

    quant_model = torch.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    print("  ✓ INT8 dynamic quantization applied  (Linear layers)")

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        scripted = torch.jit.script(quant_model)
        scripted._save_for_lite_interpreter(TORCHSCRIPT_PATH)
        print(f"  ✓ TorchScript (.ptl) saved  → {TORCHSCRIPT_PATH}")
    except Exception as e:
        print(f"  ⚠ TorchScript export failed ({e})")
        print(f"    Falling back to torch.jit.trace ...")
        traced = torch.jit.trace(quant_model, dummy_input)
        traced._save_for_lite_interpreter(TORCHSCRIPT_PATH)
        print(f"  ✓ Traced TorchScript (.ptl) saved  → {TORCHSCRIPT_PATH}")

    size_mb = os.path.getsize(TORCHSCRIPT_PATH) / (1024 ** 2)
    print(f"\n  Size   : {size_mb:.2f} MB  (target ≤ {SIZE_THRESHOLD_MB} MB)")
    print(f"  {'✓ Size target MET' if size_mb <= SIZE_THRESHOLD_MB else '⚠ Exceeds 15 MB'}")

    print("\n─── ONNX Export (optional cross-framework path) ───")
    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}},
        )
        onnx_mb = os.path.getsize(ONNX_PATH) / (1024 ** 2)
        print(f"  ✓ ONNX saved  → {ONNX_PATH}  ({onnx_mb:.2f} MB)")
        print(f"  To convert to TFLite: pip install onnx2tf && onnx2tf -i {ONNX_PATH}")
    except Exception as e:
        print(f"  ⚠ ONNX export failed: {e}")

    print("\n─── Latency Benchmark (CPU inference, 50 runs) ───")
    print("  Note: H1 target ≤100ms is for Android ARM — re-test on device.")

    lite_module = torch.jit.load(TORCHSCRIPT_PATH)
    lite_module.eval()
    warmup_inp = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    for _ in range(5):
        with torch.no_grad():
            _ = lite_module(warmup_inp)

    run_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = lite_module(warmup_inp)
        run_times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(run_times))
    p95_ms  = float(np.percentile(run_times, 95))
    print(f"  Mean : {mean_ms:.1f} ms")
    print(f"  P95  : {p95_ms:.1f} ms")
    print(f"  {'✓ Under 100ms on CPU too' if mean_ms <= LATENCY_TARGET_MS else '⚠ Exceeds 100ms on CPU — re-test on Android'}")

    print("\n─── Accuracy Drop Check (≤ 2% threshold) ───")

    def _get_preds(m, loader, device='cpu'):
        m.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                out  = m(imgs.to(device))
                pred = out.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                labels_all.extend(lbls.numpy())
        return np.array(preds), np.array(labels_all)

    test_dataset_cpu = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'test'), transform=eval_transforms)
    test_loader_cpu  = DataLoader(
        test_dataset_cpu, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    float_preds,  y_true = _get_preds(model_cpu,    test_loader_cpu)
    quant_preds,  _      = _get_preds(quant_model,  test_loader_cpu)

    float_acc = np.mean(float_preds == y_true) * 100
    quant_acc = np.mean(quant_preds == y_true) * 100
    drop      = float_acc - quant_acc

    print(f"  Float32 accuracy   : {float_acc:.2f}%")
    print(f"  INT8 accuracy      : {quant_acc:.2f}%")
    print(f"  Drop               : {drop:.2f}%  (threshold ≤ {ACC_DROP_THRESHOLD}%)")
    print(f"  {'✓ Within tolerance' if drop <= ACC_DROP_THRESHOLD else '⚠ Exceeds — consider QAT'}")

    model.to(DEVICE)

    return size_mb, mean_ms


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================

def main():
    print("─── Step 1 / 5 : Loading dataset ───")
    train_loader, val_loader, test_loader, class_names = load_datasets()

    print("─── Step 2 / 5 : Training ───")
    model, h_frozen, h_finetune = train(train_loader, val_loader,
                                        skip_phase1=False)
    plot_history(h_frozen, h_finetune)

    print("\n─── Step 3 / 5 : Evaluation ───")
    results = evaluate(model, test_loader, class_names)

    print("\n─── Grad-CAM visualisations ───")
    try:
        import cv2
        save_gradcam_grid(model, test_loader, class_names)
    except ImportError:
        print("  opencv-python not installed — skipping Grad-CAM.")
        print("  Install with: pip install opencv-python")

    print("\n─── Step 4 / 5 : Quantization + Export ───")
    size_mb, mean_ms = quantize_and_export(model, test_loader, class_names)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n─── Step 5 / 5 : Summary ───")
    h1_auc     = results["roc_auc"] >= H1_AUC_THRESHOLD
    h1_size    = size_mb  <= SIZE_THRESHOLD_MB
    h1_latency = mean_ms  <= LATENCY_TARGET_MS
    h1_all     = h1_auc and h1_size and h1_latency

    print(f"\n{'='*62}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*62}")
    print(f"  ROC-AUC  : {results['roc_auc']:.4f}  "
          f"95% CI [{results['roc_lo']:.4f}, {results['roc_hi']:.4f}]")
    print(f"  AUPRC    : {results['auprc']:.4f}  "
          f"95% CI [{results['prc_lo']:.4f}, {results['prc_hi']:.4f}]")
    print(f"  Size     : {size_mb:.2f} MB")
    print(f"  Latency  : {mean_ms:.1f} ms  (re-test on Android for H1 check)")
    print()
    print("  Hypothesis H1 targets:")
    print(f"    ROC-AUC ≥ {H1_AUC_THRESHOLD}  → "
          f"{'✓ SUPPORTED' if h1_auc else '✗ NOT MET'}")
    print(f"    Size    ≤ {SIZE_THRESHOLD_MB} MB → "
          f"{'✓ SUPPORTED' if h1_size else '✗ NOT MET'}")
    print(f"    Latency ≤ {LATENCY_TARGET_MS} ms → "
          f"{'✓ SUPPORTED' if h1_latency else '✗ NOT MET (CPU — re-test on Android)'}")
    print()
    if not h1_auc:
        lo, hi = results['roc_lo'], results['roc_hi']
        if hi >= H1_AUC_THRESHOLD:
            print(f"  ⚠  Point estimate below threshold, but 95% CI upper bound")
            print(f"     [{lo:.4f}, {hi:.4f}] overlaps ≥{H1_AUC_THRESHOLD}.")
            print(f"     Interpret as statistically inconclusive given n=150 test set.")
        else:
            print(f"  ⚠  Both point estimate and CI fall below threshold.")
            print(f"     Address in domain shift discussion (Section 5.x).")
    print()
    print(f"  Overall H1 verdict : "
          f"{'✓ SUPPORTED' if h1_all else '✗ NOT FULLY MET'}")
    print()
    print("  Output files:")
    print(f"    {CHECKPOINT_PATH_P1}")
    print(f"    {CHECKPOINT_PATH_P2}")
    print(f"    {TORCHSCRIPT_PATH}")
    print(f"    {ONNX_PATH}")
    print(f"    {PLOT_DIR}{os.sep}training_history.png")
    print(f"    {PLOT_DIR}{os.sep}evaluation.png")
    print(f"    {PLOT_DIR}{os.sep}gradcam.png")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
